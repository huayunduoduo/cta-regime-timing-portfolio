"""Long/cash ETF allocation research. Close decisions, next-open execution.
No fitted AR-GARCH dependency. Raw-price inventory + dated cash entitlements.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import hashlib
import html
import json
import platform
import sys
import numpy as np
import pandas as pd
from scipy.special import ndtri

ROOT = Path(__file__).resolve().parent
CODES = ('510300', '518880', '511010')

@dataclass(frozen=True)
class Config:
    trend: str = 'none'
    allocation: str = 'equal'
    frequency: str = 'monthly'
    es_budget: float = 0.0

    @property
    def name(self):
        return f'{self.trend}_{self.allocation}_{self.frequency}_es{self.es_budget:g}'


def load_data(directory):
    d = pd.read_csv(directory/'etf_daily.csv', dtype={'code': str}, parse_dates=['date'])
    div = pd.read_csv(directory/'etf_dividend.csv', dtype={'code': str}, parse_dates=['ex_date', 'pay_date'])
    parts = {}
    for code in CODES:
        g = d[d.code == code].sort_values('date').set_index('date')
        if g.empty or g.index.duplicated().any():
            raise ValueError(f'{code}: empty history or duplicate dates')
        p = g[['open','high','low','close','adj_factor']].to_numpy()
        if not np.isfinite(p).all() or (p <= 0).any():
            raise ValueError(f'{code}: invalid prices')
        if (g.high < g[['open','low','close']].max(axis=1)).any() or (g.low > g[['open','high','close']].min(axis=1)).any():
            raise ValueError(f'{code}: invalid OHLC')
        parts[code] = g
    # Do not silently drop a missing bar inside the overlapping history.
    lower = max(g.index.min() for g in parts.values())
    upper = min(g.index.max() for g in parts.values())
    dates = parts[CODES[0]].loc[lower:upper].index
    for code, g in parts.items():
        if not g.loc[lower:upper].index.equals(dates):
            raise ValueError(f'{code}: trading calendars differ; repair data first')
    fields = {col: np.column_stack([parts[c].loc[dates, col].to_numpy(float) for c in CODES])
              for col in ('open','close','adj_factor')}
    if div.duplicated(['code','ex_date','pay_date']).any():
        raise ValueError('Duplicate dividend events')
    if ((div.pay_date < div.ex_date) | (div.cash_dividend_per_unit < 0)).any():
        raise ValueError('Invalid dividend event')
    return dates, fields, div


def capped_weights(raw, cap=.60):
    """Long-only fully allocated base weights; 60% single-asset cap."""
    raw = np.maximum(np.asarray(raw, float), 0)
    if raw.sum() <= 0:
        return np.zeros(len(raw))
    w = np.zeros(len(raw)); free = np.ones(len(raw), dtype=bool)
    for _ in range(len(raw)+1):
        left = 1 - w.sum()
        proposal = raw[free] / raw[free].sum() * left
        indices = np.flatnonzero(free)
        over = proposal > cap + 1e-12
        if not over.any():
            w[indices] = proposal; break
        w[indices[over]] = cap; free[indices[over]] = False
    return w


def risk_parity(cov):
    # Equal risk contributions on the positive simplex, then apply the same cap.
    cov = np.asarray(cov) + np.eye(3)*1e-10
    x = 1 / np.sqrt(np.diag(cov))
    x /= np.sqrt(x @ cov @ x)
    for _ in range(60):
        old = x.copy()
        for j in range(3):
            b = cov[j] @ x - cov[j,j]*x[j]
            x[j] = (-b + np.sqrt(b*b + 4*cov[j,j]/3)) / (2*cov[j,j])
        if np.max(np.abs(x-old)) < 1e-9:
            break
    return capped_weights(x)


def historical_es(loss, alpha=.975):
    # Fractional order statistic tail: exact empirical mass (1-alpha)*n.
    ordered = np.sort(np.asarray(loss, float))[::-1]
    mass = (1-alpha)*len(ordered)
    if mass <= 0:
        return np.nan
    k = int(np.floor(mass)); fraction = mass-k
    total = ordered[:k].sum()
    if fraction > 1e-12:
        total += fraction*ordered[min(k,len(ordered)-1)]
    return max(float(total/mass), 0.)


def features(dates, fields):
    close = pd.DataFrame(fields['close']*fields['adj_factor'], index=dates)
    ret = close.pct_change(fill_method=None)
    ma120 = (close > close.rolling(120).mean()).astype(float)
    ensemble = sum((close > close.rolling(n).mean()).astype(float) for n in (60,120,240))/3
    # No full-window => no signal, including for the shorter ensemble components.
    ma120[close.rolling(120).count() < 120] = 0.
    ensemble[close.rolling(240).count() < 240] = 0.
    return ret.to_numpy(), {'none': np.ones_like(fields['close']),
                            'ma120': ma120.to_numpy(), 'ensemble': ensemble.to_numpy()}


def schedule(dates, frequency):
    # First observed session in a new period is a decision close. No future holiday calendar.
    periods = dates.to_period('W-FRI' if frequency == 'weekly' else 'M')
    return np.r_[True, periods[1:] != periods[:-1]]


def build_targets(dates, fields, configs):
    returns, signal = features(dates, fields)
    targets = {c.name: np.zeros_like(fields['close']) for c in configs}
    flags = {f: schedule(dates,f) for f in ('monthly','weekly')}
    vol = pd.DataFrame(returns).rolling(60).std().to_numpy()
    bases = {}
    for i in range(len(dates)):
        if i < 252:
            continue
        sample = returns[i-251:i+1]
        # Only today's and earlier closes enter covariance / joint ES scenarios.
        if not np.isfinite(sample).all():
            continue
        if flags['weekly'][i] or flags['monthly'][i]:
            cov = np.cov(returns[i-119:i+1], rowvar=False)
            bases[i] = {'equal': np.ones(3)/3,
                        'invvol': capped_weights(1/np.maximum(vol[i], 1e-5)),
                        'riskparity': risk_parity(cov)}
        for c in configs:
            arr = targets[c.name]
            if not flags[c.frequency][i]:
                arr[i] = arr[i-1]; continue
            w = bases[i][c.allocation] * signal[c.trend][i]
            # Inactive sleeves remain cash. Never redistribute after gating.
            es = historical_es(-(sample @ w))
            scale = min(1., c.es_budget/es) if c.es_budget and es > 0 else 1.
            arr[i] = w*scale
    return targets, flags, returns


def simulate(dates, fields, div, target, decisions, start='2015-01-01', end='2025-12-31', fee=.001, slip=.001):
    """No leverage/shorts; target weights vs NAV including unpaid dividends.
    Sell before buy. Receivables cannot finance buys; fractional inventory assumed.
    """
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    indices = np.flatnonzero(mask)
    if not len(indices):
        raise ValueError('Empty backtest interval')
    if dates.max() < pd.Timestamp(end):
        raise ValueError(f'Data end {dates.max().date()} before --end {end}')
    if not np.isfinite(target).all() or (target < -1e-12).any() or (target.sum(axis=1) > 1+1e-10).any():
        raise ValueError('Invalid target weights')
    ex = {}
    for r in div.itertuples():
        if str(r.code) in CODES:
            ex.setdefault(r.ex_date, []).append((CODES.index(str(r.code)), r.pay_date, r.cash_dividend_per_unit))
    cash = 1.; qty = np.zeros(3); pending = []; records = []; fills = []
    for i in indices:
        date = dates[i]; op = fields['open'][i]; cl = fields['close'][i]
        old_qty=qty.copy(); dividend_pnl=np.zeros(3); costs_by_asset=np.zeros(3)
        prior_nav=cash+qty@(fields['close'][i-1] if i>0 else op)+sum(x[1] for x in pending)
        for j, pay, amount in ex.get(date, []):
            pending.append((pay, float(qty[j]*amount)))
            dividend_pnl[j]+=qty[j]*amount
        unpaid = []
        for pay, amount in pending:
            if pay <= date:
                cash += amount
            else:
                unpaid.append((pay,amount))
        pending = unpaid
        receivable = sum(x[1] for x in pending)
        nav_open = cash + qty @ op + receivable
        cost = turnover = 0.
        rebalance = i > 0 and (i == indices[0] or decisions[i-1])
        if rebalance:
            delta = target[i-1]*nav_open/op - qty
            # Ignore changes below 1bp NAV to avoid floating point micro-orders.
            delta[np.abs(delta*op)/nav_open < .0001] = 0
            for j in np.flatnonzero(delta < 0):
                amount = -delta[j]; px = op[j]*(1-slip); comm = amount*px*fee
                cash += amount*px-comm; qty[j] -= amount
                costs_by_asset[j]+=amount*op[j]*slip+comm
                cost += amount*op[j]*slip+comm; turnover += amount*op[j]/nav_open
                fills.append((date,CODES[j],'sell',amount,op[j],px,comm))
            buys = np.maximum(delta,0)
            need = np.sum(buys*op*(1+slip)*(1+fee))
            if need > 0:
                buys *= min(1., max(cash,0)/need)
            for j in np.flatnonzero(buys > 0):
                amount = buys[j]; px = op[j]*(1+slip); comm = amount*px*fee
                cash -= amount*px+comm; qty[j] += amount
                costs_by_asset[j]+=amount*op[j]*slip+comm
                cost += amount*op[j]*slip+comm; turnover += amount*op[j]/nav_open
                fills.append((date,CODES[j],'buy',amount,op[j],px,comm))
        nav = cash + qty @ cl + receivable
        assert cash > -1e-10 and (qty >= -1e-10).all() and nav > 0
        row = dict(date=date,nav=nav,cash=cash,receivable=receivable,turnover=turnover,cost=cost,
                   identity_error=abs(nav-(cash+qty@cl+receivable)))
        price_pnl=old_qty*(op-(fields['close'][i-1] if i>0 else op))+qty*(cl-op)
        net_pnl=price_pnl+dividend_pnl-costs_by_asset
        row['pnl_reconstruction_error']=abs(net_pnl.sum()-(nav-prior_nav))
        assert row['pnl_reconstruction_error']<1e-10
        for j,c in enumerate(CODES):
            row[f'price_pnl_{c}']=price_pnl[j]
            row[f'dividend_pnl_{c}']=dividend_pnl[j]
            row[f'cost_{c}']=costs_by_asset[j]
            row[f'net_pnl_{c}']=net_pnl[j]
        row.update({f'quantity_{c}':qty[j] for j,c in enumerate(CODES)})
        row.update({f'weight_{c}':qty[j]*cl[j]/nav for j,c in enumerate(CODES)})
        records.append(row)
    return pd.DataFrame(records), pd.DataFrame(fills,columns=['date','code','side','quantity','reference_price','fill_price','commission'])


def stats(nav, start=None, end=None):
    selected = nav
    if start is not None:
        selected = selected[selected.date >= pd.Timestamp(start)]
    if end is not None:
        selected = selected[selected.date <= pd.Timestamp(end)]
    if len(selected) < 2:
        raise ValueError('Insufficient evaluation rows')
    prior = nav[nav.date < selected.date.iloc[0]]
    base = prior.nav.iloc[-1] if len(prior) else 1.
    v = np.r_[base,selected.nav.to_numpy()]/base
    ret = v[1:]/v[:-1]-1
    n = len(ret); ann = v[-1]**(252/n)-1
    dd = np.min(v/np.maximum.accumulate(v)-1)
    sd = np.std(ret,ddof=1)*np.sqrt(252)
    return dict(cagr=ann,max_drawdown=dd,calmar=ann/abs(dd) if dd else 0.,
                volatility=sd,sharpe=np.mean(ret)*252/sd if sd else 0.,
                annual_turnover=selected.turnover.sum()*252/n,
                cost_initial_capital=selected.cost.sum()/base,days=n,total_return=v[-1]-1)


def rank_candidates(nav_by_name, cutoff):
    """Fixed objective: return/drawdown (6% denominator floor) minus churn penalty.
    Years through cutoff only. A floor avoids selecting essentially-cash accounts.
    """
    rows = []
    for name, nav in nav_by_name.items():
        m = stats(nav,'2015-01-01',cutoff)
        score = m['cagr']/max(.06,abs(m['max_drawdown'])) - .02*m['annual_turnover']
        rows.append(dict(candidate=name,score=score,**m))
    return pd.DataFrame(rows).sort_values(['score','candidate'],ascending=[False,True])


def select_top3(rank, byname):
    # ES on/off variants of the same base rule must not count as two ensemble members.
    chosen = []; families = set()
    for name in rank.candidate:
        c = byname[name]; family = (c.trend,c.allocation,c.frequency)
        if family not in families:
            chosen.append(name); families.add(family)
        if len(chosen) == 3: break
    return chosen


def risk_forecasts(nav):
    """One-day, pre-return forecasts. Joint-path strategy P&L history, no future fit."""
    # copy=True: pandas>=3 / CoW returns a read-only view, and we overwrite r[0] below
    r = nav.nav.pct_change().to_numpy(copy=True); r[0] = nav.nav.iloc[0]-1
    rows = []
    z = ndtri(.975); phi = np.exp(-z*z/2)/np.sqrt(2*np.pi)
    for i in range(252,len(nav)):
        past = r[i-252:i]
        ew = .94**np.arange(251,-1,-1); ew /= ew.sum()
        sigma = np.sqrt(np.sum(ew*past**2))
        # Filter each past observation by its *past* 60-day realized volatility.
        local = np.array([np.std(r[max(0,k-60):k],ddof=1) if k>=20 else np.nan for k in range(i-252,i)])
        standardized = past[np.isfinite(local)] / np.maximum(local[np.isfinite(local)],1e-5)
        for method, loss in [('historical',-past),('filtered',-standardized*sigma)]:
            var = max(float(np.quantile(loss,.975)),0.); es = historical_es(loss)
            rows.append(dict(date=nav.date.iloc[i],method=method,var=var,es=es,realized_loss=-r[i],breach=bool(-r[i]>var)))
        rows.append(dict(date=nav.date.iloc[i],method='ewma_normal',var=z*sigma,es=sigma*phi/.025,realized_loss=-r[i],breach=bool(-r[i]>z*sigma)))
    return pd.DataFrame(rows)


def build_report(out, summary, yearly, selections, final_rank, manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    full_label = f"2020-{manifest['end'][:4]}"
    names = ['walkforward_top3','fixed_train_top3','benchmark_equal_monthly','benchmark_invvol_monthly','cta_original_equal','cta_improved_equal']
    fig, axes = plt.subplots(2,1,figsize=(11,7),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    for name in names:
        path = out/f'{name}_nav.csv'
        if not path.exists():
            continue
        d = pd.read_csv(path,parse_dates=['date']); d = d[d.date>='2020-01-01']
        # All saved evaluated ledgers have a fresh 2020 account.
        v = np.r_[1.,d.nav.to_numpy()]
        axes[0].plot(d.date,d.nav,label=name,lw=1.4)
        axes[1].plot(d.date,(v/np.maximum.accumulate(v)-1)[1:]*100,lw=1.)
    axes[0].set_ylabel('Net NAV'); axes[1].set_ylabel('Drawdown %')
    axes[0].legend(fontsize=8,ncol=2); axes[0].grid(alpha=.2); axes[1].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/'comparison.png',dpi=150); plt.close(fig)
    import base64
    picture=base64.b64encode((out/'comparison.png').read_bytes()).decode()
    def table(df):
        df=df.copy()
        for c in ['cagr','max_drawdown','volatility','total_return','calendar_return','mean_weight','price_pnl','dividend_pnl','cost','net_pnl']:
            if c in df:
                df[c]=df[c].map(lambda x:f'{x:.2%}')
        for c in ['calmar','sharpe','annual_turnover','score']:
            if c in df:
                df[c]=df[c].map(lambda x:f'{x:.3f}')
        labels={'walkforward_top3':'年度滚动前三组合','fixed_train_top3':'2019年底固定前三组合','fixed_train_best':'2019年底固定第一名','benchmark_equal_monthly':'月度等权配置','benchmark_invvol_monthly':'月度逆波动配置（主方案）','trend_ensemble_equal':'三均线趋势＋等权','trend_ensemble_invvol':'三均线趋势＋逆波动','trend_ensemble_invvol_es':'三均线趋势＋逆波动＋ES','cta_original_equal':'原多空CTA等分初始资金','cta_improved_equal':'no_rho_3多头CTA等分初始资金'}
        if 'strategy' in df:df['strategy']=df.strategy.map(lambda x:labels.get(x,x))
        df=df.rename(columns={'strategy':'方案','period':'期间','cagr':'年化收益','max_drawdown':'最大回撤','calmar':'卡玛比','sharpe':'夏普比','annual_turnover':'年化单边换手','year':'年份','calendar_return':'当年实际收益','code':'ETF代码','mean_weight':'平均权重','price_pnl':'价格贡献','dividend_pnl':'分红贡献','cost':'成本贡献','net_pnl':'净贡献','cutoff':'选择数据截至','first':'第一成员','second':'第二成员','third':'第三成员','cost_multiplier':'成本倍数'})
        return df.to_html(index=False,escape=True,border=0)
    def section(title, text):
        return f'<h2>{title}</h2>{text}'
    body = f'<h1>股债金策略改进：实际回测与滚动选择</h1><p>期间：2020-01-01 至 {html.escape(manifest["end"])}；按252个交易日年化；多头/现金，无杠杆。所有收益已扣每次成交的佣金1‰及滑点1‰；现金收益为0。</p>'
    body += '<p class="note">推荐的是预先固定选择算法的 walkforward_top3，不是事后收益最高的单一参数。2020—2025在此前研究中已被查看，因此此处只称时间顺序检验，不宣称全新样本外或未来最优。宏观数据没有加入；未使用发布日期不明的数据。</p>'
    body += '<p><b>本轮结论：</b>以收益和回撤平衡、规则简洁为目标，优先保留月度逆波动配置（benchmark_invvol_monthly）作为后续验证主方案；它是低换手股债金配置，不能称作研报CTA复现成功。复杂趋势、滚动选参和ES没有显示稳定增量。追求更高历史收益可比较等权，但承担更大回撤。</p>'
    body += section('1. 主要结果',table(summary[summary.period==full_label][['strategy','cagr','max_drawdown','calmar','sharpe','annual_turnover']]))
    body += f'<img alt="Net NAV and drawdown" src="data:image/png;base64,{picture}">'
    attrib=pd.read_csv(out/'attribution.csv')
    body += section('2. 主要方案的收益来源', '<p>net_pnl是对初始资金的累计贡献，三项相加等于组合累计收益。mean_weight为每日实际持仓权重均值，允许调仓之间漂移。</p>'+table(attrib[attrib.strategy.isin(['benchmark_invvol_monthly','benchmark_equal_monthly','walkforward_top3'])]))
    body += section('3. 按阶段检验',table(summary[summary.period!=full_label][['strategy','period','cagr','max_drawdown','calmar']]))
    body += section('4. 逐年收益与风险',table(yearly[['strategy','year','total_return','max_drawdown','calmar']].rename(columns={'total_return':'calendar_return'})))
    body += section('5. 每年实际选择的三个方案',table(selections))
    body += section('6. 截至数据末日的下一期配置规则', '<p>滚动组合的部署规则见 deployment.json。这三个方案由截至末日的历史评分选出；不能把它们的回溯表现当成独立验证。实用主方案仍为月度逆波动配置。</p>'+table(final_rank[final_rank.candidate.isin([c['name'] for c in json.loads((out/'deployment.json').read_text(encoding='utf-8'))['next_period_top3']])][['candidate','score','cagr','max_drawdown','annual_turnover']]))
    stress=pd.read_csv(out/'cost_stress.csv')
    body += section('7. 成本压力测试',table(stress[['strategy','cost_multiplier','cagr','max_drawdown','calmar']]))
    body += section('8. ES与单纯降仓对照', '<p>本轮0.015的一日ES预算没有改变三均线逆波动组合的实际路径；固定降仓比例只用2015—2019平均目标仓位推算。结果相同表示此处没有观察到ES增量。</p>'+table(pd.read_csv(out/'es_ablation.csv')))
    coverage=pd.read_csv(out/'risk_coverage.csv')
    body += section('9. 一日VaR/ES诊断', '<p>97.5% VaR理论超限率2.5%。仅以252个历史交易日预测下一日；ES超限日损失均值只是描述性诊断，不能当作ES校准通过。ES预算限制模型估计风险，不保证实际损失上限。</p>'+table(coverage))
    body += section('10. 方法和边界', '<ul><li>趋势：无过滤、收盘价&gt;120日均线、60/120/240日三信号均值；失效的份额留现金。</li><li>配置：等权、60日逆波动、120日协方差风险平价。单品种目标上限60%；不借钱。</li><li>周/月首个已观察交易日收盘决策，下一交易日开盘执行，权重真实漂移；ES使用过去252日的联合冲击。</li><li>滚动选择：每年初用2015年至上年底评分，前3个不同基础规则等权混合（同规则ES开关不能重复占席）；仅在成员需要调仓时执行。</li><li>选择分数=CAGR/max(6%,|最大回撤|)−0.02×年化单边换手；按名字打破同分。36个候选和评分规则均见源码。</li><li>账户用未复权价格成交，除息日确认应收股息，派息日入现金；复权价格只用于信号，不重复计入收益。</li><li>原CTA等权为3个独立账户各1/3初始资金、不再平衡，属于历史原规则对照，包含理想化空头。改进CTA用no_rho_3，各1/3独立资金。新增组合可实际多头持有。</li><li>假定可按小数份额成交，不计最低佣金、整手约束、税收差异或冲击；这是单位资金研究账本，实盘需按资金量接整手执行。</li><li>没有分钟级止损，也没有将研报的ATR执行与新组合机制混称同一个策略。所有失败候选保留。</li></ul>')
    body += '<p>完整候选结果、每日数量/现金/应收股息、成交、选参审计、风险预测和数据哈希均在本目录CSV/JSON中。</p>'
    css='body{max-width:1250px;margin:35px auto;padding:0 20px;font:16px/1.65 system-ui;color:#172230}table{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow:auto}th,td{padding:7px 10px;border-bottom:1px solid #dde3e9;white-space:nowrap;text-align:right}th{background:#edf2f7}h1,h2{color:#14385c}img{width:100%}.note{background:#fff4cf;padding:14px}h2{margin-top:35px}'
    (out/'portfolio_report.html').write_text(f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>股债金策略改进</title><style>{css}</style><body>{body}</body></html>',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=ROOT/'data')
    ap.add_argument('--out',type=Path,default=ROOT/'results'/'portfolio')
    ap.add_argument('--end',default='2025-12-31')
    args=ap.parse_args(); full_label=f'2020-{args.end[:4]}'; later_label=f'2023-{args.end[:4]}'; out=args.out; out.mkdir(parents=True,exist_ok=True)
    dates,fields,div=load_data(args.data)
    # Truncate before generating any features/rankings. Rerunning on a prefix is causal.
    keep=dates<=pd.Timestamp(args.end); dates=dates[keep];fields={k:v[keep] for k,v in fields.items()}
    if args.end < '2020-12-31':
        raise ValueError('--end must include at least 2020')
    configs=[Config(t,a,f,b) for t in ('none','ma120','ensemble') for a in ('equal','invvol','riskparity') for f in ('monthly','weekly') for b in (0.,.015)]
    print(f'1/6 Constructing {len(configs)} causal candidates...',flush=True)
    targets,flags,returns=build_targets(dates,fields,configs)
    navs={}; candidate_rows=[]
    for n,c in enumerate(configs,1):
        nav,_=simulate(dates,fields,div,targets[c.name],flags[c.frequency],end=args.end)
        navs[c.name]=nav
        for period,lo,hi in [('train_2015-2019','2015-01-01','2019-12-31'),('validation_2020-2022','2020-01-01','2022-12-31'),('later_2023-2025','2023-01-01',args.end),(full_label,'2020-01-01',args.end)]:
            if pd.Timestamp(lo)<=nav.date.max():
                candidate_rows.append(dict(strategy=c.name,period=period,**stats(nav,lo,min(hi,args.end))))
        nav.to_csv(out/f'candidate_{c.name}_nav.csv',index=False)
        if n%6==0: print(f'  candidates {n}/{len(configs)}',flush=True)
    pd.DataFrame(candidate_rows).to_csv(out/'all_candidates.csv',index=False)
    byname={c.name:c for c in configs}; select=[]; rank_audit=[]
    dynamic=np.zeros_like(fields['close']); dynamic_flags=np.zeros(len(dates),bool)
    fixed=np.zeros_like(dynamic); fixed_flags=np.zeros(len(dates),bool)
    initial=select_top3(rank_candidates(navs,'2019-12-31'),byname)
    # Fixed top3 learned before 2020; internal sleeve targets are mixed with no extra redistributions.
    for name in initial:
        fixed+=targets[name]/3;fixed_flags|=flags[byname[name].frequency]
    last_year=min(int(args.end[:4])+1,int(dates.max().year)+1)
    for year in range(2020,last_year+1):
        cutoff=f'{year-1}-12-31';rank=rank_candidates(navs,cutoff)
        rank.insert(0,'selection_year',year); rank['cutoff']=cutoff; rank_audit.append(rank)
        top=select_top3(rank,byname)
        select.append(dict(year=year,cutoff=cutoff,first=top[0],second=top[1],third=top[2]))
        # target at previous year's final close can execute at the first new-year open.
        effective_year=np.r_[dates[1:].year,dates[-1].year+int(dates[-1].month==12 and dates[-1].day>=28)]
        mask=effective_year==year
        for name in top:
            dynamic[mask]+=targets[name][mask]/3
            dynamic_flags[mask]|=flags[byname[name].frequency][mask]
        previous=np.flatnonzero(mask)
        if len(previous):dynamic_flags[previous[0]]=True
    selection=pd.DataFrame(select);selection.to_csv(out/'selections.csv',index=False)
    pd.concat(rank_audit).to_csv(out/'selection_audit.csv',index=False)
    print('2/6 Running continuous cash/inventory ledgers...',flush=True)
    accounts={}; strategies={
        'walkforward_top3':(dynamic,dynamic_flags),
        'fixed_train_top3':(fixed,fixed_flags),
        'benchmark_equal_monthly':(targets[Config().name],flags['monthly']),
        'benchmark_invvol_monthly':(targets[Config(allocation='invvol').name],flags['monthly']),
        'trend_ensemble_equal':(targets[Config(trend='ensemble').name],flags['monthly']),
        'trend_ensemble_invvol':(targets[Config(trend='ensemble',allocation='invvol').name],flags['monthly']),
        'trend_ensemble_invvol_es':(targets[Config(trend='ensemble',allocation='invvol',es_budget=.015).name],flags['monthly'])}
    # Train-only single champion is visible, never promoted by full-period return.
    single=initial[0];strategies['fixed_train_best']=(targets[single],flags[byname[single].frequency])
    for name,(target,decision) in strategies.items():
        nav,tx=simulate(dates,fields,div,target,decision,start='2020-01-01',end=args.end)
        accounts[name]=nav;nav.to_csv(out/f'{name}_nav.csv',index=False);tx.to_csv(out/f'{name}_fills.csv',index=False)
        pd.DataFrame(target,columns=CODES,index=dates).rename_axis('date').to_csv(out/f'{name}_targets_close.csv')
    # Preserve the same original independent single-asset ledgers when present.
    if args.end=='2025-12-31':
        for label,suffix,folder in [('cta_original_equal','full',ROOT/'results'),('cta_improved_equal','no_rho_3',ROOT/'results'/'improvements')]:
            paths=[folder/f'{c}_{suffix}_nav.csv' for c in CODES]
            if all(p.exists() for p in paths):
                parts=[pd.read_csv(p,parse_dates=['date']).set_index('date') for p in paths]
                if not all(p.index.equals(parts[0].index) for p in parts):raise ValueError('CTA calendars differ')
                nav=pd.DataFrame({'date':parts[0].index,'nav':sum(p.nav.to_numpy() for p in parts)/3,
                                  'turnover':sum(p.turnover.to_numpy()*np.r_[1.,p.nav.to_numpy()[:-1]] for p in parts)/sum(np.r_[1.,p.nav.to_numpy()[:-1]] for p in parts),
                                  'cost':sum(p.commission.to_numpy()+p.slippage_cash.to_numpy() for p in parts)/3})
                accounts[label]=nav;nav.to_csv(out/f'{label}_nav.csv',index=False)
    attribution=[]
    for name,nav in accounts.items():
        if 'weight_510300' not in nav:continue
        for code in CODES:
            attribution.append(dict(strategy=name,code=code,mean_weight=nav[f'weight_{code}'].mean(),
                price_pnl=nav[f'price_pnl_{code}'].sum(),dividend_pnl=nav[f'dividend_pnl_{code}'].sum(),
                cost=nav[f'cost_{code}'].sum(),net_pnl=nav[f'net_pnl_{code}'].sum()))
    pd.DataFrame(attribution).to_csv(out/'attribution.csv',index=False)
    rows=[];yearly=[]
    for name,nav in accounts.items():
        for period,lo,hi in [(full_label,'2020-01-01',args.end),('2020-2022','2020-01-01','2022-12-31'),(later_label,'2023-01-01',args.end)]:
            if pd.Timestamp(lo)<=nav.date.max():rows.append(dict(strategy=name,period=period,**stats(nav,lo,min(hi,args.end))))
        for y in sorted(nav.date.dt.year.unique()):
            yearly.append(dict(strategy=name,year=int(y),**stats(nav,f'{y}-01-01',f'{y}-12-31')))
    summary=pd.DataFrame(rows);summary.to_csv(out/'summary.csv',index=False)
    years=pd.DataFrame(yearly);years.to_csv(out/'yearly.csv',index=False)
    print('3/6 Cost stress and ES ablations...',flush=True)
    stress=[]
    for name in ('walkforward_top3','fixed_train_top3','benchmark_equal_monthly','benchmark_invvol_monthly'):
        target,decision=strategies[name]
        for multiple in (0.,1.,2.):
            nav,_=simulate(dates,fields,div,target,decision,start='2020-01-01',end=args.end,fee=.001*multiple,slip=.001*multiple)
            stress.append(dict(strategy=name,cost_multiplier=multiple,**stats(nav)))
    pd.DataFrame(stress).to_csv(out/'cost_stress.csv',index=False)
    # Matched exposure diagnostic based only on pre-2020 average target exposure.
    esname=Config(trend='ensemble',allocation='invvol',es_budget=.015).name
    noname=Config(trend='ensemble',allocation='invvol').name
    train=(dates>='2015-01-01')&(dates<='2019-12-31')
    ratio=targets[esname][train].sum()/targets[noname][train].sum()
    control=targets[noname]*ratio
    cn,ct=simulate(dates,fields,div,control,flags['monthly'],start='2020-01-01',end=args.end)
    cn.to_csv(out/'es_matched_exposure_control_nav.csv',index=False)
    pd.DataFrame([dict(strategy=n,**stats(v)) for n,v in [('no_es',accounts['trend_ensemble_invvol']),('es',accounts['trend_ensemble_invvol_es']),('fixed_exposure_control',cn)]]).to_csv(out/'es_ablation.csv',index=False)
    print('4/6 One-day VaR/ES diagnostics...',flush=True)
    forecasts=risk_forecasts(accounts['walkforward_top3']);forecasts.to_csv(out/'risk_forecasts.csv',index=False)
    coverage=[]
    for method,p in forecasts.groupby('method'):
        hit=p.breach.to_numpy(bool)
        coverage.append(dict(method=method,observations=len(p),breach_rate=hit.mean(),expected_rate=.025,
                             consecutive_breach_pairs=int(np.sum(hit[1:]&hit[:-1])),mean_predicted_es=p.es.mean(),
                             mean_loss_on_var_breach=p.loc[p.breach,'realized_loss'].mean()))
    pd.DataFrame(coverage).to_csv(out/'risk_coverage.csv',index=False)
    print('5/6 Preparing next-period rules and audit...',flush=True)
    final_rank=rank_candidates(navs,args.end);final_rank.to_csv(out/'deployment_ranking.csv',index=False)
    chosen=select_top3(final_rank,byname)
    next_weight=sum(targets[name][-1] for name in chosen)/3
    deployment=dict(recommended_research_rule=asdict(Config(allocation='invvol')),recommendation_note='Simple monthly inverse-volatility allocation is the next validation baseline, selected with knowledge of this retrospective study; not a claim of future optimum.',as_of=args.end,next_period_top3=[asdict(byname[n])|{'name':n} for n in chosen],
                    target_after_next_eligible_close={c:float(w) for c,w in zip(CODES,next_weight)},cash_weight=float(1-next_weight.sum()),
                    note='Targets are last scheduled close targets, not an execution recommendation. Recompute on fresh data before next trading period.')
    (out/'deployment.json').write_text(json.dumps(deployment,indent=2,ensure_ascii=False),encoding='utf-8')
    manifest=dict(start='2020-01-01',end=args.end,candidates=len(configs),candidate_configs=[asdict(c) for c in configs],
                  train='2015-2019',validation='2020-2022',later='2023-2025 (previously inspected; not pristine OOS)',
                  selection='annual expanding train top3 distinct base rules (ES variants not duplicated); score CAGR/max(.06,abs(MDD))-.02*annual_one_way_turnover',
                  execution='close decision/next open; weekly or monthly; raw-price ledger + dividend receivables',
                  fee=.001,slippage=.001,source_hashes={f:hashlib.sha256((args.data/f).read_bytes()).hexdigest() for f in ['etf_daily.csv','etf_dividend.csv']},
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),python=sys.version,platform=platform.platform(),
                  exposure_control_train_ratio=float(ratio),macro_included=False,maximum_pnl_reconstruction_error=max(float(n.pnl_reconstruction_error.max()) for n in accounts.values() if 'pnl_reconstruction_error' in n),maximum_identity_error=max(float(n.identity_error.max()) for n in accounts.values() if 'identity_error' in n),
                  adjustments='Only causal adjclose ratios used for signals; dividends counted once in raw-price ledger.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('6/6 Building self-contained HTML...',flush=True)
    build_report(out,summary,years,selection,final_rank,manifest)
    print(summary[summary.period==full_label][['strategy','cagr','max_drawdown','calmar','annual_turnover']].to_string(index=False))
    print('HTML:',(out/'portfolio_report.html').resolve(),flush=True)

if __name__=='__main__':
    main()
