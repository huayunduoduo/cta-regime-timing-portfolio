from pathlib import Path
import base64,html,json,re
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
NAMES={'510300':'沪深300ETF','518880':'黄金ETF','511010':'国债ETF'}
LABEL={'full':'完整策略','always_trend':'恒趋势','always_range':'恒震荡','no_atr':'无ATR（反向退出假设）','buyhold':'买入持有','no_atr_literal':'无ATR（字面删除）','rsi_wilder':'Wilder RSI','atr_local14':'14日局部TR截断','atr_formula_no_ratchet':'吊灯直接公式（不强制单向）','zero_cost_diagnostic':'零成本诊断','double_cost_stress':'成本翻倍'}
EN={'full':'Full rules','always_trend':'Always trend','always_range':'Always range','no_atr':'No ATR: opposite exit','buyhold':'Buy & hold'}
COLORS={'full':'#007c83','always_trend':'#aa6f00','always_range':'#a05da8','no_atr':'#ce5c4c','buyhold':'#53647a'}

def table(df,columns=None):
    z=df.copy() if columns is None else df[columns].copy()
    if 'code' in z:z['code']=z.code.map(lambda c:f'{c} {NAMES[str(c)]}')
    if 'strategy' in z:z['strategy']=z.strategy.map(lambda x:LABEL.get(x,x))
    for c in z:
        if c in ['total_return','cagr','volatility','max_drawdown','win_rate','average_abs_exposure']:z[c]=z[c].map(lambda v:f'{v:.2%}' if pd.notna(v) else '—')
        elif c in ['sharpe','calmar','payoff_ratio','annual_turnover']:z[c]=z[c].map(lambda v:f'{v:.2f}' if pd.notna(v) else '—')
    z=z.rename(columns={'code':'资产','strategy':'方案','year':'年份','total_return':'累计收益','cagr':'年化收益','volatility':'年化波动','max_drawdown':'最大回撤','sharpe':'夏普','calmar':'卡玛','closed_trades':'已平仓笔数','win_rate':'费用后胜率','payoff_ratio':'盈亏比','annual_turnover':'年化单边换手倍数','average_abs_exposure':'平均绝对敞口','open_position':'期末方向','max_underwater_days':'最长水下交易日'})
    return '<div class="scroll">'+z.to_html(index=False,border=0,na_rep='—',classes='metrics')+'</div>'

def md_to_html(text):
    # Minimal safe Markdown renderer for the bundled rules (no external dependency).
    out=[];lines=text.splitlines();i=0
    while i<len(lines):
        line=lines[i]
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                row=lines[i];i+=1
                if re.fullmatch(r'[| :\-]+',row):continue
                rows.append([html.escape(t.strip()) for t in row.strip('|').split('|')])
            out.append('<div class="scroll"><table><thead><tr>'+''.join('<th>'+v+'</th>' for v in rows[0])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+v+'</td>' for v in row)+'</tr>' for row in rows[1:])+'</tbody></table></div>');continue
        if line.startswith('#'):
            n=len(line)-len(line.lstrip('#'));out.append(f'<h{min(n+1,4)}>'+html.escape(line.lstrip('# '))+f'</h{min(n+1,4)}>')
        elif line.strip():
            line=html.escape(line);line=re.sub(r'\*\*(.*?)\*\*',r'<strong>\1</strong>',line);line=re.sub(r'`(.*?)`',r'<code>\1</code>',line)
            out.append('<p>'+line+'</p>')
        i+=1
    return '\n'.join(out)

def main():
    summary=pd.read_csv(OUT/'summary.csv',dtype={'code':str});sen=pd.read_csv(OUT/'sensitivity.csv',dtype={'code':str});yearly=pd.read_csv(OUT/'yearly.csv',dtype={'code':str})
    refs=[]
    values={'510300':[(.0469,-.4216,.11,None,None,None),(.1429,-.1275,1.12,44,.7045,1.64),(.0671,-.3385,.20,22,.4091,2.71),(.0106,-.2909,.04,58,.6379,.71),(.1249,-.3326,.38,None,None,None)],'518880':[(.1799,-.2489,.72,None,None,None),(.2170,-.1111,1.95,59,.6780,2.21),(.1503,-.2332,.64,53,.5472,2.60),(.0151,-.1943,.08,70,.5857,.84),(.2294,-.1713,1.34,None,None,None)],'511010':[(.0310,-.0492,.63,None,None,None),(.0233,-.0193,1.21,96,.4896,2.57),(.0227,-.0301,.75,97,.4948,2.46),(-.0092,-.0625,-.15,88,.3409,1.12),(.0254,-.0492,.52,None,None,None)]}
    for c,vals in values.items():
        for strategy,row in zip(['buyhold','full','always_trend','always_range','no_atr'],vals):refs.append(dict(code=c,strategy=strategy,cagr=row[0],max_drawdown=row[1],calmar=row[2],closed_trades=row[3],win_rate=row[4],payoff_ratio=row[5],source_page={'510300':14,'518880':15,'511010':16}[c],period='2020-01 to 2026-05',definition='paper no_atr exit rule unspecified' if strategy=='no_atr' else 'paper reference'))
    reference=pd.DataFrame(refs);reference.to_csv(OUT/'paper_reference.csv',index=False)
    sections=[]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'figure.facecolor':'white'})
    for c,name in NAMES.items():
        s=pd.read_csv(OUT/f'{c}_signals.csv',parse_dates=['date']);ss=s[s.date>='2020-01-01']
        fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True,gridspec_kw={'height_ratios':[1.1,1.1,.65]})
        navs={v:pd.read_csv(OUT/f'{c}_{v}_nav.csv',parse_dates=['date']) for v in EN}
        for state,color in [(1,'#cef0e1'),(-1,'#f7d5d1'),(0,'#eef1f5')]:axes[0].fill_between(ss.date,0,1,where=ss.state==state,transform=axes[0].get_xaxis_transform(),color=color,step='post',alpha=.9)
        for v in ('full','buyhold'):
            z=navs[v];axes[0].plot(z.date,z.nav,label=EN[v],color=COLORS[v],lw=1.8)
        axes[0].set_title(f'{c} | Full strategy vs benchmark | green: up, red: down, gray: range',loc='left',weight='bold');axes[0].set_ylabel('NAV');axes[0].legend(loc='upper left')
        for v,z in navs.items():axes[1].plot(z.date,z.nav,label=EN[v],color=COLORS[v],lw=1.4)
        axes[1].set_title('Same dates and costs | No-ATR exit is an explicit assumption',loc='left');axes[1].set_ylabel('NAV');axes[1].legend(ncol=3,loc='best',fontsize=8)
        for v in ('full','buyhold'):
            z=navs[v];vv=z.nav.to_numpy();dd=vv/np.maximum.accumulate(np.r_[1,vv])[1:]-1;axes[2].plot(z.date,100*dd,label=EN[v],color=COLORS[v])
        axes[2].set_ylabel('Drawdown (%)');axes[2].set_title('Drawdowns include initial capital and entry costs',loc='left');fig.tight_layout();path=OUT/f'{c}_comparison.png';fig.savefig(path,dpi=140);plt.close(fig)
        encoded=base64.b64encode(path.read_bytes()).decode()
        sections.append(f'<section><h2>{c} · {name}</h2><img alt="{name}净值、状态与回撤" src="data:image/png;base64,{encoded}">'+table(summary[summary.code==c],['strategy','cagr','sharpe','max_drawdown','calmar','closed_trades','win_rate','payoff_ratio','annual_turnover','average_abs_exposure'])+'<details><summary>逐年表现（包含全部年份）</summary>'+table(yearly[(yearly.code==c)&(yearly.strategy!='no_atr_literal')],['strategy','year','total_return','sharpe','max_drawdown','calmar'])+'</details></section>')
    costs=[]
    for c in NAMES:
        for label in ['zero_cost_diagnostic','full','double_cost_stress']:
            row=(summary if label=='full' else sen);r=row[(row.code==c)&(row.strategy==label)].iloc[0];costs.append(r)
    costtab=table(pd.DataFrame(costs),['code','strategy','cagr','max_drawdown','calmar','win_rate'])
    audit=pd.read_csv(OUT/'model_audit.csv',dtype={'code':str});fresh=json.loads((OUT/'cache_recompute_audit.json').read_text());ledgers=json.loads((OUT/'ledger_audit.json').read_text())
    core=table(summary[summary.strategy=='full'],['code','cagr','sharpe','max_drawdown','calmar','closed_trades'])
    report='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>时序CTA研报日频复现与差异审计</title><style>
:root{--ink:#193447;--muted:#617783;--accent:#007c83}*{box-sizing:border-box}body{margin:0;background:#f2f5f7;color:var(--ink);font:15px/1.8 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}main{max-width:1280px;margin:auto;padding:34px 28px 70px}header{padding:34px;border-radius:16px;background:#133848;color:white;margin-bottom:24px}header p{color:#d6e5eb}h1{font-size:31px;line-height:1.4;margin:12px 0}h2{font-size:23px;margin:0 0 18px}h3{font-size:18px;margin-top:28px}.eyebrow{letter-spacing:2px;font-size:12px;color:#a1e2d9}.tag{display:inline-block;margin:4px 8px 0 0;padding:3px 10px;border:1px solid #6b929b;border-radius:20px;font-size:12px}section{background:white;border-radius:14px;margin:22px 0;padding:28px;box-shadow:0 2px 10px #15384808}.callout{background:#fff5e6;border-left:4px solid #d99c3c;padding:16px 20px;margin:20px 0}.fact{background:#edf7f5;padding:16px 20px;border-radius:8px}p{margin:12px 0}img{width:100%;height:auto}.scroll{overflow:auto;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:13px;text-align:left;line-height:1.6}th{background:#eaf2f4;font-weight:650;white-space:nowrap}td,th{padding:10px 12px;border-bottom:1px solid #e0e9ed;vertical-align:top}td{white-space:nowrap}section:last-child td{white-space:normal;min-width:100px}tbody tr:hover{background:#f4faf9}details{border:1px solid #dce6ea;border-radius:8px;padding:12px 16px;margin-top:18px}summary{cursor:pointer;font-weight:650;color:var(--accent)}code,pre{font:13px/1.7 ui-monospace,Consolas,monospace}pre{overflow:auto;background:#eef3f5;padding:16px;border-radius:8px;white-space:pre-wrap}footer{color:var(--muted);font-size:13px;padding:10px}nav a{color:#b5e9e4;margin-right:20px}a{color:var(--accent)}@media(max-width:700px){main{padding:14px}header,section{padding:20px}h1{font-size:25px}h2{font-size:20px}}
</style></head><body><main><header><div class="eyebrow">RESEARCH REPRODUCTION · 2026-11-01</div><h1>时序CTA研报日频复现<br>公开规则、实际结果与差异审计</h1><p>三资产 · 多空双向 · 固定规则与成本 · 不含宏观和组合增量</p><span class="tag">2020-01-02 — 2025-12-31</span><span class="tag">15个主结果 + 18个敏感性结果</span><span class="tag">每模型窗口1000次有效重采样</span><nav><a href="#results">实际结果</a><a href="#cost">成本诊断</a><a href="#rules">逐项规则</a></nav></header>
<section><h2>先看结论：未成功复现原报告业绩</h2><div class="callout"><strong>本版已经修正上一版的多头限制、RSI退出与ATR参考价，并补齐对照。但这不等于原文业绩已被验证。</strong>当前主实现中，沪深300略低于买入持有的年化收益、回撤较小；黄金与国债亏损，未复现研报的高卡玛。</div>'''+core+'''<p>这些数值是新多空交易引擎的真实输出，不是上一版组合结果。原文区间到2026年5月，本地只到2025年底；原文没有公布足以唯一确定实现的全部参数。因此不能将差异只归因于缺少2026年，也不能将测试通过解释为复现成功。</p><div class="fact"><strong>ATR可以用日线做：</strong>盘前固定止损价，用当日最低/最高价检查触发，跳空按开盘参考价处理。触发后按止损价加固定滑点撮合，是假设；它不能恢复分钟级跳价、价差或真实成交顺序。</div></section>
<section><h2>复现思路：每一步只使用当时已知的信息</h2><p>① 在复权价格上计算ER30，决定21—250日模型窗口；窗口内MAD识别异常，Huber拟合替换异常价格。② 联合估计AR(1)-GARCH(1,1)，通过1000次残差重采样重估得到TSI置信区间。③ 综合TSI、rho和波动分位数，连续10天确认趋势；趋势用5—20日自适应均线，震荡用RSI14。</p><p>④ 空仓日收盘生成订单，下一交易日开盘成交。趋势持仓只处理吊灯止损，震荡持仓用RSI40/60逻辑退出，入场时的交易类型固定。⑤ 记录真实价格份额、现金、分红应收和到账；多空两边每次成交各扣1‰手续费与1‰逆向滑点。⑥ 统一日期、规则、费用计算净值、回撤、逐年与交易指标，不按结果调参。</p><p>吊灯主实现：多头下一日止损=max(旧止损，持仓最高价−k×ATR14)，空头取min(旧止损，持仓最低价+k×ATR14)。k分别为4.5、3.0、1.5。当天最高/最低价只影响下一日止损，不提前用于当天撮合。</p><p><strong>无ATR对照的歧义：</strong>主表采用反向完整信号触发退出这一明确假设；另附“只删除ATR、趋势单不再退出”的字面版本。原文没有给出该对照的完整退出程序，两者都不能冒充作者原始代码。</p></section><div id="results"></div>'''+''.join(sections)+'''<section id="cost"><h2>交易成本诊断：尤其影响低波动国债</h2><p>单次成交的滑点和手续费分别1‰，一次往返约4‰。国债完整策略年化单边换手约27.72倍；零成本年化约−0.10%，主费用下约−5.49%。费用是重要拖累，但零成本也没有复现原文的盈利。零成本只用于诊断，不能替换报告规定的费用。</p>'''+costtab+'''</section><section><h2>未披露参数的敏感性：全部保留，不挑最好结果</h2><p>Wilder RSI、ATR截断窗口、吊灯单向约束都会改变交易路径。字面无ATR的黄金结果明显好于主实现，但其趋势单可能长期不退出，交易机制不同，不能据此宣布完整策略复现成功。以下18个结果均使用相同区间。</p>'''+table(sen,['code','strategy','cagr','sharpe','max_drawdown','calmar','closed_trades'])+'''</section><section><h2>原文参考表：不是等区间成绩比较</h2><p>来源：研报第14—16页，2020年1月至2026年5月。仅用于溯源展示，不计算本地与原文的“超额收益”，也不把原文无ATR表格等同于本项目的反向退出假设。</p>'''+table(reference,['code','strategy','cagr','max_drawdown','calmar','closed_trades','win_rate','payoff_ratio'])+'''</section><section><h2>验证、可复现性与剩余缺口</h2><p>15项自动测试通过：多空吊灯、跳空、日内触发、严格不等号、入场日分支、RSI40/60、分红应收与支付、空头分红负债、费用、未来数据不改写过去、原生梯度、联合似然及1000次重采样确定性。18个日账本另以原始价格独立重算净值，并核对累计成交份额、手续费、滑点与日历一致。</p><p>8,848个模型窗口来自上一轮同算法缓存，输入哈希与窗口完整性均匹配，每窗1000个有效重采样。本轮没有重新估计全部窗口；每资产抽检一个窗口，独立重新估计与重采样，关键数值最大绝对误差为0。全部交易和指标已重跑。</p>'''+table(audit,['code','windows','bootstrap_valid','bootstrap_attempts','models_recomputed_this_run'])+'''<details><summary>独立抽检记录与测试输出</summary><pre>'''+html.escape(json.dumps(fresh,ensure_ascii=False,indent=2))+'\n'+html.escape((ROOT/'test_results.txt').read_text())+'''</pre></details><p><strong>剩余缺口：</strong>2026年1—5月行情与分红、作者原始代码及其收益率/RSI/分位数/无ATR退出设置、分钟级执行路径、与作者相同的数据供应商口径。输入分红完整性未在本轮逐笔独立对照管理人公告。ETF空头还缺券源、融券费率与保证金约束；当前多空模型只用于机制研究。</p><p>夏普按日均收益/日标准差×√252计算，无风险收益为0；卡玛为复合年化/最大回撤绝对值。胜率只统计费用后已平仓交易，期末持仓按市值计净值但不假造平仓。预热数据用于统计初始化；这是一套固定规则历史复现，不能称为经过独立验证集选择后的严格样本外业绩。</p></section><section id="rules">'''+md_to_html((ROOT/'RULES.md').read_text())+'''</section><footer>交付包含源代码、模型缓存、完整信号/成交/平仓明细、年度与敏感性结果、输入哈希及验证记录。原始行情与研报PDF不随GitHub代码包分发。HTML图片内嵌，可离线打开。</footer></main></body></html>'''
    (OUT/'report.html').write_text(report,encoding='utf-8');print('Created',OUT/'report.html')
if __name__=='__main__':main()
