"""Retrospective, asset-specific long/cash research using the original ledger.

Candidate design follows inspection of 2020-2025. Training-only ranking is a
chronological diagnostic, NOT a pristine out-of-sample claim. No macro inputs.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import html
import json
import sys

import numpy as np
import pandas as pd
from engine import indicators, backtest, metrics
from improvement_signals import make_signals

ROOT = Path(__file__).resolve().parent
CODES = ('510300', '518880', '511010')
ORIGINAL_K = {'510300': 4.5, '518880': 3., '511010': 1.5}


@dataclass(frozen=True)
class Rule:
    name: str
    family: str = 'ar'
    state: str = 'no_rho_10'
    atr_k: float = 3.
    stop_floor: float = 0.
    ma: int = 0
    band: float = 0.
    confirm: int = 5
    eligible: bool = False


def candidates(code):
    k = ORIGINAL_K[code]
    rules = [Rule(v, state=v, atr_k=k) for v in
             ('baseline_long', 'neutral_10', 'no_rho_10', 'no_rho_3')]
    rules.append(Rule('buyhold', family='buyhold', atr_k=k))
    if code == '510300':
        rules = [Rule(**(asdict(r) | {'eligible': r.name == 'no_rho_3'})) for r in rules]
    if code == '518880':
        # Existing ablation anchor + two wider exits + one extra slow entry gate.
        rules = [Rule(**(asdict(r) | {'eligible': r.name == 'no_rho_10'})) for r in rules]
        rules += [Rule('ar10_k4.5', atr_k=4.5, eligible=True),
                  Rule('ar10_k6', atr_k=6., eligible=True),
                  Rule('ar10_k4.5_ma120', atr_k=4.5, ma=120, eligible=True)]
    if code == '511010':
        rules += [Rule('ar10_k4.5', atr_k=4.5, eligible=True),
                  Rule('ar10_k4.5_floor0.5pct', atr_k=4.5, stop_floor=.005, eligible=True)]
        for ma in (60, 120, 240):
            rules.append(Rule(f'slow_ma{ma}_band0.25pct_5d', family='slow',
                              ma=ma, band=.0025, confirm=5, eligible=True))
        rules.append(Rule('slow_ma120_band0.5pct_5d', family='slow',
                          ma=120, band=.005, confirm=5, eligible=True))
    return rules


PRESET = {'510300': 'no_rho_3', '518880': 'ar10_k4.5',
          '511010': 'slow_ma120_band0.25pct_5d'}
ANCHOR = PRESET | {'518880': 'no_rho_10'}


def slow_signals(features, rule):
    s = features.copy()
    ma = s.adjclose.rolling(rule.ma, min_periods=rule.ma).mean()
    above = s.adjclose > ma*(1+rule.band)
    below = s.adjclose < ma*(1-rule.band)
    enter = above.rolling(rule.confirm, min_periods=rule.confirm).sum().eq(rule.confirm)
    leave = below.rolling(rule.confirm, min_periods=rule.confirm).sum().eq(rule.confirm)
    s['slow_ma'] = ma
    s['signal'] = np.select([enter, leave], [1, -1], default=0)
    # A ledger entry type, not a claim that AR-GARCH confirmed an uptrend.
    s['state'] = 1
    s['atr'] = np.nan
    return s


def signals(features, fits, rule):
    if rule.family == 'slow':
        return slow_signals(features, rule)
    s = make_signals(features, fits, 'buyhold' if rule.family == 'buyhold' else rule.state)
    if rule.ma:
        ma = s.adjclose.rolling(rule.ma, min_periods=rule.ma).mean()
        s.loc[~(s.adjclose > ma), 'signal'] = 0
        s['slow_ma'] = ma
    if rule.stop_floor:
        s['atr'] = np.fmax(s.atr, rule.stop_floor*s.adjclose/rule.atr_k)
    return s


def account(g, s, div, rule, start, end, multiple=1.):
    mode = 'buyhold' if rule.family == 'buyhold' else 'no_atr' if rule.family == 'slow' else 'full'
    nav, fills, trades, info = backtest(g, s, div, rule.atr_k, variant=mode,
        start=start, end=end, fee=.001*multiple, slip=.001*multiple, long_only=True)
    if len(trades) and rule.family == 'slow':
        trades['reason'] = trades.reason.replace({'opposite_signal_assumption': 'confirmed_ma_exit'})
    if (nav.quantity < -1e-12).any():
        raise AssertionError('Short inventory in long-only experiment')
    raw = g.set_index('date').close.loc[nav.date].to_numpy()
    error = np.max(np.abs(nav.nav - (nav.cash + nav.quantity*raw + nav.receivable)))
    if error > 1e-10 or nav.cash.min() < -1e-10:
        raise AssertionError('Ledger identity or cash constraint failed')
    return nav, fills, trades, info, float(error)


def evaluate(nav, lo=None, hi=None):
    p = nav
    if lo is not None:
        p = p[p.date >= pd.Timestamp(lo)]
    if hi is not None:
        p = p[p.date <= pd.Timestamp(hi)]
    if len(p) < 2:
        raise ValueError('Insufficient evaluation history')
    prior = nav[nav.date < p.date.iloc[0]]
    base = prior.nav.iloc[-1] if len(prior) else 1.
    out = metrics(p.nav/base, p.turnover.sum())
    if 'position' in p:
        out.update(mean_exposure=p.position.mean(), cash_day_share=p.quantity.eq(0).mean())
    if 'commission' in p:
        out['cost_initial_capital'] = (p.commission.sum()+p.slippage_cash.sum())/base
    return out


def selection_score(m):
    return m['cagr']/max(.06, abs(m['max_drawdown'])) - .02*m['annual_turnover']


def combine(accounts):
    """Equal initial capital subaccounts, no subsequent cross-sleeve rebalance."""
    dates = accounts[0].date
    if not all(x.date.reset_index(drop=True).equals(dates.reset_index(drop=True)) for x in accounts):
        raise ValueError('Different sleeve calendars')
    nav = sum(x.nav.to_numpy() for x in accounts)/len(accounts)
    prior = [np.r_[1., x.nav.to_numpy()[:-1]] for x in accounts]
    turnover = sum(x.turnover.to_numpy()*b for x, b in zip(accounts, prior))/sum(prior)
    return pd.DataFrame({'date': dates.to_numpy(), 'nav': nav, 'turnover': turnover})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=ROOT/'data')
    ap.add_argument('--out', type=Path, default=ROOT/'results')
    ap.add_argument('--end', default='2025-12-31')
    a = ap.parse_args()
    if pd.Timestamp(a.end) < pd.Timestamp('2025-12-31'):
        raise ValueError('This fixed research protocol requires the complete 2020-2025 window')
    a.out.mkdir(parents=True, exist_ok=True)
    design = {'design_status': 'retrospective; designed after viewing 2020-2025',
        'run_started_utc': datetime.now(timezone.utc).isoformat(),
        'preset_chosen_before_this_run': PRESET, 'all_candidates': {c: [asdict(r) for r in candidates(c)] for c in CODES},
        'selection': '2015-2019 only; CAGR/max(.06,abs(MDD))-.02*annual_turnover; alphabetical tie break',
        'prohibitions': 'No claim of pristine OOS; no selection using 2020-2025 scores; all failures retained'}
    (a.out/'design.json').write_text(json.dumps(design, ensure_ascii=False, indent=2), encoding='utf-8')
    d = pd.read_csv(a.data/'etf_daily.csv', dtype={'code': str}, parse_dates=['date'])
    div = pd.read_csv(a.data/'etf_dividend.csv', dtype={'code': str}, parse_dates=['ex_date', 'pay_date'])
    if div.duplicated(['code', 'ex_date', 'pay_date']).any() or (div.pay_date < div.ex_date).any() or (div.cash_dividend_per_unit < 0).any():
        raise ValueError('Invalid dividend events')
    rows, years, training, checks, stresses, selections, cache_audit = [], [], [], [], [], [], []
    saved, contexts, chosen = {}, {}, {}
    for code in CODES:
        g = d[d.code == code].sort_values('date').reset_index(drop=True)
        if g.empty or g.date.duplicated().any() or g.date.max() < pd.Timestamp(a.end):
            raise ValueError(f'{code}: missing/duplicate dates or endpoint')
        p = g[['open', 'high', 'low', 'close', 'adj_factor']]
        if not np.isfinite(p).all().all() or (p <= 0).any().any() or (g.high < g[['open', 'low', 'close']].max(axis=1)).any() or (g.low > g[['open', 'high', 'close']].min(axis=1)).any():
            raise ValueError(f'{code}: invalid OHLC or factor')
        key = hashlib.sha256(g.to_csv(index=False, lineterminator='\n').encode('utf-8')).hexdigest()[:16]
        cache = ROOT/'cache'/f'{code}_{key}.json'
        if not cache.exists():
            raise FileNotFoundError(f'Matching cache missing: {cache.name}; rebuild with the ORIGINAL project run.py; do not rename cache')
        fits = json.loads(cache.read_text(encoding='utf-8'))
        f = indicators(g)
        expected = [i for i, r in f.iterrows() if i >= max(30, int(r.window))]
        if sorted(x['i'] for x in fits) != expected or not all(x['fit_ok'] and x['bootstrap_valid'] == 1000 for x in fits):
            raise ValueError('Incomplete model cache')
        cache_audit.append({'code': code, 'input_hash': key,
                            'cache_sha256': hashlib.sha256(cache.read_bytes()).hexdigest()})
        ev = div[div.code == code]
        local_train = []
        contexts[code] = (g, ev, {})
        for rule in candidates(code):
            s = signals(f, fits, rule)
            contexts[code][2][rule.name] = (rule, s)
            s.to_csv(a.out/f'{code}_{rule.name}_signals.csv', index=False)
            train, _, _, _, _ = account(g, s, ev, rule, '2015-01-01', '2019-12-31')
            tm = evaluate(train)
            record = dict(code=code, strategy=rule.name, eligible=rule.eligible, score=selection_score(tm), **tm)
            training.append(record)
            if rule.eligible:
                local_train.append(record)
            nav, fills, trades, info, err = account(g, s, ev, rule, '2020-01-01', a.end)
            for suffix, frame in [('nav', nav), ('fills', fills), ('trades', trades)]:
                frame.to_csv(a.out/f'{code}_{rule.name}_{suffix}.csv', index=False)
            saved[(code, rule.name)] = nav
            m = evaluate(nav)
            rows.append(dict(code=code, strategy=rule.name, period='2020-end',
                             closed_trades=len(trades), **m))
            for label, lo, hi in [('2020-2022', '2020-01-01', '2022-12-31'),
                                  ('2023-2025', '2023-01-01', '2025-12-31'),
                                  ('2020-2024_without_2025', '2020-01-01', '2024-12-31')]:
                rows.append(dict(code=code, strategy=rule.name, period=label, **evaluate(nav, lo, hi)))
            for year in sorted(nav.date.dt.year.unique()):
                years.append(dict(code=code, strategy=rule.name, year=int(year),
                                  **evaluate(nav, f'{year}-01-01', f'{year}-12-31')))
            match = None
            reference = ROOT/'reference'/f'{code}_{rule.name}_nav.csv'
            if reference.exists() and a.end == '2025-12-31':
                old = pd.read_csv(reference)
                np.testing.assert_allclose(nav.nav, old.nav, rtol=1e-12, atol=1e-12)
                match = float(np.max(np.abs(nav.nav-old.nav)))
            checks.append(dict(code=code, strategy=rule.name, ledger_error=err,
                               min_quantity=float(nav.quantity.min()), prior_result_max_difference=match))
        winner = sorted(local_train, key=lambda x: (-x['score'], x['strategy']))[0]['strategy']
        chosen[code] = winner
        selections.append({'code': code, 'train_end': '2019-12-31', 'train_selected': winner,
                           'research_preset': PRESET[code]})
        # Costs never change rules or selection; generate zero/base/double costs.
        for name in dict.fromkeys([PRESET[code], winner, ANCHOR[code], 'buyhold']+
                                 [r.name for r in candidates(code) if r.eligible]):
            rule, s = contexts[code][2][name]
            for multiple in (0., 1., 2.):
                nav, _, _, _, _ = account(g, s, ev, rule, '2020-01-01', a.end, multiple)
                stresses.append(dict(code=code, strategy=name, cost_multiplier=multiple, **evaluate(nav)))
        print(f'DONE {code}: preset={PRESET[code]}, train_selected={winner}', flush=True)
    portfolios = []
    for name, mapping in [('tailored_preset_equal_initial', PRESET), ('train_selected_equal_initial', chosen),
                          ('ablation_anchor_equal_initial', ANCHOR),
                          ('old_no_rho_3_equal_initial', {c: 'no_rho_3' for c in CODES}),
                          ('buyhold_equal_initial', {c: 'buyhold' for c in CODES})]:
        nav = combine([saved[(c, mapping[c])] for c in CODES])
        nav.to_csv(a.out/f'{name}_nav.csv', index=False)
        for label, lo, hi in [('2020-end', '2020-01-01', a.end), ('2020-2024_without_2025', '2020-01-01', '2024-12-31')]:
            portfolios.append(dict(strategy=name, period=label, **evaluate(nav, lo, hi)))
    if a.end == '2025-12-31':
        for name in ('benchmark_invvol_monthly', 'benchmark_equal_monthly'):
            nav = pd.read_csv(ROOT/'reference'/f'{name}_nav.csv', parse_dates=['date'])
            portfolios.append(dict(strategy=name, period='2020-end', **evaluate(nav)))
    frames = {'summary': pd.DataFrame(rows), 'yearly': pd.DataFrame(years),
              'train_ranking': pd.DataFrame(training), 'selections': pd.DataFrame(selections),
              'checks': pd.DataFrame(checks), 'cost_stress': pd.DataFrame(stresses),
              'portfolio_comparison': pd.DataFrame(portfolios)}
    for name, frame in frames.items():
        frame.to_csv(a.out/f'{name}.csv', index=False)
    manifest = dict(design, cache_audit=cache_audit, models_reestimated=False,
        source_hashes={x: hashlib.sha256((ROOT/x).read_bytes()).hexdigest() for x in
                       ('tailored_long.py', 'engine.py', 'improvement_signals.py')},
        data_hashes={x: hashlib.sha256((a.data/x).read_bytes()).hexdigest() for x in
                     ('etf_daily.csv', 'etf_dividend.csv')},
        fee=.001, slip=.001, cash_rate=0, end=a.end, python=sys.version,
        accounting='raw price + receivables; fractional units; no shorts/leverage; no minimum commission',
        portfolio='equal INITIAL capital independent sleeves; NOT monthly equal rebalancing',
        original_ledger_change='Optional long_only flag suppresses negative entries, retains negative exit signals, and forbids financing purchases with unpaid receivables; legacy default unchanged',
        max_ledger_error=max(x['ledger_error'] for x in checks))
    (a.out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    def table(frame):
        frame = frame.copy()
        for c in ('cagr', 'max_drawdown', 'mean_exposure', 'total_return', 'volatility'):
            if c in frame:
                frame[c] = frame[c].map(lambda x: f'{x:.2%}')
        for c in ('annual_turnover', 'calmar', 'sharpe'):
            if c in frame:
                frame[c] = frame[c].map(lambda x: f'{x:.3f}')
        return frame.to_html(index=False, border=0)
    body = '<!doctype html><meta charset="utf-8"><title>仅多头分资产策略研究</title><style>body{font:16px system-ui;max-width:1300px;margin:32px auto;padding:0 20px;line-height:1.65}table{border-collapse:collapse;font-size:14px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}th{background:#eef3f8}.note{background:#fff3dc;padding:16px}</style>'
    body += '<h1>仅多头分资产策略：回顾性研究</h1><p class="note">候选规则在查看过2020—2025历史结果后设计。训练期排序只用2015—2019，但这不消除研究设计的历史反馈。全部候选保留；不将测试期赢家写成预先最优。现金零收益，按252日年化；每次佣金1‰、滑点1‰。</p>'
    body += '<h2>预先给定研究方案与训练期选择</h2>'+table(frames['selections'])
    cols = ['code', 'strategy', 'cagr', 'max_drawdown', 'calmar', 'mean_exposure', 'annual_turnover', 'closed_trades']
    body += '<h2>全部候选：2020至终点</h2>'+table(frames['summary'].query("period == '2020-end'")[cols])
    body += '<h2>组合比较</h2><p>新策略组合为各1/3初始资金的独立账户，期间不在账户间再平衡；逆波动基准的配置机制不同，不能仅凭收益差认定择时alpha。</p>'+table(frames['portfolio_comparison'][['strategy','period','cagr','max_drawdown','calmar','annual_turnover']])
    body += '<h2>排除2025年的敏感性检查</h2>'+table(frames['summary'].query("period == '2020-2024_without_2025'")[[c for c in cols if c != 'closed_trades']])
    body += '<h2>固定信号的成本压力</h2>'+table(frames['cost_stress'][['code','strategy','cost_multiplier','cagr','max_drawdown','annual_turnover']])
    body += '<h2>逐年收益</h2>'+table(frames['yearly'][['code','strategy','year','total_return','max_drawdown','annual_turnover']])
    body += '<h2>规则与执行</h2><p>黄金以no_rho_10为锚，比较ATR倍数3/4.5/6和一个MA120入场过滤；国债比较宽ATR、0.5%价格距离下限及MA60/120/240缓冲确认。慢均线版本只有价格确认退出，没有ATR或RSI退出。状态失效不使AR版本立即清仓。新开仓当日不做盘中止损，日内不重入。国债0.5%距离下限不等于保证亏损上限，且随ATR及价格更新。</p>'
    body += '<p>扩展行情前必须重新生成匹配的原模型缓存。没有加入宏观，没有新增全局最优或实盘业绩声明。</p>'
    (a.out/'report.html').write_text(body, encoding='utf-8')
    print(frames['portfolio_comparison'].to_string(index=False), flush=True)
    print('REPORT', (a.out/'report.html').resolve(), flush=True)


if __name__ == '__main__':
    main()
