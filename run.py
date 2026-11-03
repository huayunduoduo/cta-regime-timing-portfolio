"""Reproduce paper rules with declared daily execution assumptions, no macro."""
from pathlib import Path
import argparse,json,hashlib,platform,sys,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from engine import indicators,market_signals,backtest,metrics
from model import clean,estimate,library
ROOT=Path(__file__).resolve().parent
CODES={'510300':4.5,'518880':3.,'511010':1.5}
VARIANTS=['full','always_trend','always_range','no_atr','buyhold']

def fit(task):
    code,i,px=task
    return code,dict(i=i,**estimate(np.diff(np.log(clean(px)))*100,1000,20260610+int(code)*10000+i))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--start',default='2020-01-01');ap.add_argument('--end',default='2025-12-31');ap.add_argument('--rebuild-models',action='store_true');args=ap.parse_args()
    out=ROOT/'results';out.mkdir(exist_ok=True);cache=ROOT/'cache';cache.mkdir(exist_ok=True)
    d=pd.read_csv(args.data/'etf_daily.csv',dtype={'code':str},parse_dates=['date']);div=pd.read_csv(args.data/'etf_dividend.csv',dtype={'code':str},parse_dates=['ex_date','pay_date']);ds={c:d[d.code==c].sort_values('date').reset_index(drop=True) for c in CODES};fits={};cache_audit=[];inputs={}
    for c,g in ds.items():
        if g.date.duplicated().any() or (g[['open','high','low','close','adj_factor']]<=0).any().any() or (g.high<g[['open','close','low']].max(axis=1)).any() or (g.low>g[['open','close','high']].min(axis=1)).any():raise ValueError('Invalid bars')
        if g.date.max()<pd.Timestamp(args.end):raise ValueError(f'{c} history ends {g.date.max()} before requested {args.end}')
        key=hashlib.sha256(g.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()[:16];path=cache/f'{c}_{key}.json';inputs[c]=dict(hash=key,cache=str(path.name))
        if path.exists() and not args.rebuild_models:fits[c]=json.loads(path.read_text())
    tasks=[]
    for c,g in ds.items():
        if c in fits:continue
        fits[c]=[];s=indicators(g)
        for i,r in s.iterrows():
            if i>=max(30,int(r.window)):tasks.append((c,i,s.adjclose.iloc[i-int(r.window):i+1].to_numpy()))
    if tasks:
        library();now=time.time()
        with ProcessPoolExecutor(args.workers) as pool:
            for n,(c,rr) in enumerate(pool.map(fit,tasks,chunksize=10),1):
                fits[c].append(rr)
                if n%300==0:print(f'Models {n}/{len(tasks)} elapsed {time.time()-now:.0f}s',flush=True)
        for c in ds:(cache/inputs[c]['cache']).write_text(json.dumps(fits[c],default=lambda x:x.item()))
    summary=[];yearly=[];sensitivity=[]
    for c,k in CODES.items():
        g=ds[c];s=market_signals(indicators(g),fits[c]);s.to_csv(out/f'{c}_signals.csv',index=False)
        expected=[i for i,r in s.iterrows() if i>=max(30,int(r.window))]
        if sorted(x['i'] for x in fits[c])!=expected:raise ValueError('Incomplete or mismatched cache')
        if not all(x['fit_ok'] and x['bootstrap_valid']==1000 and x['bootstrap_requested']==1000 for x in fits[c]):raise ValueError('Incomplete bootstrap')
        cache_audit.append(dict(code=c,windows=len(fits[c]),bootstrap_valid=sum(x['bootstrap_valid'] for x in fits[c]),bootstrap_attempts=sum(x['bootstrap_attempts'] for x in fits[c]),cache_sha256=hashlib.sha256((cache/inputs[c]['cache']).read_bytes()).hexdigest(),input_dataframe_sha256=inputs[c]['hash'],models_recomputed_this_run=bool(tasks)))
        for variant in VARIANTS+['no_atr_literal']:
            nav,tx,tr,account=backtest(g,s,div[div.code==c],k,variant,args.start,args.end)
            nav.to_csv(out/f'{c}_{variant}_nav.csv',index=False);tx.to_csv(out/f'{c}_{variant}_fills.csv',index=False);tr.to_csv(out/f'{c}_{variant}_trades.csv',index=False)
            row=dict(code=c,strategy=variant,**metrics(nav.nav,account['turnover'],tr),**account,average_abs_exposure=nav.position.abs().mean())
            (sensitivity if variant=='no_atr_literal' else summary).append(row)
            for year,part in nav.groupby(nav.date.dt.year):
                prev=nav.loc[nav.date<part.date.min(),'nav'];base=prev.iloc[-1] if len(prev) else 1.
                yearly.append(dict(code=c,strategy=variant,year=year,**metrics(part.nav/base,part.turnover.sum())))
        for label,ss,ratchet in [('rsi_wilder',market_signals(indicators(g,rsi_method='wilder'),fits[c]),True),('atr_local14',market_signals(indicators(g,atr_method='local_sma'),fits[c]),True),('atr_formula_no_ratchet',s,False)]:
            nav,tx,tr,ac=backtest(g,ss,div[div.code==c],k,'full',args.start,args.end,ratchet=ratchet)
            nav.to_csv(out/f'{c}_{label}_nav.csv',index=False)
            sensitivity.append(dict(code=c,strategy=label,**metrics(nav.nav,ac['turnover'],tr),**ac,average_abs_exposure=nav.position.abs().mean()))
        for label,cost in [('zero_cost_diagnostic',0.),('double_cost_stress',.002)]:
            nav,tx,tr,ac=backtest(g,s,div[div.code==c],k,'full',args.start,args.end,fee=cost,slip=cost)
            nav.to_csv(out/f'{c}_{label}_nav.csv',index=False)
            sensitivity.append(dict(code=c,strategy=label,**metrics(nav.nav,ac['turnover'],tr),**ac,average_abs_exposure=nav.position.abs().mean()))
        print('DONE',c,flush=True)
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False);pd.DataFrame(yearly).to_csv(out/'yearly.csv',index=False);pd.DataFrame(sensitivity).to_csv(out/'sensitivity.csv',index=False);pd.DataFrame(cache_audit).to_csv(out/'model_audit.csv',index=False)
    manifest=dict(status='COMPLETED',scope='paper single-asset experiments, daily execution; NOT exact minute or full-period replication',start=args.start,end=args.end,bootstrap=1000,models_recomputed_this_run=len(tasks),input_sha256={f:hashlib.sha256((args.data/f).read_bytes()).hexdigest() for f in ['etf_daily.csv','etf_dividend.csv']},source_model_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['model.py','native/model.cpp']},python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,fee=.001,slip=.001,seed_rule='20260610+code*10000+row',rsi='14-day simple gains/losses, Wilder sensitivity',atr='expanding per-day 5/95% TR clip (min 30), SMA14; local14 sensitivity',short_model='idealized 100% signed notional; no borrowing fee/margin constraint/interest',cash_rate=0.,terminal='mark-to-market; open positions excluded from trade win rate')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False));print(pd.DataFrame(summary)[['code','strategy','cagr','max_drawdown','calmar','closed_trades','win_rate']].to_string(index=False))
if __name__=='__main__':main()
