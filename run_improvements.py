"""Run all state improvements with identical fees and execution, no model tuning."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np,pandas as pd
from engine import indicators,backtest,metrics
from improvement_signals import make_signals,VARIANTS
ROOT=Path(__file__).resolve().parent
CODES={'510300':4.5,'518880':3.,'511010':1.5}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,default=ROOT/'results/improvements');ap.add_argument('--start',default='2020-01-01');ap.add_argument('--end',default='2025-12-31');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(a.data/'etf_daily.csv',dtype={'code':str},parse_dates=['date']);div=pd.read_csv(a.data/'etf_dividend.csv',dtype={'code':str},parse_dates=['ex_date','pay_date']);rows=[];years=[];audit=[];inputs={}
    for code,k in CODES.items():
        g=d[d.code==code].sort_values('date').reset_index(drop=True)
        if len(g)==0 or g.date.duplicated().any() or g.date.max()<pd.Timestamp(a.end):raise ValueError(f'{code}: missing history/duplicate date')
        if not np.isfinite(g[['open','high','low','close','adj_factor']]).all().all() or (g[['open','high','low','close','adj_factor']]<=0).any().any():raise ValueError('Invalid prices')
        if (g.high<g[['open','close','low']].max(axis=1)).any() or (g.low>g[['open','close','high']].min(axis=1)).any():raise ValueError('Invalid OHLC')
        key=hashlib.sha256(g.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()[:16];cache=ROOT/'cache'/f'{code}_{key}.json'
        if not cache.exists():raise FileNotFoundError(f'{cache.name}: run python run.py --data YOUR_DATA first to build matching model cache')
        fits=json.loads(cache.read_text());f=indicators(g)
        expected=[i for i,r in f.iterrows() if i>=max(30,int(r.window))]
        if sorted(x['i'] for x in fits)!=expected or not all(x['fit_ok'] and x['bootstrap_valid']==1000 for x in fits):raise ValueError('Incomplete cache')
        inputs[code]=dict(input_hash=key,cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest())
        baseline=None
        for v in VARIANTS:
            s=make_signals(f,fits,v);s.to_csv(a.out/f'{code}_{v}_signals.csv',index=False)
            z,tx,tr,ac=backtest(g,s,div[div.code==code],k,'buyhold' if v=='buyhold' else 'full',a.start,a.end)
            if v=='baseline_full':baseline=z.date.tolist()
            assert z.date.tolist()==baseline
            reconstructed=z.cash.to_numpy()+z.quantity.to_numpy()*g.set_index('date').close.loc[z.date].to_numpy()+z.receivable.to_numpy()
            error=float(np.max(abs(reconstructed-z.nav)));assert error<1e-10
            if v!='baseline_full':assert (z.quantity>=0).all()
            z.to_csv(a.out/f'{code}_{v}_nav.csv',index=False);tx.to_csv(a.out/f'{code}_{v}_fills.csv',index=False);tr.to_csv(a.out/f'{code}_{v}_trades.csv',index=False)
            rows.append(dict(code=code,strategy=v,**metrics(z.nav,ac['turnover'],tr),mean_exposure=z.position.abs().mean(),cash_day_share=(z.quantity==0).mean(),trend_share=(s.loc[(s.date>=a.start)&(s.date<=a.end),'state']!=0).mean()))
            for year,part in z.groupby(z.date.dt.year):
                previous=z.loc[z.date<part.date.min(),'nav'];base=previous.iloc[-1] if len(previous) else 1.
                years.append(dict(code=code,strategy=v,year=year,**metrics(part.nav/base,part.turnover.sum())))
            old=ROOT/'results'/f'{code}_full_nav.csv';match=None
            if v=='baseline_full' and old.exists() and a.start=='2020-01-01' and a.end=='2025-12-31':
                previous=pd.read_csv(old);np.testing.assert_allclose(z.nav,previous.nav,atol=1e-12,rtol=1e-12);match=True
            audit.append(dict(code=code,strategy=v,nav_identity_error=error,long_only=v!='baseline_full',original_baseline_matches=match))
        print('DONE',code,flush=True)
    pd.DataFrame(rows).to_csv(a.out/'summary.csv',index=False);pd.DataFrame(years).to_csv(a.out/'yearly.csv',index=False);pd.DataFrame(audit).to_csv(a.out/'checks.csv',index=False)
    (a.out/'manifest.json').write_text(json.dumps(dict(start=a.start,end=a.end,variants=VARIANTS,fee=.001,slip=.001,models_reestimated=False,inputs=inputs,source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['engine.py','model.py','native/model.cpp','improvement_signals.py','run_improvements.py']},note='Fixed retrospective ablations; no tuning or new out-of-sample claim'),indent=2))
    print(pd.DataFrame(rows)[['code','strategy','cagr','max_drawdown','calmar']].to_string(index=False))
if __name__=='__main__':main()
