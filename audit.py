"""Diagnose replication gaps. No baseline thresholds/parameters are changed.
Direction-only accounts are counterfactuals, not the paper strategy.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,pandas as pd
from engine import indicators,market_signals,backtest,metrics
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'/'audit'
CODES={'510300':4.5,'518880':3.,'511010':1.5}

def optimizer_checks(ds):
    from scipy.optimize import minimize
    from model import clean,library
    lib=library();rows=[]
    for code,g in ds.items():
        s=indicators(g)
        for date in ['2020-03-19','2022-06-01','2024-03-05','2025-03-19','2025-09-02']:
            i=g.index[g.date==date][0];n=int(s.window.iloc[i]);y=np.ascontiguousarray(np.diff(np.log(clean(s.adjclose.iloc[i-n:i+1])))*100);z=np.zeros(5);out=np.zeros(7)
            ok=lib.fit_model(y,len(y),out,z);f0=lib.objective(z,y,len(y));best=f0;zbest=z.copy()
            x=np.c_[np.ones(len(y)-1),y[:-1]];ols=np.linalg.lstsq(x,y[1:],rcond=None)[0]
            for shift in [np.zeros(5),np.array([0,0,0,-2,2]),np.array([0,0,0,2,-2])]:
                rr=minimize(lambda v:lib.objective(np.ascontiguousarray(v),y,len(y)),z+shift,method='L-BFGS-B',bounds=[(-100,100),(-4,4),(-25,10),(-20,20),(-20,20)],options={'ftol':1e-12,'gtol':1e-7,'maxiter':1000})
                if rr.fun<best:best=rr.fun;zbest=rr.x
            rows.append(dict(code=code,date=date,window=n,objective_native=f0,objective_independent=best,likelihood_improvement=f0-best,rho_native=out[1],rho_independent=np.tanh(zbest[1]),rho_ols=ols[1],fit_native_ok=bool(ok)))
    pd.DataFrame(rows).to_csv(OUT/'independent_optimizer.csv',index=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--optimizer-check',action='store_true');args=ap.parse_args();OUT.mkdir(exist_ok=True)
    d=pd.read_csv(args.data/'etf_daily.csv',dtype={'code':str},parse_dates=['date']);div=pd.read_csv(args.data/'etf_dividend.csv',dtype={'code':str},parse_dates=['ex_date','pay_date']);ds={c:d[d.code==c].sort_values('date').reset_index(drop=True) for c in CODES}
    gates=[];groups=[];closed=[];directions=[];byyear=[];flags=[]
    for code,k in CODES.items():
        g=ds[code];s=pd.read_csv(ROOT/'results'/f'{code}_signals.csv',parse_dates=['date']);ss=s[s.date>='2020-01-01'];a=(ss.ci_low>0)|(ss.ci_high<0)|(ss.tsi_percentile>.5);b=ss.rho>0;c=(ss.rho_t>1.5)|(ss.sigma_percentile<.95)
        gates.append(dict(code=code,days=len(ss),tsi_pass=a.mean(),rho_positive=b.mean(),risk_pass=c.mean(),all_daily=(a&b&c).mean(),confirmed_trend=(ss.state!=0).mean(),uptrend=(ss.state==1).mean(),downtrend=(ss.state==-1).mean(),range=(ss.state==0).mean(),rho_median=ss.rho.median(),rho_t_missing=ss.rho_t.isna().mean()))
        for year,z in ss.groupby(ss.date.dt.year):
            byyear.append(dict(code=code,year=year,trend_days=(z.state!=0).sum(),range_days=(z.state==0).sum(),rho_positive=(z.rho>0).mean()))
        nav=pd.read_csv(ROOT/'results'/f'{code}_full_nav.csv',parse_dates=['date']);tr=pd.read_csv(ROOT/'results'/f'{code}_full_trades.csv',parse_dates=['entry_date','exit_date'])
        # Attribute absolute NAV increments (initial capital=1), including transaction costs and distribution entitlement.
        changes=nav.set_index('date').nav.diff();changes.iloc[0]=nav.nav.iloc[0]-1.;covered=set();attrib=[]
        for _,trade in tr.iterrows():
            part=changes.loc[trade.entry_date:trade.exit_date];idx=part.index
            assert not (set(idx)&covered);covered.update(idx)
            before=nav.loc[nav.date<trade.entry_date,'nav'];entry_equity=before.iloc[-1] if len(before) else 1.
            pnl=part.sum();assert abs(pnl-entry_equity*trade.return_net)<1e-10
            row=trade.to_dict()|dict(code=code,entry_equity=entry_equity,pnl_initial_capital=pnl,status='closed');attrib.append(row);closed.append(row)
        remain=changes[~changes.index.isin(covered)];open_days=nav[(nav.quantity!=0)&(~nav.date.isin(covered))]
        if len(open_days):
            first=open_days.date.min();part=changes.loc[first:];dirn=int(np.sign(open_days.quantity.iloc[0]));kind=open_days.kind.iloc[0]
            attrib.append(dict(code=code,direction=dirn,kind=kind,pnl_initial_capital=part.sum(),status='open',return_net=np.nan));covered.update(part.index)
        rest=changes[~changes.index.isin(covered)].sum();assert abs(rest)<1e-10
        assert abs(sum(x['pnl_initial_capital'] for x in attrib)-(nav.nav.iloc[-1]-1))<1e-10
        for (kind,dirn,status),z in pd.DataFrame(attrib).groupby(['kind','direction','status']):
            groups.append(dict(code=code,kind=kind,direction=dirn,status=status,trades=len(z),pnl_initial_capital=z.pnl_initial_capital.sum(),average_trade_return=z.return_net.mean(),win_rate=(z.return_net.dropna()>0).mean()))
        for variant in ['full','long_only','short_only']:
            sx=s.copy()
            if variant=='long_only':sx.loc[sx.signal<0,'signal']=0
            elif variant=='short_only':sx.loc[sx.signal>0,'signal']=0
            z,tx,t,ac=backtest(g,sx,div[div.code==code],k)
            if variant=='full':np.testing.assert_allclose(z.nav,nav.nav,rtol=1e-12,atol=1e-12)
            z.to_csv(OUT/f'{code}_{variant}_nav.csv',index=False);tx.to_csv(OUT/f'{code}_{variant}_fills.csv',index=False);t.to_csv(OUT/f'{code}_{variant}_trades.csv',index=False)
            directions.append(dict(code=code,strategy=variant,**metrics(z.nav,ac['turnover'],t),average_abs_exposure=z.position.abs().mean()))
        # Counterfactual direction removal changes subsequent entry opportunities: not additive attribution.
        flags.append(dict(code=code,full_replay_matches=True,pnl_attribution_reconciles=True,flat_day_pnl=rest))
    pd.DataFrame(gates).to_csv(OUT/'state_gates.csv',index=False);pd.DataFrame(groups).to_csv(OUT/'pnl_attribution.csv',index=False);pd.DataFrame(closed).sort_values('pnl_initial_capital').to_csv(OUT/'closed_trade_attribution.csv',index=False);pd.DataFrame(directions).to_csv(OUT/'direction_counterfactuals.csv',index=False);pd.DataFrame(byyear).to_csv(OUT/'state_yearly.csv',index=False)
    (OUT/'audit_checks.json').write_text(json.dumps(flags,indent=2))
    if args.optimizer_check:optimizer_checks(ds)
    print(pd.DataFrame(gates).to_string(index=False));print(pd.DataFrame(groups).to_string(index=False));print(pd.DataFrame(directions)[['code','strategy','cagr','max_drawdown','calmar']].to_string(index=False))
if __name__=='__main__':main()
