"""Daily, signed inventory ledger for the paper's single-asset experiments.
No intraday stop on entry day: paper chooses entry/exit branch by start-day inventory.
"""
import numpy as np
import pandas as pd


def indicators(g, rsi_method='sma', atr_method='expanding_sma'):
    c=g.close*g.adj_factor; delta=c.diff()
    er=((c-c.shift(30)).abs()/delta.abs().rolling(30).sum()).clip(0,1).fillna(0)
    window=(250-229*er).round().astype(int); ma_window=(20-15*er).round().astype(int)
    up=delta.clip(lower=0); down=-delta.clip(upper=0)
    if rsi_method=='sma':
        gain=up.rolling(14).mean(); loss=down.rolling(14).mean()
    else:
        # Wilder smoothing seeded by first 14 changes, rather than ewm first price.
        def wilder(x):
            a=np.full(len(x),np.nan)
            if len(x)>14:
                a[14]=x.iloc[1:15].mean()
                for i in range(15,len(x)):a[i]=(13*a[i-1]+x.iloc[i])/14
            return pd.Series(a,index=x.index)
        gain=wilder(up);loss=wilder(down)
    rsi=(100*gain/(gain+loss)).mask((gain+loss)==0,50)
    hi=g.high*g.adj_factor;lo=g.low*g.adj_factor
    tr=pd.concat([hi-lo,(hi-c.shift()).abs(),(lo-c.shift()).abs()],axis=1).max(axis=1)
    if atr_method=='local_sma':
        atr=tr.rolling(14).apply(lambda x:np.clip(x,*np.quantile(x,[.05,.95])).mean(),raw=True)
    else:
        # Clip each TR by information available on its own date, never later quantiles.
        clipped=tr.clip(lower=tr.expanding(30).quantile(.05),upper=tr.expanding(30).quantile(.95))
        atr=clipped.rolling(14).mean()
    return pd.DataFrame(dict(date=g.date,window=window,ma_window=ma_window,er=er,
        ma=[c.iloc[max(0,i-int(ma_window.iloc[i])+1):i+1].mean() for i in range(len(g))],
        rsi=rsi,atr=atr,tr=tr,adjclose=c,adjhigh=hi,adjlow=lo))


def market_signals(s,fits):
    s=s.join(pd.DataFrame(fits).set_index('i')); ts=[];vs=[];streak=0;rows=[]
    for _,r in s.iterrows():
        state=0;good=False;tp=sp=np.nan
        if r.get('fit_ok')==True:
            ts.append(abs(r.tsi));vs.append(r.sigma)
            tp=np.mean(np.asarray(ts)<=abs(r.tsi));sp=np.mean(np.asarray(vs)<=r.sigma)
            good=((r.ci_low>0 or r.ci_high<0) or tp>.5) and r.rho>0 and (r.rho_t>1.5 or sp<.95)
            streak=streak+1 if good else 0
            if streak>=10:state=int(np.sign(r.tsi))
        else:streak=0
        signal=(1 if state>0 and r.adjclose>r.ma else -1 if state<0 and r.adjclose<r.ma else 0) if state else (1 if r.rsi<30 else -1 if r.rsi>70 else 0)
        rows.append((state,signal,bool(good),streak,tp,sp))
    s[['state','signal','trend_condition','confirmation_days','tsi_percentile','sigma_percentile']]=rows
    return s


def backtest(g,s,div,k,variant='full',start='2020-01-01',end='2025-12-31',fee=.001,slip=.001,ratchet=True):
    start=pd.Timestamp(start);end=pd.Timestamp(end)
    cash=1.;qty=0.;receivable=0.;pending_div=[];order=0;order_kind='';exit_next=False
    peak=trough=stop=np.nan;entry=None;rows=[];fills=[];trades=[];turnover=0.;commission=0.;slippage=0.
    sel=g[(g.date>=start)&(g.date<=end)]
    def target(rr):
        if variant=='buyhold':return 1,'buyhold'
        if variant=='always_trend':return int(np.sign(rr.adjclose-rr.ma)),'trend'
        if variant=='always_range':return (1 if rr.rsi<30 else -1 if rr.rsi>70 else 0),'range'
        return int(rr.signal),'trend' if rr.state else 'range'
    first=sel.index[0]
    if first>0:order,order_kind=target(s.iloc[first-1])
    elif variant=='buyhold':order,order_kind=1,'buyhold'
    for i,r in sel.iterrows():
        date=r.date;nav_before=cash+qty*r.open+receivable
        # Signed entitlement: short positions owe the distribution.
        for _,ev in div[div.ex_date==date].iterrows():
            if qty:
                amt=qty*ev.cash_dividend_per_unit;pending_div.append((ev.pay_date,amt));receivable+=amt
        for event in pending_div[:]:
            if event[0]<=date:cash+=event[1];receivable-=event[1];pending_div.remove(event)
        was=qty!=0;notional=0.;day_fee=day_slip=0.;stop_today=stop/r.adj_factor if np.isfinite(stop) else np.nan
        def execute(delta,reference,reason):
            nonlocal cash,qty,notional,day_fee,day_slip
            fill=reference*(1+slip*np.sign(delta));fee_cash=abs(delta*fill)*fee
            cash-=delta*fill+fee_cash;qty+=delta;notional+=abs(delta*reference)
            day_fee+=fee_cash;day_slip+=abs(delta)*(abs(fill-reference))
            fills.append(dict(date=date,side='buy' if delta>0 else 'sell',quantity=abs(delta),reference_price=reference,fill_price=fill,commission=fee_cash,slippage_cash=abs(delta)*abs(fill-reference),reason=reason))
            return fill
        closed=False
        if was:
            price=None;reason=''
            if exit_next:price=r.open;reason='rsi_exit' if entry['kind']=='range' else 'opposite_signal_assumption'
            elif entry['kind']=='trend' and variant not in ('no_atr','no_atr_literal','buyhold') and np.isfinite(stop_today):
                if (qty>0 and r.open<stop_today) or (qty<0 and r.open>stop_today):price=r.open;reason='gap_atr'
                elif (qty>0 and r.low<stop_today) or (qty<0 and r.high>stop_today):price=stop_today;reason='daily_atr'
            if price is not None:
                fill=execute(-qty,price,reason);closed=True
                equity=cash+receivable
                trades.append(dict(entry_date=entry['date'],exit_date=date,direction=entry['direction'],kind=entry['kind'],entry_price=entry['price'],exit_price=fill,return_net=equity/entry['equity']-1,reason=reason))
                entry=None;stop=np.nan
        elif order:
            # Same absolute initial exposure for either direction, 100% after entry costs.
            sign=order;equity=cash+receivable;fill_price=r.open*(1+slip*sign)
            amount=equity/(fill_price*(1+fee));fill=execute(sign*amount,r.open,'entry')
            entry=dict(date=date,direction=sign,kind=order_kind,equity=equity,price=fill)
            peak=r.open*r.adj_factor;trough=peak;stop=np.nan
        exit_next=False;order=0
        if qty:
            peak=max(peak,s.adjhigh.iloc[i]);trough=min(trough,s.adjlow.iloc[i])
            if entry['kind']=='trend' and variant!='buyhold':
                new=peak-k*s.atr.iloc[i] if qty>0 else trough+k*s.atr.iloc[i]
                if np.isfinite(new):stop=new if not np.isfinite(stop) or not ratchet else max(stop,new) if qty>0 else min(stop,new)
                if variant=='no_atr':exit_next=(qty>0 and s.signal.iloc[i]<0) or (qty<0 and s.signal.iloc[i]>0)
            elif entry['kind']=='range':exit_next=(qty>0 and s.rsi.iloc[i]>40) or (qty<0 and s.rsi.iloc[i]<60)
        # No intraday re-entry. A close-generated order may execute the next day.
        else:
            order,order_kind=target(s.iloc[i])
        nav=cash+qty*r.close+receivable
        if nav<=0:raise RuntimeError('Account insolvent: explicit margin/liquidation model required')
        traded=notional/max(nav_before,1e-12);turnover+=traded;commission+=day_fee;slippage+=day_slip
        rows.append(dict(date=date,nav=nav,cash=cash,quantity=qty,receivable=receivable,position=qty*r.close/nav,stop_today=stop_today,stop_next_adjusted=stop,turnover=traded,commission=day_fee,slippage_cash=day_slip,state=s.state.iloc[i],kind=entry['kind'] if entry else 'cash'))
    # Mark to market at the endpoint; no invented terminal trade or artificial win rate.
    return pd.DataFrame(rows),pd.DataFrame(fills),pd.DataFrame(trades),dict(open_position=int(np.sign(qty)),commission=commission,slippage_cash=slippage,turnover=turnover)


def metrics(nav,turnover=0,trades=None):
    v=np.r_[1.,np.asarray(nav)];r=v[1:]/v[:-1]-1;n=len(r);cagr=v[-1]**(252/n)-1;dd=v/np.maximum.accumulate(v)-1;sd=np.std(r,ddof=1)
    longest=cur=0
    for below in dd<0:cur=cur+1 if below else 0;longest=max(longest,cur)
    out=dict(total_return=v[-1]-1,cagr=cagr,volatility=sd*np.sqrt(252),sharpe=np.mean(r)/sd*np.sqrt(252) if sd else np.nan,max_drawdown=dd.min(),calmar=cagr/abs(dd.min()) if dd.min() else np.nan,annual_turnover=turnover*252/n,max_underwater_days=longest,days=n)
    if trades is not None:
        rr=trades.return_net if len(trades) else pd.Series(dtype=float);win=rr[rr>0];loss=rr[rr<0]
        out.update(closed_trades=len(rr),win_rate=(rr>0).mean(),payoff_ratio=win.mean()/abs(loss.mean()) if len(win) and len(loss) else np.nan)
    return out
