"""Causal, preregistered state ablations; retain original model estimates."""
import numpy as np
import pandas as pd
from engine import market_signals

VARIANTS=('baseline_full','baseline_long','neutral_10','no_rho_10','no_rho_3','no_rho_3of5','risk_separate_3of5','buyhold')

def make_signals(features,fits,variant):
    if variant not in VARIANTS:raise ValueError(variant)
    s=market_signals(features.copy(),fits)
    if variant in ('baseline_full','buyhold'):return s
    if variant=='baseline_long':
        s.loc[s.signal<0,'signal']=0
        return s
    tsi=(s.ci_low>0)|(s.ci_high<0)|(s.tsi_percentile>.5)
    risk=(s.rho_t>1.5)|(s.sigma_percentile<.95)
    if variant=='risk_separate_3of5':risk=s.sigma_percentile<.95
    good=(s.fit_ok.eq(True)&tsi&risk).to_numpy()
    if variant=='neutral_10':good = good & (s.rho>0).to_numpy()
    raw=np.where(good,np.sign(s.tsi.fillna(0)).astype(int),0)
    state=np.zeros(len(s),dtype=int);streak=0
    for i in range(len(s)):
        streak=streak+1 if good[i] else 0
        if variant in ('no_rho_3of5','risk_separate_3of5'):
            # Require a full five-day window and current evidence in same direction.
            if i>=4 and raw[i]!=0 and np.sum(raw[i-4:i+1]==raw[i])>=3:state[i]=raw[i]
        elif streak >= (3 if variant=='no_rho_3' else 10):state[i]=raw[i]
    s['state']=state;s['raw_direction']=raw;s['trend_condition']=good
    s['signal']=((state==1)&(s.adjclose>s.ma)).astype(int)
    s['regime']=np.where(state>0,'uptrend',np.where(state<0,'downtrend','uncertain'))
    return s
