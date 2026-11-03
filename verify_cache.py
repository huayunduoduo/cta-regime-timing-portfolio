"""Independently recompute one cached 1000-draw fit per asset and check input hashes."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,pandas as pd
from model import clean,estimate
from engine import indicators
ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);args=ap.parse_args()
    d=pd.read_csv(args.data/'etf_daily.csv',dtype={'code':str},parse_dates=['date']);audit=[]
    for c in ('510300','518880','511010'):
        g=d[d.code==c].sort_values('date').reset_index(drop=True);key=hashlib.sha256(g.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()[:16]
        fits=json.loads((ROOT/'cache'/f'{c}_{key}.json').read_text());ref=fits[len(fits)//2];i=ref['i'];s=indicators(g);n=int(s.window.iloc[i]);px=s.adjclose.iloc[i-n:i+1]
        fresh=estimate(np.diff(np.log(clean(px)))*100,1000,20260610+int(c)*10000+i)
        errors={k:abs(fresh[k]-ref[k]) for k in ('mu','rho','sigma','tsi','ci_low','ci_high')}
        if max(errors.values())>1e-10:raise AssertionError(errors)
        audit.append(dict(code=c,row=i,date=str(g.date.iloc[i].date()),bootstrap=1000,max_absolute_difference=max(errors.values()),input_hash=key))
    (ROOT/'results/cache_recompute_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
