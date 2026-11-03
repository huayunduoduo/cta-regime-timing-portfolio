"""Independent reconciliation of every primary daily ledger using raw prices."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);args=ap.parse_args()
    d=pd.read_csv(args.data/'etf_daily.csv',dtype={'code':str},parse_dates=['date']);audit=[];dates=None
    for c in ('510300','518880','511010'):
        prices=d[d.code==c].set_index('date').close
        for variant in ('full','always_trend','always_range','no_atr','no_atr_literal','buyhold'):
            z=pd.read_csv(ROOT/'results'/f'{c}_{variant}_nav.csv',parse_dates=['date']);tx=pd.read_csv(ROOT/'results'/f'{c}_{variant}_fills.csv',parse_dates=['date'])
            calc=z.cash.to_numpy()+z.quantity.to_numpy()*prices.loc[z.date].to_numpy()+z.receivable.to_numpy()
            err=np.max(abs(calc-z.nav));assert err<1e-10
            flows=tx.assign(signed_quantity=np.where(tx.side=='buy',tx.quantity,-tx.quantity)).groupby('date').signed_quantity.sum().reindex(z.date,fill_value=0).cumsum()
            assert np.max(abs(flows.to_numpy()-z.quantity))<1e-10
            assert abs(tx.commission.sum()-z.commission.sum())<1e-10
            assert abs(tx.slippage_cash.sum()-z.slippage_cash.sum())<1e-10
            dd=z.nav.to_numpy()/np.maximum.accumulate(np.r_[1,z.nav.to_numpy()])[1:]-1
            assert np.isfinite(dd).all()
            if dates is None:dates=z.date.tolist()
            assert dates==z.date.tolist()
            audit.append(dict(code=c,strategy=variant,days=len(z),nav_identity_max_error=float(err),fills=len(tx),commission_reconciled=True,quantity_reconciled=True))
    (ROOT/'results/ledger_audit.json').write_text(json.dumps(audit,indent=2));print(f'{len(audit)} ledgers reconciled; all dates match')
if __name__=='__main__':main()
