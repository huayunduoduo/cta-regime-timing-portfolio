"""Accounting and causality tests for the new portfolio engine."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
import numpy as np
import pandas as pd
from portfolio_research import Config, build_targets, simulate, capped_weights, historical_es, rank_candidates


def fixture(n=8):
    dates=pd.bdate_range('2020-01-01',periods=n)
    fields={'open':np.ones((n,3))*10,'close':np.ones((n,3))*10,'adj_factor':np.ones((n,3))}
    div=pd.DataFrame(columns=['code','ex_date','pay_date','cash_dividend_per_unit'])
    return dates,fields,div

class PortfolioTests(unittest.TestCase):
    def test_signal_executes_next_open(self):
        dates,f,d=fixture();w=np.zeros((len(dates),3));w[3:,0]=1
        nav,tx=simulate(dates,f,d,w,np.ones(len(dates),bool),start=str(dates[1].date()),end=str(dates[-1].date()),fee=0,slip=0)
        self.assertEqual(tx.date.iloc[0],dates[4]);self.assertEqual(nav.quantity_510300.iloc[2],0)

    def test_dividend_entitlement_pay_date_and_no_double_count(self):
        dates,f,d=fixture();f['open'][3:,0]=9;f['close'][3:,0]=9;f['adj_factor'][3:,0]=10/9
        d=pd.DataFrame([dict(code='510300',ex_date=dates[3],pay_date=dates[5],cash_dividend_per_unit=1.)])
        w=np.tile([1.,0.,0.],(len(dates),1));flags=np.zeros(len(dates),bool)
        nav,tx=simulate(dates,f,d,w,flags,start=str(dates[1].date()),end=str(dates[-1].date()),fee=0,slip=0)
        np.testing.assert_allclose(nav.nav,1)
        self.assertAlmostEqual(nav.loc[nav.date==dates[3],'receivable'].iloc[0],.1)
        self.assertAlmostEqual(nav.loc[nav.date==dates[4],'cash'].iloc[0],0)
        self.assertAlmostEqual(nav.loc[nav.date==dates[5],'cash'].iloc[0],.1)

    def test_unpaid_dividend_cannot_finance_purchase(self):
        dates,f,d=fixture();f['open'][3:,0]=9;f['close'][3:,0]=9
        d=pd.DataFrame([dict(code='510300',ex_date=dates[3],pay_date=dates[6],cash_dividend_per_unit=1.)])
        w=np.tile([1.,0.,0.],(len(dates),1));w[2:]=[0.,1.,0.]
        nav,tx=simulate(dates,f,d,w,np.ones(len(dates),bool),start=str(dates[1].date()),end=str(dates[-1].date()),fee=0,slip=0)
        row=nav[nav.date==dates[3]].iloc[0]
        self.assertAlmostEqual(row.quantity_518880,.09);self.assertGreaterEqual(nav.cash.min(),-1e-12)

    def test_entry_costs_and_sale_before_purchase(self):
        dates,f,d=fixture();w=np.tile([1.,0.,0.],(len(dates),1));w[3:]=[0.,1.,0.]
        nav,tx=simulate(dates,f,d,w,np.ones(len(dates),bool),start=str(dates[1].date()),end=str(dates[-1].date()))
        self.assertAlmostEqual(nav.quantity_510300.iloc[0],1/(10*1.001*1.001))
        self.assertTrue((nav.cash>=-1e-12).all());self.assertGreater(tx.commission.sum(),0)
        t=tx[tx.date==dates[4]];self.assertEqual(t.side.tolist(),['sell','buy'])
        # Independent reconstruction, rather than trusting the ledger's stored error.
        amounts=nav[['quantity_510300','quantity_518880','quantity_511010']].to_numpy()
        np.testing.assert_allclose(nav.nav,nav.cash+np.sum(amounts*f['close'][1:],axis=1)+nav.receivable,atol=1e-12)

    def test_future_price_changes_do_not_change_past_targets_or_fills(self):
        n=390;dates,f,d=fixture(n);rng=np.random.default_rng(18)
        f['close']=10*np.exp(np.cumsum(rng.normal(0,.005,(n,3)),axis=0));f['open']=f['close'].copy()
        configs=[Config(trend='ensemble',allocation='invvol',es_budget=.015),Config(allocation='riskparity',frequency='weekly')]
        a,flags,_=build_targets(dates,f,configs)
        prefix=340;b,_,_=build_targets(dates[:prefix],{k:v[:prefix] for k,v in f.items()},configs)
        for c in configs:np.testing.assert_allclose(a[c.name][:prefix],b[c.name],rtol=0,atol=0)
        mutated={k:v.copy() for k,v in f.items()};mutated['close'][360:]*=3
        cc,_,_=build_targets(dates,mutated,configs)
        np.testing.assert_allclose(a[configs[0].name][:360],cc[configs[0].name][:360],rtol=0,atol=0)

    def test_ranking_ignores_future_nav(self):
        dates=pd.bdate_range('2015-01-01',periods=900)
        nav=pd.DataFrame(dict(date=dates,nav=np.exp(np.arange(900)*.0001),turnover=0.,cost=0.))
        changed=nav.copy();changed.loc[changed.date>'2016-12-31','nav']*=4
        a=rank_candidates({'a':nav},'2016-12-31');b=rank_candidates({'a':changed},'2016-12-31')
        pd.testing.assert_frame_equal(a,b)

    def test_cap_and_fractional_es_tail(self):
        w=capped_weights([100,2,1]);self.assertAlmostEqual(w.sum(),1);self.assertLessEqual(w.max(),.6)
        # 25% tail of 6 samples = worst + half of second-worst.
        self.assertAlmostEqual(historical_es([1,2,3,4,5,6],alpha=.75),(6+.5*5)/1.5)

if __name__=='__main__':unittest.main()
