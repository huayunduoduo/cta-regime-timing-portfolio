import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
import numpy as np,pandas as pd
from engine import backtest,indicators,market_signals

def fixture(prices,signal=1,state=1):
    n=len(prices);g=pd.DataFrame(prices,columns=['open','high','low','close'])
    g['date']=pd.date_range('2020-01-01',periods=n);g['adj_factor']=1.
    s=pd.DataFrame(dict(signal=[signal]*n,state=[state]*n,atr=[1.]*n,rsi=[20.]*n,adjhigh=g.high,adjlow=g.low,adjclose=g.close,ma=[9.]*n))
    div=pd.DataFrame(columns=['ex_date','pay_date','cash_dividend_per_unit'])
    return g,s,div

class EngineTests(unittest.TestCase):
    def run_case(self,prices,signal=1,state=1,**kw):
        g,s,d=fixture(prices,signal,state);return backtest(g,s,d,1,start=g.date.iloc[1],end=g.date.iloc[-1],fee=0,slip=0,**kw)
    def test_long_stop_uses_high_not_close(self):
        nav,fill,tr,_=self.run_case([(10,10,10,10),(10,15,9,11),(11,12,10,11)])
        self.assertEqual(fill.iloc[-1].reference_price,11);self.assertEqual(tr.iloc[-1].reason,'gap_atr')
        self.assertEqual(nav.stop_today.iloc[-1],14)
    def test_short_stop_uses_low(self):
        nav,fill,tr,_=self.run_case([(10,10,10,10),(10,11,5,9),(8,10,7,9)],signal=-1)
        self.assertEqual(nav.stop_today.iloc[-1],6);self.assertEqual(fill.iloc[-1].reference_price,8)
    def test_new_entry_not_stopped_same_day(self):
        nav,fill,tr,_=self.run_case([(10,10,10,10),(10,15,1,10)])
        self.assertEqual(len(fill),1);self.assertEqual(len(tr),0)
    def test_intraday_fixed_stop_filled_at_stop(self):
        nav,fill,tr,_=self.run_case([(10,10,10,10),(10,11,10,10),(11,12,9,11)])
        self.assertEqual(fill.iloc[-1].reference_price,10);self.assertEqual(tr.reason.iloc[0],'daily_atr')
    def test_exact_touch_does_not_trigger_strict_rule(self):
        nav,fill,tr,_=self.run_case([(10,10,10,10),(10,11,10,10),(11,12,10,11)])
        self.assertEqual(len(fill),1)
    def test_rsi_exit_thresholds(self):
        for direction,levels in [(1,[20,41,50]),(-1,[80,59,50])]:
            g,s,d=fixture([(10,10,10,10)]*3,direction,0);s['rsi']=levels
            nav,fill,tr,_=backtest(g,s,d,1,start=g.date.iloc[1],fee=0,slip=0)
            self.assertEqual(len(tr),1);self.assertEqual(tr.reason.iloc[0],'rsi_exit')
    def test_signed_dividend_and_pay_date(self):
        for sign in (1,-1):
            g,s,d=fixture([(10,10,10,10),(10,10,10,10),(9,9,9,9),(9,9,9,9)],sign,1)
            g.loc[2:,'adj_factor']=10/9;s.adjhigh=g.high*g.adj_factor;s.adjlow=g.low*g.adj_factor
            d=pd.DataFrame([dict(ex_date=g.date.iloc[2],pay_date=g.date.iloc[3],cash_dividend_per_unit=1.)])
            nav,_,_,_=backtest(g,s,d,1,variant='no_atr_literal',start=g.date.iloc[1],fee=0,slip=0)
            np.testing.assert_allclose(nav.nav,1);self.assertAlmostEqual(nav.receivable.iloc[1],sign*.1);self.assertEqual(nav.receivable.iloc[2],0)
    def test_costs_both_directions(self):
        for sign in (1,-1):
            g,s,d=fixture([(10,10,10,10)]*2,sign)
            nav,fill,_,_=backtest(g,s,d,1,start=g.date.iloc[1])
            self.assertGreater(fill.commission.iloc[0],0);self.assertGreater(fill.slippage_cash.iloc[0],0);self.assertLess(nav.nav.iloc[0],1)
    def test_no_atr_literal_keeps_trend_trade(self):
        _,_,tr,ac=self.run_case([(10,10,10,10),(10,10,1,1),(1,1,.5,.5)],variant='no_atr_literal')
        self.assertEqual(len(tr),0);self.assertEqual(ac['open_position'],1)
    def test_stop_ratchet_cannot_loosen(self):
        g,s,d=fixture([(10,10,10,10)]*4);s['atr']=[1,1,5,5]
        nav,_,_,_=backtest(g,s,d,1,start=g.date.iloc[1],fee=0,slip=0)
        np.testing.assert_allclose(nav.stop_next_adjusted,9)
    def test_future_cannot_change_past(self):
        g,s,d=fixture([(10,10,10,10)]*5)
        n1,_,_,_=backtest(g,s,d,1,start=g.date.iloc[1],fee=0,slip=0)
        g.loc[4,['open','high','low','close']]=[100,100,100,100]
        n2,_,_,_=backtest(g,s,d,1,start=g.date.iloc[1],fee=0,slip=0)
        pd.testing.assert_frame_equal(n1.iloc[:3],n2.iloc[:3])
    def test_indicators_causal(self):
        rng=np.random.default_rng(12);px=100*np.exp(np.cumsum(rng.normal(0,.01,100)))
        g=pd.DataFrame(dict(date=pd.date_range('2019-01-01',periods=100),open=px,high=px*1.01,low=px*.99,close=px,adj_factor=1.))
        a=indicators(g);b=indicators(g.iloc[:80])
        pd.testing.assert_frame_equal(a.iloc[:80],b)
if __name__=='__main__':unittest.main()
