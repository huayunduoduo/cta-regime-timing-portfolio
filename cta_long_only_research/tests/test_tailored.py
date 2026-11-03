import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import indicators, backtest
from tailored_long import Rule, slow_signals, combine


def bars(px):
    return pd.DataFrame({'date':pd.bdate_range('2020-01-01',periods=len(px)),
        'open':px, 'high':px, 'low':px, 'close':px, 'adj_factor':1.})


class TailoredTests(unittest.TestCase):
    def test_negative_exit_never_becomes_short_entry(self):
        g=bars([10.]*8);s=indicators(g);s['state']=1;s['signal']=[-1,1,0,-1,-1,-1,-1,-1]
        div=pd.DataFrame(columns=['ex_date','pay_date','cash_dividend_per_unit'])
        nav,tx,tr,_=backtest(g,s,div,3,variant='no_atr',start=g.date.iloc[1],end=g.date.iloc[-1],long_only=True,fee=0,slip=0)
        self.assertTrue((nav.quantity>=0).all())
        self.assertEqual(tx.side.tolist(),['buy','sell'])
        self.assertEqual(tx.date.tolist(),[g.date.iloc[2],g.date.iloc[4]])
        self.assertEqual(len(tr),1)

    def test_slow_confirmation_full_window_and_no_future(self):
        px=np.r_[np.arange(100.,150.),np.arange(149.,99.,-1)]
        g=bars(px);rule=Rule('slow',family='slow',ma=20,band=.0025,confirm=5)
        s=slow_signals(indicators(g),rule)
        self.assertTrue(s.signal.iloc[:23].eq(0).all())
        self.assertEqual(s.signal.iloc[23],1)
        prefix=75
        b=slow_signals(indicators(g.iloc[:prefix]),rule)
        pd.testing.assert_frame_equal(s.iloc[:prefix],b)
        altered=g.copy();altered.loc[80:,['open','high','low','close']]*=4
        c=slow_signals(indicators(altered),rule)
        pd.testing.assert_frame_equal(s.iloc[:80],c.iloc[:80])
        self.assertTrue(s.signal.eq(-1).any())

    def test_unpaid_distribution_cannot_finance_reentry(self):
        g=bars([10.,10.,9.,9.,9.,9.,9.,9.]);g.loc[2:,'adj_factor']=10/9
        s=indicators(g);s['state']=1;s['signal']=[1,-1,1,1,1,1,1,1]
        div=pd.DataFrame([{'ex_date':g.date.iloc[2], 'pay_date':g.date.iloc[6],
                           'cash_dividend_per_unit':1.}])
        nav,tx,_,_=backtest(g,s,div,3,variant='no_atr',start=g.date.iloc[1],
                            end=g.date.iloc[-1],long_only=True,fee=0,slip=0)
        row=nav[nav.date==g.date.iloc[3]].iloc[0]
        self.assertAlmostEqual(row.receivable,.1)
        self.assertAlmostEqual(row.quantity,.1)
        self.assertAlmostEqual(row.cash,0.)
        self.assertGreaterEqual(nav.cash.min(),-1e-12)
        np.testing.assert_allclose(nav.nav,1.)

    def test_combined_accounts_use_capital_weighted_turnover(self):
        dates=pd.bdate_range('2020-01-01',periods=2)
        a=pd.DataFrame({'date':dates,'nav':[2.,2.],'turnover':[1.,.5]})
        b=pd.DataFrame({'date':dates,'nav':[1.,1.],'turnover':[0.,0.]})
        c=combine([a,b])
        np.testing.assert_allclose(c.nav,[1.5,1.5])
        np.testing.assert_allclose(c.turnover,[.5,1/3])


if __name__ == '__main__':
    unittest.main()
