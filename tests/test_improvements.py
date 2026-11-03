import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
import numpy as np,pandas as pd
from improvement_signals import make_signals

def fixture(n=20,rho=-.1):
    s=pd.DataFrame(dict(date=pd.date_range('2020-01-01',periods=n),adjclose=np.ones(n)*11,ma=np.ones(n)*10,rsi=np.ones(n)*20))
    fits=[dict(i=i,tsi=1.,sigma=1.,ci_low=.5,ci_high=1.5,rho=rho,rho_t=-1.,fit_ok=True) for i in range(n)]
    return s,fits
class Improvements(unittest.TestCase):
    def test_uncertain_no_rsi_trade(self):
        s,f=fixture();z=make_signals(s,f,'neutral_10');self.assertTrue((z.signal==0).all())
        self.assertTrue((make_signals(s,f,'baseline_long').signal==1).all())
    def test_remove_rho_and_wait_confirmation(self):
        s,f=fixture();z=make_signals(s,f,'no_rho_3');self.assertTrue((z.signal.iloc[:2]==0).all());self.assertEqual(z.signal.iloc[2],0)
        # Equal sigma sits at percentile 100%; use positive rho t branch to isolate rho gate.
        for r in f:r['rho_t']=2.
        z=make_signals(s,f,'no_rho_3');self.assertEqual(z.signal.iloc[2],1)
        self.assertTrue((make_signals(s,f,'neutral_10').signal==0).all())
    def test_future_does_not_change_past(self):
        s,f=fixture();a=make_signals(s,f,'no_rho_3of5');b=make_signals(s.iloc[:12],f[:12],'no_rho_3of5');pd.testing.assert_frame_equal(a.iloc[:12],b)
    def test_downtrend_never_opens_short(self):
        s,f=fixture(rho=.2)
        for r in f:r.update(tsi=-1.,ci_low=-1.5,ci_high=-.5,rho_t=2.)
        z=make_signals(s,f,'no_rho_3of5');self.assertTrue((z.signal==0).all());self.assertEqual(z.state.iloc[-1],-1)
if __name__=='__main__':unittest.main()
