import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
import numpy as np, pandas as pd
from scipy.optimize import minimize
from model import library, estimate

class ModelTests(unittest.TestCase):

    def test_native_exact_gradient(self):
        y = np.ascontiguousarray(np.random.default_rng(5).normal(size=100))
        z = np.array([0.1, 0.1, -2.0, 0.0, 2.0])
        lib = library()
        g = np.zeros(5)
        lib.gradient(z, y, len(y), g)
        num = []
        for j in range(5):
            a = z.copy()
            b = z.copy()
            a[j] += 1e-05
            b[j] -= 1e-05
            num.append((lib.objective(a, y, len(y)) - lib.objective(b, y, len(y))) / 2e-05)
        np.testing.assert_allclose(g, num, rtol=1e-05, atol=1e-05)

    def test_joint_fit_matches_scipy_objective(self):
        y = np.ascontiguousarray(np.random.default_rng(8).normal(size=180))
        o = np.zeros(7)
        z = np.zeros(5)
        lib = library()
        lib.fit_model(y, len(y), o, z)
        v = lib.objective(z, y, len(y))
        r = minimize(lambda x: lib.objective(np.ascontiguousarray(x), y, len(y)), z, method='BFGS')
        self.assertLess(v - r.fun, 0.02)

    def test_full_bootstrap_deterministic(self):
        y = np.random.default_rng(3).normal(size=80)
        a = estimate(y)
        b = estimate(y)
        self.assertEqual(a['bootstrap_requested'], 1000)
        self.assertEqual(a['ci_low'], b['ci_low'])
        self.assertGreaterEqual(a['bootstrap_valid'], 950)


if __name__=='__main__':unittest.main()
