from pathlib import Path
import ctypes as ct, subprocess, platform
import numpy as np
from scipy.optimize import least_squares
ROOT = Path(__file__).parent
LIB = None

def library():
    global LIB
    if LIB is None:
        path = ROOT / 'native' / ('model.dll' if platform.system() == 'Windows' else 'model.so')
        if not path.exists():
            subprocess.run(['g++', '-O3', '-std=c++17', '-shared', *(['-fPIC'] if platform.system() != 'Windows' else []), str(ROOT / 'native/model.cpp'), '-o', str(path)], check=True)
        LIB = ct.CDLL(str(path))
        arr = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        LIB.fit_model.argtypes = [arr, ct.c_int, arr, arr]
        LIB.fit_model.restype = ct.c_int
        LIB.bootstrap.argtypes = [arr, ct.c_int, arr, ct.c_int, ct.c_ulonglong, arr]
        LIB.bootstrap.restype = ct.c_int
        LIB.gradient.argtypes = [arr, arr, ct.c_int, arr]
        LIB.objective.argtypes = [arr, arr, ct.c_int]
        LIB.objective.restype = ct.c_double
    return LIB

def clean(c):
    c = np.array(c, float)
    med = np.median(c)
    mad = np.median(abs(c - med))
    if mad < 1e-12:
        return c
    bad = abs(0.67448975 * (c - med) / mad) > 3.5
    if bad.any():
        x = np.linspace(-1, 1, len(c))
        o = least_squares(lambda b: (b[0] + b[1] * x - c) / mad, [med, 0], loss='huber')
        c[bad] = (o.x[0] + o.x[1] * x)[bad]
    return c

def estimate(y, B=1000, seed=42):
    y = np.ascontiguousarray(y, dtype=float)
    out = np.zeros(7)
    z = np.zeros(5)
    lib = library()
    ok = lib.fit_model(y, len(y), out, z)
    if not ok:
        from scipy.optimize import minimize

        def fun(v):
            v = np.ascontiguousarray(v)
            gg = np.zeros(5)
            lib.gradient(v, y, len(y), gg)
            return (lib.objective(v, y, len(y)), gg)
        sol = minimize(fun, z, jac=True, method='L-BFGS-B', bounds=[(-100, 100), (-4, 4), (-25, 10), (-20, 20), (-20, 20)], options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-06})
        if sol.success:
            z = np.ascontiguousarray(sol.x)
            aa, bb = np.exp(z[3:])
            out[:5] = [z[0], np.tanh(z[1]), np.exp(z[2]), 0.999 * aa / (1 + aa + bb), 0.999 * bb / (1 + aa + bb)]
            hh = max(np.var(y), 1e-06)
            old = 0.0
            for i in range(1, len(y)):
                if i > 1:
                    hh = out[2] + out[3] * old ** 2 + out[4] * hh
                old = y[i] - out[0] - out[1] * y[i - 1]
            out[5] = np.sqrt(hh)
            out[6] = out[0] / out[5]
            ok = 1
    H = np.zeros((5, 5))
    for j in range(5):
        step = 0.0001 * (1 + abs(z[j]))
        a = z.copy()
        b = z.copy()
        a[j] += step
        b[j] -= step
        ga = np.zeros(5)
        gb = np.zeros(5)
        lib.gradient(a, y, len(y), ga)
        lib.gradient(b, y, len(y), gb)
        H[:, j] = (ga - gb) / (2 * step)
    H = (H + H.T) / 2
    cov = np.linalg.pinv(H)
    se = (1 - out[1] ** 2) * np.sqrt(cov[1, 1]) if np.linalg.eigvalsh(H).min() > 0 and cov[1, 1] > 0 else np.nan
    boot = np.zeros(B)
    valid = lib.bootstrap(y, len(y), z, B, seed, boot) if ok else 0
    if not ok:
        boot[:] = np.nan
    attempts = B if ok else 0
    for retry in range(1000):
        missing = np.flatnonzero(~np.isfinite(boot))
        if not len(missing) or not ok:
            break
        extra = np.zeros(len(missing))
        lib.bootstrap(y, len(y), z, len(extra), seed + 1000000 * (retry + 1), extra)
        boot[missing] = extra
        attempts += len(extra)
    valid = int(np.isfinite(boot).sum()) if ok else 0
    if not ok or valid != B:
        raise RuntimeError(f'Incomplete fit/bootstrap: point={ok}, valid={valid}/{B}')
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return dict(zip(['mu', 'rho', 'omega', 'alpha', 'beta', 'sigma', 'tsi'], out)) | dict(rho_t=out[1] / se if se > 0 else np.nan, ci_low=lo, ci_high=hi, fit_ok=bool(ok), bootstrap_valid=valid, bootstrap_requested=B, bootstrap_attempts=attempts)
