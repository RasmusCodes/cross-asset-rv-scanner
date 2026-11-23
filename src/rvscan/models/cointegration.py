"""Unit-root and cointegration tests implemented from first principles.

- ADF regression with AIC/BIC lag selection and MacKinnon (2010) response-surface p-values
  (coefficients transcribed from MacKinnon, "Critical Values for Cointegration Tests", QED WP 1227, table 2,
  for N = 1..4 series and constant / constant + trend; checked against statsmodels in tests/).
- Engle-Granger two-step test (both orderings available) with the cointegration p-value surface.
- Johansen trace and maximum-eigenvalue tests (constant restricted to the cointegrating space is not used;
  the model is det_order=0: unrestricted constant), critical values from Osterwald-Lenum (1992) as tabulated by
  MacKinnon-Haug-Michelis for up to 6 variables.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

# ------------------------------------------------------------------------------------------------ MacKinnon surfaces
# Asymptotic p-value approximation for tau statistics (MacKinnon 1994, "Approximate asymptotic distribution
# functions for unit-root and cointegration tests", JBES 12; coefficients as tabulated in statsmodels' adfvalues).
# Rows: N = 1..6 variables in the cointegrating regression (N=1 is the plain ADF test). Regression "n" = no
# constant, "c" = constant, "ct" = constant + trend. tau_min/tau_max bound the polynomial regions.
_INF = float("inf")
_TAU_STAR = {"n": [-1.04, -1.53, -2.68, -3.09, -3.07, -3.77], "c": [-1.61, -2.62, -3.13, -3.47, -3.78, -3.93],
             "ct": [-2.89, -3.19, -3.50, -3.65, -3.80, -4.36]}
_TAU_MIN = {"n": [-19.04, -19.62, -21.21, -23.25, -21.63, -25.74], "c": [-18.83, -18.86, -23.48, -28.07, -25.96, -23.27],
            "ct": [-16.18, -21.15, -25.37, -26.63, -26.53, -26.18]}
_TAU_MAX = {"n": [_INF, 1.51, 0.86, 0.88, 1.05, 1.24], "c": [2.74, 0.92, 0.55, 0.61, 0.79, 1.0],
            "ct": [0.7, 0.63, 0.71, 0.93, 1.19, 1.42]}
_SMALL_SCALE = np.array([1.0, 1.0, 1e-2])
_SMALL_P = {
    "n": [[0.6344, 1.2378, 3.2496], [1.9129, 1.3857, 3.5322], [2.7648, 1.4502, 3.4186], [3.4336, 1.4835, 3.19],
          [4.0999, 1.5533, 3.59], [4.5388, 1.5344, 2.9807]],
    "c": [[2.1659, 1.4412, 3.8269], [2.92, 1.5012, 3.9796], [3.4699, 1.4856, 3.164], [3.9673, 1.4777, 2.6315],
          [4.5509, 1.5338, 2.9545], [5.1399, 1.6036, 3.4445]],
    "ct": [[3.2512, 1.6047, 4.9588], [3.6646, 1.5419, 3.6448], [4.0983, 1.5173, 2.9898], [4.5844, 1.5338, 2.8796],
           [5.0722, 1.5634, 2.9472], [5.53, 1.5914, 3.0392]],
}
_LARGE_SCALE = np.array([1.0, 1e-1, 1e-1, 1e-2])
_LARGE_P = {
    "n": [[0.4797, 9.3557, -0.6999, 3.3066], [1.5578, 8.558, -2.083, -3.3549], [2.2268, 6.8093, -3.2362, -5.4448],
          [2.7654, 6.4502, -3.0811, -4.4946], [3.2684, 6.8051, -2.6778, -3.4972], [3.7268, 7.167, -2.3648, -2.8288]],
    "c": [[1.7339, 9.3202, -1.2745, -1.0368], [2.1945, 6.4695, -2.9198, -4.2377], [2.5893, 4.5168, -3.6529, -5.0074],
          [3.0387, 4.5452, -3.3666, -4.1921], [3.5049, 5.2098, -2.9158, -3.3468], [3.9489, 5.8933, -2.5359, -2.721]],
    "ct": [[2.5261, 6.1654, -3.7956, -6.0285], [2.85, 5.272, -3.6622, -5.1695], [3.221, 5.255, -3.2685, -4.1501],
           [3.652, 5.9758, -2.7483, -3.2081], [4.0712, 6.6428, -2.3464, -2.546], [4.4735, 7.1757, -2.0681, -2.1196]],
}

# Finite-sample critical values, regression "c" (MacKinnon 2010, table 2): crit = b0 + b1/T + b2/T^2 + b3/T^3
_CRIT_SURFACE = {
    1: {0.01: (-3.43035, -6.5393, -16.786, -79.433), 0.05: (-2.86154, -2.8903, -4.234, -40.040), 0.10: (-2.56677, -1.5384, -2.809, 0.0)},
    2: {0.01: (-3.89644, -10.9519, -33.527, 0.0), 0.05: (-3.33613, -6.1101, -6.823, 0.0), 0.10: (-3.04445, -4.2412, -2.720, 0.0)},
    3: {0.01: (-4.29374, -14.4354, -33.195, 47.433), 0.05: (-3.74066, -8.5632, -10.852, 27.982), 0.10: (-3.45218, -6.2143, -3.718, 0.0)},
    4: {0.01: (-4.64332, -18.1031, -37.972, 0.0), 0.05: (-4.09600, -11.2349, -11.175, 0.0), 0.10: (-3.81020, -8.3931, -4.137, 0.0)},
    5: {0.01: (-4.95756, -21.8883, -45.142, 0.0), 0.05: (-4.41519, -14.0405, -12.575, 0.0), 0.10: (-4.13157, -10.7417, -3.784, 0.0)},
    6: {0.01: (-5.24568, -25.6688, -57.737, 88.639), 0.05: (-4.70693, -16.9178, -17.492, 60.007), 0.10: (-4.42501, -13.1875, -5.104, 27.877)},
}


def mackinnon_p(tau: float, n_vars: int = 1, regression: str = "c") -> float:
    """Asymptotic MacKinnon (1994) p-value for an ADF / Engle-Granger tau statistic."""
    i = n_vars - 1
    if tau > _TAU_MAX[regression][i]:
        return 1.0
    if tau < _TAU_MIN[regression][i]:
        return 0.0
    if tau <= _TAU_STAR[regression][i]:
        coefs = np.array(_SMALL_P[regression][i]) * _SMALL_SCALE
    else:
        coefs = np.array(_LARGE_P[regression][i]) * _LARGE_SCALE
    return float(stats.norm.cdf(np.polyval(coefs[::-1], tau)))


def mackinnon_crit(n_vars: int, nobs: int) -> dict[float, float]:
    """Finite-sample 1/5/10% critical values for the constant-only case."""
    surf = _CRIT_SURFACE[n_vars]
    return {lvl: b[0] + b[1] / nobs + b[2] / nobs ** 2 + b[3] / nobs ** 3 for lvl, b in surf.items()}


# ------------------------------------------------------------------------------------------------ OLS helper
def ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (beta, se, resid, sigma2) for y = X b + e."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    sigma2 = float(resid @ resid / dof)
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(xtx_inv) * sigma2, 0, None))
    return beta, se, resid, sigma2


# ------------------------------------------------------------------------------------------------ ADF
@dataclass
class ADFResult:
    stat: float
    pvalue: float
    lags: int
    nobs: int
    crit: dict = field(default_factory=dict)
    rho: float = float("nan")          # coefficient on the lagged level (gamma); AR(1) persistence = 1 + rho


def _adf_design(x: np.ndarray, lags: int, regression: str):
    dx = np.diff(x)
    T = len(dx) - lags
    y = dx[lags:]
    cols = [x[lags:-1]]
    for k in range(1, lags + 1):
        cols.append(dx[lags - k:-k])
    if regression in ("c", "ct"):
        cols.append(np.ones(T))
    if regression == "ct":
        cols.append(np.arange(T, dtype=float))
    return y, np.column_stack(cols)


def adf(x, max_lags: int | None = None, regression: str = "c", autolag: str | None = "aic",
        n_vars: int = 1) -> ADFResult:
    """Augmented Dickey-Fuller test of a unit root in `x`.

    autolag in {"aic", "bic", None}: lags chosen on a common sample (as in statsmodels), then the regression is
    re-run on the largest sample for that lag. `n_vars` > 1 switches to Engle-Granger residual p-values.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if max_lags is None:
        max_lags = int(np.ceil(12.0 * (n / 100.0) ** 0.25))
    max_lags = max(0, min(max_lags, n // 2 - 3))
    if autolag:
        best, best_ic = 0, np.inf
        y, X = _adf_design(x, max_lags, regression)   # common sample for every candidate lag length
        T = len(y)
        for p in range(max_lags + 1):
            Xp = np.column_stack([X[:, :1 + p], X[:, 1 + max_lags:]]) if X.shape[1] > 1 + max_lags else X[:, :1 + p]
            _, _, resid, _ = ols(y, Xp)
            llf = -T / 2 * (np.log(2 * np.pi) + np.log(resid @ resid / T) + 1)
            k = Xp.shape[1]
            ic = -2 * llf + (2 * k if autolag == "aic" else np.log(T) * k)
            if ic < best_ic - 1e-12:
                best, best_ic = p, ic
        lags = best
    else:
        lags = max_lags
    y, X = _adf_design(x, lags, regression)
    beta, se, _, _ = ols(y, X)
    stat = float(beta[0] / se[0]) if se[0] > 0 else float("nan")
    p = mackinnon_p(stat, n_vars, regression) if np.isfinite(stat) else float("nan")
    crit = mackinnon_crit(min(n_vars, 6), len(y)) if regression == "c" else {}
    return ADFResult(stat=stat, pvalue=p, lags=lags, nobs=len(y), crit=crit, rho=float(beta[0]))


# ------------------------------------------------------------------------------------------------ Engle-Granger
@dataclass
class EGResult:
    stat: float
    pvalue: float
    beta: np.ndarray            # hedge coefficients on the regressors (excluding the constant)
    alpha: float
    resid: np.ndarray
    lags: int
    nobs: int
    dependent: int = 0          # index of the dependent variable in the input


def engle_granger(y, X, regression: str = "c", autolag: str | None = "aic", max_lags: int | None = None) -> EGResult:
    """Two-step Engle-Granger test: OLS levels regression, then ADF (no constant) on residuals.

    `X` may be 1-D (pair) or 2-D (basket). p-values use the N-variable MacKinnon surface.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[mask], X[mask]
    Z = np.column_stack([np.ones(len(y)), X]) if regression in ("c", "ct") else X
    if regression == "ct":
        Z = np.column_stack([Z, np.arange(len(y), dtype=float)])
    coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
    resid = y - Z @ coef
    res = adf(resid, max_lags=max_lags, regression="n", autolag=autolag, n_vars=1)
    n_vars = X.shape[1] + 1
    p = mackinnon_p(res.stat, n_vars, "c" if regression != "ct" else "ct")
    alpha = float(coef[0]) if regression in ("c", "ct") else 0.0
    beta = coef[1:1 + X.shape[1]] if regression in ("c", "ct") else coef
    return EGResult(stat=res.stat, pvalue=p, beta=np.asarray(beta), alpha=alpha, resid=resid, lags=res.lags, nobs=res.nobs)


def engle_granger_both(a, b, **kw) -> tuple[EGResult, EGResult]:
    """EG is not symmetric; return both orderings (a on b, b on a). Report the weaker (max p) to be conservative."""
    return engle_granger(a, b, **kw), engle_granger(b, a, **kw)


# ------------------------------------------------------------------------------------------------ Johansen
# Critical values (90%, 95%, 99%) for det_order = 0 (constant in VAR, no trend), by n - r (1..6).
_JOH_TRACE = {1: (2.7055, 3.8415, 6.6349), 2: (13.4294, 15.4943, 19.9349), 3: (27.0669, 29.7961, 35.4628),
              4: (44.4929, 47.8545, 54.6815), 5: (65.8202, 69.8189, 77.8202), 6: (91.1090, 95.7542, 104.9637)}
_JOH_MAXEIG = {1: (2.7055, 3.8415, 6.6349), 2: (12.2971, 14.2639, 18.5200), 3: (18.8928, 21.1314, 25.8650),
               4: (25.1236, 27.5858, 32.7172), 5: (31.2379, 33.8777, 39.3693), 6: (37.2786, 40.0763, 45.8662)}


@dataclass
class JohansenResult:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray       # columns = cointegrating vectors (normalised on the first variable)
    trace: np.ndarray
    max_eig: np.ndarray
    trace_crit: np.ndarray          # rows r = 0..n-1, cols 90/95/99
    max_crit: np.ndarray
    rank_trace_95: int
    rank_maxeig_95: int
    nobs: int
    k_ar_diff: int

    def trace_level(self, r: int = 0) -> str:
        """Significance bracket of the trace test of H0: rank <= r ("<1%", "1-5%", "5-10%", ">10%")."""
        t, c = self.trace[r], self.trace_crit[r]
        return "<1%" if t > c[2] else "1-5%" if t > c[1] else "5-10%" if t > c[0] else ">10%"


def johansen(data, k_ar_diff: int = 1) -> JohansenResult:
    """Johansen (1991) reduced-rank test with an unrestricted constant (det_order = 0)."""
    Y = np.asarray(data, dtype=float)
    Y = Y[np.all(np.isfinite(Y), axis=1)]
    T_full, n = Y.shape
    if n > 6:
        raise ValueError("critical values tabulated for up to 6 variables")
    dY = np.diff(Y, axis=0)
    lagged = Y[k_ar_diff:-1]
    Z0 = dY[k_ar_diff:]
    if k_ar_diff > 0:
        Zlags = np.column_stack([dY[k_ar_diff - i:-i] for i in range(1, k_ar_diff + 1)])
        Zdet = np.column_stack([Zlags, np.ones(len(Z0))])
    else:
        Zdet = np.ones((len(Z0), 1))
    # demean via regression on lags + constant
    def resid(A):
        B, *_ = np.linalg.lstsq(Zdet, A, rcond=None)
        return A - Zdet @ B
    R0, R1 = resid(Z0), resid(lagged)
    T = len(R0)
    S00 = R0.T @ R0 / T
    S11 = R1.T @ R1 / T
    S01 = R0.T @ R1 / T
    S10 = S01.T
    # solve |lambda S11 - S10 S00^-1 S01| = 0 via Cholesky of S11
    L = np.linalg.cholesky(S11)
    Linv = np.linalg.inv(L)
    M = Linv @ S10 @ np.linalg.solve(S00, S01) @ Linv.T
    eigval, eigvec = np.linalg.eigh(M)
    order = np.argsort(eigval)[::-1]
    eigval = np.clip(eigval[order], 0, 1 - 1e-12)
    vecs = Linv.T @ eigvec[:, order]
    vecs = vecs / vecs[0:1, :]
    trace = np.array([-T * np.sum(np.log(1 - eigval[r:])) for r in range(n)])
    maxeig = np.array([-T * np.log(1 - eigval[r]) for r in range(n)])
    trace_crit = np.array([_JOH_TRACE[n - r] for r in range(n)])
    max_crit = np.array([_JOH_MAXEIG[n - r] for r in range(n)])
    rank_t = next((r for r in range(n) if trace[r] < trace_crit[r, 1]), n)
    rank_m = next((r for r in range(n) if maxeig[r] < max_crit[r, 1]), n)
    return JohansenResult(eigval, vecs, trace, maxeig, trace_crit, max_crit, rank_t, rank_m, T, k_ar_diff)


# ------------------------------------------------------------------------------------------------ KPSS (stationarity)
def kpss(x, lags: int | None = None) -> tuple[float, float]:
    """KPSS level-stationarity statistic with Bartlett long-run variance; returns (stat, p interpolated)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    e = x - x.mean()
    if lags is None:
        lags = int(np.ceil(12 * (n / 100) ** 0.25))
    s = np.cumsum(e)
    lrv = e @ e / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        lrv += 2 * w * (e[k:] @ e[:-k]) / n
    stat = float(s @ s / (n ** 2 * lrv))
    crit = np.array([0.347, 0.463, 0.574, 0.739])   # 10%, 5%, 2.5%, 1%
    pv = np.array([0.10, 0.05, 0.025, 0.01])
    p = float(np.interp(stat, crit, pv, left=0.10, right=0.01))
    return stat, p
