"""Stability of relationships over time.

- rolling_eg: Engle-Granger p-value, hedge ratio and half-life in rolling windows (share of windows significant,
  dispersion of the hedge ratio).
- cusum: Brown-Durbin-Evans CUSUM of recursive residuals of the cointegrating regression, with the standard 5%
  boundary a = 0.948 (1%: 1.143, 10%: 0.850).
- persistence: does in-sample significance carry over to the next window (formation -> test)?
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cointegration import engle_granger
from .ou import fit_ou

CUSUM_A = {0.10: 0.850, 0.05: 0.948, 0.01: 1.143}


@dataclass
class RollingStability:
    table: pd.DataFrame          # one row per window end: p, beta, half_life
    share_significant: float
    beta_cv: float               # coefficient of variation of the hedge ratio across windows
    half_life_median: float
    half_life_iqr: tuple


def rolling_eg(y: pd.Series, x: pd.Series, window: int = 504, step: int = 63, alpha: float = 0.05) -> RollingStability:
    df = pd.concat([y, x], axis=1, join="inner").dropna()
    rows = []
    for end in range(window, len(df) + 1, step):
        w = df.iloc[end - window:end]
        res = engle_granger(w.iloc[:, 0].values, w.iloc[:, 1].values)
        ou = fit_ou(res.resid)
        rows.append({"end": w.index[-1], "p": res.pvalue, "beta": float(res.beta[0]), "half_life": ou.half_life_days})
    tab = pd.DataFrame(rows).set_index("end") if rows else pd.DataFrame(columns=["p", "beta", "half_life"])
    if tab.empty:
        return RollingStability(tab, float("nan"), float("nan"), float("nan"), (float("nan"), float("nan")))
    hl = tab["half_life"].replace(np.inf, np.nan)
    beta_cv = float(tab["beta"].std() / abs(tab["beta"].mean())) if tab["beta"].mean() != 0 else float("inf")
    return RollingStability(tab, float((tab["p"] < alpha).mean()), beta_cv, float(hl.median()),
                            (float(hl.quantile(0.25)), float(hl.quantile(0.75))))


@dataclass
class CusumResult:
    stat_path: np.ndarray
    bound: np.ndarray
    crossed: bool
    first_cross: int | None


def cusum(y, x, alpha: float = 0.05, k_min: int | None = None) -> CusumResult:
    """CUSUM of standardised recursive residuals of y on [1, x]."""
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(y)), np.asarray(x, dtype=float)])
    T, k = X.shape
    k0 = k_min or max(k + 10, 30)
    w = np.full(T, np.nan)
    XtX = X[:k0].T @ X[:k0]
    Xty = X[:k0].T @ y[:k0]
    for t in range(k0, T):
        inv = np.linalg.pinv(XtX)
        b = inv @ Xty
        xt = X[t]
        f = 1.0 + xt @ inv @ xt
        w[t] = (y[t] - xt @ b) / np.sqrt(f)
        XtX += np.outer(xt, xt)
        Xty += xt * y[t]
    wr = w[k0:]
    sd = np.std(wr, ddof=1)
    W = np.cumsum(wr) / sd
    n = len(wr)
    r = np.arange(1, n + 1)
    a = CUSUM_A[alpha]
    bound = a * np.sqrt(n) + 2 * a * r / np.sqrt(n)
    over = np.where(np.abs(W) > bound)[0]
    return CusumResult(W, bound, bool(len(over)), int(over[0]) + k0 if len(over) else None)


def persistence(y: pd.Series, x: pd.Series, formation: int = 504, test: int = 252, alpha: float = 0.05) -> pd.DataFrame:
    """For consecutive (formation, test) windows: was the pair significant in formation, and again in test?"""
    df = pd.concat([y, x], axis=1, join="inner").dropna()
    rows = []
    for start in range(0, len(df) - formation - test + 1, test):
        f = df.iloc[start:start + formation]
        t = df.iloc[start + formation:start + formation + test]
        pf = engle_granger(f.iloc[:, 0].values, f.iloc[:, 1].values).pvalue
        pt = engle_granger(t.iloc[:, 0].values, t.iloc[:, 1].values).pvalue
        rows.append({"formation_end": f.index[-1], "p_formation": pf, "p_test": pt,
                     "sig_formation": pf < alpha, "sig_test": pt < alpha})
    return pd.DataFrame(rows)
