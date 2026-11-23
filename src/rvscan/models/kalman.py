"""Kalman filter for dynamic hedge ratios.

Observation:  y_t = alpha_t + beta_t' x_t + e_t,          e_t ~ N(0, R)
State:        theta_t = theta_{t-1} + w_t,                w_t ~ N(0, Q),  Q = q * diag(scale)

The state noise is parameterised by `delta` (the share of the state variance refreshed each step, as in Chan 2013)
or estimated by maximising the prediction-error likelihood over a log grid (concentrated in R). The filter is
one-sided: the hedge ratio at t uses data up to t only, and `beta_prior` is the estimate available before
observing y_t (the one a trader can use to size the position at t).

A known property worth stating: in random-walk-coefficient models the likelihood often keeps rising as Q grows
because the states absorb mean-reverting spread persistence. The MLE then sits on the grid's upper boundary and
the "hedge ratio" mostly re-centres the spread. `fit_mle` reports when that happens (`at_boundary`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanResult:
    alpha: np.ndarray          # filtered intercept path (posterior, uses y_t)
    beta: np.ndarray           # filtered slope path(s) (T x k)
    beta_prior: np.ndarray     # one-step-ahead slope(s) (T x k) for trading at t
    alpha_prior: np.ndarray
    spread: np.ndarray         # prediction error e_t = y_t - prior fit (tradable innovation)
    spread_var: np.ndarray     # predictive variance S_t
    zscore: np.ndarray         # e_t / sqrt(S_t)
    loglik: float
    delta: float
    R: float


def kalman_regression(y, x, delta: float = 1e-4, R: float | None = None, init_window: int = 60,
                      intercept: bool = True) -> KalmanResult:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    T, k = x.shape
    H = np.column_stack([np.ones(T), x]) if intercept else x
    m = H.shape[1]
    n0 = min(init_window, max(T // 4, m + 2))
    # Initialise from OLS on the first window (diffuse-ish prior around it).
    theta, *_ = np.linalg.lstsq(H[:n0], y[:n0], rcond=None)
    resid0 = y[:n0] - H[:n0] @ theta
    if R is None:
        R = float(np.var(resid0)) if n0 > m else float(np.var(y[:n0]) * 0.1 + 1e-12)
        R = max(R, 1e-12)
    xtx_inv = np.linalg.pinv(H[:n0].T @ H[:n0])
    P = xtx_inv * R * 10.0
    # State noise in units of "one observation's worth" of coefficient variance, so delta is scale-free:
    # Q = delta / (1 - delta) * R * diag((H'H / n0)^-1).
    Q = np.diag(np.maximum(np.diag(xtx_inv) * R * n0, 1e-14)) * (delta / (1 - delta))
    out_theta = np.full((T, m), np.nan)
    out_prior = np.full((T, m), np.nan)
    e = np.full(T, np.nan)
    S_all = np.full(T, np.nan)
    ll = 0.0
    for t in range(T):
        if not (np.isfinite(y[t]) and np.all(np.isfinite(H[t]))):
            out_theta[t] = theta
            out_prior[t] = theta
            P = P + Q
            continue
        P_prior = P + Q
        out_prior[t] = theta
        h = H[t]
        S = float(h @ P_prior @ h + R)
        err = y[t] - float(h @ theta)
        K = P_prior @ h / S
        theta = theta + K * err
        P = P_prior - np.outer(K, h) @ P_prior
        out_theta[t] = theta
        e[t] = err
        S_all[t] = S
        if t >= n0:
            ll += -0.5 * (np.log(2 * np.pi * S) + err * err / S)
    if intercept:
        alpha, beta = out_theta[:, 0], out_theta[:, 1:]
        alpha_p, beta_p = out_prior[:, 0], out_prior[:, 1:]
    else:
        alpha = alpha_p = np.zeros(T)
        beta, beta_p = out_theta, out_prior
    z = e / np.sqrt(S_all)
    return KalmanResult(alpha, beta, beta_p, alpha_p, e, S_all, z, float(ll), delta, R)


@dataclass
class KalmanFit:
    delta: float
    loglik: float
    grid: np.ndarray
    logliks: np.ndarray
    at_boundary: bool
    result: KalmanResult


def fit_mle(y, x, grid=None, **kw) -> KalmanFit:
    """Choose delta by maximum likelihood over a log grid (R re-estimated from the burn-in window)."""
    if grid is None:
        grid = np.logspace(-7, -1, 13)
    lls = []
    results = []
    for d in grid:
        r = kalman_regression(y, x, delta=float(d), **kw)
        lls.append(r.loglik)
        results.append(r)
    lls = np.array(lls)
    i = int(np.nanargmax(lls))
    return KalmanFit(float(grid[i]), float(lls[i]), np.asarray(grid), lls, i in (0, len(grid) - 1), results[i])


def rolling_ols_beta(y, x, window: int = 126) -> tuple[np.ndarray, np.ndarray]:
    """Trailing-window OLS (alpha, beta) using data up to and including t (vectorised via cumulative sums)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(y)
    alpha = np.full(n, np.nan)
    beta = np.full(n, np.nan)
    ok = np.isfinite(y) & np.isfinite(x)
    yy, xx = np.where(ok, y, 0.0), np.where(ok, x, 0.0)
    c = np.cumsum(np.r_[0, ok.astype(float)])
    sx, sy = np.cumsum(np.r_[0, xx]), np.cumsum(np.r_[0, yy])
    sxx, sxy = np.cumsum(np.r_[0, xx * xx]), np.cumsum(np.r_[0, xx * yy])
    for t in range(window - 1, n):
        a, b = t + 1 - window, t + 1
        m = c[b] - c[a]
        if m < window * 0.8:
            continue
        mx, my = (sx[b] - sx[a]) / m, (sy[b] - sy[a]) / m
        vx = (sxx[b] - sxx[a]) / m - mx * mx
        if vx <= 0:
            continue
        beta[t] = ((sxy[b] - sxy[a]) / m - mx * my) / vx
        alpha[t] = my - beta[t] * mx
    return alpha, beta
