"""Ornstein-Uhlenbeck mean reversion: estimation, half-life, convergence forecasts.

dS = theta (mu - S) dt + sigma dW. With a constant step dt (trading days / 252) the exact discretisation is the AR(1)
S_{t+1} = a + b S_t + eps,  b = exp(-theta dt),  a = mu (1 - b),  Var(eps) = sigma^2 (1 - b^2) / (2 theta).
The conditional MLE equals OLS of S_{t+1} on S_t. Half-life = ln 2 / theta. Stationary sd = sigma / sqrt(2 theta).

Half-life uncertainty matters as much as the point estimate: near b = 1 a small error in b is a huge error in
theta. `fit_ou` returns a delta-method interval and `half_life_bootstrap` a block-bootstrap interval.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

DT = 1.0 / 252.0


@dataclass
class OUFit:
    a: float
    b: float
    theta: float                 # per year
    mu: float
    sigma: float                 # per sqrt(year)
    sigma_eq: float              # stationary standard deviation
    half_life_days: float
    half_life_ci: tuple          # 90% interval (days), from the delta method on b
    resid_sd: float
    nobs: int
    mean_reverting: bool          # b < 1 with 95% one-sided confidence

    def expected(self, s0: float, h_days: float) -> float:
        return self.mu + (s0 - self.mu) * np.exp(-self.theta * h_days * DT)

    def sd_at(self, h_days: float) -> float:
        if not self.mean_reverting or self.theta <= 0:
            return self.resid_sd * np.sqrt(max(h_days, 1.0))
        return self.sigma_eq * np.sqrt(1 - np.exp(-2 * self.theta * h_days * DT))


def fit_ou(s, n_vars: int = 1, alpha: float = 0.05) -> OUFit:
    """Fit an OU process to a spread.

    `mean_reverting` uses the Dickey-Fuller distribution, not a normal t-test: a plain t-test on (1 - b) flags
    roughly 40% of pure random walks at the 5% level. Pass n_vars = 2 (or more) when the spread itself was built
    with an estimated hedge ratio (Engle-Granger critical values).
    """
    from .cointegration import mackinnon_p
    s = np.asarray(s, dtype=float)
    s = s[np.isfinite(s)]
    x, y = s[:-1], s[1:]
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    resid = y - X @ coef
    rsd = float(np.sqrt(resid @ resid / max(n - 2, 1)))
    se_b = rsd / np.sqrt(np.sum((x - x.mean()) ** 2)) if n > 2 else np.inf
    tau = (b - 1) / se_b if np.isfinite(se_b) and se_b > 0 else 0.0
    mean_rev = (b < 1) and (mackinnon_p(tau, n_vars, "c") < alpha)
    if 0 < b < 1:
        theta = -np.log(b) / DT
        mu = a / (1 - b)
        sigma = rsd * np.sqrt(2 * theta / (1 - b * b))
        sigma_eq = sigma / np.sqrt(2 * theta)
        hl = np.log(2) / (theta * DT)
        z = stats.norm.ppf(0.95)
        b_lo, b_hi = max(b - z * se_b, 1e-6), min(b + z * se_b, 1 - 1e-9)
        hl_ci = (np.log(2) / -np.log(b_lo), np.log(2) / -np.log(b_hi) if b_hi < 1 else np.inf)
    else:
        theta, mu, sigma, sigma_eq, hl = 0.0, float(np.mean(s)), rsd * np.sqrt(252), float("inf"), float("inf")
        hl_ci = (float("inf"), float("inf"))
    return OUFit(a, b, float(theta), float(mu), float(sigma), float(sigma_eq), float(hl), tuple(map(float, hl_ci)),
                 rsd, n, bool(mean_rev))


def half_life_bootstrap(s, n_boot: int = 300, block: int = 20, seed: int = 0) -> tuple[float, float]:
    """Moving-block bootstrap 90% interval for the half-life (days); inf when b >= 1 in a resample."""
    s = np.asarray(s, dtype=float)
    s = s[np.isfinite(s)]
    ds = np.diff(s)
    lag = s[:-1]
    n = len(ds)
    rng = np.random.default_rng(seed)
    hls = []
    nb = int(np.ceil(n / block))
    for _ in range(n_boot):
        starts = rng.integers(0, n - block, nb)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        X = np.column_stack([np.ones(n), lag[idx]])
        coef, *_ = np.linalg.lstsq(X, ds[idx], rcond=None)
        b = 1 + coef[1]
        hls.append(np.log(2) / -np.log(b) if 0 < b < 1 else np.inf)
    hls = np.array(hls)
    return float(np.percentile(hls, 5)), float(np.percentile(hls, 95))


def first_passage_prob(fit: OUFit, s0: float, level: float, horizon_days: int, n_paths: int = 4000,
                       seed: int = 0) -> tuple[float, float]:
    """P(spread crosses `level` within horizon) and median crossing day, by exact-discretisation Monte Carlo."""
    rng = np.random.default_rng(seed)
    if not fit.mean_reverting:
        b, a = 1.0, 0.0
    else:
        b, a = fit.b, fit.a
    s = np.full(n_paths, s0, dtype=float)
    hit_day = np.full(n_paths, np.nan)
    up = level > s0
    for d in range(1, horizon_days + 1):
        s = a + b * s + fit.resid_sd * rng.standard_normal(n_paths)
        crossed = (s >= level) if up else (s <= level)
        new = crossed & np.isnan(hit_day)
        hit_day[new] = d
    p = float(np.mean(np.isfinite(hit_day)))
    med = float(np.nanmedian(hit_day)) if p > 0 else float("nan")
    return p, med
