"""Bayesian spread forecasting.

1. Model comparison + model averaging between
      M1 (mean reversion):  d_t = a + b d_{t-1} + e_t
      M0 (random walk):     d_t = d_{t-1} + c + e_t
   where d = spread - centre (structural fair value when one exists, otherwise the sample mean), with conjugate
   Normal-Inverse-Gamma priors. Marginal likelihoods are closed-form, so P(M1 | data) is a proper posterior
   probability of mean reversion. A flat-prior posterior on b alone is NOT: for pure random walks it reports
   P(b < 1) of about 0.99 because of the Dickey-Fuller downward bias in b (checked in tests).
   Forecasts are posterior-predictive mixtures of both models, so fragile half-lives widen the forecast instead of
   promising convergence.

2. Bayesian online change-point detection (Adams & MacKay 2007) with a Normal-Gamma model and constant hazard:
   the posterior over the current run length flags relationship breaks in real time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special


@dataclass
class NIGPosterior:
    m: np.ndarray       # posterior mean of the coefficients
    V: np.ndarray       # posterior scale matrix (sigma^2 units)
    a: float            # inverse-gamma shape
    b: float            # inverse-gamma scale
    logml: float        # log marginal likelihood
    nobs: int

    def sample(self, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
        sig2 = self.b / rng.gamma(self.a, 1.0, size=n)
        k = len(self.m)
        L = np.linalg.cholesky(self.V + 1e-15 * np.eye(k))
        z = rng.standard_normal((n, k))
        coef = self.m[None, :] + (z @ L.T) * np.sqrt(sig2)[:, None]
        return coef, sig2

    @property
    def sigma_mean(self) -> float:
        return float(np.sqrt(self.b / (self.a - 1))) if self.a > 1 else float("nan")


def nig_regression(y, X, m0, V0, a0: float, b0: float) -> NIGPosterior:
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n = len(y)
    V0_inv = np.linalg.inv(V0)
    prec = V0_inv + X.T @ X
    Vn = np.linalg.inv(prec)
    mn = Vn @ (V0_inv @ m0 + X.T @ y)
    an = a0 + n / 2
    bn = b0 + 0.5 * float(y @ y + m0 @ V0_inv @ m0 - mn @ prec @ mn)
    bn = max(bn, 1e-300)
    _, ld_n = np.linalg.slogdet(Vn)
    _, ld_0 = np.linalg.slogdet(V0)
    logml = (-n / 2 * np.log(2 * np.pi) + 0.5 * (ld_n - ld_0) + a0 * np.log(b0) - an * np.log(bn)
             + special.gammaln(an) - special.gammaln(a0))
    return NIGPosterior(mn, Vn, an, bn, float(logml), n)


@dataclass
class BayesSpreadModel:
    centre: float
    ar: NIGPosterior            # M1 coefficients (a, b) on the centred spread
    rw: NIGPosterior            # M0 drift
    p_mean_reverting: float     # P(M1 | data) with the given prior odds
    b_mean: float
    half_life_days: float       # from the posterior mean of b under M1

    def expected_half_life_range(self, q=(0.05, 0.95), n: int = 4000, seed: int = 0) -> tuple[float, float]:
        rng = np.random.default_rng(seed)
        coef, _ = self.ar.sample(n, rng)
        b = coef[:, 1]
        hl = np.where((b > 0) & (b < 1), np.log(2) / -np.log(np.clip(b, 1e-9, 1 - 1e-12)), np.inf)
        return float(np.quantile(hl, q[0])), float(np.quantile(hl, q[1]))


def fit_spread_model(s, fair_value: float | None = None, b_prior: float = 0.95, b_prior_sd: float = 0.05,
                     prior_odds: float = 1.0) -> BayesSpreadModel:
    """Fit M1 and M0 to a spread. `fair_value` = structural equilibrium (tightens M1's intercept prior)."""
    s = np.asarray(s, dtype=float)
    s = s[np.isfinite(s)]
    centre = float(np.mean(s)) if fair_value is None else float(fair_value)
    d = s - centre
    y, x = d[1:], d[:-1]
    n = len(y)
    var_guess = float(np.var(np.diff(d))) + 1e-18
    a0, b0 = 2.0, var_guess            # prior mean of sigma^2 = var of spread changes
    v_a = 0.01 if fair_value is not None else 0.05
    X1 = np.column_stack([np.ones(n), x])
    ar = nig_regression(y, X1, np.array([0.0, b_prior]), np.diag([v_a, b_prior_sd ** 2 / var_guess]), a0, b0)
    X0 = np.ones((n, 1))
    rw = nig_regression(y - x, X0, np.array([0.0]), np.array([[0.01]]), a0, b0)
    log_odds = ar.logml - rw.logml + np.log(prior_odds)
    p1 = float(1.0 / (1.0 + np.exp(-np.clip(log_odds, -700, 700))))
    b = float(ar.m[1])
    hl = float(np.log(2) / -np.log(b)) if 0 < b < 1 else float("inf")
    return BayesSpreadModel(centre, ar, rw, p1, b, hl)


@dataclass
class SpreadForecast:
    horizon: int
    mean: np.ndarray          # expected path of the spread (original units)
    lo: np.ndarray            # 5th percentile path
    hi: np.ndarray            # 95th percentile path
    prob_hit: float           # P(spread reaches the target within the horizon, before the stop)
    prob_stop: float          # P(stop reached first)
    expected_gain: float      # E[gain in spread units, closing at target / stop / horizon]
    prob_mean_reverting: float


def forecast(model: BayesSpreadModel, s0: float, horizon: int, target: float, stop: float | None = None,
             n_paths: int = 4000, seed: int = 0) -> SpreadForecast:
    """Posterior-predictive Monte Carlo of a convergence trade, averaging over M1 and M0."""
    rng = np.random.default_rng(seed)
    n1 = int(round(n_paths * model.p_mean_reverting))
    coef1, sig1 = model.ar.sample(max(n1, 1), rng)
    coef0, sig0 = model.rw.sample(max(n_paths - n1, 1), rng)
    a = np.r_[coef1[:n1, 0], coef0[: n_paths - n1, 0]]
    b = np.r_[np.clip(coef1[:n1, 1], -0.999, 1.02), np.ones(n_paths - n1)]
    sd = np.sqrt(np.r_[sig1[:n1], sig0[: n_paths - n1]])
    c = model.centre
    d = np.full(n_paths, s0 - c, dtype=float)
    tgt, stp = target - c, (None if stop is None else stop - c)
    direction = np.sign(tgt - d[0]) or 1.0
    paths = np.empty((n_paths, horizon))
    done = np.zeros(n_paths, dtype=bool)
    hit = np.zeros(n_paths, dtype=bool)
    stopped = np.zeros(n_paths, dtype=bool)
    gain = np.full(n_paths, np.nan)
    d0 = d[0]
    for h in range(horizon):
        d = a + b * d + sd * rng.standard_normal(n_paths)
        paths[:, h] = d + c
        reach = ~done & ((d - tgt) * direction >= 0)
        hit |= reach
        gain[reach] = (tgt - d0) * direction
        done |= reach
        if stp is not None:
            st = ~done & ((d - stp) * direction <= 0)
            stopped |= st
            gain[st] = (stp - d0) * direction
            done |= st
    gain[~done] = (d[~done] - d0) * direction
    return SpreadForecast(horizon, paths.mean(axis=0), np.percentile(paths, 5, axis=0), np.percentile(paths, 95, axis=0),
                          float(hit.mean()), float(stopped.mean()), float(gain.mean()), model.p_mean_reverting)


def flat_posterior_prob_b_below_one(s) -> float:
    """P(b < 1) under a flat prior on (a, b) (Student-t marginal). Kept to document why it misleads."""
    from scipy import stats
    s = np.asarray(s, dtype=float)
    s = s[np.isfinite(s)]
    y, x = s[1:], s[:-1]
    X = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = len(y) - 2
    s2 = resid @ resid / dof
    se_b = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    return float(stats.t.cdf((1 - coef[1]) / se_b, df=dof))


# ------------------------------------------------------------------------------------------------ BOCPD
def bocpd(x, hazard: float = 1 / 250, mu0: float = 0.0, kappa0: float = 1.0, alpha0: float = 1.0,
          beta0: float | None = None, max_run: int = 1500) -> dict:
    """Adams-MacKay online change-point detection with a Normal-Gamma model (unknown mean and precision).

    Returns {"map_run": most probable run length at each t, "p_change": P(run length < 5 | data up to t)}.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    T = len(x)
    if beta0 is None:
        beta0 = float(np.var(x[: min(T, 250)])) if T > 5 else 1.0
        beta0 = max(beta0, 1e-12)
    R = np.array([1.0])
    mu, kappa, alpha, beta = np.array([mu0]), np.array([kappa0]), np.array([alpha0]), np.array([beta0])
    map_run = np.zeros(T, dtype=int)
    p_change = np.zeros(T)
    for t in range(T):
        df = 2 * alpha
        scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
        z = (x[t] - mu) / scale
        logpred = special.gammaln((df + 1) / 2) - special.gammaln(df / 2) - 0.5 * np.log(np.pi * df) - np.log(scale) \
            - (df + 1) / 2 * np.log1p(z * z / df)
        pred = np.exp(logpred - logpred.max())
        growth = R * pred * (1 - hazard)
        cp = (R * pred * hazard).sum()
        Rn = np.concatenate([[cp], growth])
        Rn = Rn / Rn.sum()
        mu_n = (kappa * mu + x[t]) / (kappa + 1)
        kappa_n = kappa + 1
        alpha_n = alpha + 0.5
        beta_n = beta + kappa * (x[t] - mu) ** 2 / (2 * (kappa + 1))
        mu = np.concatenate([[mu0], mu_n])
        kappa = np.concatenate([[kappa0], kappa_n])
        alpha = np.concatenate([[alpha0], alpha_n])
        beta = np.concatenate([[beta0], beta_n])
        if len(Rn) > max_run:
            Rn, mu, kappa, alpha, beta = Rn[:max_run], mu[:max_run], kappa[:max_run], alpha[:max_run], beta[:max_run]
            Rn = Rn / Rn.sum()
        R = Rn
        map_run[t] = int(np.argmax(R))
        p_change[t] = float(R[:5].sum())
    return {"map_run": map_run, "p_change": p_change}
