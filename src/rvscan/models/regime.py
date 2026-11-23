"""Two-state Markov-switching AR(1) for spreads (Hamilton 1989 filter, Kim 1994 smoother, EM estimation).

    s_t = c_k + phi_k s_{t-1} + sigma_k e_t,   k in {0, 1},   P[k_t = j | k_{t-1} = i] = P_ij

State 0 is relabelled after estimation as the more strongly mean-reverting regime (lower phi). The filtered
probability of the mean-reverting regime uses information up to t only and can gate trades; the smoothed
probability uses the whole sample and is for analysis only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MSARFit:
    c: np.ndarray
    phi: np.ndarray
    sigma: np.ndarray
    P: np.ndarray
    filtered: np.ndarray        # T x 2, P[k_t | s_1..s_t]
    smoothed: np.ndarray        # T x 2, P[k_t | s_1..s_T]
    loglik: float
    iterations: int
    converged: bool

    @property
    def half_life_days(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where((self.phi > 0) & (self.phi < 1), np.log(2) / -np.log(self.phi), np.inf)

    @property
    def expected_duration_days(self) -> np.ndarray:
        return 1.0 / np.maximum(1 - np.diag(self.P), 1e-9)

    def prob_mean_reverting(self, smoothed: bool = False) -> np.ndarray:
        return (self.smoothed if smoothed else self.filtered)[:, 0]


def _norm_pdf(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (np.sqrt(2 * np.pi) * sd)


def hamilton_filter(y, x, c, phi, sigma, P, p0=None):
    T = len(y)
    filt = np.zeros((T, 2))
    pred = np.zeros((T, 2))
    lik = np.zeros(T)
    prob = np.array([0.5, 0.5]) if p0 is None else p0
    for t in range(T):
        pr = prob @ P
        dens = _norm_pdf(y[t], c + phi * x[t], sigma)
        joint = pr * dens
        tot = joint.sum()
        if tot <= 0 or not np.isfinite(tot):
            joint = pr * 1e-300 + 1e-300
            tot = joint.sum()
        pred[t] = pr
        filt[t] = joint / tot
        lik[t] = tot
        prob = filt[t]
    return filt, pred, lik


def kim_smoother(filt, pred, P):
    T = len(filt)
    sm = np.zeros_like(filt)
    sm[-1] = filt[-1]
    xi = np.zeros((T - 1, 2, 2))
    for t in range(T - 2, -1, -1):
        ratio = np.where(pred[t + 1] > 0, sm[t + 1] / pred[t + 1], 0.0)
        # joint P[k_t = i, k_{t+1} = j | all]
        joint = filt[t][:, None] * P * ratio[None, :]
        xi[t] = joint
        sm[t] = joint.sum(axis=1)
    return sm, xi


def fit_msar(s, max_iter: int = 200, tol: float = 1e-6, seed: int = 0) -> MSARFit:
    s = np.asarray(s, dtype=float)
    s = s[np.isfinite(s)]
    y, x = s[1:], s[:-1]
    T = len(y)
    # Initialise: OLS fit; regime 0 = lower residual variance half, regime 1 = higher variance / persistence
    X = np.column_stack([np.ones(T), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    rsd = np.std(y - X @ coef)
    c = np.array([coef[0], coef[0]])
    phi = np.array([min(coef[1], 0.995) * 0.97, min(coef[1] + 0.02, 1.02)])
    sigma = np.array([rsd * 0.8, rsd * 1.3])
    P = np.array([[0.98, 0.02], [0.03, 0.97]])
    prev = -np.inf
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        good = (c.copy(), phi.copy(), sigma.copy(), P.copy())
        try:
            filt, pred, lik = hamilton_filter(y, x, c, phi, sigma, P)
            ll = float(np.sum(np.log(lik)))
            sm, xi = kim_smoother(filt, pred, P)
            if not (np.isfinite(ll) and np.all(np.isfinite(sm)) and np.all(np.isfinite(xi))):
                raise FloatingPointError("non-finite E-step")
            for k in range(2):
                w = sm[:, k]
                sw = w.sum()
                if sw < 5:
                    continue
                Xw = X * w[:, None]
                beta = np.linalg.solve(Xw.T @ X + 1e-10 * np.eye(2), Xw.T @ y)
                c[k], phi[k] = beta
                r = y - X @ beta
                sigma[k] = max(np.sqrt((w * r * r).sum() / sw), 1e-8)
            num = xi.sum(axis=0)
            den = sm[:-1].sum(axis=0)
            P = num / np.maximum(den[:, None], 1e-12)
            P = P / P.sum(axis=1, keepdims=True)
            if not (np.all(np.isfinite(c)) and np.all(np.isfinite(phi)) and np.all(np.isfinite(P))):
                raise FloatingPointError("non-finite M-step")
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            c, phi, sigma, P = good          # keep the last well-defined parameters
            break
        if abs(ll - prev) < tol * (1 + abs(ll)):
            converged = True
            break
        prev = ll
    filt, pred, lik = hamilton_filter(y, x, c, phi, sigma, P)
    sm, _ = kim_smoother(filt, pred, P)
    ll = float(np.sum(np.log(lik)))
    if phi[0] > phi[1]:  # relabel: state 0 = mean-reverting
        order = [1, 0]
        c, phi, sigma = c[order], phi[order], sigma[order]
        P = P[np.ix_(order, order)]
        filt, sm = filt[:, order], sm[:, order]
    pad = np.full((1, 2), np.nan)
    return MSARFit(c, phi, sigma, P, np.vstack([pad, filt]), np.vstack([pad, sm]), ll, it, converged)


def filter_with_params(s, fit: MSARFit) -> np.ndarray:
    """Run the Hamilton filter on new data with frozen parameters (out-of-sample regime probabilities)."""
    s = np.asarray(s, dtype=float)
    out = np.full((len(s), 2), np.nan)
    ok = np.isfinite(s)
    idx = np.where(ok)[0]
    if len(idx) < 3:
        return out
    v = s[idx]
    filt, _, _ = hamilton_filter(v[1:], v[:-1], fit.c, fit.phi, fit.sigma, fit.P)
    out[idx[1:]] = filt
    return out
