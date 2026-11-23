"""Multiple-testing corrections and backtest-overfitting statistics.

- Family-wise / false-discovery adjusted p-values: Bonferroni, Holm, Benjamini-Hochberg, Benjamini-Yekutieli.
- Null-rate calibration: share of independent random-walk pairs that an Engle-Granger screen flags.
- Probabilistic and deflated Sharpe ratios (Bailey & Lopez de Prado 2012, 2014).
- Stationary bootstrap (Politis & Romano 1994) and White's (2000) reality check for the best of many strategies.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def bonferroni(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.minimum(p * len(p), 1.0)


def holm(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def benjamini_hochberg(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def benjamini_yekutieli(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    m = len(p)
    c_m = np.sum(1.0 / np.arange(1, m + 1))
    return np.minimum(benjamini_hochberg(p) * c_m, 1.0)


def adjust(p, method: str) -> np.ndarray:
    return {"bonferroni": bonferroni, "holm": holm, "bh": benjamini_hochberg, "by": benjamini_yekutieli}[method](p)


# ------------------------------------------------------------------------------------------------ null-rate calibration
def eg_null_rate(n_obs: int, n_pairs: int = 500, alpha: float = 0.05, seed: int = 0, drift: float = 0.0,
                 vol: float = 0.01) -> float:
    """Share of independent geometric random-walk pairs flagged by Engle-Granger at `alpha`."""
    from .cointegration import engle_granger
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_pairs):
        a = np.cumsum(drift + vol * rng.standard_normal(n_obs))
        b = np.cumsum(drift + vol * rng.standard_normal(n_obs))
        if engle_granger(a, b).pvalue < alpha:
            hits += 1
    return hits / n_pairs


# ------------------------------------------------------------------------------------------------ Sharpe statistics
def probabilistic_sharpe(sr: float, n: int, skew: float = 0.0, kurt: float = 3.0, sr_benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe > benchmark) given an observed per-period Sharpe over n returns."""
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr * sr, 1e-12))
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(max(n - 1, 1)) / denom))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Expected maximum of n_trials per-period Sharpe estimates under the null of zero skill."""
    if n_trials < 2:
        return 0.0
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_sr) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(returns, n_trials: int, var_sr_trials: float) -> dict:
    """Deflated Sharpe ratio of a selected strategy's per-period returns after n_trials attempts."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    sd = r.std(ddof=1)
    sr = r.mean() / sd if sd > 0 else 0.0
    sk = float(stats.skew(r)) if n > 3 else 0.0
    ku = float(stats.kurtosis(r, fisher=False)) if n > 3 else 3.0
    sr0 = expected_max_sharpe(n_trials, var_sr_trials)
    return {"sr": sr, "sr_annual": sr * np.sqrt(252), "sr0": sr0, "sr0_annual": sr0 * np.sqrt(252),
            "psr": probabilistic_sharpe(sr, n, sk, ku, 0.0), "dsr": probabilistic_sharpe(sr, n, sk, ku, sr0),
            "skew": sk, "kurt": ku, "n": n}


# ------------------------------------------------------------------------------------------------ bootstrap
def stationary_bootstrap_indices(n: int, mean_block: float, rng) -> np.ndarray:
    idx = np.empty(n, dtype=int)
    p = 1.0 / mean_block
    i = rng.integers(n)
    for t in range(n):
        if t > 0 and rng.random() < p:
            i = rng.integers(n)
        else:
            i = (i + 1) % n if t > 0 else i
        idx[t] = i
    return idx


def reality_check(perf: np.ndarray, n_boot: int = 1000, mean_block: float = 20.0, seed: int = 0) -> dict:
    """White's reality check. perf: T x K matrix of strategy excess returns (vs a zero benchmark).

    H0: no strategy has positive expected excess return. Returns the p-value for the best strategy.
    """
    perf = np.asarray(perf, dtype=float)
    perf = perf[np.all(np.isfinite(perf), axis=1)]
    T, K = perf.shape
    means = perf.mean(axis=0)
    stat = np.sqrt(T) * means.max()
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_boot):
        idx = stationary_bootstrap_indices(T, mean_block, rng)
        boot_means = perf[idx].mean(axis=0)
        v = np.sqrt(T) * (boot_means - means).max()
        if v >= stat:
            exceed += 1
    return {"best": int(np.argmax(means)), "stat": float(stat), "pvalue": exceed / n_boot, "K": K, "T": T}
