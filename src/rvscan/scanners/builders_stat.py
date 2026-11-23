"""Statistical pairs (index RV, sector and stock pairs) with causal, re-estimated hedge ratios.

Hedge ratio methods (all use data up to t-1 for the position held over day t):
  rolling  - trailing-window OLS of log prices (default, 252 days)
  kalman   - random-walk-coefficient Kalman filter (delta = 1e-4), one-step-ahead estimate
  static   - OLS on a formation window, frozen afterwards (classic Gatev-Goetzmann-Rouwenhorst style)
  ratio    - 1:1 dollar-neutral log ratio (no estimation)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.universe import load_universe
from ..engine import costs as costmod
from ..models.kalman import kalman_regression, rolling_ols_beta
from .base import Relationship, SpreadData, cash_pair_costs, leg_costs_rt, license_of, max_sync_gap, tr_usd


def hedge_path(la: pd.Series, lb: pd.Series, method: str, window: int = 252, formation_end=None) -> tuple[pd.Series, pd.Series]:
    """(alpha_t, beta_t) usable for the position held over day t+1 (already lagged by one day)."""
    if method == "rolling":
        a, b = rolling_ols_beta(la.values, lb.values, window)
        alpha, beta = pd.Series(a, index=la.index), pd.Series(b, index=la.index)
    elif method == "kalman":
        res = kalman_regression(la.values, lb.values, delta=1e-4, init_window=window)
        alpha = pd.Series(res.alpha, index=la.index)
        beta = pd.Series(res.beta[:, 0], index=la.index)
        alpha.iloc[:window] = np.nan
        beta.iloc[:window] = np.nan
    elif method == "static":
        end = formation_end or la.index[min(window, len(la) - 1)]
        m = la.index <= end
        coef = np.polyfit(lb[m].values, la[m].values, 1)
        beta = pd.Series(np.where(la.index > end, coef[0], np.nan), index=la.index)
        alpha = pd.Series(np.where(la.index > end, coef[1], np.nan), index=la.index)
    elif method == "ratio":
        beta = pd.Series(1.0, index=la.index)
        alpha = (la - lb).rolling(window, min_periods=window // 2).mean()
    else:
        raise ValueError(method)
    return alpha.shift(1), beta.clip(-5, 5).shift(1)


def stat_pair(rel: Relationship, method: str | None = None, window: int | None = None) -> SpreadData:
    a, b = rel.legs[:2]
    method = method or rel.params.get("hedge", "rolling")
    window = window or rel.params.get("window", 252)
    u = load_universe()
    df = pd.concat({"a": tr_usd(a), "b": tr_usd(b)}, axis=1, join="inner").dropna()
    la, lb = np.log(df["a"]), np.log(df["b"])
    alpha, beta = hedge_path(la, lb, method, window, rel.params.get("formation_end"))
    spread = 1e4 * (la - beta * lb - alpha)
    ra, rb = df["a"].pct_change(), df["b"].pct_change()
    bad = (ra.abs() > 0.5) | (rb.abs() > 0.5)        # >50% one-day moves in these large funds/stocks = data errors
    ra, rb = ra.mask(bad), rb.mask(bad)
    pnl = 1e4 * (ra - beta * rb)
    leg_pnl = pd.DataFrame({a: 1e4 * ra, b: -1e4 * beta * rb})
    ok = spread.notna()
    spread, pnl, beta, leg_pnl = spread[ok], pnl[ok], beta[ok], leg_pnl[ok]
    idx = spread.index
    stress = costmod.stress_multiplier(np.log(df["a"]).diff()).reindex(idx).fillna(1.0)
    ca, cb = costmod.leg_cost(u[a]), costmod.leg_cost(u[b])
    cost_rt = (2 * (ca.half_spread_bp + beta.abs() * cb.half_spread_bp) * stress
               + 2 * (ca.commission_bp + beta.abs() * cb.commission_bp))
    rebal = beta.diff().abs().fillna(0.0) * (cb.half_spread_bp + cb.commission_bp)
    fee = max(costmod.borrow(u[a])[1], costmod.borrow(u[b])[1])
    # Excess-return convention: interest on any net cash (beta != 1) is not counted as edge; the holding cost is
    # the funding spread on the long leg, the rebate spread and borrow fee on the short leg.
    _, hold = cash_pair_costs(idx, 1.0, float(beta.abs().median()), fee)
    carry = 0.0
    return SpreadData(rel, spread, "bp", 0.0, np.inf, np.inf, pnl, carry, hold, cost_rt, max_sync_gap([a, b]),
                      license_of([a, b]), rebal_cost=rebal, leg_pnl=leg_pnl, est_hedge=(method != "ratio"),
                      notes=list(rel.constraints), meta={"hedge": method, "window": window, "beta": beta})
