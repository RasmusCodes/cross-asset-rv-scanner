"""Constant-maturity bond arithmetic from yield series.

Daily return of a par bond held for one day at constant maturity m (semi-annual coupons):
    r_t = y_{t-1} * dt  -  D(y_{t-1}, m) * dy  +  0.5 * C(y_{t-1}, m) * dy^2
where D is modified duration and C convexity of a par bond. This is the standard approximation used to build
total-return series from constant-maturity yields; it ignores roll-down unless `rolldown_slope` is supplied.
For yield-spread trades the engine works in DV01 units: P&L in bp of yield = change in the weighted yield spread,
plus carry (yield minus funding) and roll-down, both converted to yield bp per day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def modified_duration(y_pct, m: float, freq: int = 2):
    y = np.asarray(y_pct, dtype=float) / 100.0
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (1.0 - (1.0 + y / freq) ** (-freq * m)) / y
    return np.where(np.abs(y) < 1e-6, m, d)


def convexity(y_pct, m: float, freq: int = 2, h: float = 1e-4):
    y = np.asarray(y_pct, dtype=float) / 100.0

    def price(yy):
        n = int(round(freq * m))
        k = np.arange(1, n + 1)
        disc = (1.0 + np.asarray(yy)[..., None] / freq) ** (-k)
        return (y[..., None] / freq * disc).sum(axis=-1) + disc[..., -1]

    p0, pu, pd_ = price(y), price(y + h), price(y - h)
    return (pu + pd_ - 2 * p0) / (p0 * h * h)


def par_bond_returns(yields_pct: pd.Series, m: float, rolldown_slope_bp_per_year: pd.Series | None = None) -> pd.Series:
    y = yields_pct.dropna()
    y_prev = y.shift(1)
    dy = (y - y_prev) / 100.0
    dt = y.index.to_series().diff().dt.days / 365.0
    D = pd.Series(modified_duration(y_prev.fillna(y.iloc[0]).values, m), index=y.index)
    C = pd.Series(convexity(y_prev.fillna(y.iloc[0]).values, m), index=y.index)
    r = y_prev / 100.0 * dt - D * dy + 0.5 * C * dy * dy
    if rolldown_slope_bp_per_year is not None:
        slope = rolldown_slope_bp_per_year.reindex(y.index).ffill().shift(1)
        r = r + D * slope / 1e4 * dt
    return r.iloc[1:]


def carry_bp_per_day(yield_pct: pd.Series, funding_pct: pd.Series, m: float) -> pd.Series:
    """Carry of a long position of one DV01 unit, in yield bp per calendar day (accrued daily)."""
    df = pd.concat({"y": yield_pct, "f": funding_pct}, axis=1).ffill().dropna()
    D = modified_duration(df["y"].values, m)
    return pd.Series((df["y"] - df["f"]).values * 100.0 / 365.0 / D, index=df.index)


def rolldown_bp_per_day(curve: pd.DataFrame, m: float) -> pd.Series:
    """Roll-down of a constant-maturity point in yield bp per day: slope of the curve just below maturity m.

    curve: DataFrame of yields (percent) with maturities (years) as columns."""
    mats = sorted(c for c in curve.columns if c < m)
    if not mats:
        return pd.Series(0.0, index=curve.index)
    lower = mats[-1]
    slope = (curve[m] - curve[lower]) * 100.0 / (m - lower)     # bp per year of maturity
    return slope / 365.0
