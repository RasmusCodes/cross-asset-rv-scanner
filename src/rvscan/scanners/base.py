"""Core abstractions shared by every scanner.

A Relationship says which instruments are linked, why, and how strongly (tier). Building it produces SpreadData:
the monitored spread, its structural fair value and no-arbitrage band (if any), the daily P&L of a one-unit
position, carry, round-trip costs, timing (synchronicity) and licence. Scanners, the backtester and the demo all
consume SpreadData, so every opportunity type is evaluated with the same signal, cost and risk machinery.

Units: `unit` = "bp" (log spread x 1e4 or rate spread in bp), "yield_bp", or "usd". P&L, carry and costs are in
the same unit per one unit of position (for equity-like spreads: bp of the first leg's notional).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from ..data import fx, store
from ..data.universe import Instrument, load_universe, sync_gap_hours, venues
from ..engine import costs as costmod

TIERS = {
    "arbitrage": "Arbitrage relationship: a contractual or creation/redemption mechanism can force convergence (net of costs).",
    "quasi": "Economic link: a convergence mechanism exists but cannot be forced on any timetable.",
    "statistical": "Statistical only: co-movement without a mechanism. Signals are bets, never arbitrage.",
}

TYPES = {
    "etf_nav": "ETF vs underlying basket (NAV)",
    "same_index_etf": "ETFs on the same index",
    "futures_basis": "Futures basis",
    "calendar": "Calendar spreads",
    "cross_listing": "Cross-listed securities (interlisted, dual-listed, share classes)",
    "adr": "ADR conversion parity",
    "fx_rv": "FX-adjusted relative value",
    "commodity_curve": "Commodity term structures and spreads",
    "index_rv": "Equity index relative value",
    "sector_pairs": "Sector and industry pairs",
    "bond_spreads": "Bond spread dislocations",
    "carry_funding": "Carry and funding",
    "cointegration": "Statistical cointegration scan",
}


@dataclass
class Relationship:
    id: str
    type: str
    tier: str
    legs: list
    name: str
    mechanism: str
    params: dict = field(default_factory=dict)
    constraints: list = field(default_factory=list)

    def instruments(self) -> list[Instrument]:
        u = load_universe()
        return [u[i] for i in self.legs if i in u]


@dataclass
class SpreadData:
    """Everything the scanner, backtester and demo need for one relationship.

    Sign convention: position +1 = long the spread (gains when `spread` rises).
    pnl, carry, hold_cost, rebal_cost and cost_rt are in `pnl_unit` per one unit of position.
    """
    rel: Relationship
    spread: pd.Series
    unit: str                             # unit of the spread: "bp", "yield_bp", "usd"
    fair: pd.Series | float | None        # structural fair value (None = rolling mean, i.e. no structural anchor)
    band_up: pd.Series | float            # deviation above fair needed before the arbitrage pays (inf = no bound)
    band_down: pd.Series | float          # deviation below fair needed before the arbitrage pays
    pnl: pd.Series | None                 # daily P&L of +1 unit; None = monitor only (not tradable)
    carry: pd.Series | float              # directional daily carry of +1 unit
    hold_cost: pd.Series | float          # daily cost of holding one unit in either direction (funding, borrow)
    cost_rt: pd.Series | float            # round-trip cost of one unit (stress-scaled spreads + fixed taxes/fees)
    sync_gap_h: float
    license: str                          # "open" or "vendor"
    pnl_unit: str = "bp"
    freq: str = "D"
    no_trade: pd.DatetimeIndex | None = None    # dates on which the position must be flat (futures contract switch)
    roll_days: pd.DatetimeIndex | None = None   # dates on which holding a position costs a re-trade (spread rolls)
    rebal_cost: pd.Series | float = 0.0   # daily hedge-rebalancing cost per unit held (dynamic hedge ratios)
    leg_pnl: pd.DataFrame | None = None   # per-leg P&L of +1 unit (attribution)
    direction: str = "both"               # both | long_only | short_only (borrow / conversion limits)
    est_hedge: bool = False               # spread built with an estimated hedge ratio (Engle-Granger critical values)
    notes: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def fair_series(self) -> pd.Series:
        if self.fair is None:
            w = self.meta.get("fair_window", 36 if self.freq == "M" else 252)
            return self.spread.rolling(w, min_periods=max(12, w // 4)).mean()
        if isinstance(self.fair, pd.Series):
            return self.fair.reindex(self.spread.index)
        return pd.Series(float(self.fair), index=self.spread.index)

    @property
    def deviation(self) -> pd.Series:
        return self.spread - self.fair_series

    @property
    def tradable(self) -> bool:
        return self.pnl is not None

    def periods_per_year(self) -> float:
        return 12.0 if self.freq == "M" else 252.0

    def at(self, x, date=None) -> float:
        """Latest (or as-of date) value of a scalar-or-series field."""
        if isinstance(x, pd.Series):
            s = x.dropna()
            if date is not None:
                s = s[s.index <= date]
            return float(s.iloc[-1]) if len(s) else float("nan")
        return float(x)


# ------------------------------------------------------------------------------------------------ data helpers
def license_of(ids: list[str]) -> str:
    u = load_universe()
    return "vendor" if any(u[i].is_vendor for i in ids if i in u) else "open"


@lru_cache(maxsize=512)
def tr_usd(iid: str) -> pd.Series:
    """Total-return index in USD (Yahoo adjclose converted at the H.10 rate on the same date)."""
    inst = load_universe()[iid]
    s = store.series(iid, total_return=True)
    if inst.price_ccy != "USD":
        s = fx.to_usd(s, inst.price_ccy)
    return s.dropna()


@lru_cache(maxsize=512)
def px_usd(iid: str) -> pd.Series:
    inst = load_universe()[iid]
    s = store.series(iid, total_return=False)
    if inst.price_ccy != "USD":
        s = fx.to_usd(s, inst.price_ccy)
    return s.dropna()


def simple_ret(s: pd.Series) -> pd.Series:
    return s / s.shift(1) - 1.0


def daily_vol(iid: str) -> float:
    r = np.log(tr_usd(iid)).diff().dropna()
    return float(r.iloc[-252:].std()) if len(r) > 20 else float("nan")


@lru_cache(maxsize=32)
def overnight_rate(ccy: str) -> pd.Series:
    """Overnight / short funding rate in percent, spliced from the best available open series."""
    def get(iid):
        return store.series(iid) if store.has(iid) else pd.Series(dtype=float)
    if ccy == "USD":
        parts = [get("SOFR"), get("EFFR"), get("TBILL_3M")]
    elif ccy == "EUR":
        parts = [get("ESTR"), get("RATE3M_EUR")]
    elif ccy == "GBP":
        parts = [get("SONIA"), get("RATE3M_GBP")]
    elif ccy == "CAD":
        parts = [get("CORRA"), get("RATE3M_CAD")]
    else:
        parts = [get(f"RATE3M_{ccy}")]
    out = pd.Series(dtype=float)
    for p in parts:
        if p.empty:
            continue
        out = p if out.empty else out.combine_first(p)
    if out.empty:
        return out
    idx = pd.bdate_range(out.index[0], pd.Timestamp.today().normalize())
    return out.reindex(out.index.union(idx)).ffill().reindex(idx)


def lagged_monthly(s: pd.Series, lag_months: int = 1) -> pd.Series:
    """Monthly series made daily with a publication lag (value for month m usable from the start of m+1+lag)."""
    m = s.copy()
    m.index = m.index.to_period("M").to_timestamp(how="start") + pd.DateOffset(months=lag_months + 0)
    m.index = m.index + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)
    idx = pd.bdate_range(m.index[0], pd.Timestamp.today().normalize())
    return m.reindex(m.index.union(idx)).ffill().reindex(idx)


def day_fraction(index: pd.DatetimeIndex) -> pd.Series:
    """Calendar days since the previous observation / 365 (interest accrues over weekends)."""
    return pd.Series(index.to_series().diff().dt.days.fillna(1).clip(lower=1).values / 365.0, index=index)


def cash_pair_costs(index: pd.DatetimeIndex, w_long: float, w_short: float, borrow_fee_bp: float) -> tuple[pd.Series, pd.Series]:
    """Funding of a cash long/short pair per unit (bp): returns (directional carry of +1, symmetric hold cost).

    Long notional pays the overnight rate + spread; short proceeds earn the rate - rebate spread and pay borrow.
    The net cash (w_long - w_short) earns/pays the overnight rate: that part is directional.
    """
    ls, ss = costmod.funding_spreads()
    r = overnight_rate("USD").reindex(index).ffill().fillna(0.0) * 100.0      # bp per year
    dt = day_fraction(index)
    carry = -(w_long - w_short) * r * dt
    hold = (w_long * ls + w_short * (ss + (borrow_fee_bp if np.isfinite(borrow_fee_bp) else 0.0))) * dt
    return carry, hold


def leg_costs_rt(insts: list[Instrument], weights: list[float], stress: pd.Series | None = None,
                 notional_usd: float | None = None, first_sides: list[str] | None = None) -> pd.Series | float:
    """Round-trip cost of a unit position in bp of the unit notional: spreads/impact stress-scaled, taxes fixed."""
    variable = 0.0
    fixed = 0.0
    for k, (inst, w) in enumerate(zip(insts, weights)):
        if not inst.tradable or w == 0:
            continue
        vol = daily_vol(inst.id) if inst.src == "yahoo" else None
        adv = None
        if notional_usd and inst.src == "yahoo":
            v = store.volume_usd(inst.id)
            adv = float(v.dropna().iloc[-1]) if v.notna().any() else None
        lc = costmod.leg_cost(inst, notional_usd, vol, adv)
        side = (first_sides[k] if first_sides else ("buy" if w > 0 else "sell"))
        other = "sell" if side == "buy" else "buy"
        tax = (lc.tax_buy_bp if side == "buy" else lc.tax_sell_bp) + (lc.tax_buy_bp if other == "buy" else lc.tax_sell_bp)
        variable += abs(w) * 2 * (lc.half_spread_bp + lc.impact_bp)
        fixed += abs(w) * (2 * (lc.commission_bp + lc.fx_bp) + tax)
    if stress is None:
        return variable + fixed
    return variable * stress + fixed


def max_sync_gap(ids: list[str], extra_venues: list[str] | None = None) -> float:
    u = load_universe()
    insts = [u[i] for i in ids]
    gaps = [0.0]
    for i in range(len(insts)):
        for j in range(i + 1, len(insts)):
            g = sync_gap_hours(insts[i], insts[j])
            if np.isfinite(g):
                gaps.append(g)
    if extra_venues:
        vs = venues()
        import datetime as dt
        for v in extra_venues:
            for inst in insts:
                if inst.freq == "M":
                    continue
                for d in (dt.date(2026, 1, 15), dt.date(2026, 7, 15)):
                    gaps.append(abs(vs[v].close_utc_hours(d) - vs[inst.venue].close_utc_hours(d)))
    return float(max(gaps))
