"""Liquidity-aware transaction costs, taxes, FX conversion, borrow and funding."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import yaml

from .. import paths
from ..data.universe import Instrument


@lru_cache(maxsize=1)
def cfg() -> dict:
    return yaml.safe_load((paths.CONFIG / "costs.yaml").read_text("utf-8"))


@dataclass
class LegCost:
    half_spread_bp: float
    commission_bp: float
    tax_buy_bp: float
    tax_sell_bp: float
    fx_bp: float
    impact_bp: float

    def one_way(self, side: str) -> float:
        tax = self.tax_buy_bp if side == "buy" else self.tax_sell_bp
        return self.half_spread_bp + self.commission_bp + tax + self.fx_bp + self.impact_bp

    def round_trip(self, first_side: str = "buy") -> float:
        other = "sell" if first_side == "buy" else "buy"
        return self.one_way(first_side) + self.one_way(other)


def leg_cost(inst: Instrument, notional_usd: float | None = None, daily_vol: float | None = None,
             adv_usd: float | None = None) -> LegCost:
    c = cfg()
    tier = c["liquidity_tiers"][inst.liq]
    half = float(tier.get("half_spread_bp", 0.0))
    comm = float(tier.get("commission_bp", 0.0))
    taxes = c["instrument_taxes"].get(inst.id) or c["venue_taxes"].get(inst.venue) or {}
    fx_bp = 0.0
    if inst.price_ccy != "USD" and inst.asset_class in ("equity", "etf", "equity_index"):
        fx_bp = c["fx_conversion_bp"]["em" if inst.price_ccy in c["em_currencies"] else "major"]
    impact = 0.0
    k = float(tier.get("impact_k", 0.0))
    if k and notional_usd and adv_usd and daily_vol and adv_usd > 0:
        impact = k * daily_vol * np.sqrt(notional_usd / adv_usd) * 1e4
    return LegCost(half, comm, float(taxes.get("buy", 0.0)), float(taxes.get("sell", 0.0)), fx_bp, float(impact))


def yield_cost_bp(inst: Instrument) -> float:
    """Half-spread in yield bp for bond legs (multiply by DV01 per unit notional to get price bp)."""
    return float(cfg()["liquidity_tiers"][inst.liq].get("yield_half_spread_bp", 0.5))


def borrow(inst: Instrument) -> tuple[bool, float]:
    t = cfg()["borrow_tiers"][inst.borrow]
    return bool(t["available"]), float(t["fee_bp"]) if t["available"] else float("nan")


def stress_multiplier(returns: pd.Series) -> pd.Series:
    """Cost multiplier from realised volatility relative to its trailing median (uses data up to t only)."""
    s = cfg()["stress_scaling"]
    vol = returns.rolling(s["window"], min_periods=10).std()
    base = vol.rolling(s["baseline"], min_periods=60).median()
    return (vol / base).clip(lower=s["floor"], upper=s["cap"]).fillna(1.0)


def funding_spreads() -> tuple[float, float]:
    f = cfg()["funding"]
    return float(f["long_spread_bp"]), float(f["short_rebate_spread_bp"])
