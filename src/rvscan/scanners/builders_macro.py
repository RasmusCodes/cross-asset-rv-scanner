"""Builders for term structures, commodity spreads, bond spreads, credit ETFs, funding spreads and FX carry."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import bonds, calendars, fx, store
from ..data.universe import load_universe
from ..engine import costs as costmod
from .base import (Relationship, SpreadData, cash_pair_costs, day_fraction, lagged_monthly, leg_costs_rt, license_of,
                   max_sync_gap, overnight_rate, tr_usd)

EIA_NEXT = {"WTI1": "WTI2", "WTI2": "WTI3", "WTI3": "WTI4", "NG1": "NG2", "NG2": "NG3", "NG3": "NG4",
            "RBOB1": "RBOB2", "RBOB2": "RBOB3", "RBOB3": "RBOB4", "HO1": "HO2", "HO2": "HO3", "HO3": "HO4"}
EIA_ROOT = {"WTI": "CL", "NG": "NG", "RBOB": "RB", "HO": "HO"}
SEASONAL_ROOTS = {"NG", "RB", "HO"}
CUSHING_STORAGE_USD_BBL_MONTH = 0.40      # assumed cost of storing crude at Cushing (for the full-carry bound)


def _root(iid: str) -> str:
    return EIA_ROOT[iid.rstrip("0123456789")]


def roll_adjusted(iid: str) -> pd.Series:
    """Daily return of holding EIA 'contract k' through rolls (needs contract k+1 for the roll day)."""
    nxt = EIA_NEXT[iid]
    front, n = store.series(iid), store.series(nxt)
    rolls = calendars.roll_days(_root(iid), front.index)
    return calendars.roll_adjusted_returns(front, n, rolls)


def seasonal_fair(spread: pd.Series, years: int = 5, min_years: int = 3) -> pd.Series:
    """Same-calendar-month average of the previous `years` years (strictly earlier years: no look-ahead)."""
    m = spread.groupby([spread.index.year, spread.index.month]).mean()
    tab = m.unstack()                                      # rows = year, cols = month
    fair = tab.shift(1).rolling(years, min_periods=min_years).mean()
    keys = list(zip(spread.index.year, spread.index.month))
    vals = [fair.at[y, mo] if (y in fair.index and mo in fair.columns) else np.nan for y, mo in keys]
    return pd.Series(vals, index=spread.index, dtype=float)


# ------------------------------------------------------------------------------------------------ calendar spreads
def calendar(rel: Relationship) -> SpreadData:
    front, back = rel.legs
    root = rel.params["root"]
    c1, c2 = store.series(front), store.series(back)
    df = pd.concat({"c1": c1, "c2": c2}, axis=1, join="inner").dropna()
    df = df[(df > 0).all(axis=1)]                         # WTI contract 1 settled at -37.63 on 2020-04-20
    spread = 1e4 * np.log(df["c2"] / df["c1"])            # contango in bp of the front price
    r1, r2 = roll_adjusted(front).reindex(df.index), roll_adjusted(back).reindex(df.index)
    pnl = 1e4 * (r2 - r1)                                  # long the spread = long deferred, short front
    rolls = calendars.roll_days(root, store.series(front).index)
    u = load_universe()
    stress = pd.Series(1.0, index=df.index)
    vol = np.log(df["c1"]).diff()
    stress = costmod.stress_multiplier(vol).reindex(df.index).fillna(1.0)
    cost_rt = leg_costs_rt([u[front], u[back]], [1, -1], stress)
    fair = seasonal_fair(spread) if root in SEASONAL_ROOTS else None
    meta = {"root": root, "data_end": str(df.index[-1].date())}
    if root == "CL":
        r = overnight_rate("USD").reindex(df.index).ffill() / 100.0
        meta["full_carry_bp"] = 1e4 * np.log(1 + r / 12 + CUSHING_STORAGE_USD_BBL_MONTH / df["c1"])
    notes = list(rel.constraints) + ["EIA stopped publishing NYMEX contract prices on 2024-04-05; history ends there.",
                                     "Positions are rolled each month at a full round-trip cost (conservative)."]
    if root in SEASONAL_ROOTS:
        notes.append("Fair value = same calendar month in the previous five years (seasonal curve).")
    return SpreadData(rel, spread, "bp", fair, np.inf, np.inf, pnl, 0.0, 0.0, cost_rt, 0.0, "open",
                      roll_days=rolls, notes=notes, meta=meta)


# ------------------------------------------------------------------------------------------------ commodity spreads
def commodity_spread(rel: Relationship) -> SpreadData:
    kind = rel.params["kind"]
    u = load_universe()
    if kind == "crack":
        rb, ho, wti = rel.legs
        df = pd.concat({"rb": store.series(rb), "ho": store.series(ho), "wti": store.series(wti)}, axis=1, join="inner").dropna()
        df = df[(df > 0).all(axis=1)]
        crack = (2 * 42 * df["rb"] + 42 * df["ho"] - 3 * df["wti"]) / 3.0       # $/bbl
        notional = pd.DataFrame({rb: 2 * 42 * df["rb"], ho: 42 * df["ho"], wti: -3 * df["wti"]})
        gross = notional.abs().sum(axis=1)
        w = notional.div(gross, axis=0).shift(1)                                # weights known before the move
        rets = pd.DataFrame({k: roll_adjusted(k).reindex(df.index) for k in (rb, ho, wti)})
        leg_pnl = 1e4 * w * rets
        pnl = leg_pnl.sum(axis=1, min_count=3)
        vol = np.log(df["wti"]).diff()
        stress = costmod.stress_multiplier(vol).reindex(df.index).fillna(1.0)
        cost_rt = leg_costs_rt([u[rb], u[ho], u[wti]], [0.4, 0.2, 0.4], stress)
        rolls = calendars.roll_days("CL", store.series(wti).index).union(calendars.roll_days("RB", store.series(rb).index))
        return SpreadData(rel, crack, "usd", seasonal_fair(crack), np.inf, np.inf, pnl, 0.0, 0.0, cost_rt, 0.0, "open",
                          roll_days=rolls, leg_pnl=leg_pnl,
                          notes=list(rel.constraints) + ["Seasonal fair value (gasoline cracks peak in spring).",
                                                         "EIA contract data end 2024-04-05."],
                          meta={"data_end": str(df.index[-1].date())})
    a, b = rel.legs
    if kind in ("location",):
        pa, pb = store.series(a), store.series(b)
        df = pd.concat({"a": pa, "b": pb}, axis=1, join="inner").dropna()
        spread = df["a"] - df["b"]
        notes = list(rel.constraints) + ["Monitor only: spot assessments (or continuous futures with different roll "
                                         "dates) are not a tradable spread."]
        return SpreadData(rel, spread, "usd", None, np.inf, np.inf, None, 0.0, 0.0, 0.0, max_sync_gap([a, b]),
                          license_of([a, b]), notes=notes, meta={"fair_window": 252})
    if kind == "metal_etf":
        df = pd.concat({"a": tr_usd(a), "b": store.series(b)}, axis=1, join="inner").dropna()
        spread = 1e4 * np.log(df["a"] / df["b"])
        notes = list(rel.constraints) + ["Monitor only: the continuous gold future jumps at every roll (carry), "
                                         "which a z-score mistakes for mean reversion."]
        return SpreadData(rel, spread, "bp", None, np.inf, np.inf, None, 0.0, 0.0, 0.0, max_sync_gap([a, b]),
                          license_of([a, b]), notes=notes, meta={"fair_window": 252})
    if kind == "roll_etf":
        r_f = roll_adjusted(b)
        fut_idx = (1 + r_f.fillna(0)).cumprod()
        df = pd.concat({"etf": tr_usd(a), "fut": fut_idx}, axis=1, join="inner").dropna()
        spread = 1e4 * (np.log(df["etf"]) - np.log(df["fut"]))
        pnl = 1e4 * (df["etf"].pct_change() - df["fut"].pct_change())
        stress = costmod.stress_multiplier(np.log(df["fut"]).diff()).reindex(df.index).fillna(1.0)
        cost_rt = leg_costs_rt([u[a], u[b]], [1, -1], stress)
        carry, hold = cash_pair_costs(df.index, 1.0, 0.0, costmod.borrow(u[a])[1])
        return SpreadData(rel, spread, "bp", None, np.inf, np.inf, pnl, carry, hold, cost_rt, max_sync_gap([a, b]),
                          license_of([a, b]), notes=list(rel.constraints) + [
                              "USO rolls ahead of expiry and changed its contract mix in April 2020: the tracking "
                              "spread has structural breaks, not a stable mean."], meta={"fair_window": 252})
    raise ValueError(kind)


# ------------------------------------------------------------------------------------------------ bond spreads
INDEX_YIELD_HALF_SPREAD = {"corporate": 3.0, "sovereign": 0.75}
MATURITY_DEFAULT = {"MOODY_AAA": 25.0, "MOODY_BAA": 25.0}


def _maturity(inst) -> float | None:
    if inst.id in MATURITY_DEFAULT:
        return MATURITY_DEFAULT[inst.id]
    if inst.mat is not None:
        return float(inst.mat)
    if inst.src in ("treasury", "mof"):
        return float(inst.key)
    return None


def bond_spread(rel: Relationship) -> SpreadData:
    u = load_universe()
    legs, w = rel.legs, [float(x) for x in rel.params["weights"]]
    insts = [u[i] for i in legs]
    monthly = any(i.freq == "M" for i in insts)
    ys = pd.concat({i: store.series(i) for i in legs}, axis=1, join="inner").dropna()
    if monthly:
        ys = ys.groupby(ys.index.to_period("M")).last()
        ys.index = ys.index.to_timestamp(how="end").normalize()
    spread = 100.0 * sum(wi * ys[i] for wi, i in zip(w, legs))
    is_oas = any(i.id.startswith("ICE_") for i in insts)
    pnl = None if is_oas else spread.diff()
    # carry of +1 (long spread = short the w>0 bonds, long the w<0 bonds), yield bp per DV01 unit per period
    carry = pd.Series(0.0, index=spread.index)
    hold = pd.Series(0.0, index=spread.index)
    dt = day_fraction(spread.index)
    for wi, inst in zip(w, insts):
        m = _maturity(inst)
        if m is None or is_oas:
            continue
        f = overnight_rate(inst.price_ccy).reindex(spread.index, method="ffill") if not overnight_rate(inst.price_ccy).empty \
            else pd.Series(np.nan, index=spread.index)
        D = pd.Series(bonds.modified_duration(ys[inst.id].values, m), index=spread.index)
        c_i = (ys[inst.id] - f) * 100.0 * dt / D
        carry = carry + (-wi) * c_i.fillna(0.0)
        hold = hold + abs(wi) * 10.0 * dt / D                    # repo spread 10 bp/yr on the financed notional
    half = []
    for inst in insts:
        tier = costmod.cfg()["liquidity_tiers"].get(inst.liq, {})
        if "yield_half_spread_bp" in tier:
            half.append(tier["yield_half_spread_bp"])
        else:
            half.append(INDEX_YIELD_HALF_SPREAD["corporate" if inst.asset_class == "corporate" else "sovereign"])
    cost_rt = float(sum(abs(wi) * 2 * h for wi, h in zip(w, half)))
    hypothetical = any(i.liq == "index" for i in insts)
    notes = list(rel.constraints)
    if hypothetical:
        notes.append("Index yields (not a single bond): P&L is hypothetical, costs assumed at "
                     f"{max(half):.2f} bp of yield per leg.")
    if is_oas:
        notes.append("Monitor only: option-adjusted spreads over 3 years of data.")
    return SpreadData(rel, spread, "yield_bp", None, np.inf, np.inf, pnl, carry, hold, cost_rt,
                      max_sync_gap(legs), license_of(legs), pnl_unit="yield_bp", freq="M" if monthly else "D",
                      notes=notes, meta={"weights": w, "hypothetical": hypothetical,
                                         "fair_window": 36 if monthly else 252})


# ------------------------------------------------------------------------------------------------ credit ETFs
def etf_credit(rel: Relationship) -> SpreadData:
    a, b = rel.legs
    u = load_universe()
    df = pd.concat({"a": tr_usd(a), "b": tr_usd(b)}, axis=1, join="inner").dropna()
    ra, rb = df["a"].pct_change(), df["b"].pct_change()
    beta = (ra.rolling(126, min_periods=60).cov(rb) / rb.rolling(126, min_periods=60).var()).clip(0.0, 3.0).shift(1)
    hedged = (ra - beta * rb)
    spread = 1e4 * hedged.fillna(0.0).cumsum()
    pnl = 1e4 * hedged
    stress = costmod.stress_multiplier(np.log(df["a"]).diff()).reindex(df.index).fillna(1.0)
    ca, cb = costmod.leg_cost(u[a]), costmod.leg_cost(u[b])
    cost_rt = (2 * (ca.half_spread_bp + beta.abs().fillna(1.0) * cb.half_spread_bp) * stress
               + 2 * (ca.commission_bp + beta.abs().fillna(1.0) * cb.commission_bp))
    rebal = beta.diff().abs().fillna(0.0) * (cb.half_spread_bp + cb.commission_bp)
    _, hold = cash_pair_costs(df.index, 1.0, float(beta.abs().median()), costmod.borrow(u[b])[1])
    carry = 0.0                                             # excess-return convention (see builders_stat)
    return SpreadData(rel, spread, "bp", None, np.inf, np.inf, pnl, carry, hold, cost_rt, max_sync_gap([a, b]),
                      license_of([a, b]), rebal_cost=rebal, est_hedge=True, notes=list(rel.constraints),
                      meta={"hedge": "rolling 126-day return beta", "beta": beta, "fair_window": 252})


# ------------------------------------------------------------------------------------------------ funding spreads
def funding(rel: Relationship) -> SpreadData:
    a, b = rel.legs
    sa = store.series(a)
    if a == "SOFR" or a == "EFFR":
        pass
    sb = store.series(b)
    if b == "IORB" and store.has("IOER"):
        sb = sb.combine_first(store.series("IOER"))       # IOER until 2021-07-28, IORB after
    df = pd.concat({"a": sa, "b": sb}, axis=1, join="inner").dropna()
    spread = 100.0 * (df["a"] - df["b"])
    return SpreadData(rel, spread, "bp", None, np.inf, np.inf, None, 0.0, 0.0, 0.0, 0.0, license_of([a, b]),
                      notes=list(rel.constraints) + ["Monitor only: trading it needs direct money-market access."],
                      meta={"fair_window": 252})


# ------------------------------------------------------------------------------------------------ FX carry
def carry_rate(ccy: str) -> pd.Series:
    """3-month rate (OECD, monthly average) usable from the start of the month after next (one-month lag)."""
    iid = f"RATE3M_{ccy}"
    return lagged_monthly(store.series(iid), 1) if store.has(iid) else pd.Series(dtype=float)


def fx_carry(rel: Relationship) -> SpreadData:
    ccy = rel.legs[0]
    u = load_universe()
    s = fx.usd_per(ccy)
    r_c, r_u = carry_rate(ccy), carry_rate("USD")
    df = pd.concat({"s": s, "rc": r_c, "ru": r_u}, axis=1).dropna(subset=["s"])
    df[["rc", "ru"]] = df[["rc", "ru"]].ffill()
    df = df.dropna()
    diff_bp = 100.0 * (df["rc"] - df["ru"])                          # annual carry of long ccy vs USD, bp
    dt = day_fraction(df.index)
    pnl = 1e4 * df["s"].pct_change() + diff_bp.shift(1) * dt         # spot return + accrued rate differential
    vol = np.log(df["s"]).diff().rolling(63, min_periods=20).std() * np.sqrt(252) * 1e4
    tier = "fx_em" if ccy in costmod.cfg()["em_currencies"] else "fx_major"
    cost_rt = 2 * costmod.cfg()["liquidity_tiers"][tier]["half_spread_bp"]
    iid = fx.FX_IDS[ccy]
    return SpreadData(rel, diff_bp, "bp", None, np.inf, np.inf, pnl, 0.0, 0.0, float(cost_rt), 0.0,
                      license_of([iid]), notes=list(rel.constraints),
                      meta={"carry_to_vol": diff_bp / vol, "vol_bp": vol, "ccy": ccy})
