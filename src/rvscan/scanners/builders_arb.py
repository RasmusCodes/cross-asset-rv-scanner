"""Builders for relationships with a structural link: ETF vs NAV, same-index ETFs, futures basis, ADR parity,
interlisted shares, share classes, dual-listed companies, FX triangles and FX-adjusted index tracking."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import calendars, fx, store
from ..data.universe import load_universe
from ..engine import costs as costmod
from .base import (Relationship, SpreadData, cash_pair_costs, day_fraction, leg_costs_rt, license_of, max_sync_gap,
                   overnight_rate, px_usd, simple_ret, tr_usd)

# Basket-side frictions for ETF arbitrage, bp (one way): what an authorised participant pays to trade the basket.
BASKET_HALF_BP = {"equity": 2.0, "equity_intl": 6.0, "bond": 20.0, "bond_intl": 25.0, "commodity": 3.0}
NAV_SYNC_GAP_H = {"equity": 0.0, "bond": 1.0, "commodity": 6.0, "equity_intl": 14.0, "bond_intl": 10.0}


def _stress(ids: list[str]) -> pd.Series:
    rets = pd.concat([np.log(tr_usd(i)).diff() for i in ids if store.has(i)], axis=1).mean(axis=1).dropna()
    return costmod.stress_multiplier(rets)


def _on(s, idx):
    return s.reindex(idx) if isinstance(s, pd.Series) else s


# ------------------------------------------------------------------------------------------------ ETF vs NAV
def etf_nav(rel: Relationship) -> SpreadData:
    etf, kind = rel.legs[0], rel.params["kind"]
    inst = load_universe()[etf]
    px = store.series(etf)
    nav = store.load_nav(etf)["nav"].dropna()
    ev = store.load_events(etf)
    for _, row in ev[ev["kind"] == "split"].iterrows():      # Yahoo close is split-adjusted, SSGA NAV is not
        nav = nav.where(nav.index >= row["date"], nav / row["value"])
    df = pd.concat({"px": px, "nav": nav}, axis=1, join="inner").dropna()
    spread = 1e4 * np.log(df["px"] / df["nav"])
    bad = spread.abs() > 1500                               # residual corporate-action mismatches
    notes = list(rel.constraints)
    if bad.any():
        notes.append(f"{int(bad.sum())} days with |premium| > 15% dropped as data errors.")
        spread = spread[~bad]
    idx = spread.index
    stress = _stress([etf]).reindex(idx).ffill().fillna(1.0)
    lc = costmod.leg_cost(inst)
    creation = costmod.cfg()["etf_creation_bp"]["bond" if kind.startswith("bond") else kind if kind in ("equity", "commodity", "equity_intl") else "equity"]
    basket = BASKET_HALF_BP[kind]
    band = creation + basket + lc.half_spread_bp + lc.commission_bp
    cost_rt = 2 * (lc.half_spread_bp + basket) * stress + 2 * lc.commission_bp
    carry, hold = cash_pair_costs(idx, 1.0, 1.0, costmod.borrow(inst)[1])
    pnl = spread.diff()                                     # long ETF / short basket gains the premium change
    return SpreadData(rel, spread, "bp", 0.0, band, band, pnl, carry * 0.0, hold, cost_rt,
                      NAV_SYNC_GAP_H[kind], license_of([etf]), notes=notes,
                      meta={"kind": kind, "band_bp": band, "nav_first": str(nav.index[0].date())})


# ------------------------------------------------------------------------------------------------ same-index ETFs
def same_index_etf(rel: Relationship) -> SpreadData:
    a, b = rel.legs
    ta, tb = tr_usd(a), tr_usd(b)
    df = pd.concat({"a": ta, "b": tb}, axis=1, join="inner").dropna()
    spread = 1e4 * (np.log(df["a"]) - np.log(df["b"]))
    idx = spread.index
    pnl = 1e4 * (df["a"].pct_change() - df["b"].pct_change())
    pnl = pnl.mask(pnl.abs() > 500)              # funds on one index cannot diverge 5% in a day: data errors
    stress = _stress([a, b]).reindex(idx).ffill().fillna(1.0)
    u = load_universe()
    cost_rt = leg_costs_rt([u[a], u[b]], [1, -1], stress)
    carry, hold = cash_pair_costs(idx, 1.0, 1.0, max(costmod.borrow(u[a])[1], costmod.borrow(u[b])[1]))
    return SpreadData(rel, spread, "bp", None, np.inf, np.inf, pnl, carry, hold, cost_rt, max_sync_gap([a, b]),
                      license_of([a, b]), notes=list(rel.constraints), meta={"fair_window": 126})


# ------------------------------------------------------------------------------------------------ futures basis
def trailing_dividend_yield(etf: str) -> pd.Series:
    px = store.series(etf)
    ev = store.load_events(etf)
    div = ev[ev["kind"] == "div"].set_index("date")["value"].sort_index()
    daily = div.reindex(px.index.union(div.index), fill_value=0.0).groupby(level=0).sum().reindex(px.index, fill_value=0.0)
    ttm = daily.rolling("365D").sum()
    return (ttm / px).where(px.index >= px.index[0] + pd.Timedelta(days=365))


def infer_front_expiry(lb: pd.Series, carry_rate: pd.Series, window: int = 15) -> pd.Series:
    """Which quarterly contract does a continuous futures series quote on each date?

    For every quarter the series quotes contract e_j until a roll date within the last `window` trading days before
    e_j and contract e_{j+1} afterwards. The roll date minimises the absolute gap between the observed log basis and
    the fair carry of the quoted contract. Returns the expiry date of the quoted contract per day."""
    idx = lb.index
    exps = calendars.quarterly_expiries(idx[0] - pd.Timedelta(days=100), idx[-1] + pd.Timedelta(days=200))
    out = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns]")
    for j in range(len(exps) - 1):
        e, e_next = exps[j], exps[j + 1]
        prev = exps[j - 1] if j > 0 else e - pd.Timedelta(days=95)
        mask = (idx > prev) & (idx <= e)
        days = idx[mask]
        if len(days) == 0:
            continue
        out[mask] = e
        cand = days[-window:]
        tau_e = (e - cand).days.values / 365.0
        tau_n = (e_next - cand).days.values / 365.0
        r = carry_rate.reindex(cand).values
        obs = lb.reindex(cand).values
        err_e = np.abs(obs - r * tau_e)
        err_n = np.abs(obs - r * tau_n)
        best, best_cost = len(cand), np.nansum(err_e)
        for k in range(len(cand)):
            c = np.nansum(err_e[:k]) + np.nansum(err_n[k:])
            if c < best_cost - 1e-12:
                best, best_cost = k, c
        if best < len(cand):
            out[cand[best:]] = e_next
    # dates after the last tested quarter keep the first expiry after them
    missing = out.isna()
    if missing.any():
        out[missing] = [exps[exps > d][0] for d in idx[missing]]
    return out


def futures_basis(rel: Relationship) -> SpreadData:
    fut, ind, etf = rel.legs
    F, S = store.series(fut), store.series(ind)
    q = trailing_dividend_yield(etf)
    r = overnight_rate("USD") / 100.0
    df = pd.concat({"F": F, "S": S, "q": q, "r": r}, axis=1).dropna(subset=["F", "S"])
    df["q"] = df["q"].ffill()
    df["r"] = df["r"].ffill()
    df = df.dropna()
    lb = np.log(df["F"] / df["S"])
    carry_rate = df["r"] - df["q"]
    expiry = infer_front_expiry(lb, carry_rate)
    tau = pd.Series((expiry.values - df.index.values).astype("timedelta64[D]").astype(float) / 365.0, index=df.index)
    fair = carry_rate * tau
    spread = 1e4 * (lb - fair)                                   # mispricing vs cost of carry, bp
    implied_fin = 1e4 * (lb / tau.clip(lower=5 / 365) + df["q"] - df["r"])   # implied financing over overnight, bp/yr
    switch = expiry.ne(expiry.shift(1)) & expiry.shift(1).notna()
    no_trade = df.index[switch.values]
    pnl = spread.diff().where(~switch)                            # contract switches are not P&L
    u = load_universe()
    stress = _stress([etf]).reindex(df.index).ffill().fillna(1.0)
    cost_rt = leg_costs_rt([u[fut], u[etf]], [1, -1], stress)
    ls, ss = costmod.funding_spreads()
    hold = (ss + costmod.borrow(u[etf])[1]) * day_fraction(df.index)  # short-basket rebate spread + borrow
    band = (cost_rt / 2 if isinstance(cost_rt, float) else cost_rt.median() / 2) + 20.0 * tau  # + dividend uncertainty (20 bp/yr)
    notes = list(rel.constraints) + [
        "Index basket traded through the ETF as a proxy (tracking error), dividends from the ETF's trailing 12 months.",
        "Contract quoted by the continuous series inferred from the basis (Yahoo does not say which month it shows)."]
    sd = SpreadData(rel, spread, "bp", 0.0, band, band, pnl, 0.0, hold, cost_rt, max_sync_gap([fut, ind]),
                    license_of(rel.legs), no_trade=pd.DatetimeIndex(no_trade), notes=notes,
                    meta={"implied_financing_bp": implied_fin, "tau": tau, "expiry": expiry, "div_yield": df["q"]})
    return sd


# ------------------------------------------------------------------------------------------------ parity pairs
def _parity(rel: Relationship, a: str, b: str, ratio: float, structural: bool, fx_venue: bool = True) -> SpreadData:
    """Two lines of the same economic claim: spread = log(P_a / (ratio * P_b)) in USD, bp."""
    u = load_universe()
    pa, pb = px_usd(a), px_usd(b)
    df = pd.concat({"a": pa, "b": pb}, axis=1, join="inner").dropna()
    spread = 1e4 * np.log(df["a"] / (ratio * df["b"]))
    # Vendor histories can carry a ticker back before the share class existed (Yahoo's GOOG before April 2014 is the
    # old class A line): the "spread" is then a series against itself. Drop such a degenerate leading segment.
    live = spread.diff().abs().rolling(60, min_periods=40).std() > 0.5
    if live.any() and not live.iloc[:80].any():
        first = live.idxmax() - pd.Timedelta(days=90)
        df = df[df.index >= first]
        spread = spread[spread.index >= first]
    ta, tb = tr_usd(a).reindex(df.index), tr_usd(b).reindex(df.index)
    pnl = 1e4 * (ta.pct_change() - tb.pct_change())
    leg_pnl = pd.DataFrame({a: 1e4 * ta.pct_change(), b: -1e4 * tb.pct_change()})
    # Two lines of one claim cannot diverge 15% in a day: such days are corporate actions the vendor adjusted on one
    # listing only (spin-offs, rights issues, stock dividends). They are excluded from P&L and reported.
    glitch = pnl.abs() > 1500
    # A permanent step in a same-claim spread exactly at a corporate-action date is a vendor adjustment that differs
    # between the listings (a spin-off booked as a "split" on one line, a bonus issue adjusted on one line only), not a
    # market move. Re-base the earlier history onto current terms and drop P&L around the event.
    repairs = []
    ev_dates = set()
    for leg in (a, b):
        ev = store.load_events(leg)
        ev_dates |= set(pd.to_datetime(ev.loc[ev["kind"] == "split", "date"]))
    for d in sorted(ev_dates):
        before = spread[spread.index < d - pd.Timedelta(days=4)].iloc[-20:]
        after = spread[spread.index > d + pd.Timedelta(days=4)].iloc[:20]
        if len(before) < 15 or len(after) < 15:
            continue
        jump = float(after.median() - before.median())
        if abs(jump) > 300:
            spread = spread.where(spread.index >= d, spread + jump)
            near = (spread.index >= d - pd.Timedelta(days=6)) & (spread.index <= d + pd.Timedelta(days=6))
            glitch = glitch | pd.Series(near, index=spread.index)
            repairs.append((str(d.date()), round(jump)))
    n_glitch = int(glitch.sum())
    pnl = pnl.mask(glitch)
    leg_pnl = leg_pnl.mask(glitch, axis=0)
    idx = spread.index
    stress = _stress([a, b]).reindex(idx).ffill().fillna(1.0)
    ia, ib = u[a], u[b]
    cost_rt = leg_costs_rt([ia, ib], [1, -1], stress)
    av_a, fee_a = costmod.borrow(ia)
    av_b, fee_b = costmod.borrow(ib)
    direction = "both"
    if not av_b and av_a:
        direction = "short_only"          # can only short the spread: short line a, buy line b
    elif not av_a and av_b:
        direction = "long_only"
    elif not av_a and not av_b:
        direction = "none"
    fees = [f for f in (fee_a, fee_b) if np.isfinite(f)]
    carry, hold = cash_pair_costs(idx, 1.0, 1.0, max(fees) if fees else 0.0)
    extra = ["FED_NOON"] if (ia.price_ccy != "USD" or ib.price_ccy != "USD") else None
    gap = max_sync_gap([a, b], extra)
    lc_a, lc_b = costmod.leg_cost(ia), costmod.leg_cost(ib)
    conv_bp = 0.0
    if rel.type == "adr":
        conv_bp = costmod.cfg()["adr_conversion_fee_usd"] / float(df["a"].iloc[-1]) * 1e4
    # Arbitrage when line a is rich: sell a, buy b (and convert). When a is cheap: buy a, sell b.
    band_up = conv_bp + lc_a.half_spread_bp + lc_b.half_spread_bp + lc_a.commission_bp + lc_b.commission_bp \
        + lc_b.tax_buy_bp + lc_a.tax_sell_bp + lc_a.fx_bp + lc_b.fx_bp
    band_down = conv_bp + lc_a.half_spread_bp + lc_b.half_spread_bp + lc_a.commission_bp + lc_b.commission_bp \
        + lc_a.tax_buy_bp + lc_b.tax_sell_bp + lc_a.fx_bp + lc_b.fx_bp
    level = float(spread.iloc[-min(252, len(spread)):].median())
    notes = list(rel.constraints)
    meta = {"ratio": ratio, "median_premium_1y_bp": level, "sync_gap_h": gap, "pnl_glitch_days": n_glitch,
            "repairs": repairs}
    for d, jump in repairs:
        notes.append(f"History before {d} re-based by {jump:+d} bp: the vendor's corporate-action adjustment differs "
                     "between the two listings on that date.")
    if n_glitch:
        notes.append(f"{n_glitch} day(s) with a >15% one-day gap between the two lines' total returns excluded from "
                     "P&L (corporate action adjusted on one listing only).")
    if structural and abs(level) > 1500:
        notes.append(f"Ratio check failed: median 1-year premium {level:.0f} bp; check the ADR ratio or splits.")
        meta["ratio_check"] = "failed"
    # level shifts: a structural parity spread whose 60-day median moves > 10% suggests a ratio change in the data
    med = spread.rolling(60, min_periods=30).median()
    jumps = med.diff(60).abs() > 1000
    if structural and jumps.any():
        meta["level_shift_dates"] = [str(d.date()) for d in med.index[jumps.values]][:5]
    return SpreadData(rel, spread, "bp", 0.0 if structural else None, band_up, band_down, pnl, carry, hold, cost_rt,
                      gap, license_of([a, b]), leg_pnl=leg_pnl, direction=direction, notes=notes, meta=meta)


def adr(rel: Relationship) -> SpreadData:
    us, home = rel.legs
    sd = _parity(rel, us, home, float(rel.params["ratio"]), structural=(rel.tier == "arbitrage"))
    if rel.tier != "arbitrage":
        sd.band_up = sd.band_down = np.inf
    return sd


def interlisted(rel: Relationship) -> SpreadData:
    us, to = rel.legs
    sd = _parity(rel, us, to, 1.0, structural=True)
    sd.notes.append("Both lines close at 16:00 New York time, but the CAD rate is the noon H.10 fixing: deviations "
                    "smaller than a few tens of bp are FX-timing noise.")
    return sd


def share_class(rel: Relationship) -> SpreadData:
    a, b = rel.legs
    ratio = float(rel.params["ratio"])
    sd = _parity(rel, a, b, ratio, structural=False)
    if rel.tier == "arbitrage":
        # One-way conversion a -> ratio * b: b can't trade above a/ratio by more than costs (spread can't fall below
        # -band_down); a may trade at any premium.
        sd.band_up = np.inf
        sd.meta["bound"] = "spread >= -band_down (class A converts into class B)"
    else:
        sd.band_up = sd.band_down = np.inf
    return sd


def dual_listed(rel: Relationship) -> SpreadData:
    a, b = rel.legs
    sd = _parity(rel, a, b, float(rel.params.get("ratio", 1.0)), structural=False)
    sd.band_up = sd.band_down = np.inf
    return sd


# ------------------------------------------------------------------------------------------------ FX
def fx_triangular(rel: Relationship) -> SpreadData:
    """ECB cross vs the cross implied by two H.10 rates: the triangle holds for synchronous quotes, but the
    fixings are taken about six hours apart, so any 'deviation' is mostly the market move between the fixings."""
    cross_id, base, quote = rel.legs
    u = load_universe()
    h10 = {c: store.series(fx.FX_IDS[c]) for c in (base, quote)}        # H.10 only (no ECB extension here)
    implied = h10[base] / h10[quote]                                     # quote per base, NY noon
    ecb = store.series(cross_id)                                         # quote per base, 14:15 CET
    df = pd.concat({"ecb": ecb, "h10": implied}, axis=1, join="inner").dropna()
    spread = 1e4 * np.log(df["ecb"] / df["h10"])
    pnl = spread.diff()
    cost = 2 * 2 * costmod.cfg()["liquidity_tiers"]["fx_major"]["half_spread_bp"]
    gap = max_sync_gap([cross_id, fx.FX_IDS[base]])
    return SpreadData(rel, spread, "bp", 0.0, cost, cost, pnl, 0.0, 0.0, cost, gap, "open",
                      notes=list(rel.constraints) + ["Fixings 6 hours apart: not executable; kept as a false-positive benchmark."],
                      meta={"executable_in_principle": False})


def fx_adjusted(rel: Relationship) -> SpreadData:
    etf, ind = rel.legs
    ccy = rel.params["ccy"]
    u = load_universe()
    t_etf = tr_usd(etf)
    idx_usd = fx.to_usd(store.series(ind), ccy).dropna()
    df = pd.concat({"etf": t_etf, "ind": idx_usd}, axis=1, join="inner").dropna()
    spread = 1e4 * (np.log(df["etf"]) - np.log(df["ind"]))
    pnl = 1e4 * (df["etf"].pct_change() - df["ind"].pct_change())
    stress = _stress([etf]).reindex(spread.index).ffill().fillna(1.0)
    lc_e = costmod.leg_cost(u[etf])
    fut_half = costmod.cfg()["liquidity_tiers"]["fut_eq"]["half_spread_bp"] + 0.5   # index leg via local futures
    cost_rt = 2 * (lc_e.half_spread_bp + fut_half) * stress + 2 * lc_e.commission_bp + 2 * 1.0
    # Long ETF (total return) vs short index futures: the futures leg pays away the index dividend yield relative
    # to the price index used here, so carry of +1 = -dividend yield (ETF's trailing yield as the proxy).
    q = trailing_dividend_yield(etf).reindex(spread.index).ffill().fillna(0.0)
    carry = -q * 1e4 * day_fraction(spread.index)
    _, hold = cash_pair_costs(spread.index, 1.0, 0.0, 0.0)
    gap = max_sync_gap([etf, ind], ["FED_NOON"])
    return SpreadData(rel, spread, "bp", None, np.inf, np.inf, pnl, carry, hold, cost_rt, gap, license_of([etf, ind]),
                      notes=list(rel.constraints) + ["Index leg would be traded through local futures (not in the data)."],
                      meta={"fair_window": 252})
