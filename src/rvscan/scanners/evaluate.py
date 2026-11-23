"""Turn a spread into an opportunity assessment: signal, models, expected convergence, costs, carry, liquidity,
borrow, currency exposure, basis risk, historical convergence and an executability verdict.

Verdicts:
  executable                    arbitrage-tier link, deviation outside the no-arbitrage band, synchronous prices,
                                positive expected edge after costs, short leg borrowable, data fresh
  executable (not enforceable)  the same for an economic (quasi) link: convergence is likely but cannot be forced
  statistical signal            a statistical relationship with |z| above the entry level: a bet, not an arbitrage
  watch                         |z| above entry but at least one check fails (reasons listed)
  no signal                     |z| below the entry level
  monitor                       not tradable as specified (spot assessments, money-market rates, OAS indices)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import store
from ..data.universe import load_universe
from ..engine import costs as costmod
from ..engine.backtest import TradeParams, convergence_episodes, zscore
from ..models import bayes, cointegration as ci, ou, regime
from .base import SpreadData

SYNC_LIMIT_H = 1.0          # legs whose closing prints are further apart cannot be compared as an arbitrage
STALE_BDAYS = 3


def latest_market_date() -> pd.Timestamp:
    return store.series("SPY").index[-1]


def _short_legs(sd: SpreadData, position: int) -> list[str]:
    legs = sd.rel.legs
    t = sd.rel.params.get("builder")
    if position == 0 or t in ("etf_nav", "futures_basis", "fx_triangular", "fx_carry", "funding", "bond_spread",
                              "calendar", "commodity_spread", "fx_adjusted"):
        return []
    if len(legs) >= 2:
        return [legs[1]] if position > 0 else [legs[0]]
    return []


def _liquidity(sd: SpreadData) -> tuple[float, list]:
    u = load_universe()
    advs = []
    for leg in sd.rel.legs:
        inst = u.get(leg)
        if inst is None or inst.src != "yahoo" or not inst.tradable:
            continue
        v = store.volume_usd(leg).dropna()
        if len(v):
            advs.append((leg, float(v.iloc[-1])))
    if not advs:
        return float("nan"), []
    return min(a for _, a in advs), advs


def evaluate(sd: SpreadData, p: TradeParams | None = None, window: int | None = None, asof=None,
             with_regime: bool = True, with_history: bool = True) -> dict:
    p = p or TradeParams()
    u = load_universe()
    monthly = sd.freq == "M"
    window = window or (72 if monthly else 504)
    s = sd.spread.dropna()
    if asof is not None:
        s = s[s.index <= pd.Timestamp(asof)]
    out: dict = {"id": sd.rel.id, "name": sd.rel.name, "type": sd.rel.type, "tier": sd.rel.tier,
                 "legs": sd.rel.legs, "unit": sd.unit, "pnl_unit": sd.pnl_unit, "license": sd.license,
                 "freq": sd.freq, "tradable": sd.tradable, "sync_gap_h": round(sd.sync_gap_h, 2)}
    if len(s) < 60:
        out.update({"status": "insufficient data", "reasons": ["fewer than 60 observations"]})
        return out
    last = s.index[-1]
    fair = sd.fair_series.reindex(s.index)
    dev = (s - fair)
    z = zscore(sd, p.lookback).reindex(s.index)
    z0, d0, s0, f0 = float(z.iloc[-1]), float(dev.iloc[-1]), float(s.iloc[-1]), float(fair.iloc[-1])
    sig = float(dev.rolling(p.lookback, min_periods=p.lookback // 2).std().iloc[-1])
    hist = dev.dropna()
    pct = float((hist <= d0).mean()) if len(hist) else float("nan")
    out.update({"date": str(last.date()), "spread": s0, "fair": f0, "deviation": d0, "z": z0, "sigma": sig,
                "percentile": pct, "first_date": str(s.index[0].date()), "obs": int(len(s))})
    # --- models on the trailing window ---------------------------------------------------------------------
    # A structural fair value comes from an arbitrage mechanism (parity, NAV, cost of carry). Statistical residuals
    # are also centred at zero but have no mechanism behind that level.
    structural = sd.rel.tier == "arbitrage" and sd.fair is not None and not isinstance(sd.fair, pd.Series)
    tail = (s if not isinstance(sd.fair, pd.Series) else dev).iloc[-window:].dropna()
    n_vars = 2 if sd.est_hedge else 1
    o = ou.fit_ou(tail.values, n_vars=n_vars)
    adf = ci.adf(tail.values, regression="c", n_vars=n_vars)
    bm = bayes.fit_spread_model(tail.values, fair_value=f0 if structural else None)
    hl = o.half_life_days if o.mean_reverting else float("inf")
    horizon = int(np.clip(hl * 2 if np.isfinite(hl) else p.max_hold, 5, p.max_hold))
    position = 0
    is_carry = sd.rel.params.get("builder") == "fx_carry"
    if is_carry:
        ctv = sd.meta["carry_to_vol"].reindex(s.index)
        ctv0 = float(ctv.iloc[-1]) if np.isfinite(ctv.iloc[-1]) else 0.0
        out["carry_to_vol"] = ctv0
        if abs(ctv0) >= p.min_carry_to_vol:
            position = 1 if s0 > 0 else -1
    elif np.isfinite(z0) and abs(z0) >= p.entry_z:
        position = -1 if z0 > 0 else 1
    target = f0 + np.sign(d0) * p.exit_z * sig if np.isfinite(sig) else f0
    stop = f0 + np.sign(d0) * p.stop_z * sig if np.isfinite(sig) else None
    fc = bayes.forecast(bm, s0=s0, horizon=horizon, target=target, stop=stop, n_paths=2000)
    exp_conv = (s0 - o.mu) * (1 - np.exp(-o.theta * horizon / 252.0)) if o.mean_reverting else 0.0
    out.update({
        "half_life": hl, "half_life_ci": o.half_life_ci, "ou_mu": o.mu, "ou_mean_reverting": o.mean_reverting,
        "adf_stat": adf.stat, "adf_p": adf.pvalue, "bayes_p_mr": bm.p_mean_reverting, "bayes_p_hit": fc.prob_hit,
        "bayes_p_stop": fc.prob_stop, "bayes_exp_gain": fc.expected_gain, "horizon": horizon,
        "expected_convergence": float(abs(exp_conv)) if position else 0.0, "position": position,
    })
    if with_regime and len(tail) > 150:
        try:
            ms = regime.fit_msar(tail.values, max_iter=60)
            out["regime_p_mr"] = float(ms.filtered[-1, 0])
            out["regime_half_lives"] = [float(x) for x in ms.half_life_days]
        except Exception:  # noqa: BLE001
            out["regime_p_mr"] = float("nan")
    # --- costs, carry, band ------------------------------------------------------------------------------------
    cost_rt = sd.at(sd.cost_rt, last) * p.cost_mult
    carry = sd.at(sd.carry, last)
    hold = sd.at(sd.hold_cost, last)
    band = sd.at(sd.band_up if d0 > 0 else sd.band_down, last)
    # Convergence is measured in spread units; convert to P&L units when they differ (crack spread in $/bbl).
    scale = 1.0
    if sd.unit != sd.pnl_unit and sd.pnl is not None:
        ds = s.diff().iloc[-window:]
        pn = sd.pnl.reindex(ds.index)
        ok = ds.notna() & pn.notna()
        scale = float(np.polyfit(ds[ok], pn[ok], 1)[0]) if ok.sum() > 30 else 0.0
    carry_h = (position * carry - (hold if position else 0.0)) * horizon
    edge = abs(exp_conv) * abs(scale) - cost_rt + carry_h if position else float("nan")
    if is_carry and position:
        horizon = 63
        carry_h = abs(s0) * horizon / 252.0               # rate differential earned over a quarter, bp
        exp_conv = 0.0
        edge = carry_h - cost_rt
        out["horizon"] = horizon
        out["expected_convergence"] = 0.0
    out.update({"cost_rt": cost_rt, "carry_per_period": carry, "hold_per_period": hold, "carry_horizon": carry_h,
                "band": band, "outside_band": bool(abs(d0) > band) if np.isfinite(band) else None,
                "net_edge": edge})
    # --- liquidity, borrow, currency, basis risk ---------------------------------------------------------------
    adv, advs = _liquidity(sd)
    cap = costmod.cfg()["max_adv_share"] * adv if np.isfinite(adv) else float("nan")
    shorts = _short_legs(sd, position)
    borrow_ok, borrow_notes = True, []
    for leg in shorts:
        av, fee = costmod.borrow(u[leg])
        if not av:
            borrow_ok = False
            borrow_notes.append(f"{leg}: not borrowable (assumption)")
        else:
            borrow_notes.append(f"{leg}: {u[leg].borrow} borrow, {fee:.0f} bp/yr")
    ccys = sorted({u[l].price_ccy for l in sd.rel.legs if l in u and u[l].price_ccy != "USD"})
    pnl_tail = sd.pnl.reindex(s.index).iloc[-window:].dropna() if sd.pnl is not None else pd.Series(dtype=float)
    basis_sd = float(pnl_tail.std()) if len(pnl_tail) > 20 else float(s.diff().iloc[-window:].std())
    out.update({"adv_min_usd": adv, "capacity_usd": cap, "advs": advs, "borrow_ok": borrow_ok, "borrow": borrow_notes,
                "currencies": ccys, "fx_exposure": ("hedged in USD terms (parity legs move with FX)" if
                                                    sd.rel.type in ("adr", "cross_listing") and ccys else
                                                    ", ".join(ccys) if ccys else "USD only"),
                "basis_risk_sd": basis_sd, "loss_99_10d": 2.33 * np.sqrt(10 if not monthly else 1) * basis_sd})
    # --- historical convergence ---------------------------------------------------------------------------------
    if with_history:
        ep = convergence_episodes(sd, p.entry_z, p.exit_z, p.lookback, horizon=252 if not monthly else 36)
        if len(ep):
            out.update({"episodes": int(len(ep)), "reverted_share": float(ep["reverted"].mean()),
                        "median_periods_to_revert": float(ep["periods_to_revert"].median()),
                        "mae_p95": float(ep["max_adverse"].quantile(0.95))})
        else:
            out.update({"episodes": 0})
    # --- verdict ------------------------------------------------------------------------------------------------
    reasons = []
    bdays_old = len(pd.bdate_range(last, latest_market_date())) - 1
    if bdays_old > STALE_BDAYS:
        reasons.append(f"data end {last.date()} ({bdays_old} business days old)")
    if sd.sync_gap_h > SYNC_LIMIT_H:
        reasons.append(f"closing prices {sd.sync_gap_h:.1f} h apart: deviation partly a timing artefact")
    if sd.meta.get("ratio_check") == "failed":
        reasons.append("ratio check failed")
    if not sd.tradable:
        status = "monitor"
    elif position == 0:
        status = "no signal"
    else:
        if np.isfinite(band) and abs(d0) <= band and p.use_band:
            reasons.append("inside the no-arbitrage band")
        if not structural and not o.mean_reverting and not is_carry:
            reasons.append(f"no evidence of mean reversion in the trailing window (ADF p = {adf.pvalue:.2f})")
        if np.isfinite(hl) and hl < 1.0 and not is_carry:
            reasons.append("half-life under one period: mostly closing-print noise that reverts before a next-day trade")
        if not (np.isfinite(edge) and edge > 0):
            reasons.append("expected convergence does not cover costs")
        if not borrow_ok:
            reasons.append("short leg not borrowable")
        if sd.direction == "short_only" and position > 0 or sd.direction == "long_only" and position < 0:
            reasons.append("direction blocked by conversion / borrow limits")
        if reasons:
            status = "watch"
        elif sd.rel.tier == "arbitrage":
            status = "executable"
        elif sd.rel.tier == "quasi":
            status = "executable (not enforceable)"
        else:
            status = "statistical signal"
    if status in ("no signal", "monitor") and not reasons:
        reasons = []
    out.update({"status": status, "reasons": reasons, "notes": sd.notes})
    return out
