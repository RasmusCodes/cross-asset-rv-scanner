"""Build the JSON files behind the public web demo, enforcing the data-licence policy.

Policy (decided for the public site): row-level series are published only for instruments whose source is public
domain or open with attribution (Treasury, Fed, NY Fed, ECB, Bundesbank, MoF Japan, BoC, RBA, BoE, EIA, OECD).
Relationships that use any vendor series (Yahoo prices, issuer NAVs, Moody's / ICE yields) are published as derived
analytics only: the spread, its z-score inputs, hedge ratios, the per-unit P&L of the hedged position, costs and
backtest results. Their legs' own prices or returns are never written. `assert_policy` re-checks every file.
"""
from __future__ import annotations

import dataclasses
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths
from ..data import store
from ..data.universe import ASSET_CLASS_LABELS, load_universe
from ..engine import backtest as bt
from ..scanners import catalog
from ..scanners.base import TIERS, TYPES, SpreadData

EPOCH = pd.Timestamp("1970-01-01")
DAILY_START = pd.Timestamp("2000-01-01")

# Open-data series offered in the pair builder, grouped for lazy loading. kind: "yield" (percent; spreads in bp,
# P&L = change in spread per DV01 unit) or "price" (log levels; P&L = hedged simple returns).
OPEN_GROUPS = {
    "rates": {"label": "Sovereign yields", "kind": "yield", "ids": [
        "UST_3M", "UST_1Y", "UST_2Y", "UST_3Y", "UST_5Y", "UST_7Y", "UST_10Y", "UST_20Y", "UST_30Y", "TIPS_5Y",
        "TIPS_10Y", "TIPS_30Y", "BUND_2Y", "BUND_5Y", "BUND_10Y", "BUND_30Y", "JGB_2Y", "JGB_5Y", "JGB_10Y", "JGB_30Y",
        "GILT_5Y", "GILT_10Y", "GILT_20Y", "CGB_2Y", "CGB_5Y", "CGB_10Y", "ACGB_2Y", "ACGB_5Y", "ACGB_10Y",
        "EA_AAA_2Y", "EA_AAA_10Y", "EA_ALL_2Y", "EA_ALL_10Y"]},
    "money": {"label": "Money markets and funding", "kind": "yield", "ids": [
        "SOFR", "TGCR", "BGCR", "EFFR", "IORB", "TBILL_3M", "CP_FIN_3M", "CP_NONFIN_3M", "ESTR", "SONIA", "CORRA"]},
    "fx": {"label": "FX (USD per unit, H.10 noon)", "kind": "price", "ids": [
        "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK", "DKK", "MXN", "BRL", "ZAR", "KRW", "INR",
        "CNY", "HKD", "SGD", "TWD", "THB"]},
    "energy": {"label": "Energy (EIA)", "kind": "price", "ids": [
        "WTI1", "WTI2", "WTI3", "WTI4", "NG1", "NG2", "NG3", "NG4", "RBOB1", "RBOB2", "HO1", "HO2", "WTI_SPOT",
        "BRENT_SPOT", "HH_SPOT", "NYH_GASOLINE_SPOT", "NYH_ULSD_SPOT"]},
    "monthly": {"label": "Monthly yields (euro area, US corporate HQM)", "kind": "yield", "freq": "M", "ids": [
        "DE_10Y_M", "FR_10Y_M", "IT_10Y_M", "ES_10Y_M", "PT_10Y_M", "GR_10Y_M", "NL_10Y_M", "BE_10Y_M", "AT_10Y_M",
        "IE_10Y_M", "HQM_2Y", "HQM_5Y", "HQM_10Y", "HQM_30Y", "GS2_M", "GS5_M", "GS10_M", "GS30_M"]},
}

ASSET_CLASS_OF_TYPE = {
    "etf_nav": ["etf"], "same_index_etf": ["etf"], "futures_basis": ["future", "equity_index"], "calendar": ["future", "commodity"],
    "cross_listing": ["equity"], "adr": ["equity", "fx"], "fx_rv": ["fx", "equity_index", "etf"],
    "commodity_curve": ["commodity", "future"], "index_rv": ["equity_index", "etf"], "sector_pairs": ["etf", "equity"],
    "bond_spreads": ["sovereign", "corporate"], "carry_funding": ["rates", "fx"], "cointegration": ["etf"],
}


def _days(idx: pd.DatetimeIndex) -> list[int]:
    return ((idx - EPOCH).days).astype(int).tolist()


def _arr(s, nd: int = 2):
    out = []
    for v in (s.tolist() if hasattr(s, "tolist") else s):
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            out.append(None)
        else:
            out.append(round(float(v), nd))
    return out


def _scalar_or_series(x, idx, nd=2):
    if isinstance(x, pd.Series):
        s = x.reindex(idx).ffill()
        if s.nunique(dropna=True) <= 1:
            v = s.dropna()
            return float(v.iloc[0]) if len(v) else 0.0
        return _arr(s, nd)
    return float(x) if np.isfinite(x) else None


def _finite(x):
    try:
        return float(x) if x is not None and np.isfinite(float(x)) else None
    except (TypeError, ValueError):
        return None


def relationship_file(sd: SpreadData, scan_row: dict | None) -> dict:
    u = load_universe()
    s = sd.spread.dropna()
    if sd.freq == "D":
        s = s[s.index >= DAILY_START]
    idx = s.index
    fair_mode = "rolling" if sd.fair is None else ("series" if isinstance(sd.fair, pd.Series) else "const")
    data = {
        "id": sd.rel.id, "d": _days(idx), "s": _arr(s, 3),
        "fair_mode": fair_mode, "fair_window": sd.meta.get("fair_window", 36 if sd.freq == "M" else 252),
        "fair_const": float(sd.fair) if fair_mode == "const" else None,
        "f": _arr(sd.fair.reindex(idx), 2) if fair_mode == "series" else None,
        "pnl": _arr(sd.pnl.reindex(idx), 2) if sd.pnl is not None else None,
        "cost": _scalar_or_series(sd.cost_rt, idx, 2), "carry": _scalar_or_series(sd.carry, idx, 4),
        "hold": _scalar_or_series(sd.hold_cost, idx, 4), "rebal": _scalar_or_series(sd.rebal_cost, idx, 4),
        "band_up": _finite(sd.at(sd.band_up)), "band_down": _finite(sd.at(sd.band_down)),
        "no_trade": _days(pd.DatetimeIndex([d for d in (sd.no_trade if sd.no_trade is not None else []) if d >= idx[0]])),
        "roll": _days(pd.DatetimeIndex([d for d in (sd.roll_days if sd.roll_days is not None else []) if d >= idx[0]])),
        "direction": sd.direction, "freq": sd.freq, "unit": sd.unit, "pnl_unit": sd.pnl_unit,
    }
    beta = sd.meta.get("beta")
    if isinstance(beta, pd.Series):
        data["beta"] = {"rolling": _arr(beta.reindex(idx), 4)}
        if sd.rel.params.get("builder") == "stat_pair":
            from ..scanners.builders_stat import stat_pair
            k = stat_pair(sd.rel, method="kalman")
            data["beta"]["kalman"] = _arr(k.meta["beta"].reindex(idx), 4)
            data["alt"] = {"kalman": {"s": _arr(k.spread.reindex(idx), 2), "pnl": _arr(k.pnl.reindex(idx), 2)}}
            r = stat_pair(sd.rel, method="ratio")
            data["alt"]["ratio"] = {"s": _arr(r.spread.reindex(idx), 2), "pnl": _arr(r.pnl.reindex(idx), 2)}
    if sd.rel.params.get("builder") == "futures_basis":
        data["implied_fin"] = _arr(sd.meta["implied_financing_bp"].reindex(idx), 1)
    if "full_carry_bp" in sd.meta:
        data["full_carry"] = _arr(sd.meta["full_carry_bp"].reindex(idx), 1)
    # leg-level P&L only where every leg is open data
    if sd.leg_pnl is not None and sd.license == "open":
        data["legpnl"] = {k: _arr(v.reindex(idx), 2) for k, v in sd.leg_pnl.items()}
    if sd.rel.params.get("builder") == "fx_carry":
        data["ctv"] = _arr(sd.meta["carry_to_vol"].reindex(idx), 5)
    if sd.pnl is not None:
        # Python engine on exactly the published window: the browser engine must reproduce it (self-check).
        trimmed = dataclasses.replace(sd, spread=s, pnl=sd.pnl.reindex(idx),
                                      meta={**sd.meta, **({"carry_to_vol": sd.meta["carry_to_vol"].reindex(idx)} if "carry_to_vol" in sd.meta else {})})
        if sd.fair is None or isinstance(sd.fair, pd.Series):
            pass
        m = bt.run(trimmed, bt.TradeParams()).metrics
        data["py_check"] = {"sharpe": m["sharpe"], "trades": m["trades"], "net_total": m["net_total"]}
    return data


def _dumps(obj) -> str:
    """Strict JSON for browsers: every NaN / inf becomes null (JSON has no Infinity), and anything left fails loudly."""
    return json.dumps(_clean(obj), separators=(",", ":"), allow_nan=False)


def _clean(v):
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    return str(v)


def catalog_entry(sd: SpreadData, scan_row: dict | None, bt_rows: dict) -> dict:
    u = load_universe()
    legs = []
    for leg in sd.rel.legs:
        inst = u.get(leg)
        if inst is None:
            continue
        legs.append({"id": leg, "name": inst.name, "symbol": str(inst.key) if inst.src == "yahoo" else leg,
                     "asset_class": inst.asset_class, "ccy": inst.price_ccy, "venue": inst.venue, "license": inst.license,
                     "source": inst.credit})
    classes = sorted({l["asset_class"] for l in legs})
    entry = {"id": sd.rel.id, "name": sd.rel.name, "type": sd.rel.type, "type_label": TYPES.get(sd.rel.type, sd.rel.type),
             "tier": sd.rel.tier, "legs": legs, "asset_classes": classes, "mechanism": sd.rel.mechanism,
             "notes": sd.notes, "license": sd.license, "unit": sd.unit, "pnl_unit": sd.pnl_unit, "freq": sd.freq,
             "tradable": sd.tradable, "sync_gap_h": round(sd.sync_gap_h, 1), "direction": sd.direction,
             "est_hedge": sd.est_hedge, "first": str(sd.spread.dropna().index[0].date()),
             "last": str(sd.spread.dropna().index[-1].date()),
             "fair_mode": "rolling" if sd.fair is None else ("series" if isinstance(sd.fair, pd.Series) else "const")}
    if scan_row:
        keep = ("date", "spread", "fair", "deviation", "z", "sigma", "percentile", "half_life", "ou_mu", "ou_mean_reverting",
                "adf_p", "bayes_p_mr", "bayes_p_hit", "bayes_p_stop", "regime_p_mr", "horizon", "expected_convergence",
                "position", "cost_rt", "carry_per_period", "hold_per_period", "carry_horizon", "band", "outside_band",
                "net_edge", "adv_min_usd", "capacity_usd", "borrow_ok", "borrow", "currencies", "fx_exposure",
                "basis_risk_sd", "loss_99_10d", "episodes", "reverted_share", "median_periods_to_revert", "mae_p95",
                "status", "reasons", "carry_to_vol")
        entry["scan"] = {k: _clean(scan_row.get(k)) for k in keep if k in scan_row}
    if bt_rows:
        entry["backtest"] = {v: {k: _finite(r.get(k)) for k in ("sharpe", "gross_sharpe", "net_per_year", "trades", "hit_rate",
                                                                "avg_net_per_trade", "max_drawdown", "years", "is_sharpe",
                                                                "oos_sharpe", "breakeven_cost_mult", "time_in_market")}
                             for v, r in bt_rows.items()}
    return entry


def open_group_file(group: str) -> dict:
    cfg = OPEN_GROUPS[group]
    u = load_universe()
    series = {}
    for iid in cfg["ids"]:
        inst = u[iid]
        assert inst.license != "vendor", iid
        if not store.has(iid):
            continue
        if inst.asset_class == "fx":
            from ..data import fx
            s = fx.usd_per(inst.ccy)
        else:
            s = store.series(iid)
        if cfg.get("freq") != "M":
            s = s[s.index >= DAILY_START]
        series[iid] = s
    allidx = sorted(set().union(*[set(s.index) for s in series.values()]))
    idx = pd.DatetimeIndex(allidx)
    out = {"group": group, "label": cfg["label"], "kind": cfg["kind"], "freq": cfg.get("freq", "D"), "d": _days(idx),
           "series": {}}
    for iid, s in series.items():
        inst = u[iid]
        nd = 6 if inst.asset_class == "fx" else 4 if cfg["kind"] == "yield" else 3
        out["series"][iid] = {"name": inst.name, "ccy": inst.price_ccy, "source": inst.credit, "license": inst.license,
                              "venue": inst.venue, "v": _arr(s.reindex(idx), nd)}
    return out


def assert_policy(out_dir: Path) -> dict:
    """Re-read every written file and check no vendor instrument's own series is present."""
    u = load_universe()
    vendor_ids = {i for i, inst in u.items() if inst.is_vendor}
    checked = 0
    for f in (out_dir / "rel").glob("*.json"):
        d = json.loads(f.read_text("utf-8"))
        if "legpnl" in d:
            assert not (set(d["legpnl"]) & vendor_ids), f"vendor leg P&L in {f.name}"
        checked += 1
    for f in (out_dir / "open").glob("*.json"):
        d = json.loads(f.read_text("utf-8"))
        assert not (set(d["series"]) & vendor_ids), f"vendor series in {f.name}"
        checked += 1
    return {"files_checked": checked}


def build(out_dir: Path | None = None) -> dict:
    out_dir = Path(out_dir or paths.DEMO_DATA)
    if (out_dir / "rel").exists():
        shutil.rmtree(out_dir / "rel")
    (out_dir / "rel").mkdir(parents=True, exist_ok=True)
    (out_dir / "open").mkdir(parents=True, exist_ok=True)
    res = paths.RESULTS
    scan = {r["id"]: r for r in json.loads((res / "scan_latest.json").read_text("utf-8"))} if (res / "scan_latest.json").exists() else {}
    bts = {}
    if (res / "backtests.csv").exists():
        df = pd.read_csv(res / "backtests.csv")
        for _, r in df.iterrows():
            bts.setdefault(r["id"], {})[r["variant"]] = r.to_dict()
    entries = []
    for rel in catalog.relationships():
        sd = catalog.build(rel)
        if sd is None:
            continue
        row = scan.get(rel.id)
        entries.append(catalog_entry(sd, row, bts.get(rel.id, {})))
        (out_dir / "rel" / f"{rel.id}.json").write_text(_dumps(relationship_file(sd, row)), "utf-8")
    for g in OPEN_GROUPS:
        (out_dir / "open" / f"{g}.json").write_text(_dumps(open_group_file(g)), "utf-8")
    summaries = {}
    for name in ("e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9"):
        p = res / f"{name}_summary.json"
        if p.exists():
            summaries[name] = json.loads(p.read_text("utf-8"))
    meta = {"asof": str(store.series("SPY").index[-1].date()), "tiers": TIERS, "types": TYPES,
            "asset_classes": ASSET_CLASS_LABELS, "asset_class_of_type": ASSET_CLASS_OF_TYPE,
            "default_params": bt.TradeParams().to_dict(),
            "open_groups": {g: {"label": c["label"], "kind": c["kind"], "freq": c.get("freq", "D")} for g, c in OPEN_GROUPS.items()},
            "policy": "Row-level series only from public-domain / open sources; vendor-based relationships as derived analytics.",
            "research": summaries}
    (out_dir / "catalog.json").write_text(_dumps({"meta": meta, "relationships": entries}), "utf-8")
    check = assert_policy(out_dir)
    sizes = sum(f.stat().st_size for f in out_dir.rglob("*.json"))
    return {"relationships": len(entries), "bytes": sizes, **check}
