"""Fetch every instrument in the universe, normalise units, run data checks and write the processed store."""
from __future__ import annotations

import datetime as dt
import logging
import time
import traceback

import numpy as np
import pandas as pd

from . import sources, store
from .cache import Fetcher
from .universe import Instrument, curve_snapshot_config, load_universe

log = logging.getLogger("rvscan.data")

MONTH_CODES = "FGHJKMNQUVXZ"


class TableCache:
    """Sources that return whole tables (curves) are fetched once per run."""

    def __init__(self, f: Fetcher):
        self.f = f
        self._t: dict = {}

    def get(self, name, loader):
        if name not in self._t:
            try:
                self._t[name] = loader()
            except Exception as exc:  # noqa: BLE001 - remember the failure so each instrument doesn't retry
                self._t[name] = exc
        if isinstance(self._t[name], Exception):
            raise self._t[name]
        return self._t[name]


def _fred_like(inst: Instrument, f: Fetcher) -> pd.DataFrame:
    df = sources.fred(f, inst.key, inst.license)
    if inst.unit == "ccy_per_usd":
        df["value"] = 1.0 / df["value"]
    return df


def fetch_instrument(inst: Instrument, f: Fetcher, tables: TableCache) -> pd.DataFrame:
    src = inst.src
    if src == "yahoo":
        bars, meta, ev = sources.yahoo_chart(f, inst.key)
        if bars.empty:
            return bars
        if meta.get("currency") and meta["currency"] != inst.ccy:
            log.warning("%s: Yahoo currency %s != configured %s", inst.id, meta["currency"], inst.ccy)
        if inst.ccy == "GBp":
            for c in ("open", "high", "low", "close", "adjclose"):
                bars[c] = bars[c] / 100.0
        if not ev.empty:
            if inst.ccy == "GBp":
                ev.loc[ev["kind"] == "div", "value"] /= 100.0
            store.save_events(inst.id, ev)
        return bars
    if src.startswith("fred"):
        return _fred_like(inst, f)
    if src == "treasury":
        table = tables.get(("treasury", inst.table), lambda: sources.treasury_curve(f, inst.table))
        return table[[float(inst.key)]].rename(columns={float(inst.key): "value"}).dropna()
    if src == "eia":
        return sources.eia_series(f, inst.key, inst.area or "pet")
    if src == "ecb":
        df = sources.ecb_series(f, inst.key)
        return df
    if src == "bundesbank":
        return sources.bundesbank_series(f, inst.key)
    if src == "mof":
        table = tables.get("mof", lambda: sources.mof_jgb(f))
        return table[[float(inst.key)]].rename(columns={float(inst.key): "value"}).dropna()
    if src == "boc":
        if inst.asset_class == "sovereign":
            keys = [i.key for i in load_universe().values() if i.src == "boc" and i.asset_class == "sovereign"]
            table = tables.get("boc_bonds", lambda: sources.boc_valet(f, keys, "bond_yields", start="2000-01-01"))
        else:
            table = tables.get(("boc", inst.key), lambda: sources.boc_valet(f, [inst.key], inst.key.replace(".", "_"), start="1997-01-01"))
        return table[[inst.key]].rename(columns={inst.key: "value"}).dropna()
    if src == "rba":
        table = tables.get("rba", lambda: sources.rba_f2(f))
        return table[[inst.key]].rename(columns={inst.key: "value"}).dropna()
    if src == "boe":
        keys = [i.key for i in load_universe().values() if i.src == "boe"]
        table = tables.get("boe", lambda: sources.boe_series(f, keys, "gilt_zero_coupon"))
        return table[[inst.key]].rename(columns={inst.key: "value"}).dropna()
    if src == "nyfed":
        return sources.nyfed_rate(f, inst.key)
    raise ValueError(f"no adapter for source {src}")


def quality(inst: Instrument, df: pd.DataFrame) -> dict:
    """Simple data checks recorded in the catalog (and surfaced in the report)."""
    col = "close" if "close" in df else "value"
    s = df[col].dropna()
    out = {"first": s.index[0].date().isoformat(), "last": s.index[-1].date().isoformat(), "obs": int(len(s))}
    if len(s) < 3:
        return out
    if inst.asset_class in ("sovereign", "corporate", "rates"):
        d = s.diff().abs()
        out["max_abs_change"] = round(float(d.max()), 4)
        out["max_abs_change_date"] = d.idxmax().date().isoformat()
    else:
        pos = s[s > 0]
        r = np.log(pos).diff().abs()
        out["max_abs_logret"] = round(float(r.max()), 4)
        out["max_abs_logret_date"] = r.idxmax().date().isoformat()
    same = (s.diff() == 0).astype(int)
    runs = same.groupby((same == 0).cumsum()).sum()
    out["longest_flat_run"] = int(runs.max())
    if inst.freq == "D":
        bdays = pd.bdate_range(s.index[0], s.index[-1])
        out["coverage_bdays"] = round(len(s.index.intersection(bdays)) / max(len(bdays), 1), 3)
    if "volume" in df:
        out["zero_volume_share"] = round(float((df["volume"].fillna(0) <= 0).mean()), 3)
    return out


def snapshot_contracts(f: Fetcher) -> dict:
    """Fetch live futures contracts for the configured roots (current term structure)."""
    today = dt.date.today()
    summary = {}
    for root, cfg in curve_snapshot_config().items():
        rows = []
        months = cfg["months"]
        cands = []
        y, m = today.year, today.month
        for k in range(months * (3 if cfg.get("quarterly") else 2 if cfg.get("bimonthly") else 1) + 3):
            mm = (m - 1 + k) % 12 + 1
            yy = y + (m - 1 + k) // 12
            if cfg.get("quarterly") and mm not in (3, 6, 9, 12):
                continue
            if cfg.get("bimonthly") and mm not in (2, 4, 6, 8, 10, 12):
                continue
            cands.append((yy, mm))
        for yy, mm in cands:
            sym = f"{root}{MONTH_CODES[mm - 1]}{yy % 100:02d}.{cfg['exchange']}"
            try:
                bars, meta, _ = sources.yahoo_chart(f, sym)
            except Exception as exc:  # noqa: BLE001
                log.info("snapshot %s failed: %s", sym, exc)
                continue
            if bars.empty:
                continue
            b = bars[["close", "volume"]].copy()
            b["contract"] = sym
            b["year"], b["month"] = yy, mm
            rows.append(b.reset_index())
        if rows:
            df = pd.concat(rows, ignore_index=True)
            last = df["date"].max()
            live = df.groupby("contract")["date"].max()
            keep = live[live >= last - pd.Timedelta(days=4)].index
            df = df[df["contract"].isin(keep)].sort_values(["year", "month", "date"])
            store.save_curve(root, df)
            summary[root] = {"contracts": int(df["contract"].nunique()), "last": str(last.date())}
    return summary


def run(refresh: bool = False, only: list[str] | None = None, skip_snapshots: bool = False) -> dict:
    universe = load_universe()
    f = Fetcher(refresh=refresh)
    tables = TableCache(f)
    catalog = store.read_catalog()
    failures = {}
    t0 = time.time()
    for n, (iid, inst) in enumerate(universe.items(), 1):
        if only and iid not in only and inst.src not in only and inst.asset_class not in only:
            continue
        try:
            df = fetch_instrument(inst, f, tables)
            if df is None or df.empty:
                failures[iid] = "empty"
                log.warning("%s: no data", iid)
                continue
            store.save_instrument(iid, df)
            entry = {"asset_class": inst.asset_class, "src": inst.src, "key": str(inst.key), "license": inst.license,
                     "ccy": inst.price_ccy, "venue": inst.venue, "name": inst.name, **quality(inst, df)}
            if inst.nav:
                nav = sources.ssga_nav(f, inst.nav)
                store.save_nav(iid, nav)
                entry["nav_first"] = nav.index[0].date().isoformat()
                entry["nav_obs"] = int(len(nav))
            catalog[iid] = entry
        except Exception as exc:  # noqa: BLE001
            failures[iid] = f"{type(exc).__name__}: {exc}"
            log.error("%s failed: %s\n%s", iid, exc, traceback.format_exc(limit=2))
        if n % 25 == 0:
            log.info("%d/%d instruments (%.0fs)", n, len(universe), time.time() - t0)
            store.write_catalog(catalog)
    if not skip_snapshots:
        catalog["_curve_snapshots"] = snapshot_contracts(f)
    catalog["_built_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    catalog["_failures"] = failures
    store.write_catalog(catalog)
    return {"failures": failures, "seconds": round(time.time() - t0, 1)}
