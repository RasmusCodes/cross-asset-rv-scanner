"""Processed-data store: one parquet file per instrument plus a catalog.

Layout under data/processed/:
  instruments/<id>.parquet   daily (or monthly) series; Yahoo: open/high/low/close/adjclose/volume, others: value
  nav/<id>.parquet           issuer NAV history (nav, shares, tna) for ETFs with `nav:` in the universe
  events/<id>.parquet        dividends and splits (Yahoo)
  curves/<root>.parquet      live futures contracts for the current term-structure snapshot (long format)
  catalog.json               per-instrument metadata: first/last date, observations, licence, checks
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd

from .. import paths
from .universe import Instrument, load_universe

INSTR = paths.PROCESSED / "instruments"
NAV = paths.PROCESSED / "nav"
EVENTS = paths.PROCESSED / "events"
CURVES = paths.PROCESSED / "curves"
CATALOG = paths.PROCESSED / "catalog.json"


def _atomic_parquet(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp)
    os.replace(tmp, path)


def save_instrument(iid: str, df: pd.DataFrame) -> None:
    _atomic_parquet(df, INSTR / f"{iid}.parquet")


def save_nav(iid: str, df: pd.DataFrame) -> None:
    _atomic_parquet(df, NAV / f"{iid}.parquet")


def save_events(iid: str, df: pd.DataFrame) -> None:
    _atomic_parquet(df.reset_index(drop=True), EVENTS / f"{iid}.parquet")


def save_curve(root: str, df: pd.DataFrame) -> None:
    _atomic_parquet(df.reset_index(drop=True), CURVES / f"{root}.parquet")


def write_catalog(cat: dict) -> None:
    CATALOG.parent.mkdir(parents=True, exist_ok=True)
    tmp = CATALOG.with_suffix(".tmp")
    tmp.write_text(json.dumps(cat, indent=1, default=str), "utf-8")
    os.replace(tmp, CATALOG)


def read_catalog() -> dict:
    return json.loads(CATALOG.read_text("utf-8")) if CATALOG.exists() else {}


@lru_cache(maxsize=512)
def load_instrument(iid: str) -> pd.DataFrame:
    path = INSTR / f"{iid}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{iid}: no processed data (run scripts/fetch_data.py)")
    return pd.read_parquet(path)


def has(iid: str) -> bool:
    return (INSTR / f"{iid}.parquet").exists()


def price_field(inst: Instrument, total_return: bool = False) -> str:
    if inst.src == "yahoo":
        return "adjclose" if total_return else "close"
    return "value"


def series(iid: str, total_return: bool = False) -> pd.Series:
    """Main price/level series for an instrument (in its normalised unit)."""
    inst = load_universe()[iid]
    df = load_instrument(iid)
    s = df[price_field(inst, total_return)].astype(float).dropna()
    s.name = iid
    return s


def panel(ids: list[str], total_return: bool = False, how: str = "inner") -> pd.DataFrame:
    frames = [series(i, total_return) for i in ids]
    return pd.concat(frames, axis=1, join=how).sort_index()


@lru_cache(maxsize=128)
def load_nav(iid: str) -> pd.DataFrame:
    return pd.read_parquet(NAV / f"{iid}.parquet")


@lru_cache(maxsize=256)
def load_events(iid: str) -> pd.DataFrame:
    path = EVENTS / f"{iid}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["date", "kind", "value"])


def load_curve(root: str) -> pd.DataFrame:
    path = CURVES / f"{root}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def volume_usd(iid: str, window: int = 63) -> pd.Series:
    """Rolling median daily traded value in USD (Yahoo volume x close x FX); NaN where unknown."""
    from .fx import to_usd  # local import to avoid a cycle
    inst = load_universe()[iid]
    df = load_instrument(iid)
    if "volume" not in df:
        return pd.Series(np.nan, index=df.index)
    value_local = (df["close"] * df["volume"]).where(df["volume"] > 0)
    value_usd = to_usd(value_local, inst.price_ccy)
    return value_usd.rolling(window, min_periods=max(5, window // 3)).median()
