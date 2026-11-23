"""FX conversion helpers. All FX is expressed as USD per one unit of currency.

Primary source: Fed H.10 noon buying rates (public domain). H.10 is published weekly, so its latest days are
missing; those days are filled with ECB reference rates (14:15 CET) converted through EUR/USD, and only those
days (`ecb_extension_dates` lists them). The Taiwan dollar has no ECB rate, so TWD conversions stop at the last
H.10 date.

Timing caveat that matters for every cross-currency comparison: H.10 rates are New York noon rates, so a London
close (11:30-12:30 ET) is roughly synchronous with them but Tokyo, Taipei or Hong Kong closes are 13-16 hours
earlier and a New York close 4 hours later. Scanners report this gap instead of hiding it.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import store
from .universe import load_universe

FX_IDS = {inst.ccy: iid for iid, inst in load_universe().items()
          if inst.asset_class == "fx" and inst.src.startswith("fred")}
ECB_IDS = {inst.ccy: iid for iid, inst in load_universe().items()
           if inst.asset_class == "fx" and inst.src == "ecb"}


@lru_cache(maxsize=64)
def _ecb_usd_per(ccy: str) -> pd.Series:
    """USD per unit of ccy implied by ECB reference rates (EUR crosses through EUR/USD)."""
    if not store.has("ECB_EURUSD"):
        return pd.Series(dtype=float)
    usd_per_eur = store.series("ECB_EURUSD")
    if ccy == "EUR":
        return usd_per_eur
    iid = ECB_IDS.get(ccy)
    if iid is None or not store.has(iid):
        return pd.Series(dtype=float)
    ccy_per_eur = store.series(iid)
    return (usd_per_eur / ccy_per_eur).dropna()


@lru_cache(maxsize=64)
def usd_per(ccy: str) -> pd.Series:
    """USD value of one unit of `ccy` (daily): H.10 noon, extended with ECB-implied rates after its last date."""
    if ccy == "USD":
        raise ValueError("USD needs no conversion")
    if ccy == "GBp":
        return usd_per("GBP") / 100.0
    iid = FX_IDS.get(ccy)
    if iid is None:
        raise KeyError(f"no FX series for {ccy}")
    h10 = store.series(iid)
    ecb = _ecb_usd_per(ccy)
    tail = ecb[ecb.index > h10.index[-1]] if not ecb.empty else ecb
    return pd.concat([h10, tail]).sort_index() if len(tail) else h10


def ecb_extension_dates(ccy: str) -> pd.DatetimeIndex:
    if ccy in ("USD",):
        return pd.DatetimeIndex([])
    h10 = store.series(FX_IDS["GBP" if ccy == "GBp" else ccy])
    s = usd_per(ccy)
    return s.index[s.index > h10.index[-1]]


def to_usd(local: pd.Series, ccy: str, max_ffill: int = 3) -> pd.Series:
    """Convert a local-currency series to USD with the same-date rate (carried forward up to max_ffill days)."""
    if ccy == "USD":
        return local.astype(float)
    fxs = usd_per(ccy)
    fxs = fxs.reindex(fxs.index.union(local.index)).ffill(limit=max_ffill).reindex(local.index)
    return local.astype(float) * fxs


def fx_return(ccy: str) -> pd.Series:
    """Daily log return of holding one unit of `ccy` against USD."""
    return np.log(usd_per(ccy)).diff().dropna()
