"""Instrument universe: typed view of config/universe.yaml."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from functools import lru_cache
from zoneinfo import ZoneInfo

import yaml

from .. import paths

ASSET_CLASSES = ("equity_index", "future", "commodity", "etf", "equity", "fx", "sovereign", "corporate", "rates")
ASSET_CLASS_LABELS = {
    "equity_index": "Equity indices", "future": "Futures", "commodity": "Commodities (spot)", "etf": "ETFs",
    "equity": "Equities, ADRs and dual listings", "fx": "FX", "sovereign": "Sovereign bonds",
    "corporate": "Corporate bonds", "rates": "Interest-rate and funding instruments",
}


@dataclass(frozen=True)
class Instrument:
    id: str
    asset_class: str
    src: str
    key: object
    ccy: str
    venue: str
    liq: str
    name: str
    license: str
    credit: str
    unit: str | None = None
    borrow: str = "gc"
    nav: str | None = None
    table: str | None = None
    area: str | None = None
    mat: float | None = None
    freq: str = "D"
    mult: float | None = None
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def is_vendor(self) -> bool:
        return self.license == "vendor"

    @property
    def tradable(self) -> bool:
        return self.liq != "index"

    @property
    def price_ccy(self) -> str:
        """Currency after unit normalisation (Yahoo quotes London in pence)."""
        return "GBP" if self.ccy == "GBp" else self.ccy


@dataclass(frozen=True)
class Venue:
    code: str
    tz: str
    close: str

    def close_utc_hours(self, on: dt.date) -> float:
        """Close time on `on` expressed in hours UTC (can be negative / > 24 across midnight)."""
        h, m = map(int, self.close.split(":"))
        local = dt.datetime(on.year, on.month, on.day, h, m, tzinfo=ZoneInfo(self.tz))
        utc = local.astimezone(dt.timezone.utc)
        day_shift = (utc.date() - on).days
        return utc.hour + utc.minute / 60 + 24 * day_shift


@lru_cache(maxsize=1)
def _raw() -> dict:
    return yaml.safe_load((paths.CONFIG / "universe.yaml").read_text("utf-8"))


@lru_cache(maxsize=1)
def venues() -> dict[str, Venue]:
    return {code: Venue(code, v["tz"], v["close"]) for code, v in _raw()["venues"].items()}


@lru_cache(maxsize=1)
def load_universe() -> dict[str, Instrument]:
    raw = _raw()
    sources = raw["sources"]
    known_fields = {"id", "src", "key", "ccy", "venue", "liq", "name", "unit", "borrow", "nav", "table", "area", "mat",
                    "freq", "mult"}
    out: dict[str, Instrument] = {}
    for asset_class, items in raw["instruments"].items():
        if asset_class not in ASSET_CLASSES:
            raise ValueError(f"unknown asset class {asset_class}")
        for item in items:
            iid = item["id"]
            if iid in out:
                raise ValueError(f"duplicate instrument id {iid}")
            src = item["src"]
            if src not in sources:
                raise ValueError(f"{iid}: unknown source {src}")
            if item["venue"] not in raw["venues"]:
                raise ValueError(f"{iid}: unknown venue {item['venue']}")
            extra = {k: v for k, v in item.items() if k not in known_fields}
            out[iid] = Instrument(
                id=iid, asset_class=asset_class, src=src, key=item["key"], ccy=item["ccy"], venue=item["venue"],
                liq=item["liq"], name=item["name"], license=sources[src]["license"], credit=sources[src]["credit"],
                unit=item.get("unit"), borrow=item.get("borrow", "gc"), nav=item.get("nav"), table=item.get("table"),
                area=item.get("area"), mat=item.get("mat"), freq=item.get("freq", "D"), mult=item.get("mult"), extra=extra)
    return out


def curve_snapshot_config() -> dict:
    return _raw().get("curve_snapshots", {})


def sync_gap_hours(a: Instrument, b: Instrument, on: dt.date | None = None) -> float:
    """Absolute gap between the two legs' closing prints, in hours (max over a winter and a summer date)."""
    vs = venues()
    if a.freq == "M" or b.freq == "M":
        return float("nan")
    dates = [on] if on else [dt.date(2026, 1, 15), dt.date(2026, 7, 15)]
    gaps = [abs(vs[a.venue].close_utc_hours(d) - vs[b.venue].close_utc_hours(d)) for d in dates]
    return max(gaps)
