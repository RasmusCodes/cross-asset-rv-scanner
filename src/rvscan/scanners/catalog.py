"""Expand config/relationships.yaml into Relationship objects and build their spreads."""
from __future__ import annotations

import logging
from functools import lru_cache

import yaml

from .. import paths
from ..data import store
from ..data.universe import load_universe
from . import builders_arb as A
from . import builders_macro as M
from . import builders_stat as S
from .base import Relationship, SpreadData

log = logging.getLogger("rvscan.catalog")

MECHANISMS = {
    "etf_nav": "Authorised participants create units by delivering the basket (or redeem units for it), so the ETF's "
               "price is held near the basket's value net of the creation fee and the basket's trading costs.",
    "same_index_etf": "Each fund is arbitraged to the same basket, so their prices move together; nothing converts one "
                      "fund into the other.",
    "futures_basis": "Cash-and-carry: buy the basket and sell the future (or the reverse) and hold to expiry, when the "
                     "future must equal the index. Fair basis = (financing rate - dividend yield) x time to expiry.",
    "calendar": "Two delivery months of the same commodity. Contango is capped by the cost of storage plus financing "
                "(for those with storage); backwardation has no arbitrage bound (convenience yield).",
    "adr": "A depositary receipt is a claim on a fixed number of ordinary shares; where the depositary creates and "
           "cancels receipts freely, the ratio-adjusted prices are held together net of fees, taxes and FX costs.",
    "interlisted": "One share listed in two countries and settled across the border: the same claim in two currencies.",
    "share_class": "Two share classes of one company with identical economic rights; they differ in votes and liquidity.",
    "dual_listed": "Two legal entities run as one company under an equalisation agreement: identical dividends, no "
                   "conversion between the shares.",
    "fx_triangular": "Triangular no-arbitrage: EUR/JPY must equal EUR/USD x USD/JPY for synchronous executable quotes.",
    "fx_adjusted": "A US-listed country fund converted to local currency should track its home index, up to "
                   "differences in index definition, dividends and the hours between the two closes.",
    "commodity": "Physical processing or transport links the prices (refining margins, pipeline and export capacity) "
                 "but no contract forces convergence.",
    "stat": "No convergence mechanism: the two series have co-moved in the past.",
    "bond": "Yield spreads between bonds that share risk factors; no arbitrage ties them, carry and risk premia drive them.",
    "etf_credit": "Credit fund vs a duration-matched Treasury fund: the spread is the credit risk premium.",
    "funding": "Money-market rates linked by bank arbitrage (reserves vs repo vs bills) within balance-sheet limits.",
    "fx_carry": "Currency carry: earn the interest-rate differential, bear the exchange-rate risk. A risk premium, "
                "not an arbitrage (covered interest parity removes the riskless version).",
}


def _cfg() -> dict:
    return yaml.safe_load((paths.CONFIG / "relationships.yaml").read_text("utf-8"))


def _name(i: str) -> str:
    inst = load_universe().get(i)
    if inst is None:
        return i
    return inst.extra.get("short", inst.key if inst.src == "yahoo" else i) if inst.src == "yahoo" else i


@lru_cache(maxsize=1)
def relationships() -> list[Relationship]:
    c = _cfg()
    out: list[Relationship] = []

    def add(rid, typ, tier, legs, name, mech_key, params=None, constraints=None, builder=None):
        out.append(Relationship(rid, typ, tier, legs, name, MECHANISMS[mech_key], dict(params or {}, builder=builder),
                                list(constraints or [])))

    for etf, kind, note in c["etf_nav"]:
        add(f"NAV_{etf}", "etf_nav", "arbitrage", [etf], f"{etf} vs NAV", "etf_nav", {"kind": kind}, [note], "etf_nav")
    for a, b, note in c["same_index_etf"]:
        add(f"SAME_{a}_{b}", "same_index_etf", "quasi", [a, b], f"{a} vs {b}", "same_index_etf", {}, [note], "same_index_etf")
    for fut, ind, etf, _mult in c["futures_basis"]:
        add(f"BASIS_{fut}", "futures_basis", "arbitrage", [fut, ind, etf], f"{fut} basis vs {ind}", "futures_basis", {},
            [], "futures_basis")
    for front, back, root in c["calendar"]:
        add(f"CAL_{front}_{back}", "calendar", "quasi", [front, back], f"{front}/{back} calendar spread", "calendar",
            {"root": root}, [], "calendar")
    for us, home, ratio, tier, note in c["adr"]:
        add(f"ADR_{us}", "adr", tier, [us, home], f"{us} vs {_name(home)} (x{ratio:g})", "adr", {"ratio": ratio}, [note], "adr")
    for us, to in c["interlisted"]:
        add(f"XL_{us}", "cross_listing", "arbitrage", [us, to], f"{us} (NYSE) vs {_name(to)} (TSX)", "interlisted", {},
            [], "interlisted")
    for a, b, ratio, tier, note in c["share_class"]:
        add(f"CLASS_{a}_{b}", "cross_listing", tier, [a, b], f"{_name(a)} vs {_name(b)}" + (f" (x{ratio:g})" if ratio != 1 else ""),
            "share_class", {"ratio": ratio}, [note], "share_class")
    for a, b, ratio, note in c["dual_listed"]:
        add(f"DLC_{a}_{b}", "cross_listing", "quasi", [a, b], f"{_name(a)} vs {_name(b)}", "dual_listed",
            {"ratio": ratio}, [note], "dual_listed")
    for cross, base, quote in c["fx_triangular"]:
        add(f"TRI_{base}{quote}", "fx_rv", "arbitrage", [cross, base, quote], f"ECB {base}/{quote} vs H.10 implied cross",
            "fx_triangular", {}, [], "fx_triangular")
    for etf, ind, ccy, note in c["fx_adjusted"]:
        tier = "quasi" if etf == "FEZ" else "statistical"
        add(f"FXA_{etf}_{ind}", "fx_rv", tier, [etf, ind], f"{etf} vs {ind} in USD", "fx_adjusted", {"ccy": ccy}, [note],
            "fx_adjusted")
    for rid, kind, legs, note in c["commodity_spreads"]:
        add(rid, "commodity_curve", "quasi", legs, note.split(":")[0], "commodity", {"kind": kind}, [note], "commodity_spread")
    for a, b in c["index_rv"]:
        add(f"IDX_{a}_{b}", "index_rv", "statistical", [a, b], f"{a} vs {b}", "stat", {}, [], "stat_pair")
    for a, b, note in c["sector_pairs"]:
        add(f"PAIR_{a}_{b}", "sector_pairs", "statistical", [a, b], f"{a} vs {b}", "stat", {}, [note], "stat_pair")
    for rid, kind, legs, weights, tier, note in c["bond_spreads"]:
        add(rid, "bond_spreads", tier, legs, note.split(" (")[0].rstrip("."), "bond", {"kind": kind, "weights": weights},
            [note], "bond_spread")
    for a, b, note in c["etf_credit"]:
        add(f"CREDIT_{a}_{b}", "bond_spreads", "statistical", [a, b], f"{a} vs {b} (duration-hedged)", "etf_credit", {},
            [note], "etf_credit")
    for rid, legs, note in c["funding"]:
        tier = "quasi" if "IORB" in legs else "statistical"
        add(rid, "carry_funding", tier, legs, note.split(":")[0], "funding", {}, [note], "funding")
    for ccy in c["fx_carry"]["currencies"] + c["fx_carry"]["em_currencies"]:
        add(f"CARRY_{ccy}", "carry_funding", "statistical", [ccy], f"Long {ccy} vs USD carry", "fx_carry",
            {"em": ccy in c["fx_carry"]["em_currencies"]}, [], "fx_carry")
    return out


BUILDERS = {
    "etf_nav": A.etf_nav, "same_index_etf": A.same_index_etf, "futures_basis": A.futures_basis, "adr": A.adr,
    "interlisted": A.interlisted, "share_class": A.share_class, "dual_listed": A.dual_listed,
    "fx_triangular": A.fx_triangular, "fx_adjusted": A.fx_adjusted, "calendar": M.calendar,
    "commodity_spread": M.commodity_spread, "bond_spread": M.bond_spread, "etf_credit": M.etf_credit,
    "funding": M.funding, "fx_carry": M.fx_carry, "stat_pair": S.stat_pair,
}


def by_id() -> dict[str, Relationship]:
    return {r.id: r for r in relationships()}


def available(rel: Relationship) -> bool:
    u = load_universe()
    for leg in rel.legs:
        if leg in u and not store.has(leg):
            return False
        if leg in u and rel.params.get("builder") == "etf_nav" and u[leg].nav and not (store.NAV / f"{leg}.parquet").exists():
            return False
    return True


def build(rel: Relationship, **kw) -> SpreadData | None:
    if not available(rel):
        log.warning("%s: missing data for %s", rel.id, rel.legs)
        return None
    fn = BUILDERS[rel.params["builder"]]
    try:
        sd = fn(rel, **kw) if kw else fn(rel)
    except Exception as exc:  # noqa: BLE001
        log.error("%s: build failed: %s", rel.id, exc)
        raise
    if sd is not None and len(sd.spread.dropna()) < 60:
        log.warning("%s: fewer than 60 observations", rel.id)
    return sd
