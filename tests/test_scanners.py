"""Scanner-layer tests: configuration integrity, the cointegration screen, the demo licence policy, and (when the
processed data exist locally) end-to-end builds. Data-dependent tests are skipped on a fresh clone."""
import json

import numpy as np
import pandas as pd
import pytest

from rvscan.data import store
from rvscan.data.universe import load_universe, sync_gap_hours
from rvscan.engine import backtest as bt
from rvscan.scanners import catalog, stat_scan
from rvscan.scanners.base import TIERS, TYPES

HAVE_DATA = store.CATALOG.exists() and store.has("SPY")


def test_universe_and_relationships_are_consistent():
    u = load_universe()
    rels = catalog.relationships()
    ids = [r.id for r in rels]
    assert len(ids) == len(set(ids)), "duplicate relationship ids"
    for r in rels:
        assert r.tier in TIERS and r.type in TYPES
        assert r.params["builder"] in catalog.BUILDERS
        for leg in r.legs:
            assert leg in u or leg in {"EUR", "JPY", "GBP", "CHF"}, f"{r.id}: unknown leg {leg}"
        if r.type == "adr":
            assert r.params["ratio"] > 0
    # every statistical relationship is labelled statistical (never arbitrage)
    assert all(r.tier == "statistical" for r in rels if r.params["builder"] == "stat_pair")


def test_closing_time_gaps():
    u = load_universe()
    assert sync_gap_hours(u["RY"], u["RY_TO"]) == 0.0           # Toronto and New York close together
    assert sync_gap_hours(u["TSM"], u["2330_TW"]) > 13           # Taipei closes ~15 hours earlier
    assert 3.5 <= sync_gap_hours(u["SHEL"], u["SHEL_L"]) <= 5.5  # London 4.5-5.5 h earlier (DST)


def test_cointegration_screen_finds_planted_pair_and_rejects_walks():
    rng = np.random.default_rng(0)
    n = 900
    idx = pd.bdate_range("2018-01-01", periods=n)
    common = np.cumsum(rng.standard_normal(n)) * 0.01
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.9 * noise[t - 1] + 0.004 * rng.standard_normal()
    a = pd.Series(common + noise, index=idx)
    b = pd.Series(common, index=idx)
    walk = pd.Series(np.cumsum(rng.standard_normal(n)) * 0.01, index=idx)
    planted = stat_scan.test_pair(a, b, "A", "B", "test")
    indep = stat_scan.test_pair(a, walk, "A", "W", "test")
    assert planted.p < 0.01 and planted.johansen_rank >= 1
    assert indep.p > 0.05
    assert 3 < planted.half_life < 15


def test_policy_check_rejects_vendor_series(tmp_path):
    from rvscan.demo import export
    (tmp_path / "rel").mkdir()
    (tmp_path / "open").mkdir()
    (tmp_path / "rel" / "ok.json").write_text(json.dumps({"legpnl": {"UST_10Y": [1, 2]}}))
    (tmp_path / "open" / "ok.json").write_text(json.dumps({"series": {"UST_10Y": {}}}))
    assert export.assert_policy(tmp_path)["files_checked"] == 2
    (tmp_path / "open" / "bad.json").write_text(json.dumps({"series": {"SPY": {}}}))
    with pytest.raises(AssertionError):
        export.assert_policy(tmp_path)


@pytest.mark.skipif(not HAVE_DATA, reason="processed data not present (run scripts/fetch_data.py)")
def test_every_relationship_builds_and_backtests():
    for r in catalog.relationships():
        sd = catalog.build(r)
        assert sd is not None, r.id
        s = sd.spread.dropna()
        assert len(s) >= 60, r.id
        assert np.isfinite(s).all(), r.id
        if sd.tradable:
            res = bt.run(sd)
            d = res.daily
            assert (d["gross"] + d["carry"] + d["hold"] + d["costs"] - d["net"]).abs().max() < 1e-6, r.id


@pytest.mark.skipif(not HAVE_DATA, reason="processed data not present (run scripts/fetch_data.py)")
def test_adr_ratios_are_plausible():
    # Freely convertible ADRs should trade within 15% of parity on median over the last year.
    for r in catalog.relationships():
        if r.type == "adr" and r.tier == "arbitrage":
            sd = catalog.build(r)
            assert abs(sd.meta["median_premium_1y_bp"]) < 1500, (r.id, sd.meta["median_premium_1y_bp"])
