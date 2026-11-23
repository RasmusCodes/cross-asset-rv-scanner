"""Engine tests on synthetic spreads: timing (no look-ahead), costs, bands, direction limits, attribution, carry rule,
futures calendars and bond arithmetic. No market data needed."""
import numpy as np
import pandas as pd
import pytest

from rvscan.data import bonds, calendars
from rvscan.engine import backtest as bt
from rvscan.scanners.base import Relationship, SpreadData


def make_sd(spread, pnl=None, cost=2.0, band=np.inf, fair=0.0, direction="both", tier="arbitrage", carry=0.0,
            hold=0.0, builder="test", meta=None, no_trade=None, roll_days=None):
    idx = spread.index
    rel = Relationship("T", "adr", tier, ["A", "B"], "test", "mech", params={"builder": builder})
    return SpreadData(rel, spread, "bp", fair, band, band, spread.diff() if pnl is None else pnl, carry, hold, cost, 0.0,
                      "open", direction=direction, meta=meta or {}, no_trade=no_trade, roll_days=roll_days)


def ou_series(n=3000, phi=0.9, sd=10.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + sd * rng.standard_normal()
    return pd.Series(x, index=pd.bdate_range("2010-01-01", periods=n))


def test_next_close_execution_has_no_lookahead():
    s = ou_series()
    sd = make_sd(s, cost=0.0)
    res1 = bt.run(sd, bt.TradeParams(lag=1))
    # Perturb the spread AFTER day k: positions held through day k must not change.
    k = 1500
    s2 = s.copy()
    s2.iloc[k + 1:] += np.random.default_rng(1).standard_normal(len(s2) - k - 1) * 50
    res2 = bt.run(make_sd(s2, cost=0.0), bt.TradeParams(lag=1))
    pd.testing.assert_series_equal(res1.daily["position"].iloc[:k + 1], res2.daily["position"].iloc[:k + 1])


def test_same_close_beats_next_close_on_pure_noise():
    rng = np.random.default_rng(2)
    s = pd.Series(rng.standard_normal(4000) * 10, index=pd.bdate_range("2005-01-03", periods=4000))  # iid noise
    sd = make_sd(s, cost=0.0)
    same = bt.run(sd, bt.TradeParams(lag=0)).metrics["sharpe"]
    nxt = bt.run(sd, bt.TradeParams(lag=1)).metrics["sharpe"]
    assert same > 2.0           # trading at the print that produced the signal harvests the noise
    assert abs(nxt) < 0.5       # one close later there is nothing left


def test_costs_charged_per_unit_traded():
    s = ou_series(seed=3)
    free = bt.run(make_sd(s, cost=0.0)).metrics
    paid = bt.run(make_sd(s, cost=10.0)).metrics
    res = bt.run(make_sd(s, cost=10.0))
    traded_units = res.daily["position"].diff().abs().fillna(res.daily["position"].abs()).sum()
    assert free["net_total"] - paid["net_total"] == pytest.approx(traded_units * 5.0, rel=1e-6)


def test_band_blocks_small_deviations_and_direction_limits():
    s = ou_series(seed=4)
    no_band = bt.run(make_sd(s)).metrics["trades"]
    wide = bt.run(make_sd(s, band=1e6)).metrics["trades"]
    assert no_band > 0 and wide == 0
    pos = bt.positions(make_sd(s, direction="short_only"), bt.TradeParams())
    assert (pos <= 0).all() and (pos < 0).any()


def test_no_trade_days_force_flat():
    s = ou_series(seed=5)
    nt = s.index[::50]
    pos = bt.positions(make_sd(s, no_trade=pd.DatetimeIndex(nt)), bt.TradeParams())
    assert (pos.reindex(nt) == 0).all()


def test_roll_days_charge_retrade():
    s = ou_series(seed=6)
    rolls = s.index[::21]
    base = bt.run(make_sd(s, cost=4.0)).metrics["net_total"]
    rolled = bt.run(make_sd(s, cost=4.0, roll_days=pd.DatetimeIndex(rolls))).metrics["net_total"]
    assert rolled < base


def test_attribution_adds_up():
    s = ou_series(seed=7)
    carry = pd.Series(0.05, index=s.index)
    res = bt.run(make_sd(s, cost=3.0, carry=carry, hold=0.02))
    d = res.daily
    assert (d["gross"] + d["carry"] + d["hold"] + d["costs"] - d["net"]).abs().max() < 1e-9
    # every bp of net P&L belongs to exactly one trade (flips split their cost between the two trades)
    assert res.trades["net"].sum() == pytest.approx(d["net"].sum(), rel=1e-9, abs=1e-6)


def test_carry_rule_holds_positive_carry_side():
    idx = pd.bdate_range("2015-01-01", periods=500)
    carry = pd.Series(np.where(np.arange(500) < 250, 150.0, -150.0), index=idx)
    ctv = carry / 100.0          # carry-to-vol 1.5: above the 0.25 threshold
    sd = make_sd(carry, pnl=pd.Series(0.0, index=idx), builder="fx_carry", meta={"carry_to_vol": ctv})
    pos = bt.positions(sd, bt.TradeParams())
    assert (pos.iloc[:250] == 1).all() and (pos.iloc[250:] == -1).all()


def test_convergence_episodes_on_ou():
    s = ou_series(seed=8, phi=0.8)
    ep = bt.convergence_episodes(make_sd(s), entry_z=2.0, exit_z=0.5)
    assert len(ep) > 20 and ep["reverted"].mean() > 0.9


def test_cl_expiry_rule_and_roll_days():
    tdays = pd.bdate_range("2020-01-01", "2021-12-31")
    # Dec-2020 WTI (CLZ0) expired on 2020-11-19 (25 Nov 2020 was a Wednesday -> 3 business days before: Fri 20th?)
    # 25 Nov 2020 is a business day; 3 business days earlier = Fri 20 Nov 2020.
    assert calendars.expiry("CL", 2020, 12, tdays) == pd.Timestamp("2020-11-20")
    # Jan-2021 contract: 25 Dec 2020 is a holiday-free Friday in a plain bdate calendar -> 3 bdays before = 22 Dec.
    assert calendars.expiry("CL", 2021, 1, tdays) == pd.Timestamp("2020-12-22")
    assert calendars.expiry("NG", 2021, 1, tdays) == pd.Timestamp("2020-12-29")
    assert calendars.expiry("RB", 2021, 1, tdays) == pd.Timestamp("2020-12-31")
    rolls = calendars.roll_days("CL", tdays)
    assert pd.Timestamp("2020-11-23") in rolls


def test_roll_adjusted_returns_use_next_contract_base():
    idx = pd.bdate_range("2021-01-04", periods=5)
    front = pd.Series([50, 51, 55, 56, 57.0], index=idx)
    nxt = pd.Series([54, 55, 59, 60, 61.0], index=idx)
    r = calendars.roll_adjusted_returns(front, nxt, pd.DatetimeIndex([idx[2]]))
    assert r.iloc[2] == pytest.approx(55 / 55 - 1)       # today's contract 1 vs yesterday's contract 2
    assert r.iloc[3] == pytest.approx(56 / 55 - 1)


def test_third_friday():
    assert calendars.third_friday(2026, 12) == pd.Timestamp("2026-12-18")
    assert calendars.third_friday(2024, 3) == pd.Timestamp("2024-03-15")


def test_par_bond_duration_and_return():
    d = float(bonds.modified_duration(4.0, 10.0))
    assert 8.0 < d < 8.3                                   # 10y par bond at 4%: ~8.18
    idx = pd.bdate_range("2024-01-01", periods=3)
    y = pd.Series([4.0, 4.10, 4.10], index=idx)
    r = bonds.par_bond_returns(y, 10.0)
    assert r.iloc[0] == pytest.approx(-d * 0.001 + 0.04 / 365 + 0.5 * float(bonds.convexity(4.0, 10.0)) * 1e-6, rel=1e-3)
