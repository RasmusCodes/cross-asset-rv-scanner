"""Statistical cointegration scan across instrument groups, with multiple-testing control and out-of-sample checks.

For every pair within a group and a formation window:
  - Engle-Granger in both orderings; the reported p-value is the LARGER of the two (a pair is only called
    cointegrated if it passes whichever variable is the dependent one),
  - Johansen trace test (rank >= 1 at 95%),
  - OU half-life of the Engle-Granger residual.
Across all pairs tested in a window, p-values are adjusted with Benjamini-Hochberg (FDR) and Bonferroni (FWER).
Out of sample, pairs selected in a formation window are (a) re-tested in the next, non-overlapping window and
(b) traded in the following year with the hedge ratio, mean and volatility frozen at formation.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import fx, store
from ..data.universe import load_universe
from ..engine import costs as costmod
from ..models import cointegration as ci
from ..models import multiple_testing as mt
from ..models.ou import fit_ou
from .base import tr_usd

GROUPS = {
    "equity_etf": ["SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "VWO", "EWJ", "EWG", "EWU", "EWC", "EWA", "EWT", "EWH",
                   "EWZ", "FXI", "INDA", "EZU", "FEZ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB",
                   "XLRE", "XLC", "SMH", "SOXX", "KRE", "KBE", "XOP", "OIH", "IBB", "XBI", "ITB", "XHB", "XME", "XRT"],
    "bond_etf": ["SHY", "IEI", "IEF", "TLT", "SPTL", "AGG", "BND", "SPAB", "LQD", "VCIT", "HYG", "JNK", "EMB", "EBND",
                 "BWX", "TIP"],
    "commodity_etf": ["GLD", "IAU", "SLV", "USO", "BNO", "UNG", "DBC", "GDX", "GDXJ"],
    "stocks": ["KO", "PEP", "XOM", "CVX", "V", "MA", "HD", "LOW", "UPS", "FDX", "GS", "MS", "JPM", "BAC", "WMT", "TGT",
               "T", "VZ"],
    "sovereign_10y": ["UST_10Y", "BUND_10Y", "JGB_10Y", "GILT_10Y", "CGB_10Y", "ACGB_10Y"],
    "fx": ["EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK"],
}


def level_series(iid: str) -> pd.Series:
    """The level a cointegration test should use: log total-return price, yield in percent, or log FX rate."""
    inst = load_universe()[iid]
    if inst.asset_class == "sovereign":
        return store.series(iid)
    if inst.asset_class == "fx":
        return np.log(fx.usd_per(inst.ccy))
    return np.log(tr_usd(iid))


def group_panel(group: str) -> pd.DataFrame:
    cols = {}
    for iid in GROUPS[group]:
        if store.has(iid):
            cols[iid] = level_series(iid)
    return pd.DataFrame(cols).sort_index()


@dataclass
class PairTest:
    a: str
    b: str
    group: str
    nobs: int
    p_ab: float
    p_ba: float
    p: float                   # max of the two orderings
    beta: float                # hedge of a on b (from the a-on-b regression)
    half_life: float
    johansen_rank: int


def test_pair(x: pd.Series, y: pd.Series, a: str, b: str, group: str, max_lags: int = 8) -> PairTest | None:
    df = pd.concat([x, y], axis=1, join="inner").dropna()
    if len(df) < 250:
        return None
    va, vb = df.iloc[:, 0].values, df.iloc[:, 1].values
    r1 = ci.engle_granger(va, vb, max_lags=max_lags)
    r2 = ci.engle_granger(vb, va, max_lags=max_lags)
    hl = fit_ou(r1.resid, n_vars=2).half_life_days
    try:
        jr = ci.johansen(df.values, k_ar_diff=1).rank_trace_95
    except Exception:  # noqa: BLE001
        jr = -1
    return PairTest(a, b, group, len(df), r1.pvalue, r2.pvalue, max(r1.pvalue, r2.pvalue), float(r1.beta[0]), hl, jr)


def scan_window(panels: dict[str, pd.DataFrame], start, end, min_obs: int = 500) -> pd.DataFrame:
    rows = []
    for group, pan in panels.items():
        w = pan[(pan.index > pd.Timestamp(start)) & (pan.index <= pd.Timestamp(end))]
        ok = [c for c in w.columns if w[c].notna().sum() >= min_obs]
        for a, b in itertools.combinations(ok, 2):
            t = test_pair(w[a], w[b], a, b, group)
            if t is not None:
                rows.append(t.__dict__)
    df = pd.DataFrame(rows)
    if len(df):
        df["q_bh"] = mt.benjamini_hochberg(df["p"].values)
        df["p_bonf"] = mt.bonferroni(df["p"].values)
        df["q_bh_min"] = mt.benjamini_hochberg(np.minimum(df["p_ab"], df["p_ba"]).values)
    return df


def trade_frozen(xa: pd.Series, xb: pd.Series, form_a: pd.Series, form_b: pd.Series, iid_a: str, iid_b: str,
                 entry: float = 2.0, exit_: float = 0.5, stop: float = 4.0) -> dict:
    """Trade a pair in the test window with hedge, mean and sd frozen at formation (log-price levels only).

    Execution at the next close; P&L of long $1 a / short $beta b from total returns; costs from the cost model."""
    fa = pd.concat([form_a, form_b], axis=1, join="inner").dropna()
    beta, alpha = np.polyfit(fa.iloc[:, 1].values, fa.iloc[:, 0].values, 1)
    resid_f = fa.iloc[:, 0] - alpha - beta * fa.iloc[:, 1]
    mu, sd = resid_f.mean(), resid_f.std()
    te = pd.concat([xa, xb], axis=1, join="inner").dropna()
    if len(te) < 20 or sd <= 0:
        return {"net": np.nan, "gross": np.nan, "trades": 0, "days": 0}
    z = ((te.iloc[:, 0] - alpha - beta * te.iloc[:, 1]) - mu) / sd
    ra, rb = np.exp(te.iloc[:, 0]).pct_change(), np.exp(te.iloc[:, 1]).pct_change()
    u = load_universe()
    ca, cb = costmod.leg_cost(u[iid_a]), costmod.leg_cost(u[iid_b])
    rt = 2 * (ca.half_spread_bp + ca.commission_bp) + abs(beta) * 2 * (cb.half_spread_bp + cb.commission_bp)
    pos, cur = np.zeros(len(z)), 0.0
    for t, zt in enumerate(z.values):
        if cur != 0 and (abs(zt) <= exit_ or abs(zt) >= stop or np.sign(zt) == np.sign(cur)):
            cur = 0.0
        if cur == 0 and entry <= abs(zt) < stop:
            cur = -np.sign(zt)
        pos[t] = cur
    held = pd.Series(pos, index=z.index).shift(1).fillna(0.0)        # executed at the next close
    prev = held.shift(1).fillna(0.0)
    gross = prev * 1e4 * (ra - beta * rb).fillna(0.0)
    costs = (held - prev).abs() * rt / 2
    net = gross - costs - prev.abs() * (30 + 30 * abs(beta) + 30 * abs(beta)) / 252.0
    # close any open position at the end of the window (exit cost)
    net.iloc[-1] -= abs(held.iloc[-1]) * rt / 2
    trades = int(((held != 0) & (prev == 0)).sum())
    return {"net": float(net.sum()), "gross": float(gross.sum()), "trades": trades, "days": int(len(z)),
            "daily": net}
