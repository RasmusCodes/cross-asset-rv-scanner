"""Registered experiments. Each writes its outputs under results/ and returns a small summary dict.

Parameters are fixed in advance (entry |z| = 2, exit 0.5, stop 4, 126-period z window, next-close execution, base
costs); nothing is tuned on the data. In-sample / out-of-sample split for the backtests: 2016-01-01.

  E1 scan        current opportunity census (every relationship evaluated at the latest date)
  E2 backtests   every tradable relationship: base, stress costs (x2), same-close fills (lag 0), IS / OOS
  E3 multitest   Sharpe p-values with BH / Bonferroni, deflated Sharpe of the best, White's reality check
  E4 coint       cointegration scan: latest window, random-walk null rate, rolling formation -> test persistence
                 and frozen-hedge out-of-sample trading
  E5 hedges      hedge-ratio methods on the statistical pairs (rolling OLS, Kalman, static, 1:1)
  E6 gating      regime-switching and Bayesian gates vs the plain z-score rule, walk-forward
  E7 falsepos    false-positive gallery (timing artefacts, stale NAVs, structural premia, broken bounds)
  E8 costs       cost sensitivity and capacity
  E9 stability   rolling stationarity and half-life stability by tier
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd

from .. import paths
from ..data import store
from ..data.universe import load_universe
from ..engine import backtest as bt
from ..engine import costs as costmod
from ..models import bayes, cointegration as ci, multiple_testing as mt, regime
from ..models.ou import fit_ou
from ..scanners import catalog, evaluate, stat_scan
from ..scanners.base import SpreadData, tr_usd

log = logging.getLogger("rvscan.research")
OOS_START = pd.Timestamp("2016-01-01")
RES = paths.RESULTS


def _write_json(name: str, obj) -> None:
    RES.mkdir(parents=True, exist_ok=True)
    (RES / name).write_text(json.dumps(obj, indent=1, default=_json_default), "utf-8")


def _json_default(o):
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


@lru_cache(maxsize=1)
def spreads() -> dict[str, SpreadData]:
    out = {}
    for r in catalog.relationships():
        try:
            sd = catalog.build(r)
        except Exception as exc:  # noqa: BLE001
            log.error("%s failed: %s", r.id, exc)
            continue
        if sd is not None:
            out[r.id] = sd
    return out


def tradable() -> dict[str, SpreadData]:
    return {k: v for k, v in spreads().items() if v.tradable}


# ------------------------------------------------------------------------------------------------ E1
def e1_scan() -> dict:
    rows = []
    for rid, sd in spreads().items():
        try:
            rows.append(evaluate.evaluate(sd))
        except Exception as exc:  # noqa: BLE001
            log.error("evaluate %s: %s", rid, exc)
    df = pd.DataFrame(rows)
    RES.mkdir(parents=True, exist_ok=True)
    df.to_json(RES / "scan_latest.json", orient="records", default_handler=str, indent=1)
    flat = df.drop(columns=[c for c in ("advs", "notes", "reasons", "borrow", "legs", "half_life_ci", "regime_half_lives")
                            if c in df.columns])
    flat["reasons"] = df["reasons"].apply(lambda r: "; ".join(r) if isinstance(r, list) else "")
    flat.to_csv(RES / "scan_latest.csv", index=False)
    summary = {"n": len(df), "by_status": df["status"].value_counts().to_dict(),
               "by_tier_status": df.groupby(["tier", "status"]).size().unstack(fill_value=0).to_dict(orient="index"),
               "by_type_status": df.groupby(["type", "status"]).size().unstack(fill_value=0).to_dict(orient="index"),
               "asof": str(evaluate.latest_market_date().date())}
    _write_json("e1_summary.json", summary)
    return summary


# ------------------------------------------------------------------------------------------------ E2
VARIANTS = {
    "base": bt.TradeParams(),
    "stress_costs": bt.TradeParams(cost_mult=2.0),
    "same_close": bt.TradeParams(lag=0),
}


def _segment_metrics(daily: pd.DataFrame, ppy: float, start=None, end=None) -> dict:
    d = daily
    if start is not None:
        d = d[d.index >= start]
    if end is not None:
        d = d[d.index < end]
    if len(d) < 20:
        return {"sharpe": np.nan, "net_per_year": np.nan, "years": len(d) / ppy}
    net = d["net"]
    sd = net.std()
    return {"sharpe": float(net.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0,
            "net_per_year": float(net.sum() / (len(d) / ppy)), "years": float(len(d) / ppy)}


def e2_backtests() -> dict:
    rows = []
    daily_net = {}
    trades_all = []
    for rid, sd in tradable().items():
        ppy = sd.periods_per_year()
        for vname, p in VARIANTS.items():
            res = bt.run(sd, p)
            m = res.metrics
            row = {"id": rid, "variant": vname, "type": sd.rel.type, "tier": sd.rel.tier, "freq": sd.freq,
                   "license": sd.license, **{k: v for k, v in m.items() if not isinstance(v, (list, dict))}}
            if vname == "base":
                is_m = _segment_metrics(res.daily, ppy, end=OOS_START)
                oos_m = _segment_metrics(res.daily, ppy, start=OOS_START)
                row.update({"is_sharpe": is_m["sharpe"], "oos_sharpe": oos_m["sharpe"], "is_years": is_m["years"],
                            "oos_years": oos_m["years"], "oos_net_per_year": oos_m["net_per_year"]})
                s = res.daily["net"].copy()
                if sd.freq == "M":
                    s = s.resample("ME").sum()
                daily_net[rid] = s
                t = res.trades.copy()
                if len(t):
                    t.insert(0, "id", rid)
                    trades_all.append(t)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "backtests.csv", index=False)
    pd.DataFrame({k: v for k, v in daily_net.items()}).to_parquet(RES / "backtest_daily_net.parquet")
    if trades_all:
        pd.concat(trades_all, ignore_index=True).to_csv(RES / "backtest_trades.csv", index=False)
    base = df[df["variant"] == "base"]
    piv = df.pivot_table(index="id", columns="variant", values="sharpe")
    meta = base.set_index("id")[["type", "tier"]]
    piv = piv.join(meta)
    summary = {
        "n_strategies": int(len(base)),
        "by_tier": base.groupby("tier").agg(n=("sharpe", "size"), median_sharpe=("sharpe", "median"),
                                            share_positive=("sharpe", lambda x: float((x > 0).mean())),
                                            median_gross_sharpe=("gross_sharpe", "median")).round(3).to_dict(orient="index"),
        "by_type": base.groupby("type").agg(n=("sharpe", "size"), median_sharpe=("sharpe", "median"),
                                            share_positive=("sharpe", lambda x: float((x > 0).mean())),
                                            median_trades=("trades", "median")).round(3).to_dict(orient="index"),
        "same_close_vs_base": piv.groupby("tier")[["same_close", "base"]].median().round(3).to_dict(orient="index"),
        "same_close_sharpe_gt1": int((piv["same_close"] > 1).sum()),
        "base_sharpe_gt1": int((piv["base"] > 1).sum()),
        "is_oos_rank_corr": float(base[["is_sharpe", "oos_sharpe"]].corr(method="spearman").iloc[0, 1]),
        "oos_median_sharpe": float(base["oos_sharpe"].median()),
    }
    top_is = base.nlargest(10, "is_sharpe")
    summary["top10_is_then_oos"] = top_is[["id", "is_sharpe", "oos_sharpe"]].round(3).to_dict(orient="records")
    _write_json("e2_summary.json", summary)
    return summary


# ------------------------------------------------------------------------------------------------ E3
def e3_multiple_testing() -> dict:
    df = pd.read_csv(RES / "backtests.csv")
    base = df[df["variant"] == "base"].copy()
    base = base[base["years"] > 2]
    t = base["sharpe"] * np.sqrt(base["years"])
    from scipy import stats
    base["p_one_sided"] = 1 - stats.norm.cdf(t)
    base["q_bh"] = mt.benjamini_hochberg(base["p_one_sided"].values)
    base["p_bonf"] = mt.bonferroni(base["p_one_sided"].values)
    base[["id", "type", "tier", "sharpe", "years", "p_one_sided", "q_bh", "p_bonf"]].to_csv(RES / "e3_pvalues.csv", index=False)
    daily = pd.read_parquet(RES / "backtest_daily_net.parquet")
    # daily strategies only for the reality check (monthly ones are in the BH table)
    dstrat = [c for c in daily.columns if c in set(base[base["freq"] == "D"]["id"])]
    panel = daily[dstrat]
    panel = panel[panel.index >= "2012-01-01"].fillna(0.0)
    vol = panel.std().replace(0, np.nan)
    norm = (panel / vol).dropna(axis=1, how="all").fillna(0.0)
    rc = mt.reality_check(norm.values, n_boot=500, mean_block=20, seed=1)
    best = norm.columns[rc["best"]]
    per_sr = (norm.mean() / norm.std()).values
    n_trials = int(len(base) * len(VARIANTS))
    dsr = mt.deflated_sharpe(panel[best].values, n_trials=n_trials, var_sr_trials=float(np.var(per_sr)))
    out = {"n_tested": int(len(base)), "raw_p_lt_005": int((base["p_one_sided"] < 0.05).sum()),
           "bh_q_lt_005": int((base["q_bh"] < 0.05).sum()), "bh_q_lt_010": int((base["q_bh"] < 0.10).sum()),
           "bonf_lt_005": int((base["p_bonf"] < 0.05).sum()),
           "expected_false_raw": float(0.05 * len(base)),
           "reality_check": {"best": best, "pvalue": rc["pvalue"], "K": rc["K"], "T": rc["T"]},
           "deflated_sharpe_best": {k: dsr[k] for k in ("sr_annual", "sr0_annual", "psr", "dsr", "n")},
           "n_trials_assumed": n_trials,
           "survivors_bh_010": base[base["q_bh"] < 0.10][["id", "sharpe", "years", "q_bh"]].round(4).to_dict(orient="records")}
    _write_json("e3_summary.json", out)
    return out


# ------------------------------------------------------------------------------------------------ E4
def _scan_one_window(args):
    panels, start, end = args
    return stat_scan.scan_window(panels, start, end)


def e4_coint(workers: int = 3) -> dict:
    panels = {g: stat_scan.group_panel(g) for g in stat_scan.GROUPS}
    end = evaluate.latest_market_date()
    start = end - pd.offsets.BDay(756)
    latest = stat_scan.scan_window(panels, start, end)
    latest.to_csv(RES / "coint_latest.csv", index=False)
    null_rate = mt.eg_null_rate(756, n_pairs=1000, seed=3)
    null_rate_drift = mt.eg_null_rate(756, n_pairs=1000, seed=4, drift=0.0003)
    # rolling formation (3y) -> persistence window (next 3y) and trading year (next 1y)
    form_ends = [pd.Timestamp(f"{y}-12-31") for y in range(2008, 2023)]
    jobs = []
    for fe in form_ends:
        jobs.append((panels, fe - pd.DateOffset(years=3), fe))
        pe = fe + pd.DateOffset(years=3)
        if pe <= end:
            jobs.append((panels, fe, pe))
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for (pan, s, e), df in zip(jobs, ex.map(_scan_one_window, jobs)):
            results[(str(s.date()), str(e.date()))] = df
    rows, trade_rows = [], []
    for fe in form_ends:
        fs = fe - pd.DateOffset(years=3)
        form = results.get((str(fs.date()), str(fe.date())))
        if form is None or form.empty:
            continue
        pers = results.get((str(fe.date()), str((fe + pd.DateOffset(years=3)).date())))
        key = form.set_index(["a", "b"])
        if pers is not None and not pers.empty:
            pk = pers.set_index(["a", "b"])["p"]
            key = key.join(pk.rename("p_next"), how="left")
        for sel_name, mask in (("bh_q05", key["q_bh"] < 0.05), ("raw_p05", key["p"] < 0.05), ("all", key["p"] <= 1.0)):
            sub = key[mask]
            rows.append({"formation_end": str(fe.date()), "selection": sel_name, "n_pairs_tested": int(len(key)),
                         "n_selected": int(len(sub)),
                         "persist_rate": float((sub["p_next"] < 0.05).mean()) if "p_next" in sub and sub["p_next"].notna().any() else np.nan})
        # trade the BH and raw selections (and a random 150 of all pairs) in the following year, frozen at formation
        ty_end = fe + pd.DateOffset(years=1)
        if ty_end > end + pd.DateOffset(days=200):
            continue
        rng = np.random.default_rng(int(fe.year))
        all_pairs = list(key.index)
        sample_all = [all_pairs[i] for i in rng.choice(len(all_pairs), size=min(150, len(all_pairs)), replace=False)]
        for sel_name, pairs in (("bh_q05", list(key[key["q_bh"] < 0.05].index)),
                                ("raw_p05", list(key[key["p"] < 0.05].index)), ("random_all", sample_all)):
            for a, b in pairs:
                g = key.loc[(a, b), "group"]
                if g == "sovereign_10y":
                    continue
                pan = panels[g]
                f_a, f_b = pan[a][(pan.index > fs) & (pan.index <= fe)], pan[b][(pan.index > fs) & (pan.index <= fe)]
                t_a, t_b = pan[a][(pan.index > fe) & (pan.index <= ty_end)], pan[b][(pan.index > fe) & (pan.index <= ty_end)]
                r = stat_scan.trade_frozen(t_a, t_b, f_a, f_b, a, b)
                trade_rows.append({"formation_end": str(fe.date()), "selection": sel_name, "a": a, "b": b, "group": g,
                                   "net": r["net"], "gross": r["gross"], "trades": r["trades"]})
    roll = pd.DataFrame(rows)
    trades = pd.DataFrame(trade_rows)
    roll.to_csv(RES / "coint_rolling.csv", index=False)
    trades.to_csv(RES / "coint_oos_trades.csv", index=False)
    tsum = trades.groupby("selection").agg(pair_years=("net", "size"), mean_net=("net", "mean"),
                                           median_net=("net", "median"), share_positive=("net", lambda x: float((x > 0).mean())),
                                           mean_gross=("gross", "mean"), mean_trades=("trades", "mean")).round(2)
    out = {"latest_window": [str(start.date()), str(end.date())], "latest_pairs": int(len(latest)),
           "latest_raw_p05": int((latest["p"] < 0.05).sum()), "latest_raw_minp05": int((np.minimum(latest["p_ab"], latest["p_ba"]) < 0.05).sum()),
           "latest_bh_q05": int((latest["q_bh"] < 0.05).sum()), "latest_bonf05": int((latest["p_bonf"] < 0.05).sum()),
           "latest_johansen_rank1": int((latest["johansen_rank"] >= 1).sum()),
           "null_rate_eg_756": null_rate, "null_rate_eg_756_drift": null_rate_drift,
           "rolling": roll.groupby("selection").agg(windows=("n_selected", "size"), mean_selected=("n_selected", "mean"),
                                                    mean_persist=("persist_rate", "mean")).round(3).to_dict(orient="index"),
           "oos_trading": tsum.to_dict(orient="index")}
    _write_json("e4_summary.json", out)
    return out


# ------------------------------------------------------------------------------------------------ E5
def e5_hedges() -> dict:
    from ..scanners.builders_stat import stat_pair
    rows = []
    for rid, sd in tradable().items():
        if sd.rel.params.get("builder") != "stat_pair":
            continue
        rel = sd.rel
        first = sd.spread.index[0]
        formation_end = first + pd.DateOffset(years=3)
        variants = {}
        for method in ("rolling", "kalman", "static", "ratio"):
            rel.params["formation_end"] = formation_end
            v = stat_pair(rel, method=method)
            variants[method] = bt.run(v, bt.TradeParams(), start=formation_end + pd.DateOffset(days=1))
        rel.params.pop("formation_end", None)
        for method, res in variants.items():
            m = res.metrics
            rows.append({"id": rid, "method": method, "sharpe": m["sharpe"], "gross_sharpe": m["gross_sharpe"],
                         "trades": m["trades"], "net_per_year": m["net_per_year"], "years": m["years"],
                         "spread_sd": float(res.daily["spread"].diff().std())})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "e5_hedges.csv", index=False)
    piv = df.pivot_table(index="id", columns="method", values="sharpe")
    out = {"n_pairs": int(piv.shape[0]), "median_sharpe": piv.median().round(3).to_dict(),
           "share_positive": (piv > 0).mean().round(3).to_dict(),
           "kalman_minus_rolling_median": float((piv["kalman"] - piv["rolling"]).median()),
           "ratio_minus_rolling_median": float((piv["ratio"] - piv["rolling"]).median()),
           "best_method_counts": piv.idxmax(axis=1).value_counts().to_dict()}
    _write_json("e5_summary.json", out)
    return out


# ------------------------------------------------------------------------------------------------ E6
def walk_forward_regime(sd: SpreadData, train: int = 756, step: int = 252) -> pd.Series:
    s = sd.deviation.dropna() if isinstance(sd.fair, pd.Series) else sd.spread.dropna()
    out = pd.Series(np.nan, index=s.index)
    fit = None
    for end in range(train, len(s), step):
        try:
            fit = regime.fit_msar(s.iloc[end - train:end].values, max_iter=60)
        except Exception:  # noqa: BLE001 - keep the previous year's model if this window's fit fails
            if fit is None:
                continue
        seg = s.iloc[end - 1:min(end + step, len(s))]
        filt = regime.filter_with_params(seg.values, fit)
        out.iloc[end:min(end + step, len(s))] = filt[1:, 0]
    return out


def walk_forward_bayes(sd: SpreadData, train: int = 504, step: int = 21) -> pd.Series:
    s = sd.spread.dropna()
    structural = sd.fair is not None and not isinstance(sd.fair, pd.Series)
    out = pd.Series(np.nan, index=s.index)
    for end in range(train, len(s), step):
        tail = s.iloc[end - train:end].values
        m = bayes.fit_spread_model(tail, fair_value=float(sd.fair) if structural else None)
        out.iloc[end:min(end + step, len(s))] = m.p_mean_reverting
    return out


def e6_gating() -> dict:
    rows = []
    for rid, sd in tradable().items():
        if sd.freq != "D" or sd.rel.params.get("builder") in ("fx_carry",) or len(sd.spread) < 1500:
            continue
        if sd.rel.tier == "arbitrage":
            continue                         # gates are for relationships whose mean reversion is uncertain
        gates = {"none": None, "regime": walk_forward_regime(sd), "bayes": walk_forward_bayes(sd)}
        thr = {"none": 0.5, "regime": 0.5, "bayes": 0.9}
        start = sd.spread.index[756]
        for g, series in gates.items():
            res = bt.run(sd, bt.TradeParams(gate=g if g != "none" else None, gate_threshold=thr[g]), gate=series, start=start)
            m = res.metrics
            rows.append({"id": rid, "tier": sd.rel.tier, "type": sd.rel.type, "gate": g, "sharpe": m["sharpe"],
                         "trades": m["trades"], "net_per_year": m["net_per_year"], "time_in_market": m["time_in_market"]})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "e6_gating.csv", index=False)
    piv = df.pivot_table(index="id", columns="gate", values="sharpe")
    out = {"n": int(piv.shape[0]), "median_sharpe": piv.median().round(3).to_dict(),
           "regime_minus_none_median": float((piv["regime"] - piv["none"]).median()),
           "bayes_minus_none_median": float((piv["bayes"] - piv["none"]).median()),
           "regime_better_share": float((piv["regime"] > piv["none"]).mean()),
           "bayes_better_share": float((piv["bayes"] > piv["none"]).mean())}
    _write_json("e6_summary.json", out)
    return out


# ------------------------------------------------------------------------------------------------ E7
def e7_false_positives() -> dict:
    sds = spreads()
    out = {}
    base, same = bt.TradeParams(), bt.TradeParams(lag=0)

    def lag_pair(rid):
        sd = sds[rid]
        return {"same_close_sharpe": bt.run(sd, same).metrics["sharpe"], "next_close_sharpe": bt.run(sd, base).metrics["sharpe"]}

    # 1. FX triangle from official fixings 6 hours apart
    tri = {}
    for rid in ("TRI_EURJPY", "TRI_EURGBP", "TRI_EURCHF"):
        sd = sds[rid]
        s = sd.spread
        tri[rid] = {**lag_pair(rid), "spread_sd_bp": float(s.std()), "half_life": fit_ou(s.iloc[-1000:].values).half_life_days}
    out["fx_triangle"] = tri
    # 2. ADR timing: premium today vs home-line return tomorrow, by time-zone gap
    adr_rows = []
    for rid, sd in sds.items():
        if sd.rel.type != "adr" or not sd.tradable:
            continue
        home = sd.rel.legs[1]
        prem = sd.spread
        r_home = np.log(tr_usd(home)).diff().reindex(prem.index) * 1e4
        df = pd.concat({"prem": prem - prem.rolling(60).mean(), "r1": r_home.shift(-1)}, axis=1).dropna()
        df = df[df.index >= "2010-01-01"]
        if len(df) < 500:
            continue
        lp = lag_pair(rid)
        adr_rows.append({"id": rid, "gap_h": sd.sync_gap_h, "corr_prem_next_home": float(df["prem"].corr(df["r1"])), **lp})
    adr_df = pd.DataFrame(adr_rows)
    adr_df.to_csv(RES / "e7_adr_timing.csv", index=False)
    out["adr_timing"] = {
        "asia_median_corr": float(adr_df[adr_df["gap_h"] > 10]["corr_prem_next_home"].median()),
        "europe_median_corr": float(adr_df[(adr_df["gap_h"] > 1) & (adr_df["gap_h"] <= 10)]["corr_prem_next_home"].median()),
        "same_close_median_sharpe": float(adr_df["same_close_sharpe"].median()),
        "next_close_median_sharpe": float(adr_df["next_close_sharpe"].median()), "n": int(len(adr_df))}
    # 3. Stale NAVs: premium today vs NAV change tomorrow
    nav_rows = []
    for rid, sd in sds.items():
        if sd.rel.type != "etf_nav":
            continue
        etf = sd.rel.legs[0]
        nav = store.load_nav(etf)["nav"]
        rn = np.log(nav).diff().shift(-1).reindex(sd.spread.index) * 1e4
        df = pd.concat({"prem": sd.spread, "rn1": rn}, axis=1).dropna()
        nav_rows.append({"id": rid, "kind": sd.meta.get("kind"), "corr_prem_next_nav": float(df["prem"].corr(df["rn1"])),
                         "premium_sd_bp": float(sd.spread.std()), **lag_pair(rid)})
    nav_df = pd.DataFrame(nav_rows)
    nav_df.to_csv(RES / "e7_nav_staleness.csv", index=False)
    out["nav_staleness"] = nav_df.groupby("kind")[["corr_prem_next_nav", "premium_sd_bp", "same_close_sharpe",
                                                   "next_close_sharpe"]].median().round(3).to_dict(orient="index")
    # 4. Structural premia that never converge: TSM / Indian ADRs, Rio Tinto DLC
    struct = {}
    for rid in ("ADR_TSM", "ADR_INFY", "ADR_HDB", "ADR_IBN", "DLC_RIO_L_RIO_AX"):
        if rid in sds:
            s = sds[rid].spread
            yearly = s.groupby(s.index.year).median().round(0)
            struct[rid] = {"median_by_year_bp": yearly.iloc[-12:].to_dict(), "share_days_positive": float((s > 0).mean())}
    out["structural_premia"] = struct
    # 5. WTI calendar spread vs full carry in April 2020
    cal = sds.get("CAL_WTI1_WTI2")
    if cal is not None:
        fc = cal.meta["full_carry_bp"]
        s = cal.spread
        window = s["2020-03-01":"2020-06-30"]
        out["wti_2020"] = {"max_contango_bp": float(window.max()), "max_date": str(window.idxmax().date()),
                           "full_carry_bp_that_day": float(fc.reindex([window.idxmax()]).iloc[0]),
                           "days_above_full_carry_since_2000": int((s["2000":] > fc.reindex(s["2000":].index)).sum())}
    # 6. Berkshire conversion bound
    brk = sds.get("CLASS_BRK_A_BRK_B")
    if brk is not None:
        bd = brk.at(brk.band_down)
        s = brk.spread
        out["brk_bound"] = {"band_bp": bd, "share_days_below_minus_band": float((s < -bd).mean()),
                            "min_spread_bp": float(s.min()), "min_date": str(s.idxmin().date()),
                            "median_spread_bp": float(s.median()), **lag_pair("CLASS_BRK_A_BRK_B")}
    # 7. Futures basis: daily noise vs plausible mispricing
    bas = {}
    for rid in ("BASIS_ES1", "BASIS_NQ1", "BASIS_RTY1", "BASIS_YM1"):
        s = sds[rid].spread["2012":]
        bas[rid] = {"mispricing_sd_bp": float(s.std()), "daily_change_sd_bp": float(s.diff().std()),
                    "half_life": fit_ou(s.values).half_life_days, **lag_pair(rid)}
    out["futures_basis"] = bas
    # 8. Interlisted Canadians with a noon FX fixing
    xl = [lag_pair(rid) | {"id": rid} for rid, sd in sds.items() if rid.startswith("XL_")]
    out["interlisted"] = {"same_close_median": float(np.median([x["same_close_sharpe"] for x in xl])),
                          "next_close_median": float(np.median([x["next_close_sharpe"] for x in xl])), "n": len(xl)}
    _write_json("e7_summary.json", out)
    return out


# ------------------------------------------------------------------------------------------------ E8
def e8_costs() -> dict:
    rows = []
    for rid, sd in tradable().items():
        for mult in (0.0, 0.5, 1.0, 2.0, 3.0):
            m = bt.run(sd, bt.TradeParams(cost_mult=mult)).metrics
            rows.append({"id": rid, "type": sd.rel.type, "tier": sd.rel.tier, "cost_mult": mult, "sharpe": m["sharpe"]})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "e8_costs.csv", index=False)
    curve = df.groupby(["tier", "cost_mult"])["sharpe"].median().unstack().round(3)
    # capacity: square-root impact at larger sizes for the best base strategies
    base = pd.read_csv(RES / "backtests.csv")
    base = base[(base["variant"] == "base") & (base["years"] > 3)].nlargest(12, "sharpe")
    u = load_universe()
    cap_rows = []
    for _, r in base.iterrows():
        sd = spreads()[r["id"]]
        for size in (1e6, 1e7, 1e8):
            extra = 0.0
            for leg in sd.rel.legs:
                inst = u.get(leg)
                if inst is None or inst.src != "yahoo" or not inst.tradable:
                    continue
                v = store.volume_usd(leg).dropna()
                if not len(v):
                    continue
                vol = float(np.log(tr_usd(leg)).diff().iloc[-252:].std())
                k = costmod.cfg()["liquidity_tiers"][inst.liq].get("impact_k", 0.5)
                extra += 2 * k * vol * np.sqrt(size / float(v.iloc[-1])) * 1e4
            cap_rows.append({"id": r["id"], "size_usd": size, "impact_rt_bp": extra,
                             "avg_gross_per_trade": r.get("avg_gross_per_trade"), "avg_net_per_trade": r.get("avg_net_per_trade")})
    cap = pd.DataFrame(cap_rows)
    cap.to_csv(RES / "e8_capacity.csv", index=False)
    out = {"median_sharpe_by_tier_and_cost": {t: {str(k): v for k, v in row.items()} for t, row in curve.iterrows()}}
    _write_json("e8_summary.json", out)
    return out


# ------------------------------------------------------------------------------------------------ E9
def e9_stability(window: int = 504, step: int = 126) -> dict:
    rows = []
    for rid, sd in tradable().items():
        if sd.freq != "D" or sd.rel.params.get("builder") == "fx_carry":
            continue
        s = (sd.deviation if isinstance(sd.fair, pd.Series) else sd.spread).dropna()
        if len(s) < window + step * 4:
            continue
        n_vars = 2 if sd.est_hedge else 1
        ps, hls = [], []
        for end in range(window, len(s), step):
            w = s.iloc[end - window:end].values
            ps.append(ci.adf(w, n_vars=n_vars).pvalue)
            o = fit_ou(w, n_vars=n_vars)
            hls.append(o.half_life_days if o.mean_reverting else np.inf)
        hls = np.array(hls)
        fin = hls[np.isfinite(hls)]
        rows.append({"id": rid, "tier": sd.rel.tier, "type": sd.rel.type, "windows": len(ps),
                     "share_stationary": float(np.mean(np.array(ps) < 0.05)),
                     "hl_median": float(np.median(fin)) if len(fin) else np.inf,
                     "hl_iqr_ratio": float(np.percentile(fin, 75) / np.percentile(fin, 25)) if len(fin) > 3 else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "e9_stability.csv", index=False)
    out = {"by_tier": df.groupby("tier").agg(n=("id", "size"), median_share_stationary=("share_stationary", "median"),
                                            median_hl=("hl_median", "median"),
                                            median_hl_iqr_ratio=("hl_iqr_ratio", "median")).round(3).to_dict(orient="index")}
    _write_json("e9_summary.json", out)
    return out


EXPERIMENTS = {"e1": e1_scan, "e2": e2_backtests, "e3": e3_multiple_testing, "e4": e4_coint, "e5": e5_hedges,
               "e6": e6_gating, "e7": e7_false_positives, "e8": e8_costs, "e9": e9_stability}


def run(only: list[str] | None = None) -> dict:
    RES.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, fn in EXPERIMENTS.items():
        if only and name not in only:
            continue
        t0 = time.time()
        log.info("running %s", name)
        out[name] = fn()
        log.info("%s done in %.0fs", name, time.time() - t0)
    return out
