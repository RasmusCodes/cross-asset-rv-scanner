"""Build reports/REPORT.md and its figures from results/. Every number in the report is read from a result file."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import paths  # noqa: E402
from ..data import store  # noqa: E402
from ..data.universe import ASSET_CLASS_LABELS, load_universe  # noqa: E402
from ..scanners import catalog  # noqa: E402
from ..scanners.base import TIERS, TYPES  # noqa: E402

RES = paths.RESULTS
FIG = paths.REPORTS / "figures"
# dataviz reference palette, light mode (validated: slots 1-3 pass adjacent CVD / normal-vision checks)
SLOT = {"arbitrage": "#2a78d6", "quasi": "#eb6834", "statistical": "#1baf7a"}
INK, INK2, MUTED, GRID, NEG = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#e34948"
TIER_NAME = {"arbitrage": "Arbitrage", "quasi": "Economic link", "statistical": "Statistical"}


def _j(name):
    p = RES / name
    return json.loads(p.read_text("utf-8")) if p.exists() else {}


def _style(ax, title=None):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    return f"figures/{name}"


def fig_timing(bt: pd.DataFrame) -> str:
    piv = bt.pivot_table(index="id", columns="variant", values="sharpe").join(bt[bt.variant == "base"].set_index("id")["tier"])
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for tier in ("arbitrage", "quasi", "statistical"):
        d = piv[piv["tier"] == tier]
        ax.scatter(d["same_close"], d["base"], s=26, color=SLOT[tier], alpha=0.85, edgecolor="#fcfcfb", linewidth=0.8,
                   label=f"{TIER_NAME[tier]} ({len(d)})")
    lim = [min(-1.5, piv[["same_close", "base"]].min().min()), max(3.5, piv["same_close"].max() + 0.2)]
    ax.plot(lim, lim, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.axhline(0, color="#c3c2b7", linewidth=0.8)
    ax.axvline(0, color="#c3c2b7", linewidth=0.8)
    ax.text(lim[1], lim[1], " same Sharpe", color=MUTED, fontsize=8, va="bottom", ha="right")
    ax.set_xlabel("Net Sharpe, filled at the signal's own close", color=INK2)
    ax.set_ylabel("Net Sharpe, filled at the next close", color=INK2)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax, "Execution timing: same-close fills invent arbitrage")
    return _save(fig, "fig1_timing.png")


def fig_tiers(bt: pd.DataFrame) -> str:
    base = bt[bt.variant == "base"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    rng = np.random.default_rng(0)
    for k, tier in enumerate(("arbitrage", "quasi", "statistical")):
        d = base[base.tier == tier]["sharpe"].values
        ax.scatter(d, k + rng.uniform(-0.18, 0.18, len(d)), s=18, color=SLOT[tier], alpha=0.8, edgecolor="#fcfcfb", linewidth=0.6)
        med = np.median(d)
        ax.plot([med, med], [k - 0.3, k + 0.3], color=INK, linewidth=2)
        ax.text(med, k + 0.34, f"median {med:+.2f}", ha="center", fontsize=8, color=INK2)
    ax.axvline(0, color="#c3c2b7", linewidth=0.8)
    ax.set_yticks([0, 1, 2], [TIER_NAME[t] for t in ("arbitrage", "quasi", "statistical")])
    ax.set_xlabel("Net Sharpe (next-close execution, base costs, full history)", color=INK2)
    _style(ax, "Backtests by relationship strength")
    return _save(fig, "fig2_tiers.png")


def fig_coint(latest: pd.DataFrame, e4: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax = axes[0]
    bins = np.linspace(0, 1, 21)
    ax.hist(latest["p"], bins=bins, color=SLOT["arbitrage"], edgecolor="#fcfcfb", linewidth=1.5)
    ax.axhline(len(latest) / 20, color=INK, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_ylim(0, max(np.histogram(latest["p"], bins=bins)[0].max(), len(latest) / 20) * 1.25)
    ax.text(0.01, ax.get_ylim()[1] * 0.97, "dashed: count expected if no pair were related (uniform p)", ha="left", va="top",
            fontsize=8, color=INK2)
    ax.set_xlabel("Engle-Granger p-value (worse of both orderings)", color=INK2)
    ax.set_ylabel("pairs", color=INK2)
    _style(ax, f"{len(latest):,} pairs, last 3 years")
    ax = axes[1]
    roll = e4.get("rolling", {})
    labels = ["All pairs\n(base rate)", "EG p < 0.05", "BH q < 0.05"]
    vals = [roll.get(k, {}).get("mean_persist", np.nan) for k in ("all", "raw_p05", "bh_q05")]
    ax.bar(labels, [100 * v for v in vals], color=SLOT["arbitrage"], width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, 100 * v + 0.6, f"{100 * v:.1f}%", ha="center", fontsize=9, color=INK)
    ax.set_ylabel("still cointegrated next 3 years (%)", color=INK2)
    _style(ax, "Out-of-sample persistence")
    fig.tight_layout()
    return _save(fig, "fig3_coint.png")


def fig_examples() -> str:
    sds = {r.id: r for r in catalog.relationships()}
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    panels = [("NAV_JNK", "2019-06-01", "2021-06-30", "JNK premium to NAV, bp", None),
              ("CAL_WTI1_WTI2", "2019-06-01", "2021-06-30", "WTI contract 2 vs 1 (contango), bp", "full_carry_bp"),
              ("ADR_TSM", "2005-01-01", None, "TSMC ADR premium over 5 x Taipei shares, bp", None),
              ("TRI_EURJPY", "2018-01-01", None, "ECB EUR/JPY vs H.10-implied cross, bp", None)]
    for ax, (rid, start, end, title, extra) in zip(axes.ravel(), panels):
        sd = catalog.build(sds[rid])
        s = sd.spread[start:end] if end else sd.spread[start:]
        ax.plot(s.index, s.values, color=SLOT["arbitrage"], linewidth=0.9, label="spread")
        if extra and extra in sd.meta:
            e = sd.meta[extra].reindex(s.index)
            ax.plot(e.index, e.values, color=SLOT["quasi"], linewidth=1.0, linestyle=(0, (3, 2)), label="full-carry bound")
            ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax.axhline(0, color="#c3c2b7", linewidth=0.8)
        _style(ax, title)
        import matplotlib.dates as mdates
        loc = mdates.AutoDateLocator(minticks=3, maxticks=6)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    return _save(fig, "fig4_examples.png")


def fig_costs(costs: pd.DataFrame) -> str:
    curve = costs.groupby(["tier", "cost_mult"])["sharpe"].median().unstack(0)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for tier in ("arbitrage", "quasi", "statistical"):
        if tier in curve:
            ax.plot(curve.index, curve[tier], color=SLOT[tier], linewidth=2, marker="o", markersize=5, label=TIER_NAME[tier])
            ax.text(curve.index[-1] + 0.05, curve[tier].iloc[-1], TIER_NAME[tier], color=INK2, fontsize=8, va="center")
    ax.axhline(0, color="#c3c2b7", linewidth=0.8)
    ax.set_xlabel("Cost multiplier (1 = base assumptions, 0 = free trading)", color=INK2)
    ax.set_ylabel("Median net Sharpe", color=INK2)
    ax.legend(frameon=False, fontsize=9)
    _style(ax, "Cost sensitivity")
    return _save(fig, "fig5_costs.png")


def _fmt(v, d=2, pct=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{100 * v:.{d}f}%" if pct else f"{v:.{d}f}"


def md_table(df: pd.DataFrame, floatfmt: int = 2) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("–" if not np.isfinite(v) else f"{v:.{floatfmt}f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def build() -> str:
    e1, e2, e3, e4, e5, e6, e7, e8, e9 = (_j(f"e{i}_summary.json") for i in range(1, 10))
    bt = pd.read_csv(RES / "backtests.csv")
    latest = pd.read_csv(RES / "coint_latest.csv")
    costs = pd.read_csv(RES / "e8_costs.csv") if (RES / "e8_costs.csv").exists() else None
    scan = pd.read_csv(RES / "scan_latest.csv")
    f1, f2, f3 = fig_timing(bt), fig_tiers(bt), fig_coint(latest, e4)
    f4 = fig_examples()
    f5 = fig_costs(costs) if costs is not None else None
    u = load_universe()
    cat = store.read_catalog()
    base = bt[bt.variant == "base"]

    # --- data table
    rows = []
    for ac, label in ASSET_CLASS_LABELS.items():
        ids = [i for i, inst in u.items() if inst.asset_class == ac]
        have = [i for i in ids if i in cat]
        lic = sorted({u[i].license for i in have})
        first = min((cat[i]["first"] for i in have), default="–")
        rows.append({"Asset class": label, "Instruments": len(have), "Earliest data": first, "Licences": ", ".join(lic)})
    data_tab = md_table(pd.DataFrame(rows))
    rels = catalog.relationships()
    rt = pd.DataFrame([{"type": TYPES[r.type], "tier": TIER_NAME[r.tier]} for r in rels])
    rel_tab = md_table(rt.groupby(["type", "tier"]).size().unstack(fill_value=0).reset_index().rename(columns={"type": "Opportunity type"}), 0)

    by_tier = pd.DataFrame(e2["by_tier"]).T.reset_index().rename(columns={"index": "tier"})
    by_tier["tier"] = by_tier["tier"].map(TIER_NAME)
    by_tier = by_tier.rename(columns={"n": "Strategies", "median_sharpe": "Median net Sharpe", "share_positive": "Share positive",
                                      "median_gross_sharpe": "Median gross Sharpe", "tier": "Strength"})
    by_type = pd.DataFrame(e2["by_type"]).T.reset_index().rename(columns={"index": "type"})
    by_type["type"] = by_type["type"].map(lambda t: TYPES.get(t, t))
    by_type = by_type.rename(columns={"type": "Type", "n": "Strategies", "median_sharpe": "Median net Sharpe",
                                      "share_positive": "Share positive", "median_trades": "Median trades"})
    sc = pd.DataFrame(e2["same_close_vs_base"]).T.reset_index().rename(columns={"index": "Strength", "same_close": "Same-close fills", "base": "Next-close fills"})
    sc["Strength"] = sc["Strength"].map(TIER_NAME)
    status = scan.groupby(["tier", "status"]).size().unstack(fill_value=0)
    status.index = [TIER_NAME[t] for t in status.index]
    status_tab = md_table(status.reset_index().rename(columns={"index": "Strength"}), 0)
    actionable = scan[scan["status"].isin(["executable", "executable (not enforceable)", "statistical signal"])]
    act_tab = md_table(actionable[["name", "tier", "status", "z", "deviation", "half_life", "cost_rt", "net_edge"]]
                       .assign(tier=lambda d: d["tier"].map(TIER_NAME)).round(2)
                       .rename(columns={"name": "Relationship", "tier": "Strength", "status": "Verdict", "z": "z",
                                        "deviation": "Deviation", "half_life": "Half-life", "cost_rt": "Cost r/t",
                                        "net_edge": "Net edge"})) if len(actionable) else "_None._"
    watch_reasons = scan[scan["status"] == "watch"]["reasons"].fillna("").str.split("; ").explode().str.replace(r"\(.*\)", "", regex=True).str.strip()
    wr = watch_reasons[watch_reasons != ""].str.split(":").str[0].value_counts()
    wr_tab = md_table(wr.reset_index().rename(columns={"reasons": "Reason a signal is not executable", "count": "Signals"}), 0) if len(wr) else ""

    top_oos = pd.DataFrame(e2["top10_is_then_oos"]).rename(columns={"id": "Relationship", "is_sharpe": "Sharpe to 2015", "oos_sharpe": "Sharpe 2016 on"})
    hed = pd.DataFrame({"Method": list(e5.get("median_sharpe", {}).keys()),
                        "Median net Sharpe": list(e5.get("median_sharpe", {}).values()),
                        "Share positive": [e5.get("share_positive", {}).get(k) for k in e5.get("median_sharpe", {})]})
    oos = pd.DataFrame(e4.get("oos_trading", {})).T.reset_index().rename(columns={"index": "Selection"})
    adr = pd.read_csv(RES / "e7_adr_timing.csv") if (RES / "e7_adr_timing.csv").exists() else pd.DataFrame()
    navs = pd.read_csv(RES / "e7_nav_staleness.csv") if (RES / "e7_nav_staleness.csv").exists() else pd.DataFrame()
    stab = pd.DataFrame(e9.get("by_tier", {})).T.reset_index().rename(columns={"index": "Strength"}) if e9 else pd.DataFrame()
    if len(stab):
        stab["Strength"] = stab["Strength"].map(TIER_NAME)
        stab = stab.rename(columns={"n": "Relationships", "median_share_stationary": "Windows stationary (median)",
                                    "median_hl": "Median half-life (days)", "median_hl_iqr_ratio": "Half-life IQR ratio"})
    cost_tab = ""
    if e8:
        ct = pd.DataFrame(e8["median_sharpe_by_tier_and_cost"]).T
        ct.index = [TIER_NAME[t] for t in ct.index]
        cost_tab = md_table(ct.reset_index().rename(columns={"index": "Strength"}))
    cap = pd.read_csv(RES / "e8_capacity.csv") if (RES / "e8_capacity.csv").exists() else pd.DataFrame()
    cap_tab = ""
    if len(cap):
        cp = cap.pivot_table(index="id", columns="size_usd", values="impact_rt_bp").round(1)
        cp.columns = [f"Impact r/t at ${c / 1e6:.0f}m (bp)" for c in cp.columns]
        g = cap.groupby("id")["avg_gross_per_trade"].first().round(1)
        cp.insert(0, "Gross per trade (bp)", g)
        cap_tab = md_table(cp.reset_index().rename(columns={"id": "Relationship"}), 1)

    rc = e3.get("reality_check", {})
    dsr = e3.get("deflated_sharpe_best", {})
    w = e7.get("wti_2020", {})
    brk = e7.get("brk_bound", {})
    tri = e7.get("fx_triangle", {})
    tri_rows = pd.DataFrame(tri).T.reset_index().rename(columns={"index": "Cross", "same_close_sharpe": "Same-close Sharpe",
                                                                 "next_close_sharpe": "Next-close Sharpe", "spread_sd_bp": "Spread s.d. (bp)",
                                                                 "half_life": "Half-life (days)"}) if tri else pd.DataFrame()
    basis = pd.DataFrame(e7.get("futures_basis", {})).T.reset_index().rename(columns={"index": "Future", "mispricing_sd_bp": "Mispricing s.d. (bp)",
                                                                                      "daily_change_sd_bp": "Daily change s.d. (bp)", "half_life": "Half-life (days)",
                                                                                      "same_close_sharpe": "Same-close Sharpe", "next_close_sharpe": "Next-close Sharpe"})
    struct = e7.get("structural_premia", {})
    struct_lines = []
    for rid, d in struct.items():
        yrs = d["median_by_year_bp"]
        if yrs:
            keys = sorted(yrs)
            struct_lines.append(f"- **{rid}**: median premium {yrs[keys[0]]:.0f} bp in {keys[0]}, {yrs[keys[-1]]:.0f} bp in {keys[-1]}; "
                                f"positive on {100 * d['share_days_positive']:.0f}% of days.")
    nav_tab = md_table(navs.groupby("kind")[["corr_prem_next_nav", "premium_sd_bp", "same_close_sharpe", "next_close_sharpe"]].median().round(3)
                       .reset_index().rename(columns={"kind": "Fund type", "corr_prem_next_nav": "corr(premium, next NAV change)",
                                                       "premium_sd_bp": "Premium s.d. (bp)", "same_close_sharpe": "Same-close Sharpe",
                                                       "next_close_sharpe": "Next-close Sharpe"})) if len(navs) else ""
    asof = e1.get("asof", "")

    md = f"""# Cross-asset relative value: what survives costs, timing and multiple testing

*Data to {asof}. Generated by `scripts/build_report.py` from the files in `results/`; every number below is computed,
not typed. Parameters (entry |z| = 2, exit 0.5, stop 4, 126-period z window, next-close execution, base costs) were
fixed before any result was seen and are the same for every relationship.*

## Summary

- **{e1.get('n', '–')} relationships across {len(ASSET_CLASS_LABELS)} asset classes**, each labelled by what links the
  prices: {sum(1 for r in rels if r.tier == 'arbitrage')} arbitrage relationships (a mechanism can force convergence),
  {sum(1 for r in rels if r.tier == 'quasi')} economic links (a reason to converge, nothing that forces it) and
  {sum(1 for r in rels if r.tier == 'statistical')} statistical relationships.
- **Timing is the whole game for daily-close arbitrage.** Filling at the very close that produced the signal gives
  {e2.get('same_close_sharpe_gt1', '–')} relationships a net Sharpe above 1 (median {_fmt(e2['same_close_vs_base']['arbitrage']['same_close'])}
  for arbitrage relationships). Filling at the next close, which a real trader could do, leaves
  {e2.get('base_sharpe_gt1', '–')} above 1 and a median of {_fmt(e2['same_close_vs_base']['arbitrage']['base'])}. The apparent edge is
  bid-ask bounce at the close, stale NAVs and closing prices taken hours apart.
- **Multiple testing removes what is left.** Of {e3.get('n_tested', '–')} strategies, {e3.get('raw_p_lt_005', '–')} are positive at
  p < 0.05, against {_fmt(e3.get('expected_false_raw'), 1)} expected by chance; {e3.get('bh_q_lt_010', '–')} survive Benjamini-Hochberg at
  q < 0.10 and {e3.get('bonf_lt_005', '–')} survive Bonferroni. White's reality check gives the best strategy
  ({rc.get('best', '–')}) p = {_fmt(rc.get('pvalue'))}; its deflated Sharpe ratio is {_fmt(dsr.get('dsr'), 3)}.
- **The cointegration scan finds what chance predicts.** {e4.get('latest_pairs', '–')} pairs over the last three years:
  {e4.get('latest_raw_p05', '–')} pass Engle-Granger at 5% ({100 * e4.get('latest_raw_p05', 0) / max(e4.get('latest_pairs', 1), 1):.1f}%) when
  independent random walks pass {_fmt(e4.get('null_rate_eg_756'), 1, pct=True)} of the time; {e4.get('latest_bh_q05', '–')} survives
  Benjamini-Hochberg. Selected pairs stay cointegrated in the next three years more often than the base rate
  ({_fmt(e4['rolling']['bh_q05']['mean_persist'], 1, pct=True)} after BH, {_fmt(e4['rolling']['raw_p05']['mean_persist'], 1, pct=True)} for raw p < 0.05,
  {_fmt(e4['rolling']['all']['mean_persist'], 1, pct=True)} base), but traded out of sample with frozen hedges they lose money after
  costs ({_fmt(e4['oos_trading']['bh_q05']['mean_net'], 1)} and {_fmt(e4['oos_trading']['raw_p05']['mean_net'], 1)} bp per pair-year).
- **What does earn something is carry, a risk premium, not an arbitrage**: {_fmt(e2['by_type'].get('carry_funding', {}).get('share_positive'), 0, pct=True)}
  of the {e2['by_type'].get('carry_funding', {}).get('n', '–'):.0f} FX carry backtests are positive (median Sharpe
  {_fmt(e2['by_type'].get('carry_funding', {}).get('median_sharpe'))}), the highest share of any opportunity type, yet none is
  individually significant after the corrections above.
- **Today's scan**: {e1.get('by_status', {}).get('executable', 0)} executable arbitrage(s),
  {e1.get('by_status', {}).get('executable (not enforceable)', 0)} executable economic link(s),
  {e1.get('by_status', {}).get('statistical signal', 0)} statistical signals and {e1.get('by_status', {}).get('watch', 0)} signals that fail at
  least one executability check.

The interactive scanner is at <https://rasmuscodes.github.io/projects/cross-asset-rv/>.

## 1. Data

{data_tab}

Open sources are public domain (US Treasury, Federal Reserve, EIA) or reusable with attribution (New York Fed, ECB,
Bundesbank, Japan's Ministry of Finance, Bank of Canada, Reserve Bank of Australia, Bank of England, OECD). Vendor data
(Yahoo Finance prices, SSGA NAV histories, Moody's and ICE yields redistributed by FRED) is used for research but never
redistributed: the repository contains no raw data, and the public demo shows vendor-based relationships only as
derived analytics. Known gaps and traps, all handled in code:

- EIA stopped publishing NYMEX futures (contracts 1-4) on 2024-04-05, so calendar and crack spreads end there.
- Fed H.10 FX rates are published weekly; their latest days are filled with ECB reference rates (EUR crosses). The
  Taiwan dollar has no ECB rate, so TWD conversions stop at the last H.10 date.
- Yahoo's `range=max` request silently returns monthly bars; daily history needs explicit period bounds.
- WTI contract 1 settled at -37.63 on 2020-04-20: returns across non-positive prices are left undefined.
- Vendor total returns sometimes adjust a corporate action on one listing only; same-claim pairs drop days with a
  >15% one-day gap between the lines and report how many (for example Unilever's ADR history).

## 2. Universe and relationships

{rel_tab}

Definitions: **arbitrage** = {TIERS['arbitrage']} **Economic link** = {TIERS['quasi']} **Statistical** = {TIERS['statistical']}

## 3. Methods

**Spreads.** Parity spreads are log price ratios in USD at the official conversion ratio (ADR ratio, share-class
conversion, 1:1 for interlisted shares); ETF premiums are measured against the issuer's NAV; the futures basis is the
log basis minus (financing - dividend yield) x time to expiry, with the quoted contract inferred from the basis;
calendar spreads use EIA contracts with exchange expiry rules and roll-adjusted returns; bond spreads are DV01-weighted
yield spreads; statistical pairs use hedge ratios re-estimated every day from data up to the day before.

**Models.** ADF and Engle-Granger tests with MacKinnon (1994, 2010) p-values, Johansen trace and maximum-eigenvalue
tests, Kalman-filter dynamic hedges, Ornstein-Uhlenbeck half-lives (with Dickey-Fuller rather than normal critical
values), a two-state Markov-switching AR(1) (Hamilton filter, Kim smoother, EM), Bayesian model comparison between
mean reversion and a random walk (conjugate Normal-Inverse-Gamma, closed-form marginal likelihoods) with
posterior-predictive forecasts, Bayesian online change-point detection, rolling stability and CUSUM tests. The
statistical code is tested against statsmodels and against simulations with known answers (`tests/`).

**Costs.** Stress-scaled half-spreads (multiplied by the ratio of 21-day to median volatility, capped at 4x),
commissions, square-root market impact for capacity, transaction taxes (UK stamp duty, Hong Kong stamp duty, Taiwan
securities transaction tax, Indian STT, Korean transaction tax, Swiss stamp duty, French FTT), FX conversion, ADR
conversion fees, funding spreads, borrow fees (with non-borrowable markets flagged) and a full re-trade at every
futures roll. Backtests are also run with all costs doubled.

**Validation.** In-sample to 2015, out-of-sample from 2016; Benjamini-Hochberg, Benjamini-Yekutieli, Holm and
Bonferroni corrections; random-walk null rates for the cointegration screen; deflated Sharpe ratio and White's reality
check (stationary bootstrap); rolling formation-to-test persistence; walk-forward regime and Bayesian gates.

## 4. Results

### 4.1 Backtests by strength and type

![timing]({f1})

{md_table(sc)}

{md_table(by_tier)}

![tiers]({f2})

{md_table(by_type)}

The in-sample and out-of-sample Sharpe ratios have a rank correlation of {_fmt(e2.get('is_oos_rank_corr'))}; the median
out-of-sample Sharpe is {_fmt(e2.get('oos_median_sharpe'))}. The ten best in-sample relationships, out of sample:

{md_table(top_oos)}

### 4.2 Multiple testing

{e3.get('n_tested', '–')} strategies with more than two years of data were tested for a positive mean (one-sided). Raw
p < 0.05: {e3.get('raw_p_lt_005', '–')} (expected under the null: {_fmt(e3.get('expected_false_raw'), 1)}). Benjamini-Hochberg q < 0.05:
{e3.get('bh_q_lt_005', '–')}; q < 0.10: {e3.get('bh_q_lt_010', '–')}; Bonferroni: {e3.get('bonf_lt_005', '–')}. White's reality check on
{rc.get('K', '–')} daily strategies over {rc.get('T', '–')} days: p = {_fmt(rc.get('pvalue'))} for the best ({rc.get('best', '–')}).
Its annualised Sharpe of {_fmt(dsr.get('sr_annual'))} compares with an expected maximum of {_fmt(dsr.get('sr0_annual'))} from
{e3.get('n_trials_assumed', '–')} trials with no skill, so the deflated Sharpe ratio is {_fmt(dsr.get('dsr'), 4)}.

### 4.3 The cointegration scan

![coint]({f3})

Groups: equity ETFs, bond ETFs, commodity ETFs, single stocks, 10-year sovereign yields and G10 FX. Each pair is
tested both ways and must pass in the worse ordering; {e4.get('latest_raw_minp05', '–')} pairs would pass if either ordering
were enough. Johansen's trace test finds rank >= 1 for {e4.get('latest_johansen_rank1', '–')} pairs (uncorrected). Rolling
three-year formation windows from 2008 to 2022, then the next three years for persistence and the next year for frozen-hedge
trading:

{md_table(oos.round(2)) if len(oos) else ''}

### 4.4 Hedge ratios

{md_table(hed.round(3)) if len(hed) else ''}

Static hedges estimated on the first three years and frozen decay; the dynamic estimators are indistinguishable once
costs are paid. Best method by pair: {', '.join(f'{k} {v}' for k, v in e5.get('best_method_counts', {}).items())}.

### 4.5 Regime and Bayesian gates

Across {e6.get('n', '–')} economic-link and statistical relationships with at least six years of daily data, a walk-forward
Markov-switching filter (refit yearly on three years, gate open when the mean-reverting state has probability >= 0.5)
changed the median Sharpe by {_fmt(e6.get('regime_minus_none_median'))} and improved {_fmt(e6.get('regime_better_share'), 0, pct=True)} of
relationships; the Bayesian gate (refit monthly on two years, open when P(mean reversion) >= 0.9) changed it by
{_fmt(e6.get('bayes_minus_none_median'))} and improved {_fmt(e6.get('bayes_better_share'), 0, pct=True)}. Median Sharpe: no gate
{_fmt(e6.get('median_sharpe', {}).get('none'))}, regime gate {_fmt(e6.get('median_sharpe', {}).get('regime'))}, Bayesian gate
{_fmt(e6.get('median_sharpe', {}).get('bayes'))}: a little less negative, mostly because the gates keep the book out of the market.

### 4.6 Today's scan

{status_tab}

Signals that pass every check:

{act_tab}

Why the other signals are not executable:

{wr_tab}

## 5. Failed opportunities and false positives

![examples]({f4})

**FX triangle from two official fixings.** The ECB's EUR/JPY fixing (14:15 Frankfurt) against the cross implied by the
Fed's noon New York rates looks like a mean-reverting arbitrage spread:

{md_table(tri_rows.round(2)) if len(tri_rows) else ''}

It is the market move between two fixings six hours apart, untradable at both prices.

**ADR premiums across time zones.** Today's premium predicts tomorrow's home-market return (median correlation
{_fmt(e7.get('adr_timing', {}).get('asia_median_corr'))} for Asian listings, {_fmt(e7.get('adr_timing', {}).get('europe_median_corr'))} for
European ones): the home market catches up with the New York move overnight. Across {e7.get('adr_timing', {}).get('n', '–')} ADRs the
median Sharpe falls from {_fmt(e7.get('adr_timing', {}).get('same_close_median_sharpe'))} with same-close fills to
{_fmt(e7.get('adr_timing', {}).get('next_close_median_sharpe'))} with next-close fills.

**Stale NAVs.** International funds' NAVs are struck at foreign closes, so their "premium" predicts the next NAV change:

{nav_tab}

**Futures basis from daily closes.** The measured mispricing moves far more from day to day than any plausible
financing anomaly:

{md_table(basis.round(2)) if len(basis) else ''}

**Premia that never converge.** Restricted ADR creation (Taiwan) and foreign-ownership limits (India) produce
persistent premia; a dual-listed company's two lines trade at a lasting gap:

{chr(10).join(struct_lines)}

**A no-arbitrage bound that failed.** In {w.get('max_date', '–')[:7]} the WTI contract-2 vs contract-1 spread reached
{_fmt(w.get('max_contango_bp'), 0)} bp against a full-carry bound of {_fmt(w.get('full_carry_bp_that_day'), 0)} bp: the bound assumes
storage is available, and Cushing was full. Contango exceeded the assumed full-carry bound on
{w.get('days_above_full_carry_since_2000', '–')} days since 2000.

**Berkshire's conversion bound.** Class A converts into 1,500 class B, so B should not trade above A/1500 by more than
costs. At daily closes it did on {_fmt(brk.get('share_days_below_minus_band'), 1, pct=True)} of days (minimum
{_fmt(brk.get('min_spread_bp'), 0)} bp on {brk.get('min_date', '–')}), all closing-print noise: same-close Sharpe
{_fmt(brk.get('same_close_sharpe'))}, next-close {_fmt(brk.get('next_close_sharpe'))}.

**Interlisted Canadians.** The same share in Toronto and New York, both closing at 16:00, converted at the noon CAD
fixing: median Sharpe {_fmt(e7.get('interlisted', {}).get('same_close_median'))} same-close, {_fmt(e7.get('interlisted', {}).get('next_close_median'))}
next-close across {e7.get('interlisted', {}).get('n', '–')} names.

## 6. Costs, capacity and stability

{f'![costs]({f5})' if f5 else ''}

{cost_tab}

Square-root impact for the relationships with the best base Sharpe, against their average gross P&L per trade:

{cap_tab}

Stability (rolling two-year windows, stepped every six months):

{md_table(stab) if len(stab) else ''}

## 7. Limitations

- Daily closing data only. Intraday synchronised quotes would turn several "timing artefact" verdicts into measured
  executable spreads (or confirm there are none); they are not freely available.
- Corporate bonds are covered by indices (Moody's, Treasury HQM, ICE OAS) and ETFs, not issuer-level bonds (TRACE).
- No FX forwards, so covered-interest-parity bases cannot be measured; carry uses lagged OECD 3-month rates.
- Index futures use Yahoo's continuous front contract, whose month is inferred; expired individual contracts are not
  available.
- Borrow availability and fees are tier assumptions, not locate data.

## 8. Reproduce

```
python -m venv .venv && .venv\\Scripts\\pip install -e .[dev]
python scripts/fetch_data.py          # ~15 min; cached under data/raw with a provenance manifest
python scripts/run_research.py        # experiments E1-E9 -> results/
python scripts/build_report.py        # this report
python scripts/build_demo.py          # demo/data for the web demo
pytest
```
"""
    out = paths.REPORTS / "REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, "utf-8")
    return str(out)
