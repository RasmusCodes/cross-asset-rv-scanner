# Cross-asset relative-value scanner

A platform that searches equity indices, ETFs, futures, ADRs and dual listings, FX, sovereign and corporate bonds,
commodities and funding markets for prices that look out of line with each other, and then asks the questions a
trader would: is there a mechanism that forces convergence, are the prices even synchronous, does the expected
convergence pay for spreads, taxes, borrow and funding, and does the signal survive out of sample and after
correcting for the number of things tried?

**[Interactive demo](https://rasmuscodes.github.io/projects/cross-asset-rv/)** ·
**[Research report](reports/REPORT.md)** · [Methodology](docs/METHODOLOGY.md) · [Data and licences](docs/DATA.md) ·
[Architecture](docs/ARCHITECTURE.md)

## What it found

Data to 20 November 2026; the same pre-set rule for every relationship (enter at |z| = 2, exit at 0.5, stop at 4,
next-close execution, conservative costs).

- **191 relationships in 9 asset classes**, each labelled by what links the prices: 74 **arbitrage** relationships
  (ETF creation/redemption, cash-and-carry, ADR conversion, cross-border settlement, share-class conversion),
  40 **economic links** (convergence likely, never forced) and 77 **statistical** relationships.
- **Same-close fills invent arbitrage.** Assuming you trade at the very close that produced the signal, 73
  relationships have a net Sharpe ratio above 1 (median 2.20 for arbitrage relationships). Trading at the next close,
  which is actually possible, leaves **none** above 1 and a median of -0.40. The apparent profits are bid-ask bounce,
  stale NAVs and closing prices taken hours apart: Asian ADR premiums predict the next day's home-market return
  (median correlation 0.55), international ETF "premiums" predict the next NAV change.
- **Multiple testing removes what is left.** 10 of 181 strategies are positive at p < 0.05, against 9 expected by
  chance; none survives Benjamini-Hochberg or Bonferroni, and White's reality check gives the best one p = 0.27.
- **The cointegration screen finds what chance predicts.** Over the last three years, 52 of 1,221 pairs (4.3%) pass
  Engle-Granger at 5%; independent random walks pass 5.5% of the time; one pair survives Benjamini-Hochberg. Selected
  pairs do stay cointegrated more often than the base rate (20.6% after BH vs 5.1%), but traded out of sample with
  frozen hedges they lose money after costs.
- **Carry is the one consistent earner**, and it is a risk premium, not an arbitrage: 12 of 13 FX carry backtests
  are positive (median Sharpe 0.29), none individually significant after correction.
- Kalman, rolling and 1:1 hedges are indistinguishable once costs are paid (a frozen hedge decays); regime-switching
  and Bayesian gates change the median Sharpe by less than 0.02.

The report keeps the failures and false positives: the FX triangle from two fixings six hours apart, WTI's April 2020
contango blowing through the full-carry bound, Berkshire's conversion bound "violated" by closing-print noise, TSMC's
ADR premium that never converges, and the vendor corporate-action adjustments that had to be repaired.

## What is in the repository

| Layer | Contents |
|---|---|
| Data (`src/rvscan/data`) | 12 source adapters, 381 instruments, cached downloads with a SHA-256 / licence manifest, unit normalisation, data checks, FX conversion, futures expiry calendars, bond arithmetic |
| Models (`src/rvscan/models`) | ADF, Engle-Granger, Johansen (MacKinnon p-values, matched to statsmodels), Kalman hedges, OU half-lives, Markov-switching AR, Bayesian mean-reversion vs random walk, change-point detection, stability tests, multiple-testing corrections, deflated Sharpe, reality check |
| Scanners (`src/rvscan/scanners`) | 13 opportunity builders sharing one `SpreadData` contract; the executability evaluator; the cointegration screen |
| Engine (`src/rvscan/engine`) | liquidity-aware costs (spreads, taxes, FX, impact), borrow and funding, z-score and carry rules, backtests with leg attribution, MAE, convergence episodes |
| Research (`src/rvscan/research`) | experiments E1-E9 and the generated report |
| Demo (`demo/`) | the static web app: scanner, trade simulator, open-data pair builder; the browser engine reproduces the Python backtests |

### Opportunity types

| Type | Relationships | Strength |
|---|---|---|
| ETF vs underlying basket (NAV) | 18 | arbitrage |
| ETFs on the same index | 8 | economic link |
| Futures basis (ES, NQ, RTY, YM vs cash) | 4 | arbitrage |
| Calendar spreads (WTI, gas, RBOB, heating oil) | 6 | economic link |
| ADR conversion parity | 45 | arbitrage / economic link (conversion limits) |
| Cross-listings: interlisted, share classes, dual-listed | 22 | arbitrage / economic link |
| FX-adjusted RV (fixing triangles, country ETFs vs indices) | 10 | arbitrage (non-synchronous) / statistical |
| Commodity curves and spreads (crack, location, roll) | 5 | economic link |
| Equity index RV | 7 | statistical |
| Sector, industry and stock pairs | 21 | statistical |
| Bond spreads (curve, flies, cross-market, breakevens, credit) | 26 | statistical |
| Carry and funding (FX carry, money-market spreads) | 19 | statistical / economic link |
| Cointegration screen | 1,221 pairs | statistical |

## Data

Rates, FX fixings, sovereign yields, money-market rates and energy prices come from public-domain or open sources
(US Treasury, Federal Reserve, New York Fed, ECB, Bundesbank, Japan's Ministry of Finance, Bank of Canada, RBA,
Bank of England, EIA, OECD). Equities, ETFs, ADRs, futures and indices come from Yahoo Finance; ETF NAVs from State
Street. **No price data is committed to this repository**, and the public demo shows vendor-based relationships only as
derived analytics (spreads, hedged-position P&L, costs, results), never the instruments' own prices.

## Quick start

```
python -m venv .venv
.venv\Scripts\pip install -e .[dev]        # or: pip install -r requirements.txt && pip install -e .
python scripts/fetch_data.py               # ~15 minutes; cached under data/raw
python scripts/run_research.py             # experiments E1-E9 -> results/  (E6 takes ~20 minutes)
python scripts/build_report.py             # reports/REPORT.md
python scripts/build_demo.py               # demo/data, then: python -m http.server -d demo 8840
pytest                                     # 41 tests; data-dependent ones skip without data/
```

```python
from rvscan.scanners import catalog, evaluate
from rvscan.engine import backtest

sd = catalog.build(catalog.by_id()["ADR_TSM"])     # TSMC ADR vs 5 Taipei shares
print(evaluate.evaluate(sd)["status"])            # verdict with reasons
print(backtest.run(sd).metrics["sharpe"])         # next-close execution, base costs
```

## Tests

`tests/test_models.py` checks the statistical code against statsmodels (ADF, Engle-Granger, Johansen, MacKinnon
p-values, multiple-testing corrections) and against simulations with known answers (OU half-life, regime recovery,
Bayes factor honesty on random walks, change-point detection, CUSUM, reality check). `tests/test_engine.py` covers
no look-ahead, cost accounting, bands, direction limits, roll costs, attribution identities, futures expiry rules and
bond arithmetic. `tests/test_scanners.py` checks configuration integrity, the cointegration screen, the demo licence
policy, and (with data) that every relationship builds, backtests and passes ADR-ratio plausibility.

## License

MIT. Nothing here is investment advice.
