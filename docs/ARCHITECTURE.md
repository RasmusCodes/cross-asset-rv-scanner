# Architecture

```
config/                     universe.yaml (381 instruments, venues, sources + licences)
                            relationships.yaml (curated relationships, compact lists)
                            costs.yaml (spreads, taxes, borrow, funding, stress scaling)
src/rvscan/
  data/      cache.py       cached HTTP fetcher; raw files + provenance manifest (URL, SHA-256, licence, time)
             sources.py     one adapter per source (Yahoo, FRED, Treasury, EIA, ECB, Bundesbank, MoF, BoC, RBA,
                            BoE, NY Fed, SSGA NAVs)
             universe.py    typed Instrument / Venue view of universe.yaml; closing-time gaps between legs
             build.py       fetch + normalise (pence -> pounds, FX -> USD per unit) + data checks -> processed store
             store.py       per-instrument parquet store, catalog.json, NAVs, events, live futures curves
             fx.py          H.10 noon rates, extended with ECB reference rates over H.10's publication lag
             calendars.py   NYMEX / CME expiry rules, roll days, roll-adjusted returns
             bonds.py       par-bond duration, convexity, constant-maturity returns, carry
  models/    cointegration.py  ADF, Engle-Granger, Johansen, KPSS, MacKinnon p-values
             kalman.py      dynamic regression (scale-free state noise), MLE grid, rolling OLS
             ou.py          OU fit, half-life + intervals, Dickey-Fuller mean-reversion test, first passage
             regime.py      2-state Markov-switching AR(1): Hamilton filter, Kim smoother, EM
             bayes.py       mean reversion vs random walk (NIG marginal likelihoods), predictive forecasts, BOCPD
             stability.py   rolling Engle-Granger, CUSUM of recursive residuals, formation -> test persistence
             multiple_testing.py  Bonferroni / Holm / BH / BY, null rates, PSR / DSR, reality check
  scanners/  base.py        Relationship, SpreadData (the contract every type fills), funding and cost helpers
             builders_arb.py   ETF vs NAV, same-index ETFs, futures basis, ADR parity, interlisted, share classes,
                               dual-listed companies, FX triangle, FX-adjusted index tracking
             builders_macro.py calendar spreads, crack / location / roll spreads, bond spreads, credit ETFs,
                               funding spreads, FX carry
             builders_stat.py  statistical pairs with rolling / Kalman / static / 1:1 hedges
             catalog.py     expands relationships.yaml; dispatches builders
             evaluate.py    signal, models, expected convergence, costs, carry, liquidity, borrow, FX, basis risk,
                            historical convergence and the executability verdict
             stat_scan.py   pairwise cointegration scan with multiple-testing control and frozen-hedge OOS trading
  engine/    costs.py       leg costs (spreads, commission, taxes, FX, sqrt impact), borrow, stress multiplier
             backtest.py    positions (z-score / carry rules), backtest with costs, trades, metrics, episodes
  research/  experiments.py E1-E9 (results/)
             report.py      reports/REPORT.md + figures, all numbers read from results/
  demo/      export.py      demo JSON with the licence policy enforced and re-checked
scripts/                    fetch_data.py, run_research.py, build_report.py, build_demo.py
demo/                       static web app (index.html, style.css, js/stats.js, js/chart.js, js/app.js)
tests/                      model, engine and scanner tests
```

## Data flow

1. `fetch_data.py` downloads every instrument once into `data/raw/<source>/` (manifest with SHA-256 and licence),
   parses it, normalises units and writes `data/processed/instruments/<id>.parquet` plus `catalog.json` with data
   checks (coverage, largest move, longest flat run, zero-volume share).
2. A relationship in `relationships.yaml` names its legs and type. `catalog.build()` calls the type's builder, which
   returns `SpreadData`: the spread, its structural fair value and no-arbitrage band (if a mechanism defines one), the
   daily P&L of a one-unit position, directional carry, holding costs, stress-scaled round-trip costs, dates that
   force the position flat (futures contract switches) or cost a re-trade (rolls), the timing gap between the legs'
   closing prints and the licence of its inputs.
3. `evaluate()` turns `SpreadData` into today's opportunity assessment; `backtest.run()` into a history of trades.
   Both use the same contract, so every opportunity type gets the same signal, cost and risk machinery.
4. `run_research.py` runs E1-E9; `build_report.py` writes the report; `build_demo.py` exports the web demo's data.

## Conventions

- Position +1 = long the spread (gains when the spread rises). P&L, carry and costs are in the same unit per unit of
  position: bp of one leg's notional for price spreads, bp of yield per DV01 unit for bond spreads.
- Next-close execution: a signal from the close of day t is traded at the close of day t+1.
- Every estimate used for a position (hedge ratio, fair value, z-score scale, regime probability) uses data up to the
  day before the position is held.
