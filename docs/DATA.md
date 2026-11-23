# Data sources and licences

Every instrument's source and licence is declared in `config/universe.yaml`. Three licence tags:

| Tag | Meaning | Sources |
|---|---|---|
| `public-domain` | US federal government work | US Treasury (par and real yield curves, HQM corporate curve via FRED), Federal Reserve (H.10 FX, H.15 rates, commercial paper via FRED), EIA (energy spot and NYMEX contracts 1-4) |
| `open-attribution` | reuse permitted with attribution | New York Fed (SOFR, TGCR, BGCR, EFFR), ECB (euro-area curves, €STR, reference rates, convergence yields), Deutsche Bundesbank (Bund curve), Ministry of Finance Japan (JGB), Bank of Canada (benchmark yields, CORRA), Reserve Bank of Australia (government yields), Bank of England (gilt curve, SONIA), OECD (3-month rates via FRED) |
| `vendor` | third-party data; not redistributed | Yahoo Finance (equities, ETFs, ADRs and home listings, indices, futures), State Street (ETF NAV histories), Moody's and ICE BofA yields/spreads as redistributed by FRED |

**What is in this repository:** code, configuration, tests, the report and its figures, and summary tables of
results (statistics per relationship). **What is not:** any raw or processed price data (`data/` is git-ignored).
`python scripts/fetch_data.py` rebuilds everything; downloads are cached with a manifest
(`data/raw/manifest.json`: URL, SHA-256, size, licence, fetch time).

**Public demo policy:** the web demo publishes row-level series only for `public-domain` and `open-attribution`
instruments. Relationships with any vendor input are published as derived analytics (spread, hedged-position P&L,
hedge ratios, costs, backtest results); their legs' own prices or returns are never written.
`rvscan.demo.export.assert_policy` re-reads every exported file and fails if a vendor instrument's series appears.

## Source notes (checked November 2026)

- **Yahoo chart API**: `range=max` silently returns monthly bars; use `period1=0&period2=<now>` for daily history.
  `close` is split-adjusted, `adjclose` also dividend-adjusted; the latest bar's `adjclose` can be empty (carried
  forward). Timestamps are session starts in UTC: convert to the exchange's time zone before taking the date.
  London quotes are in pence (`GBp`) and are converted to pounds. Expired futures contracts return 404, so only the
  current curve is available contract by contract.
- **EIA**: NYMEX futures contracts 1-4 end on 2024-04-05 (EIA stopped publishing them); spot series continue.
  Natural-gas series live under `/dnav/ng/`, petroleum under `/dnav/pet/`. WTI contract 1 settled at -37.63 on
  2020-04-20.
- **Fed H.10** (via FRED): noon New York buying rates, published weekly. Mixed quote conventions (USD per EUR / GBP /
  AUD / NZD, currency per USD for the rest) are normalised to USD per unit. The latest days are filled from ECB
  reference rates; TWD has no ECB rate.
- **ECB**: data-api.ecb.europa.eu SDMX CSV. EONIA ended in January 2022 (€STR from October 2019). Country 10-year
  yields (IRS dataset) are monthly averages.
- **Bundesbank**: daily Svensson term-structure yields for listed Federal securities (residual maturity 1-30 years).
- **Ministry of Finance Japan**: `jgbcme_all.csv` (history) plus `jgbcme.csv` (current month), Shift-JIS encoded.
- **RBA**: `f2-data.csv` is daily; `f2.1-data.csv` is monthly.
- **Bank of England**: IADB CSV endpoint for nominal zero-coupon gilt yields (5, 10, 20 years).
- **NY Fed**: rates API search endpoint; SOFR/TGCR/BGCR from 2018-04-02. EFFR history comes from FRED.
- **OECD 3-month rates** (FRED `IR3TIB01*`): monthly averages; used with a one-month publication lag. India and
  Brazil series are not available; euro-area and UK series end in early 2026 (overnight rates take over).
- **SSGA**: `navhist-us-en-<ticker>.xlsx` gives NAV, shares outstanding and net assets; the NAV is not split-adjusted
  (the builder applies Yahoo's split events). iShares' equivalent download now returns an HTML page.
