# Methodology

## Relationship strength

| Tier | Definition | Examples |
|---|---|---|
| Arbitrage | A contractual or creation/redemption mechanism lets a trader force convergence, net of costs | ETF vs NAV, index futures vs cash (cash-and-carry), freely convertible ADRs, interlisted shares, Berkshire A -> B conversion, FX triangle (synchronous quotes) |
| Economic link | A reason to converge exists, but nothing forces it on any timetable | share classes without conversion, dual-listed companies, ADRs with conversion limits (Taiwan, India, Korea), same-index ETFs, calendar and crack spreads, location spreads |
| Statistical | Co-movement with no mechanism | sector and stock pairs, index pairs, curve and cross-market bond spreads, credit ETFs, FX carry |

A statistical relationship is never called an arbitrage, whatever its test statistics.

## Spreads (units in brackets)

- **Parity** (ADRs, interlisted, share classes, DLCs): `1e4 * ln(P_a / (ratio * P_b))` with both prices in USD at the
  H.10 noon rate [bp]. Fair value 0 for arbitrage tiers; trailing mean otherwise. The ratio is the security's terms
  (never estimated from prices, which would remove the premium being measured).
- **ETF vs NAV**: `1e4 * ln(close / NAV)` [bp]; no-arbitrage band = creation fee + basket half-spread + ETF half-spread.
- **Futures basis**: `1e4 * (ln(F/S) - (r - q) * tau)` [bp] with r = overnight rate, q = trailing dividend yield of
  the index ETF, tau from the inferred contract. The quoted contract of the continuous series is inferred per quarter
  by the roll date that best fits the basis; positions are flat across contract switches.
- **Calendar**: `1e4 * ln(C2 / C1)` [bp] from EIA contracts; seasonal fair value (same month, previous five years) for
  natural gas, gasoline and heating oil; returns roll-adjusted with exchange expiry rules; a full re-trade is charged
  at every roll. Full-carry bound for WTI: `ln(1 + r/12 + storage / C1)` with storage 0.40 $/bbl/month.
- **Bond spreads**: DV01-weighted yield spreads [bp of yield]; carry = (yield - local funding) / duration per day.
- **Statistical pairs**: residual `1e4 * (ln A - beta * ln B - alpha)` with (alpha, beta) estimated up to the day
  before [bp]; P&L = long $1 A, short $beta B.
- **FX carry**: rate differential (OECD 3-month, lagged a month) [bp/yr]; P&L = spot return + accrued differential.

## Signals and execution

z = (spread - fair) / trailing 126-period standard deviation of that deviation. Enter at |z| >= 2 (and outside the
no-arbitrage band where one exists), exit at |z| <= 0.5 or a sign change, stop at |z| >= 4 (then wait until |z| < 2),
time stop after 126 periods. Carry relationships hold the positive-carry side when carry / volatility >= 0.25.
Trades execute at the close after the signal. These parameters are fixed in advance and identical for every
relationship.

## Costs

Per leg and per one-way trade: stress-scaled half-spread (by liquidity tier) + commission + square-root impact (for
capacity analysis) + transaction tax on the relevant side + FX conversion for non-USD legs. Bond legs: yield
half-spread x DV01. Holding: funding spread on long cash legs, rebate spread and borrow fee on short legs (tiers:
general collateral 30 bp/yr, warm 100, hard 500, not borrowable). Hedge rebalancing when the hedge ratio moves.
Stress scaling multiplies spreads by clip(21-day vol / 252-day median vol, 1, 4).

## Executability verdict (today's scan)

A signal is **executable** only if: arbitrage tier, legs' closing prints within one hour, deviation outside the band,
expected convergence over two half-lives (capped at the time stop) exceeds round-trip costs after carry and funding,
the short leg is borrowable and the direction allowed, the spread shows mean reversion in its trailing window
(Dickey-Fuller p < 0.05 with Engle-Granger critical values when the hedge is estimated) unless an arbitrage mechanism
anchors it, its half-life exceeds one period, and the data are current. The same checks on an economic link give
"executable (not enforceable)"; on a statistical relationship, "statistical signal". Any failed check gives "watch"
with the reasons listed.

## Models

- **Unit roots and cointegration**: ADF with AIC lag choice on a common sample; Engle-Granger in both orderings (the
  larger p-value is used); Johansen trace and max-eigenvalue tests with an unrestricted constant; MacKinnon response
  surfaces for p-values and finite-sample critical values. Matched to statsmodels to 1e-6 in tests.
- **Kalman filter**: random-walk coefficients; state noise expressed in units of one observation's coefficient
  variance so `delta` is scale-free; one-step-ahead (prior) estimates are what positions use. The likelihood often
  rises with the state noise until the states absorb the spread's mean reversion, so the MLE can sit on the grid
  boundary (reported).
- **OU**: exact AR(1) discretisation; half-life ln 2 / theta; mean reversion judged with Dickey-Fuller critical
  values (a normal t-test flags about 40% of random walks at 5%; tested).
- **Regime switching**: two-state Markov-switching AR(1) by EM; the state with the smaller AR coefficient is the
  mean-reverting one; filtered probabilities (data up to t) gate trades; refit yearly in walk-forward use.
- **Bayesian**: closed-form marginal likelihoods of a mean-reverting AR(1) and a random walk with drift (Normal-
  Inverse-Gamma priors) give P(mean reversion); forecasts average both models by posterior probability. A flat-prior
  posterior on the AR coefficient alone reports ~0.99 "mean reversion" for random walks (Dickey-Fuller bias; tested),
  which is why it is not used. Online change-point detection (Adams-MacKay) flags breaks.
- **Stability**: rolling Engle-Granger and ADF significance, half-life dispersion, CUSUM of recursive residuals
  (Brown-Durbin-Evans 5% boundary), formation-to-test persistence.

## Validation

In-sample to 2015, out-of-sample from 2016 (no parameters are fitted, so this tests whether past winners keep
winning). Multiple testing across all strategies: BH / BY / Holm / Bonferroni on one-sided Sharpe p-values; deflated
Sharpe ratio for the best strategy given the number of trials; White's reality check with a stationary bootstrap. For
the cointegration screen: random-walk null rates at the same sample length, BH within each window, persistence into
the next window, frozen-hedge trading in the following year. Every backtest is repeated with doubled costs and with
same-close fills (to measure how much apparent edge is timing).
