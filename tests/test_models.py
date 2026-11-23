"""Model library tests: own implementations against statsmodels and against simulated truths."""
import numpy as np
import pytest

from rvscan.models import bayes, cointegration as ci, kalman, multiple_testing as mt, ou, regime, stability

sm_tools = pytest.importorskip("statsmodels.tsa.stattools")
from statsmodels.stats.multitest import multipletests  # noqa: E402
from statsmodels.tsa.vector_ar.vecm import coint_johansen  # noqa: E402


def ar1(n, phi, sd=1.0, seed=0, c=0.0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = c + phi * x[t - 1] + sd * rng.standard_normal()
    return x


@pytest.mark.parametrize("phi,seed", [(0.9, 1), (0.98, 2), (1.0, 3), (0.5, 4)])
def test_adf_matches_statsmodels(phi, seed):
    x = ar1(800, phi, seed=seed) + 5.0
    mine = ci.adf(x, regression="c", autolag="aic")
    ref = sm_tools.adfuller(x, regression="c", autolag="AIC")
    assert mine.lags == ref[2]
    assert mine.stat == pytest.approx(ref[0], rel=1e-6, abs=1e-8)
    assert mine.pvalue == pytest.approx(ref[1], rel=1e-6, abs=1e-8)
    for lvl, key in ((0.01, "1%"), (0.05, "5%"), (0.10, "10%")):
        assert mine.crit[lvl] == pytest.approx(ref[4][key], rel=1e-6)


def test_mackinnon_p_all_regressions_match_statsmodels():
    from statsmodels.tsa.adfvalues import mackinnonp
    for reg in ("n", "c", "ct"):
        for n_vars in range(1, 7):
            for tau in np.linspace(-6, 1.5, 31):
                assert ci.mackinnon_p(tau, n_vars, reg) == pytest.approx(mackinnonp(tau, reg, n_vars), abs=1e-10)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_engle_granger_matches_statsmodels_coint(seed):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(1000))
    y = 0.7 * x + ar1(1000, 0.95, seed=seed + 10) + 3.0
    mine = ci.engle_granger(y, x)
    stat, p, _ = sm_tools.coint(y, x, trend="c", autolag="aic")
    assert mine.stat == pytest.approx(stat, rel=1e-6)
    assert mine.pvalue == pytest.approx(p, rel=1e-6, abs=1e-10)
    assert mine.beta[0] == pytest.approx(0.7, abs=0.15)   # AR(0.95) noise: finite-sample bias is sizeable


def test_engle_granger_independent_walks_not_significant_on_average():
    rate = mt.eg_null_rate(500, n_pairs=200, seed=7)
    assert 0.0 <= rate <= 0.12   # nominal 5%; finite-sample rate is close to it


def test_johansen_matches_statsmodels():
    rng = np.random.default_rng(3)
    common = np.cumsum(rng.standard_normal(900))
    data = np.column_stack([common + ar1(900, 0.8, seed=1), 0.5 * common + ar1(900, 0.7, seed=2),
                            np.cumsum(rng.standard_normal(900))])
    mine = ci.johansen(data, k_ar_diff=1)
    ref = coint_johansen(data, det_order=0, k_ar_diff=1)
    np.testing.assert_allclose(mine.eigenvalues, ref.eig, rtol=1e-6)
    np.testing.assert_allclose(mine.trace, ref.lr1, rtol=1e-6)
    np.testing.assert_allclose(mine.max_eig, ref.lr2, rtol=1e-6)
    np.testing.assert_allclose(mine.trace_crit, ref.cvt, rtol=1e-6)
    assert mine.rank_trace_95 >= 1


def test_kalman_tiny_delta_converges_to_ols_and_tracks_drift():
    rng = np.random.default_rng(5)
    n = 1500
    x = np.cumsum(rng.standard_normal(n)) + 50
    y = 2.0 + 1.5 * x + rng.standard_normal(n) * 0.5
    res = kalman.kalman_regression(y, x, delta=1e-9)
    beta_ols = np.polyfit(x, y, 1)[0]
    assert res.beta[-1, 0] == pytest.approx(beta_ols, abs=0.02)
    # a drifting beta is tracked with a larger delta
    beta_true = np.linspace(1.0, 2.0, n)
    y2 = beta_true * x + rng.standard_normal(n) * 0.5
    fast = kalman.kalman_regression(y2, x, delta=1e-3)
    err = np.abs(fast.beta[300:, 0] - beta_true[300:]).mean()
    static = np.polyfit(x, y2, 1)[0]
    static_err = np.abs(static - beta_true[300:]).mean()
    assert err < 0.4 * static_err   # intercept and slope trade off, so tracking is good but not exact
    # prior (tradable) estimate never uses y_t: changing y_t leaves beta_prior[t] unchanged
    y3 = y2.copy()
    y3[1000] += 100
    alt = kalman.kalman_regression(y3, x, delta=1e-3)
    assert alt.beta_prior[1000, 0] == pytest.approx(fast.beta_prior[1000, 0])
    assert alt.beta[1000, 0] != pytest.approx(fast.beta[1000, 0])


def test_rolling_ols_matches_direct():
    rng = np.random.default_rng(6)
    x = rng.standard_normal(300).cumsum()
    y = 0.5 * x + rng.standard_normal(300)
    a, b = kalman.rolling_ols_beta(y, x, window=60)
    bb = np.polyfit(x[200:260], y[200:260], 1)
    assert b[259] == pytest.approx(bb[0], rel=1e-9)
    assert a[259] == pytest.approx(bb[1], rel=1e-9)


def test_ou_recovers_half_life():
    theta = np.log(2) / (20 / 252)            # 20-day half-life
    b = np.exp(-theta / 252)
    s = ar1(5000, b, sd=0.01, seed=8, c=0.5 * (1 - b))
    fit = ou.fit_ou(s)
    assert fit.mean_reverting
    assert fit.half_life_days == pytest.approx(20, rel=0.25)
    assert fit.mu == pytest.approx(0.5, abs=0.01)
    lo, hi = fit.half_life_ci
    assert lo < fit.half_life_days < hi
    assert fit.expected(fit.mu + 1, fit.half_life_days) == pytest.approx(fit.mu + 0.5, rel=1e-6)


def test_ou_random_walk_not_mean_reverting_mostly():
    flags = [ou.fit_ou(np.cumsum(np.random.default_rng(s).standard_normal(500))).mean_reverting for s in range(40)]
    assert np.mean(flags) < 0.25


def test_msar_recovers_regimes():
    rng = np.random.default_rng(9)
    n = 3000
    states = np.zeros(n, dtype=int)
    for t in range(1, n):
        stay = 0.99 if states[t - 1] == 0 else 0.98
        states[t] = states[t - 1] if rng.random() < stay else 1 - states[t - 1]
    s = np.zeros(n)
    for t in range(1, n):
        if states[t] == 0:
            s[t] = 0.8 * s[t - 1] + 0.5 * rng.standard_normal()
        else:
            s[t] = 0.999 * s[t - 1] + 1.5 * rng.standard_normal()
    fit = regime.fit_msar(s)
    assert fit.phi[0] < fit.phi[1]
    acc = np.mean((fit.smoothed[1:, 0] > 0.5) == (states[1:] == 0))
    assert acc > 0.85
    oos = regime.filter_with_params(s, fit)
    assert np.nanmax(np.abs(oos[1:] - fit.filtered[1:])) < 1e-8


def test_bayes_model_probability_and_forecast():
    b = 0.95
    s = ar1(2000, b, sd=0.1, seed=11, c=0.2 * (1 - b))
    model = bayes.fit_spread_model(s)
    X = np.column_stack([np.ones(len(s) - 1), s[:-1]])
    ols_coef = np.linalg.lstsq(X, s[1:], rcond=None)[0]
    assert model.b_mean == pytest.approx(ols_coef[1], abs=0.01)
    assert model.p_mean_reverting > 0.99
    fc = bayes.forecast(model, s0=model.centre + 0.5, horizon=60, target=model.centre + 0.1,
                        stop=model.centre + 1.0, n_paths=3000)
    analytic = model.centre + 0.5 * model.b_mean ** 30
    assert fc.mean[29] == pytest.approx(analytic, abs=0.03)
    assert 0.5 < fc.prob_hit <= 1.0


def test_bayes_factor_is_honest_about_random_walks():
    flat, bf = [], []
    for seed in range(30):
        rw = np.cumsum(np.random.default_rng(100 + seed).standard_normal(500))
        flat.append(bayes.flat_posterior_prob_b_below_one(rw))
        bf.append(bayes.fit_spread_model(rw).p_mean_reverting)
    # The flat-prior posterior "finds" mean reversion in random walks; the Bayes factor mostly does not.
    assert np.median(flat) > 0.85
    assert np.median(bf) < 0.5
    assert np.mean(np.array(bf) > 0.95) < 0.15


def test_bocpd_detects_mean_shift():
    rng = np.random.default_rng(13)
    x = np.r_[rng.standard_normal(400), 3 + rng.standard_normal(400)]
    out = bayes.bocpd(x, hazard=1 / 200)
    assert out["p_change"][400:410].max() > 0.5
    assert out["map_run"][-1] < 420


def test_multiple_testing_matches_statsmodels():
    p = np.random.default_rng(14).uniform(size=60) ** 3
    for mine, name in ((mt.benjamini_hochberg, "fdr_bh"), (mt.holm, "holm"), (mt.bonferroni, "bonferroni"),
                       (mt.benjamini_yekutieli, "fdr_by")):
        np.testing.assert_allclose(mine(p), multipletests(p, method=name)[1], rtol=1e-12)


def test_deflated_sharpe_properties():
    r = np.random.default_rng(15).normal(0.0005, 0.01, 1500)
    one = mt.deflated_sharpe(r, n_trials=1, var_sr_trials=0.001)
    many = mt.deflated_sharpe(r, n_trials=500, var_sr_trials=0.001)
    assert many["dsr"] < one["dsr"] <= one["psr"] + 1e-12
    assert mt.expected_max_sharpe(1000, 1e-3) > mt.expected_max_sharpe(10, 1e-3) > 0


def test_reality_check_rejects_only_real_edge():
    rng = np.random.default_rng(16)
    noise = rng.normal(0, 0.01, (1000, 20))
    assert mt.reality_check(noise, n_boot=300)["pvalue"] > 0.05
    edge = noise.copy()
    edge[:, 3] += 0.003
    out = mt.reality_check(edge, n_boot=300)
    assert out["best"] == 3 and out["pvalue"] < 0.05


def test_cusum_flags_break():
    rng = np.random.default_rng(17)
    x = np.cumsum(rng.standard_normal(1000))
    stable = 1.0 * x + rng.standard_normal(1000)
    broken = np.where(np.arange(1000) < 600, 1.0 * x, 1.0 * x + 8) + rng.standard_normal(1000)
    assert not stability.cusum(stable, x).crossed
    res = stability.cusum(broken, x)
    assert res.crossed and res.first_cross > 600
