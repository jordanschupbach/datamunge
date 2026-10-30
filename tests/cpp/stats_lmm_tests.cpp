#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/stats/lm.hpp>
#include <datamunge/stats/lmm.hpp>

#include <cmath>
#include <random>
#include <sstream>

using datamunge::dstruct::DataFrame;
using datamunge::stats::LM;
using datamunge::stats::LMM;
using datamunge::stats::LMMOptions;

// Reference values for the "oracle" tests below were computed with an independent,
// from-scratch numpy implementation (dense n x n matrices, no block-diagonal shortcut,
// grid + golden-section search over the variance-component parameter -- deliberately a
// different numerical approach than this engine's block-diagonal-per-group Cholesky +
// DifferentialEvolution search) since neither R's lme4/nlme nor Python's statsmodels are
// available in this environment. See scratchpad/lmm_oracle/{gen_data,oracle}.py.

namespace {

DataFrame make_oracle_dataframe() {
    const std::vector<double> group{0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                     3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5};
    const std::vector<double> x{-1.090656, -0.732967, 1.189462,  0.705019,  -0.435562, -0.668744, 0.393235,
                                 -1.253063, 0.691024,  1.767211,  -1.007017, 1.795525,  0.668950,  -1.616408,
                                 -0.232641, 1.545920,  0.789814,  -0.694109, 0.935713,  -1.119460, -1.673622,
                                 -1.360418, -0.639599, -0.139227, -0.934316, 1.263106,  -1.226822, -1.482124,
                                 -1.633341, 0.394272};
    const std::vector<double> y{2.312248,  1.187700,  2.155728,  3.254406,  2.397287,  1.828553,  4.078931,
                                 1.398947,  2.981028,  2.799915,  -0.320443, 3.132711,  0.945518,  -0.872998,
                                 0.596755,  1.562953,  1.210838,  0.030168,  1.415654,  -1.641852, -1.233617,
                                 -2.885786, -0.439631, -1.567756, -1.013474, 3.634390,  0.753544,  1.061874,
                                 -0.797234, 1.608015};
    DataFrame df;
    df.add_column("group", group);
    df.add_column("x", x);
    df.add_column("y", y);
    return df;
}

} // namespace

TEST(LMM, MatchesIndependentNumpyOracleUnderReml) {
    const auto df = make_oracle_dataframe();
    LMMOptions options;
    options.reml = true;
    LMM model(df, "y ~ x + (1 | group)", options);

    ASSERT_EQ(model.coefficients().size(), 2u);
    EXPECT_NEAR(model.coefficients()[0], 1.15891866, 1e-2);
    EXPECT_NEAR(model.coefficients()[1], 0.89524613, 1e-2);
    EXPECT_NEAR(model.residual_variance(), 0.600567, 5e-2);
    EXPECT_NEAR(model.random_effect_std_devs()[0] * model.random_effect_std_devs()[0], 1.354593, 1e-1);
    EXPECT_NEAR(model.deviance(), 84.541475, 5e-2);
}

TEST(LMM, MatchesIndependentNumpyOracleUnderMl) {
    const auto df = make_oracle_dataframe();
    LMMOptions options;
    options.reml = false;
    LMM model(df, "y ~ x + (1 | group)", options);

    ASSERT_EQ(model.coefficients().size(), 2u);
    EXPECT_NEAR(model.coefficients()[0], 1.15942435, 1e-2);
    EXPECT_NEAR(model.coefficients()[1], 0.89786139, 1e-2);
    EXPECT_NEAR(model.residual_variance(), 0.576030, 5e-2);
    EXPECT_NEAR(model.deviance(), 82.768487, 5e-2);
}

TEST(LMM, ReducesToOlsWhenTrueGroupEffectIsZero) {
    std::mt19937_64 rng(7);
    std::uniform_int_distribution<int> group_dist(0, 19);
    std::normal_distribution<double> noise(0.0, 0.5);
    std::uniform_real_distribution<double> x_dist(-3.0, 3.0);

    std::vector<double> group(300), x(300), y(300);
    for (int i = 0; i < 300; ++i) {
        group[i] = static_cast<double>(group_dist(rng));
        x[i] = x_dist(rng);
        y[i] = 2.0 + 3.0 * x[i] + noise(rng); // no group-level shift at all
    }
    DataFrame df;
    df.add_column("group", group);
    df.add_column("x", x);
    df.add_column("y", y);

    const LM ols(df, "y ~ x");
    const LMM mixed(df, "y ~ x + (1 | group)");

    EXPECT_NEAR(mixed.coefficients()[0], ols.coefficients()[0], 0.15);
    EXPECT_NEAR(mixed.coefficients()[1], ols.coefficients()[1], 0.05);
    // The estimated between-group SD should be small relative to the residual SD.
    EXPECT_LT(mixed.random_effect_std_devs()[0], 0.3 * mixed.residual_std_dev());
}

TEST(LMM, RecoversKnownSimulatedGroundTruth) {
    std::mt19937_64 rng(99);
    constexpr int n_groups = 60;
    std::uniform_int_distribution<int> group_dist(0, n_groups - 1);
    std::normal_distribution<double> group_effect(0.0, 2.0); // true random-intercept SD = 2
    std::normal_distribution<double> noise(0.0, 1.0);        // true residual SD = 1
    std::uniform_real_distribution<double> x_dist(-2.0, 2.0);

    std::vector<double> group_effects(n_groups);
    for (auto& g : group_effects) g = group_effect(rng);

    const double true_intercept = -1.0, true_slope = 2.5;
    constexpr int n = n_groups * 20;
    std::vector<double> group(n), x(n), y(n);
    for (int i = 0; i < n; ++i) {
        const int g = group_dist(rng);
        group[i] = static_cast<double>(g);
        x[i] = x_dist(rng);
        y[i] = true_intercept + true_slope * x[i] + group_effects[g] + noise(rng);
    }
    DataFrame df;
    df.add_column("group", group);
    df.add_column("x", x);
    df.add_column("y", y);

    const LMM model(df, "y ~ x + (1 | group)");
    // With a true between-group SD of 2 spread over n_groups groups, the intercept's
    // sampling SE is on the order of 2/sqrt(n_groups) ~= 0.26; allow a generous ~3 SE.
    EXPECT_NEAR(model.coefficients()[0], true_intercept, 0.8);
    EXPECT_NEAR(model.coefficients()[1], true_slope, 0.15);
    EXPECT_NEAR(model.random_effect_std_devs()[0], 2.0, 0.4);
    EXPECT_NEAR(model.residual_std_dev(), 1.0, 0.15);
    EXPECT_EQ(model.num_groups(), static_cast<std::size_t>(n_groups));
}

TEST(LMM, RandomInterceptAndSlopeFitsWithoutError) {
    std::mt19937_64 rng(3);
    std::uniform_int_distribution<int> group_dist(0, 14);
    std::normal_distribution<double> noise(0.0, 1.0);
    std::uniform_real_distribution<double> x_dist(-2.0, 2.0);

    std::vector<double> group(400), x(400), y(400);
    for (int i = 0; i < 400; ++i) {
        const int g = group_dist(rng);
        group[i] = static_cast<double>(g);
        x[i] = x_dist(rng);
        y[i] = 1.0 + 0.5 * x[i] + 0.3 * static_cast<double>(g) * 0.1 * x[i] + noise(rng);
    }
    DataFrame df;
    df.add_column("group", group);
    df.add_column("x", x);
    df.add_column("y", y);

    const LMM model(df, "y ~ x + (1 + x | group)");
    ASSERT_EQ(model.random_effect_names().size(), 2u);
    EXPECT_EQ(model.random_effect_names()[0], "(Intercept)");
    EXPECT_EQ(model.random_effect_names()[1], "x");
    EXPECT_EQ(model.num_groups(), 15u);
    EXPECT_GT(model.residual_std_dev(), 0.0);
    const auto text = model.summary();
    EXPECT_NE(text.find("Random effects"), std::string::npos);
    EXPECT_NE(text.find("Fixed effects"), std::string::npos);
}

TEST(LMM, RejectsFormulaWithoutRandomEffectsTerm) {
    const auto df = make_oracle_dataframe();
    EXPECT_THROW(LMM(df, "y ~ x"), std::invalid_argument);
}

TEST(LMM, RejectsFormulaWithTwoRandomEffectsTerms) {
    const auto df = make_oracle_dataframe();
    EXPECT_THROW(LMM(df, "y ~ x + (1 | group) + (1 | x)"), std::invalid_argument);
}

TEST(LMM, RejectsUnknownGroupingColumn) {
    const auto df = make_oracle_dataframe();
    EXPECT_THROW(LMM(df, "y ~ x + (1 | nonexistent)"), std::invalid_argument);
}

TEST(LMM, RejectsNoIntercptAndNoSlopes) {
    const auto df = make_oracle_dataframe();
    EXPECT_THROW(LMM(df, "y ~ x + (0 | group)"), std::invalid_argument);
}

TEST(LMM, PredictFallsBackToFixedEffectsOnlyForUnseenGroup) {
    const auto df = make_oracle_dataframe();
    const LMM model(df, "y ~ x + (1 | group)");

    DataFrame newdata;
    newdata.add_column("x", std::vector<double>{0.0});
    newdata.add_column("group", std::vector<double>{999.0}); // never seen while fitting
    const auto preds = model.predict(newdata);
    ASSERT_EQ(preds.size(), 1u);
    EXPECT_NEAR(preds[0], model.coefficients()[0], 1e-9); // x=0 -> intercept only, no BLUP shift
}

TEST(LMM, PredictAppliesBlupForKnownGroup) {
    const auto df = make_oracle_dataframe();
    const LMM model(df, "y ~ x + (1 | group)");

    DataFrame newdata;
    newdata.add_column("x", std::vector<double>{0.0});
    newdata.add_column("group", std::vector<double>{0.0}); // seen while fitting
    const auto preds = model.predict(newdata);
    const double expected = model.coefficients()[0] + model.random_effects()[0][0];
    ASSERT_EQ(model.group_labels()[0], "0");
    EXPECT_NEAR(preds[0], expected, 1e-9);
}

// Reference values from R's nlme 3.1-168: lme(distance ~ age * sex, random = ~ age | subject,
// data = Orthodont), with method = "REML" (default) and "ML". This is the case that exposed a
// transposed-Lambda bug (the likelihood used Lambda' Lambda, Sigma and the BLUPs Lambda Lambda'),
// so it checks the variance components, correlation and BLUPs, not just the likelihood.
TEST(LMM, RandomSlopeMatchesNlmeOnOrthodont) {
    const auto growth = datamunge::datasets::orthodont();
    const LMM reml(growth, "distance ~ age * sex + (1 + age | subject)");
    const std::vector<double> beta{17.3727272727, 0.4795454545, -1.0321022727, 0.3048295455};
    const std::vector<double> se{1.2283958, 0.1037193, 1.5957329, 0.1347353};
    for (std::size_t j = 0; j < 4; ++j) {
        EXPECT_NEAR(reml.coefficients()[j], beta[j], 1e-6);
        EXPECT_NEAR(reml.standard_errors()[j], se[j], 1e-6);
    }
    EXPECT_NEAR(reml.random_effect_std_devs()[0], 2.4055009, 1e-5);
    EXPECT_NEAR(reml.random_effect_std_devs()[1], 0.1803455, 1e-5);
    EXPECT_NEAR(reml.random_effect_correlation(0, 1), -0.668, 5e-4);
    EXPECT_NEAR(reml.residual_std_dev(), 1.3100396, 1e-6);
    EXPECT_NEAR(reml.log_likelihood(), -216.2908308, 1e-6);
    ASSERT_EQ(reml.group_labels().front(), "F01");
    EXPECT_NEAR(reml.random_effects().front()[0], -0.6413202408, 1e-5);
    EXPECT_NEAR(reml.random_effects().front()[1], -0.04475484529, 1e-6);
    EXPECT_NEAR(reml.random_effects().back()[0], -0.7832560530, 1e-5);  // M16
    EXPECT_NEAR(reml.random_effects().back()[1], -0.08864706073, 1e-6);

    const LMM ml(growth, "distance ~ age * sex + (1 + age | subject)", LMMOptions{.reml = false});
    const std::vector<double> se_ml{1.18202362104, 0.09980390455, 1.53549372554, 0.12964907511};
    for (std::size_t j = 0; j < 4; ++j) {
        EXPECT_NEAR(ml.coefficients()[j], beta[j], 1e-6);
        EXPECT_NEAR(ml.standard_errors()[j], se_ml[j], 1e-5);
    }
    EXPECT_NEAR(ml.random_effect_std_devs()[0], 2.1346882563, 1e-5);
    EXPECT_NEAR(ml.random_effect_std_devs()[1], 0.1541389639, 1e-5);
    EXPECT_NEAR(ml.log_likelihood(), -213.9029754, 1e-6);
    EXPECT_NEAR(ml.random_effects().front()[0], -0.6828078979, 1e-5);
    EXPECT_NEAR(ml.random_effects().front()[1], -0.03997159248, 1e-6);
}

// Balanced one-way layout: REML gives the ANOVA estimates sigma^2 = MSW and
// tau^2 = (MSB - MSW)/m; ML gives tau^2 = (SSB/g - MSW)/m; the BLUPs are the group-mean
// deviations shrunk by m tau^2 / (m tau^2 + sigma^2).
TEST(LMM, BalancedOneWayMatchesAnovaClosedForms) {
    const std::vector<std::vector<double>> groups{{3, 5, 4}, {8, 9, 10}, {6, 4, 8}, {11, 13, 12}};
    DataFrame df;
    std::vector<double> g, y;
    for (std::size_t k = 0; k < groups.size(); ++k)
        for (const double v : groups[k]) g.push_back(static_cast<double>(k)), y.push_back(v);
    df.add_column("g", g);
    df.add_column("y", y);
    const double m = 3, ng = 4;
    double grand = 0, ssw = 0, ssb = 0;
    for (const double v : y) grand += v / 12.0;
    std::vector<double> means;
    for (const auto& grp : groups) {
        const double mean = (grp[0] + grp[1] + grp[2]) / 3.0;
        means.push_back(mean);
        for (const double v : grp) ssw += (v - mean) * (v - mean);
        ssb += m * (mean - grand) * (mean - grand);
    }
    const double msw = ssw / (ng * (m - 1)), msb = ssb / (ng - 1);

    const LMM reml(df, "y ~ 1 + (1 | g)");
    const double tau2 = (msb - msw) / m;
    EXPECT_NEAR(reml.residual_variance(), msw, 1e-7);
    EXPECT_NEAR(reml.random_effect_covariance()(0, 0), tau2, 1e-6);
    EXPECT_NEAR(reml.coefficients()[0], grand, 1e-9);
    EXPECT_NEAR(reml.standard_errors()[0], std::sqrt(msb / (m * ng)), 1e-7);
    const double shrink = m * tau2 / (m * tau2 + msw);
    for (std::size_t k = 0; k < 4; ++k) EXPECT_NEAR(reml.random_effects()[k][0], shrink * (means[k] - grand), 1e-7);
    EXPECT_NEAR(reml.theta()[0], std::sqrt(tau2 / msw), 1e-6);
    EXPECT_NEAR(reml.deviance_at(reml.theta()), reml.deviance(), 1e-12);
    EXPECT_GT(reml.deviance_at({0.5 * reml.theta()[0]}), reml.deviance());

    const LMM ml(df, "y ~ 1 + (1 | g)", LMMOptions{.reml = false});
    EXPECT_NEAR(ml.residual_variance(), msw, 1e-7);
    EXPECT_NEAR(ml.random_effect_covariance()(0, 0), (ssb / ng - msw) / m, 1e-6);
    EXPECT_THROW((void)ml.deviance_at({1.0, 2.0}), std::invalid_argument);
}

TEST(LMM, CoefficientAndRandomEffectFrames) {
    const LMM fit(datamunge::datasets::orthodont(), "distance ~ age + (1 + age | subject)");
    const auto coef = fit.coefficient_frame();
    EXPECT_EQ(coef.columns(), (std::vector<std::string>{"term", "estimate", "std_error", "z_value", "p_value"}));
    EXPECT_EQ(coef.string_at("term", 1), "age");
    EXPECT_DOUBLE_EQ(coef.double_at("estimate", 1), fit.coefficients()[1]);
    const auto re = fit.random_effects_frame();
    EXPECT_EQ(re.columns(), (std::vector<std::string>{"subject", "n", "(Intercept)", "age"}));
    EXPECT_EQ(re.nrows(), 27u);
    EXPECT_DOUBLE_EQ(re.double_at("n", 0), 4.0);
    EXPECT_DOUBLE_EQ(re.double_at("age", 26), fit.random_effects()[26][1]);
}
