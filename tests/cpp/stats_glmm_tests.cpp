#include <gtest/gtest.h>

#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/stats/glm.hpp>
#include <datamunge/stats/glmm.hpp>

#include <cmath>
#include <random>

using datamunge::dstruct::DataFrame;
using datamunge::stats::GLM;
using datamunge::stats::GLMFamily;
using datamunge::stats::GLMM;
using datamunge::stats::GLMMFamily;
using datamunge::stats::GLMMOptions;
using datamunge::stats::GLMOptions;

// Reference values for the "oracle" test below were computed with an independent,
// from-scratch numpy implementation of penalized quasi-likelihood (dense n x n matrices,
// no block-diagonal shortcut, grid + golden-section search for the variance component at
// each outer iteration -- deliberately a different numerical approach than this engine's
// block-diagonal-per-group Cholesky + DifferentialEvolution search) since neither R's
// lme4 nor Python's statsmodels are available in this environment. See
// scratchpad/lmm_oracle/{gen_glmm_data,glmm_oracle}.py.

namespace {

DataFrame make_oracle_dataframe() {
    const std::vector<double> group{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
                                     3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
                                     6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7};
    const std::vector<double> x{0.444376,  -0.468734, 0.400282,  1.854231,  -1.215350, -0.651834, 0.273900,  0.158368,
                                 0.944194,  -0.276725, -1.769892, -1.335766, 1.029627,  0.719754,  0.806116,  -1.578959,
                                 -1.814354, 1.727624,  -0.313283, -1.864618, -1.120677, -0.040814, -0.707463, -1.587066,
                                 1.306247,  -1.488036, -1.199925, -1.319720, 1.568057,  0.954771,  1.111608,  -1.702459,
                                 -1.471793, -1.342953, -0.755928, -1.464881, -1.368975, -1.826882, -0.602272, -1.554926,
                                 -1.379802, 1.037987,  -1.087335, -0.149050, 0.765600,  0.778776,  1.393720,  1.412503,
                                 -0.361332, -0.717544, 1.613259,  -1.716074, 1.220654,  1.092091,  1.580676,  0.862569,
                                 1.041068,  1.309387,  0.954250,  0.562808,  -0.205975, 1.226108,  -0.207794, -0.592801};
    const std::vector<double> y{0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                                 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1,
                                 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0};
    DataFrame df;
    df.add_column("group", group);
    df.add_column("x", x);
    df.add_column("y", y);
    return df;
}

} // namespace

TEST(GLMM, MatchesIndependentNumpyPqlOracle) {
    const auto df = make_oracle_dataframe();
    GLMMOptions options;
    options.family = GLMMFamily::Binomial;
    const GLMM model(df, "y ~ x + (1 | group)", options);

    ASSERT_EQ(model.coefficients().size(), 2u);
    EXPECT_NEAR(model.coefficients()[0], -0.30444076, 0.15);
    EXPECT_NEAR(model.coefficients()[1], 1.10832718, 0.2);
    EXPECT_NEAR(model.random_effect_std_devs()[0], 0.639765, 0.3);
    EXPECT_NEAR(model.deviance(), 291.962510, 5.0);
}

TEST(GLMM, ReducesToGlmWhenTrueGroupEffectIsZero) {
    std::mt19937_64 rng(11);
    std::uniform_int_distribution<int> group_dist(0, 19);
    std::uniform_real_distribution<double> x_dist(-2.0, 2.0);
    std::uniform_real_distribution<double> unif01(0.0, 1.0);

    const double true_intercept = -0.3, true_slope = 1.0;
    std::vector<double> group(600), x(600), y(600);
    for (int i = 0; i < 600; ++i) {
        group[i] = static_cast<double>(group_dist(rng));
        x[i] = x_dist(rng);
        const double eta = true_intercept + true_slope * x[i]; // no group-level shift at all
        const double p = 1.0 / (1.0 + std::exp(-eta));
        y[i] = unif01(rng) < p ? 1.0 : 0.0;
    }
    DataFrame df;
    df.add_column("group", group);
    df.add_column("x", x);
    df.add_column("y", y);

    GLMOptions glm_options;
    glm_options.family = GLMFamily::Binomial;
    const GLM glm(df, "y ~ x", glm_options);

    GLMMOptions glmm_options;
    glmm_options.family = GLMMFamily::Binomial;
    const GLMM glmm(df, "y ~ x + (1 | group)", glmm_options);

    EXPECT_NEAR(glmm.coefficients()[0], glm.coefficients()[0], 0.3);
    EXPECT_NEAR(glmm.coefficients()[1], glm.coefficients()[1], 0.3);
    EXPECT_LT(glmm.random_effect_std_devs()[0], 0.5);
}

TEST(GLMM, RecoversKnownSimulatedGroundTruthForPoisson) {
    std::mt19937_64 rng(555);
    constexpr int n_groups = 40;
    std::uniform_int_distribution<int> group_dist(0, n_groups - 1);
    std::normal_distribution<double> group_effect(0.0, 0.5); // true random-intercept SD
    std::uniform_real_distribution<double> x_dist(-1.0, 1.0);

    std::vector<double> group_effects(n_groups);
    for (auto& g : group_effects) g = group_effect(rng);

    const double true_intercept = 1.0, true_slope = 0.6;
    constexpr int n = n_groups * 15;
    std::vector<double> group(n), x(n), y(n);
    for (int i = 0; i < n; ++i) {
        const int g = group_dist(rng);
        group[i] = static_cast<double>(g);
        x[i] = x_dist(rng);
        const double lambda = std::exp(true_intercept + true_slope * x[i] + group_effects[g]);
        std::poisson_distribution<int> pois(lambda);
        y[i] = static_cast<double>(pois(rng));
    }
    DataFrame df;
    df.add_column("group", group);
    df.add_column("x", x);
    df.add_column("y", y);

    GLMMOptions options;
    options.family = GLMMFamily::Poisson;
    const GLMM model(df, "y ~ x + (1 | group)", options);

    EXPECT_NEAR(model.coefficients()[0], true_intercept, 0.3);
    EXPECT_NEAR(model.coefficients()[1], true_slope, 0.2);
    EXPECT_NEAR(model.random_effect_std_devs()[0], 0.5, 0.35);
    EXPECT_EQ(model.family(), "poisson");
    EXPECT_EQ(model.num_groups(), static_cast<std::size_t>(n_groups));
}

TEST(GLMM, RandomInterceptAndSlopeFitsWithoutError) {
    std::mt19937_64 rng(3);
    std::uniform_int_distribution<int> group_dist(0, 11);
    std::uniform_real_distribution<double> x_dist(-1.5, 1.5);
    std::uniform_real_distribution<double> unif01(0.0, 1.0);

    std::vector<double> group(350), x(350), y(350);
    for (int i = 0; i < 350; ++i) {
        const int g = group_dist(rng);
        group[i] = static_cast<double>(g);
        x[i] = x_dist(rng);
        const double eta = 0.2 + 0.8 * x[i] + 0.05 * static_cast<double>(g) * x[i];
        const double p = 1.0 / (1.0 + std::exp(-eta));
        y[i] = unif01(rng) < p ? 1.0 : 0.0;
    }
    DataFrame df;
    df.add_column("group", group);
    df.add_column("x", x);
    df.add_column("y", y);

    const GLMM model(df, "y ~ x + (1 + x | group)");
    ASSERT_EQ(model.random_effect_names().size(), 2u);
    EXPECT_EQ(model.random_effect_names()[0], "(Intercept)");
    EXPECT_EQ(model.random_effect_names()[1], "x");
    EXPECT_EQ(model.num_groups(), 12u);
    const auto text = model.summary();
    EXPECT_NE(text.find("Random effects"), std::string::npos);
    EXPECT_NE(text.find("Fixed effects"), std::string::npos);
    EXPECT_NE(text.find("binomial"), std::string::npos);
}

TEST(GLMM, RejectsInvalidBinomialResponse) {
    DataFrame df;
    df.add_column("x", std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    df.add_column("group", std::vector<double>{0, 0, 1, 1, 2, 2, 3, 3});
    df.add_column("y", std::vector<double>{0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0}); // 2.0 invalid
    GLMMOptions options;
    options.family = GLMMFamily::Binomial;
    EXPECT_THROW(GLMM(df, "y ~ x + (1 | group)", options), std::invalid_argument);
}

TEST(GLMM, RejectsNegativePoissonResponse) {
    DataFrame df;
    df.add_column("x", std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    df.add_column("group", std::vector<double>{0, 0, 1, 1, 2, 2, 3, 3});
    df.add_column("y", std::vector<double>{0.0, 1.0, -2.0, 0.0, 1.0, 0.0, 1.0, 0.0});
    GLMMOptions options;
    options.family = GLMMFamily::Poisson;
    EXPECT_THROW(GLMM(df, "y ~ x + (1 | group)", options), std::invalid_argument);
}

TEST(GLMM, RejectsFormulaWithoutRandomEffectsTerm) {
    const auto df = make_oracle_dataframe();
    EXPECT_THROW(GLMM(df, "y ~ x"), std::invalid_argument);
}

TEST(GLMM, PredictFallsBackToFixedEffectsOnlyForUnseenGroup) {
    const auto df = make_oracle_dataframe();
    const GLMM model(df, "y ~ x + (1 | group)");

    DataFrame newdata;
    newdata.add_column("x", std::vector<double>{0.0});
    newdata.add_column("group", std::vector<double>{999.0});
    const auto preds = model.predict(newdata);
    ASSERT_EQ(preds.size(), 1u);
    const double expected = 1.0 / (1.0 + std::exp(-model.coefficients()[0]));
    EXPECT_NEAR(preds[0], expected, 1e-9);
}

TEST(GLMM, PredictAppliesBlupForKnownGroup) {
    const auto df = make_oracle_dataframe();
    const GLMM model(df, "y ~ x + (1 | group)");

    DataFrame newdata;
    newdata.add_column("x", std::vector<double>{0.0});
    newdata.add_column("group", std::vector<double>{0.0});
    const auto preds = model.predict(newdata);
    const double eta = model.coefficients()[0] + model.random_effects()[0][0];
    const double expected = 1.0 / (1.0 + std::exp(-eta));
    EXPECT_NEAR(preds[0], expected, 1e-9);
}
