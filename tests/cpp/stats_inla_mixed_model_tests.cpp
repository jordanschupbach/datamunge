#include <gtest/gtest.h>

#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/stats/glmm.hpp>
#include <datamunge/stats/inla_mixed_model.hpp>
#include <datamunge/stats/lmm.hpp>

#include <cmath>
#include <random>

using datamunge::dstruct::DataFrame;
using datamunge::stats::GLMM;
using datamunge::stats::GLMMFamily;
using datamunge::stats::GLMMOptions;
using datamunge::stats::INLAMixedModel;
using datamunge::stats::INLAMixedModelFamily;
using datamunge::stats::INLAMixedModelOptions;
using datamunge::stats::LMM;
using datamunge::stats::LMMOptions;

namespace {

// Same fixed dataset as stats_lmm_tests.cpp's oracle test -- reused here so INLAMixedModel's
// Gaussian fit is directly comparable to LMM's already-oracle-validated REML fit on identical
// data (loose tolerance: PQL/REML and INLA are different estimation philosophies, so this
// checks "in the right neighborhood", not numerical agreement).
DataFrame make_gaussian_dataframe() {
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

// Same fixed dataset as stats_glmm_tests.cpp's oracle test.
DataFrame make_binomial_dataframe() {
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

TEST(INLAMixedModel, GaussianRandomInterceptNearLmmReml) {
    const auto df = make_gaussian_dataframe();
    const LMM lmm(df, "y ~ x + (1 | group)");

    INLAMixedModelOptions options;
    options.family = INLAMixedModelFamily::Gaussian;
    const INLAMixedModel model(df, "y ~ x + (1 | group)", options);

    ASSERT_EQ(model.fixed_effects_mean().size(), 2u);
    EXPECT_NEAR(model.fixed_effects_mean()[0], lmm.coefficients()[0], 0.5);
    EXPECT_NEAR(model.fixed_effects_mean()[1], lmm.coefficients()[1], 0.5);
    for (const double sd : model.fixed_effects_sd()) EXPECT_GT(sd, 0.0);

    EXPECT_GT(model.residual_std_dev(), 0.0);
    ASSERT_EQ(model.random_effect_std_devs().size(), 1u);
    EXPECT_GT(model.random_effect_std_devs()[0], 0.0);
    EXPECT_TRUE(std::isfinite(model.log_marginal_likelihood()));

    ASSERT_EQ(model.random_effects_mean().size(), model.num_groups());
    ASSERT_EQ(model.random_effects_sd().size(), model.num_groups());
}

TEST(INLAMixedModel, BinomialRandomInterceptNearGlmmPql) {
    const auto df = make_binomial_dataframe();
    GLMMOptions glmm_options;
    glmm_options.family = GLMMFamily::Binomial;
    const GLMM glmm(df, "y ~ x + (1 | group)", glmm_options);

    INLAMixedModelOptions options;
    options.family = INLAMixedModelFamily::Binomial;
    const INLAMixedModel model(df, "y ~ x + (1 | group)", options);

    ASSERT_EQ(model.fixed_effects_mean().size(), 2u);
    EXPECT_NEAR(model.fixed_effects_mean()[0], glmm.coefficients()[0], 0.6);
    EXPECT_NEAR(model.fixed_effects_mean()[1], glmm.coefficients()[1], 0.6);

    EXPECT_THROW(model.residual_std_dev(), std::logic_error);
}

TEST(INLAMixedModel, PoissonSmokeTestFitsWithoutError) {
    std::mt19937_64 rng(21);
    std::normal_distribution<double> noise(0.0, 0.3);
    std::vector<double> group, x, y;
    for (int g = 0; g < 8; ++g) {
        const double group_effect = noise(rng);
        for (int j = 0; j < 10; ++j) {
            const double xi = noise(rng) * 2.0;
            const double log_rate = 1.0 + 0.4 * xi + group_effect;
            std::poisson_distribution<int> pois(std::exp(log_rate));
            group.push_back(g);
            x.push_back(xi);
            y.push_back(static_cast<double>(pois(rng)));
        }
    }
    DataFrame df;
    df.add_column("group", group);
    df.add_column("x", x);
    df.add_column("y", y);

    INLAMixedModelOptions options;
    options.family = INLAMixedModelFamily::Poisson;
    const INLAMixedModel model(df, "y ~ x + (1 | group)", options);

    ASSERT_EQ(model.fixed_effects_mean().size(), 2u);
    for (const double sd : model.fixed_effects_sd()) EXPECT_GT(sd, 0.0);
    EXPECT_TRUE(std::isfinite(model.log_marginal_likelihood()));
}

TEST(INLAMixedModel, RandomInterceptAndSlopeFitsWithoutError) {
    // theta_dim = 3 (lambda) + 1 (gaussian sigma) = 4, right at the grid-strategy dimension
    // cap -- reduce grid density/DE effort to keep this smoke test fast.
    const auto df = make_gaussian_dataframe();
    INLAMixedModelOptions options;
    options.family = INLAMixedModelFamily::Gaussian;
    options.grid_points_per_dim = 3;
    options.mode_population_size = 15;
    options.mode_max_generations = 40;
    const INLAMixedModel model(df, "y ~ x + (1 + x | group)", options);

    ASSERT_EQ(model.random_effect_names().size(), 2u);
    ASSERT_EQ(model.random_effect_std_devs().size(), 2u);
    for (const double sd : model.random_effect_std_devs()) EXPECT_GT(sd, 0.0);
}

TEST(INLAMixedModel, PredictReturnsFiniteResponseScaleValues) {
    const auto df = make_binomial_dataframe();
    INLAMixedModelOptions options;
    options.family = INLAMixedModelFamily::Binomial;
    const INLAMixedModel model(df, "y ~ x + (1 | group)", options);

    const auto preds = model.predict(df);
    ASSERT_EQ(preds.size(), df.nrows());
    for (const double p : preds) {
        ASSERT_TRUE(std::isfinite(p));
        EXPECT_GE(p, 0.0);
        EXPECT_LE(p, 1.0);
    }
}
