#include <gtest/gtest.h>

#include <datamunge/fda/pspline.hpp>
#include <datamunge/stats/gam.hpp>

#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace st = datamunge::stats;

TEST(AdditiveModel, GaussianSmoothMatchesTheCanonicalPSpline) {
    const std::vector<double> x{-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.7, 1, 1.5, 2, 3};
    const std::vector<double> y{1, 1, 2, 3, 3, 4, 5, 4, 4, 3, 2, 1};
    const auto reference = datamunge::fda::fit_pspline(x, y, 3, 7, {.lambda = 0.4});
    const auto model = st::fit_gam(y, {x}, {.smooths = {{.basis_size = 7, .lambda = 0.4}}});
    ASSERT_TRUE(model.converged());
    const auto prediction = model.predict({x});
    const auto term = model.smooth(0, x);
    EXPECT_NEAR(std::accumulate(term.begin(), term.end(), 0.), 0., 1e-10);
    EXPECT_NEAR(model.deviance(), reference.residual_sum_of_squares, 1e-10);
    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(prediction[i], reference.fitted[i], 1e-10);
        EXPECT_NEAR(prediction[i], model.intercept() + term[i], 1e-12);
    }
    EXPECT_DOUBLE_EQ(model.predict({{-5., 6.}})[0], prediction.front());
    EXPECT_DOUBLE_EQ(model.predict({{-5., 6.}})[1], prediction.back());
    EXPECT_FALSE(model.plot_smooth(0).to_svg().empty());
}

TEST(AdditiveModel, TwoCenteredLinearTermsReconstructAdditivePredictions) {
    const std::vector<double> x{0, 1, 2, 3, 4, 5}, z{3, 2, 1, 0, 2, 4};
    std::vector<double> y;
    for (std::size_t i = 0; i < x.size(); ++i) y.push_back(1 + 2 * x[i] - 3 * z[i]);
    const auto model = st::fit_gam(y, {x, z}, {.smooths = {{0}, {0}}});
    ASSERT_TRUE(model.converged());
    EXPECT_NEAR(model.predict({{7}, {2}}).front(), 9, 1e-12);
    EXPECT_NEAR(model.deviance(), 0, 1e-20);
    EXPECT_TRUE(model.predict({{}, {}}).empty());
    EXPECT_THROW((void)model.predict({x, z}, std::vector<std::string>(x.size(), "A")), std::invalid_argument);
}

TEST(AdditiveModel, GaussianGroupModesMatchBalancedAnalyticShrinkage) {
    const std::vector<double> x{-1, 0, 1, -1, 0, 1};
    const std::vector<double> y{2, 4, 6, -2, 0, 2};
    const std::vector<std::string> groups{"A", "A", "A", "B", "B", "B"};
    st::GAMMOptions options;
    options.gam.smooths = {{.basis_size = 0}};
    options.random_intercept_sd = 1;
    options.dispersion = 0.25;
    const auto model = st::fit_gamm(y, {x}, groups, options);
    ASSERT_TRUE(model.converged());
    EXPECT_EQ(model.group_levels(), (std::vector<std::string>{"A", "B"}));
    const auto effects = model.random_intercepts();
    EXPECT_NEAR(effects[0], 2 * 3. / 3.25, 1e-12);
    EXPECT_NEAR(effects[1], -effects[0], 1e-12);
    EXPECT_NEAR(model.predict({{0}}, {"new"}).front(), 2., 1e-12);
    EXPECT_NEAR(model.predict({{0}}).front(), 2., 1e-12);
    EXPECT_NEAR(model.predict({{0}}, {"A"}).front(), 2 + effects[0], 1e-12);
    const auto means = model.predict({x}, groups);
    for (std::size_t i = 0; i < x.size(); ++i)
        EXPECT_NEAR(means[i], model.fitted_values()[i], 1e-12);
}

TEST(AdditiveModel, LinearTermsMatchExistingGLMFamilyFits) {
    const std::vector<double> x{-2, -2, -1, -1, 0, 0, 1, 1, 2, 2};
    const std::vector<std::pair<st::GLMFamily, std::vector<double>>> cases{
        {st::GLMFamily::Binomial, {0, 0, 0, 1, 0, 1, 0, 1, 1, 1}},
        {st::GLMFamily::Poisson, {0, 1, 1, 2, 3, 1, 2, 3, 4, 5}},
        {st::GLMFamily::Gamma, {2, 3, 2, 3, 3, 5, 4, 6, 5, 7}}
    };
    for (const auto& [family, y] : cases) {
        datamunge::dstruct::DataFrame data;
        data.add_column("x", x);
        data.add_column("y", y);
        const st::GLM reference(data, "y ~ x", {.family = family, .tol = 1e-13});
        const auto model = st::fit_gam(y, {x}, {.family = family, .smooths = {{0}}});
        ASSERT_TRUE(reference.converged());
        ASSERT_TRUE(model.converged());
        for (std::size_t i = 0; i < x.size(); ++i)
            EXPECT_NEAR(model.fitted_values()[i], reference.fitted_values()[i], 2e-6);
    }
}

TEST(AdditiveModel, ValidatesSamplesAndReportsIterationLimit) {
    EXPECT_THROW(st::fit_gam({1, 2}, {{0}}), std::invalid_argument);
    EXPECT_THROW(st::fit_gam({1, 2}, {{0, 0}}), std::invalid_argument);
    EXPECT_THROW(st::fit_gam({1, 2}, {{0, std::numeric_limits<double>::infinity()}}), std::invalid_argument);
    EXPECT_THROW(st::fit_gam({1, 2}, {{0, 1}}, {.smooths = {{3}}}), std::invalid_argument);
    EXPECT_THROW(st::fit_gam({1, 2, 3}, {{0, 1, 2}, {0, 1, 2}}, {.smooths = {{0}, {0}}}), std::runtime_error);
    EXPECT_THROW(st::fit_gam({0, 0, 0}, {{0, 1, 2}}, {.family = st::GLMFamily::Poisson}), std::invalid_argument);
    EXPECT_THROW(st::fit_gamm({1, 2}, {{0, 1}}, {"", "B"}), std::invalid_argument);
    const auto partial = st::fit_gam({1, 2, 3, 4}, {{0, 1, 2, 3}},
                                    {.smooths = {{0}}, .max_iter = 1});
    EXPECT_FALSE(partial.converged());
    EXPECT_EQ(partial.iterations(), 1u);
    EXPECT_NEAR(partial.predict({{2}}).front(), 3., 1e-12);
}
