#include <gtest/gtest.h>

#include <datamunge/fda/pspline.hpp>
#include <datamunge/stats/gam.hpp>

#include <cmath>
#include <limits>
#include <numbers>
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

TEST(AdditiveModel, TensorPenaltyMatchesIndependentKroneckerFit) {
    std::vector<double> x, z, gaussian, poisson;
    for (int i = 0; i <= 5; ++i)
        for (int j = 0; j <= 6; ++j) {
            if ((i + 2 * j) % 8 == 3)
                continue;
            const auto a = i / 5., b = std::pow(j / 6., 1.4);
            const auto y = 1 + 2*a - .3*b + std::sin(2 * std::numbers::pi * a) * std::pow(b + .1, 2)
                         + .06 * std::cos(2 * i + j);
            x.push_back(a);
            z.push_back(b);
            gaussian.push_back(y);
            poisson.push_back(std::round(2 * y + (i + j) % 3));
        }
    const std::vector<double> qx{0,.15,.8,1.2}, qz{-.2,.45,.75,1};
    st::GAMOptions options{.smooths = {{0}, {0}}, .tensor_smooths = {{{{0,5,.05}, {1,4,1.3}}}}};
    // R splines::bs(), explicit nullspace projection of the Kronecker penalties,
    // and weighted normal equations for the Gaussian and Poisson objectives.
    const auto first = st::fit_gam(gaussian, {x,z}, options);
    ASSERT_TRUE(first.converged());
    const auto y = first.predict({qx,qz});
    const auto component = first.tensor_smooth(0, {qx,qz});
    const double expected[]{1.189744515548,1.288611518576,1.983578331252,2.708221494243};
    const double expected_tensor[]{-.070282358518,.001439541377,-.248433319069,-.083738484960};
    for (std::size_t i = 0; i < qx.size(); ++i) {
        EXPECT_NEAR(y[i], expected[i], 2e-10);
        EXPECT_NEAR(component[i], expected_tensor[i], 2e-10);
    }
    EXPECT_NEAR(first.deviance(), 2.149507260484, 1e-10);
    const auto train_component = first.tensor_smooth(0, {x,z});
    EXPECT_NEAR(std::accumulate(train_component.begin(), train_component.end(), 0.), 0, 1e-12);

    options.tensor_smooths[0].margins[0].lambda = 1.3;
    options.tensor_smooths[0].margins[1].lambda = .05;
    const auto swapped = st::fit_gam(gaussian, {x,z}, options);
    ASSERT_TRUE(swapped.converged());
    EXPECT_NEAR(swapped.predict({qx,qz})[2], 2.163295511204, 2e-10);
    EXPECT_NEAR(swapped.deviance(), 3.474283026428, 1e-10);

    options.family = st::GLMFamily::Poisson;
    const auto count = st::fit_gam(poisson, {x,z}, options);
    ASSERT_TRUE(count.converged());
    const auto rates = count.predict({qx,qz});
    const double expected_rates[]{3.038842248560,3.735997583428,4.861510793080,6.890015018966};
    for (std::size_t i = 0; i < qx.size(); ++i)
        EXPECT_NEAR(rates[i], expected_rates[i], 1e-8);
}

TEST(AdditiveModel, ThreeWayTensorAndPairCoexistWithGroupEffects) {
    std::vector<double> x, y, z, response;
    std::vector<std::string> groups;
    for (const auto* group : {"A", "B"})
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                for (int k = 0; k < 4; ++k) {
                    const auto a = i - 1.5, b = j - 1.5, c = k - 1.5;
                    x.push_back(a);
                    y.push_back(b);
                    z.push_back(c);
                    groups.push_back(group);
                    response.push_back(3 + a + a*b + a*b*c + (groups.back() == "A" ? 2 : -2));
                }
    st::GAMMOptions options;
    options.gam.smooths = {{0}, {0}, {0}};
    options.gam.tensor_smooths = {{{{0,4,.1}, {1,4,.3}}}, {{{2,4,2}, {0,4,.2}, {1,4,1}}}};
    const auto fit = st::fit_gamm(response, {x,y,z}, groups, options);
    ASSERT_TRUE(fit.converged());
    EXPECT_NEAR(fit.tensor_smooth(0, {{1.25}, {-.5}})[0], -.625, 1e-11);
    EXPECT_NEAR(fit.tensor_smooth(1, {{.75}, {1.25}, {-.5}})[0], -.46875, 1e-11);
    EXPECT_NEAR(fit.tensor_smooth(1, {{.75}, {2}, {-.5}})[0], -.5625, 1e-11);
    EXPECT_NEAR(fit.predict({{1.25}, {-.5}, {.75}}, {"A"})[0], 3 + 1.25 - .625 - .46875 + 128./65, 1e-11);
    EXPECT_TRUE(fit.tensor_smooth(1, {{}, {}, {}}).empty());
    EXPECT_THROW((void)fit.tensor_smooth(2, {{1}, {2}}), std::invalid_argument);
    EXPECT_THROW((void)fit.tensor_smooth(1, {{1}, {2}, {}}), std::invalid_argument);
}

TEST(AdditiveModel, RejectsInvalidTensorSpecs) {
    const std::vector<double> y{1,2,3,4}, x{0,1,2,3}, z{0,2,1,3};
    st::GAMOptions options{.smooths = {{0}, {0}}};
    for (const auto& spec : std::vector<st::GAMTensorSmooth>{
            {{{0,4,1}}}, {{{0,4,1}, {0,4,1}}}, {{{0,4,1}, {2,4,1}}},
            {{{0,3,1}, {1,4,1}}}, {{{0,4,-1}, {1,4,1}}},
            {{{0,std::numeric_limits<std::size_t>::max(),1}, {1,4,1}}}}) {
        options.tensor_smooths = {spec};
        EXPECT_THROW(st::fit_gam(y, {x,z}, options), std::invalid_argument);
    }
    options.tensor_smooths = {{{{0,4,1}, {1,4,1}}}, {{{1,4,1}, {0,4,1}}}};
    EXPECT_THROW(st::fit_gam(y, {x,z}, options), std::invalid_argument);
}
