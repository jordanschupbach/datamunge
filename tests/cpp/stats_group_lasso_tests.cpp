#include <datamunge/stats/group_lasso.hpp>

#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::group_lasso_path;
using datamunge::stats::GroupLassoOptions;

TEST(GroupLasso, MatchesClosedFormAndStartsAtTheNullModel) {
    const DenseMatrix<double> x(4, 3, {-1,-1,1, -1,1,-1, 1,-1,-1, 1,1,1});
    const std::vector<double> y{1.4,.6,4.4,5.6};
    constexpr std::size_t selected = 4;
    const auto group = group_lasso_path(x, y, {selected,91,selected}, {.lambdas = {1,.02}});
    ASSERT_EQ(group.size(), 2u);
    ASSERT_TRUE(group[0].converged);
    EXPECT_EQ(group[0].active_groups, (std::vector<std::size_t>{selected}));
    EXPECT_EQ(group[1].active_groups, (std::vector<std::size_t>{selected,91}));
    const auto shrink = 1 - std::sqrt(2.) / std::hypot(2., .5);
    EXPECT_NEAR(group[0].coefficients[0], 2 * shrink, 1e-12);
    EXPECT_DOUBLE_EQ(group[0].coefficients[1], 0);
    EXPECT_NEAR(group[0].coefficients[2], .5 * shrink, 1e-12);
    EXPECT_NEAR(group[0].intercept, 3, 1e-12);
    EXPECT_NEAR(group[0].predict(x)[0], 3 - 1.5 * shrink, 1e-12);

    const auto sparse = group_lasso_path(x, y, {selected,91,selected},
        {.alpha = .5, .lambdas = {1}});
    ASSERT_TRUE(sparse[0].converged);
    EXPECT_EQ(sparse[0].active_groups, (std::vector<std::size_t>{selected}));
    EXPECT_NEAR(sparse[0].coefficients[0], 1.5 - .5 * std::sqrt(2.), 1e-12);
    EXPECT_DOUBLE_EQ(sparse[0].coefficients[2], 0);
    const auto automatic = group_lasso_path(x, y, {selected,91,selected},
        {.alpha = .5, .path_length = 5, .lambda_min_ratio = .1});
    ASSERT_EQ(automatic.size(), 5u);
    EXPECT_NEAR(automatic[0].lambda, 4 / (1 + std::sqrt(2.)), 1e-12);
    EXPECT_TRUE(automatic[0].active_groups.empty());
    EXPECT_EQ(automatic[0].iterations, 0u);
    EXPECT_NEAR(automatic.back().lambda / automatic.front().lambda, .1, 1e-14);
}

TEST(GroupLasso, RecoversWeightedSparseSolutionWithNoncontiguousGroups) {
    const DenseMatrix<double> x(10, 5, {
        1,2,3,4,1, 0,1,4,2,3, -1,3,2,2,2, 2,0,1,3,1, -2,-2,1,4,-1,
        1,1,0,-2,0, 3,2,-1,0,2, -1,-3,0,2,1, 0,1,2,-1,3, 4,-2,1,1,-2});
    const std::vector<double> y{
        1.476273624515232,3.179962790227093,-1.984651797428182,5.390923571230161,.538653950808742,
        1.342757152126887,4.769569997366375,4.975131528295507,2.873611984631170,8.043026691247160};
    GroupLassoOptions options{.alpha = .35, .lambdas = {2,.4}, .standardize = false,
        .weights = {1,2,.4,1.4,1,1.5,3,1,2,.5}, .group_weights = {1.2,.7,1.8}, .tolerance = 1e-10};
    // Independently constructed from the weighted normal equations and
    // an explicit subgradient with slack on both zero slopes.
    const auto fit = group_lasso_path(x, y, {8,1,8,12,1}, options).back();
    ASSERT_TRUE(fit.converged);
    EXPECT_EQ(fit.active_groups, (std::vector<std::size_t>{1,8}));
    const double expected[]{1,-.75,0,0,.4};
    for (std::size_t i = 0; i < 5; ++i)
        EXPECT_NEAR(fit.coefficients[i], expected[i], 1e-9);
    EXPECT_DOUBLE_EQ(fit.coefficients[2], 0);
    EXPECT_DOUBLE_EQ(fit.coefficients[3], 0);
    EXPECT_NEAR(fit.intercept, 2.3, 1e-9);
    EXPECT_NEAR(fit.objective, 1.021676776581852, 1e-11);
    EXPECT_LT(fit.kkt_residual, 1e-8);
    options.lambdas = {.4};
    options.max_iterations = 1;
    EXPECT_FALSE(group_lasso_path(x, y, {8,1,8,12,1}, options).front().converged);
}

TEST(GroupLasso, HandlesConstantColumnsNullSignalAndInvalidInputs) {
    const DenseMatrix<double> x(3, 2, {1,10, 2,10, 3,10});
    auto path = group_lasso_path(x, {4,4,4}, {2,3});
    ASSERT_EQ(path.size(), 1u);
    EXPECT_DOUBLE_EQ(path[0].lambda, 0);
    EXPECT_TRUE(path[0].active_groups.empty());
    EXPECT_DOUBLE_EQ(path[0].intercept, 4);
    const auto unpenalized = group_lasso_path(x, {4,8,12}, {2,3}, {.lambdas = {0}}).front();
    EXPECT_TRUE(unpenalized.converged);
    EXPECT_NEAR(unpenalized.coefficients[0], 4, 1e-12);
    EXPECT_DOUBLE_EQ(unpenalized.coefficients[1], 0);
    EXPECT_THROW((void)group_lasso_path(x, {4,8,12}, {2}), std::invalid_argument);
    EXPECT_THROW((void)group_lasso_path(x, {4,8,12}, {2,3}, {.lambdas = {.1,.2}}), std::invalid_argument);
    EXPECT_THROW((void)group_lasso_path(x, {4,8,12}, {2,3}, {.group_weights = {1}}), std::invalid_argument);
}
}
