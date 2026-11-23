#include <datamunge/stats/lm.hpp>

#include <limits>
#include <stdexcept>
#include <gtest/gtest.h>

namespace {

using datamunge::dstruct::DataFrame;
using datamunge::linalg::DenseMatrix;
using datamunge::stats::LM;
using datamunge::stats::MarginalMeansOptions;

TEST(EstimatedMarginalMeans, AveragesAnUnbalancedWeightedInteractionAndProjectsContrastCovariance) {
    DataFrame data, grid;
    data.add_column("method", std::vector<std::string>{"A","A","A","A","A","B","B","B","B","B","C","C","C","C","C"});
    data.add_column("batch", std::vector<std::string>{
        "Cool","Cool","Cool","Warm","Warm","Cool","Cool","Warm","Warm","Warm","Cool","Cool","Warm","Warm","Warm"});
    data.add_column("y", std::vector<double>{8,9,10,12,13,12,14,16,17,18,13,15,16,19,19});
    data.add_column("w", std::vector<double>{1,2,1,1,3,2,1,1,2,1,1,2,2,1,3});
    grid.add_column("method", std::vector<std::string>{"B","A","C","B","A","C"});
    grid.add_column("batch", std::vector<std::string>{"Warm","Warm","Warm","Cool","Cool","Cool"});
    const LM fit(data, "y ~ method * batch", {.weights_column = "w"});
    MarginalMeansOptions options{
        .groups = {"B", "A", "C", "B", "A", "C"},
        .weights = {3, 3, 3, 1, 1, 1},
        .contrasts = DenseMatrix<double>(2, 3, {1, -1, 0, 0, -1, 1}),
    };
    const auto result = fit.estimated_marginal_means(grid, options);
    EXPECT_EQ(result.groups, (std::vector<std::string>{"B", "A", "C"}));
    ASSERT_EQ(result.means.estimates.size(), 3u);
    ASSERT_EQ(result.contrasts.estimates.size(), 2u);
    // Reference from R model.matrix(), coef(), vcov() and qt() on the weighted fit.
    EXPECT_NEAR(result.means.estimates[0].estimate, 15.9166666666667, 1e-11);
    EXPECT_NEAR(result.means.estimates[0].standard_error, .629421091554635, 1e-11);
    EXPECT_NEAR(result.means.estimates[1].lower, 9.99622702526164, 1e-10);
    EXPECT_NEAR(result.means.estimates[2].upper, 18.6386965857905, 1e-10);
    EXPECT_NEAR(result.means.covariance(0, 1), 0, 1e-11);
    EXPECT_NEAR(result.contrasts.estimates[0].estimate, 4.10416666666667, 1e-11);
    EXPECT_NEAR(result.contrasts.estimates[0].lower, 1.73349501283370, 1e-10);
    EXPECT_NEAR(result.contrasts.estimates[1].upper, 7.45964382208164, 1e-10);
    EXPECT_NEAR(result.contrasts.covariance(0, 1), .383391203703705, 1e-11);
    EXPECT_DOUBLE_EQ(result.contrasts.covariance(1, 0), result.contrasts.covariance(0, 1));
    options.simultaneous = false;
    const auto pointwise = fit.estimated_marginal_means(grid, options);
    EXPECT_GT(pointwise.contrasts.estimates[0].lower, result.contrasts.estimates[0].lower);
    EXPECT_LT(pointwise.means.estimates[0].upper, result.means.estimates[0].upper);
}

TEST(EstimatedMarginalMeans, AveragesTransformedModelRowsAndAllowsZeroWeights) {
    DataFrame data, grid;
    data.add_column("x", std::vector<double>{0, 1, 2, 3, 4});
    data.add_column("y", std::vector<double>{1, 3, 8, 19, 33});
    grid.add_column("x", std::vector<double>{0, 4, 100});
    const LM fit(data, "y ~ I(x^2)");
    const auto predictions = fit.predict(grid);
    const auto result = fit.estimated_marginal_means(grid, {
        .weights = {1e308, 1e308, 0},
        .contrasts = DenseMatrix<double>(1, 1, 0),
    });
    EXPECT_EQ(result.groups, (std::vector<std::string>{""}));
    EXPECT_NEAR(result.means.estimates[0].estimate, (predictions[0] + predictions[1]) / 2, 1e-12);
    EXPECT_DOUBLE_EQ(result.contrasts.estimates[0].standard_error, 0);
    EXPECT_DOUBLE_EQ(result.contrasts.estimates[0].upper, 0);
}

TEST(EstimatedMarginalMeans, RejectsInvalidGridAndAveragingSettings) {
    DataFrame data, grid, wrong_level;
    data.add_column("y", std::vector<double>{1, 2, 4, 5});
    data.add_column("factor", std::vector<std::string>{"A", "A", "B", "B"});
    grid.add_column("factor", std::vector<std::string>{"A", "B"});
    wrong_level.add_column("factor", std::vector<std::string>{"C"});
    const LM fit(data, "y ~ factor");
    EXPECT_THROW((void)fit.estimated_marginal_means(wrong_level), std::runtime_error);
    EXPECT_THROW((void)fit.estimated_marginal_means(grid, {.groups = {"only"}}), std::invalid_argument);
    EXPECT_THROW((void)fit.estimated_marginal_means(grid, {.groups = {"A", "B"}, .weights = {1, 0}}),
                 std::invalid_argument);
    EXPECT_THROW((void)fit.estimated_marginal_means(grid, {.contrasts = DenseMatrix<double>(1, 2, 1)}),
                 std::invalid_argument);
    EXPECT_THROW((void)fit.estimated_marginal_means(grid, {.confidence_level = std::numeric_limits<double>::quiet_NaN()}),
                 std::invalid_argument);
}

} // namespace
