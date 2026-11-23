#include <datamunge/stats/ordinal_regression.hpp>

#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::ordinal_regression;
using datamunge::stats::OrdinalRegressionStatus;

TEST(OrdinalRegression, MatchesPolrAndIndependentParallelSlopesScore) {
    DenseMatrix<double> x(18, 1);
    std::vector<std::int64_t> y;
    for (int i = 0; i < 6; ++i)
        for (const int category : {-4, 3, 99}) {
            x(y.size(), 0) = i;
            y.push_back(category);
        }
    const std::vector<double> w{20,70,10, 20,60,20, 20,45,35, 20,30,50, 20,17,63, 20,8,72};
    const auto fit = ordinal_regression(x, y, {.weights = w});
    ASSERT_EQ(fit.status, OrdinalRegressionStatus::Converged);
    EXPECT_EQ(fit.levels, (std::vector<std::int64_t>{-4, 3, 99}));
    EXPECT_NEAR(fit.cutpoints[0], -.6276174704181953, 1e-7);
    EXPECT_NEAR(fit.cutpoints[1], 1.252360920532047, 1e-7);
    EXPECT_NEAR(fit.coefficients[0], .3845221749419195, 1e-7);
    EXPECT_NEAR(fit.log_likelihood, -598.3610641013189, 1e-10);
    EXPECT_NEAR(fit.covariance(0, 1), .0134752275247626, 2e-8);
    EXPECT_NEAR(fit.covariance(1, 2), .00551265428470515, 2e-8);
    EXPECT_NEAR(fit.covariance(2, 2), .0023120991872687, 5e-9);
    const auto predictions = fit.predict(DenseMatrix<double>(3, 1, {2.5, -10000, 10000}));
    EXPECT_NEAR(predictions(0, 0), .169535490230849, 1e-8);
    EXPECT_NEAR(predictions(0, 1), .402719022799455, 1e-8);
    EXPECT_NEAR(predictions(0, 2), .427745486969696, 1e-8);
    EXPECT_DOUBLE_EQ(predictions(1, 0), 1);
    EXPECT_DOUBLE_EQ(predictions(2, 2), 1);
    // Independently differentiated cumulative probabilities in R give the efficient Rao statistic.
    ASSERT_TRUE(fit.parallel_slopes);
    EXPECT_EQ(fit.parallel_slopes->degrees_of_freedom, 1u);
    EXPECT_NEAR(fit.parallel_slopes->statistic, 96.29807380701635, 5e-7);
    EXPECT_NEAR(fit.parallel_slopes->p_value, 9.882613668263806e-23, 2e-29);

    for (std::size_t i = 0; i < x.rows(); ++i)
        x(i, 0) = 1000 + 7 * x(i, 0);
    const auto transformed = ordinal_regression(x, y, {.weights = w});
    ASSERT_TRUE(transformed.parallel_slopes);
    EXPECT_NEAR(transformed.coefficients[0] * 7, fit.coefficients[0], 1e-12);
    EXPECT_NEAR(transformed.cutpoints[0] - 1000 * transformed.coefficients[0], fit.cutpoints[0], 1e-12);
    EXPECT_NEAR(transformed.covariance(2, 2) * 49, fit.covariance(2, 2), 1e-12);
    EXPECT_NEAR(transformed.parallel_slopes->statistic, fit.parallel_slopes->statistic, 1e-11);
}

TEST(OrdinalRegression, InterceptOnlyMatchesMarginalFrequenciesAndBinaryMatchesOddsRatio) {
    const auto intercept = ordinal_regression(DenseMatrix<double>(5, 0), {1, 1, 2, 4, 4});
    ASSERT_EQ(intercept.status, OrdinalRegressionStatus::Converged);
    EXPECT_NEAR(intercept.cutpoints[0], std::log(2.0 / 3), 1e-13);
    EXPECT_NEAR(intercept.cutpoints[1], std::log(3.0 / 2), 1e-13);
    EXPECT_NEAR(intercept.fitted_probabilities(0, 1), .2, 1e-13);
    EXPECT_FALSE(intercept.parallel_slopes);
    const auto binary = ordinal_regression(DenseMatrix<double>(4, 1, {0, 0, 1, 1}), {0, 1, 0, 1},
                                            {.weights = {8, 2, 4, 6}});
    ASSERT_EQ(binary.status, OrdinalRegressionStatus::Converged);
    EXPECT_NEAR(binary.cutpoints[0], std::log(4.0), 1e-10);
    EXPECT_NEAR(binary.coefficients[0], std::log(6.0), 1e-10);
    EXPECT_FALSE(binary.parallel_slopes);
}

TEST(OrdinalRegression, ScoreIsZeroForIdenticalCategoryDistributions) {
    const auto fit = ordinal_regression(DenseMatrix<double>(6, 1, {0,0,0, 1,1,1}), {1,2,3, 1,2,3},
                                         {.weights = {3, 7, 2, 6, 14, 4}});
    ASSERT_TRUE(fit.parallel_slopes);
    EXPECT_NEAR(fit.parallel_slopes->statistic, 0, 1e-14);
    EXPECT_NEAR(fit.parallel_slopes->p_value, 1, 1e-12);
    EXPECT_NEAR(fit.coefficients[0], 0, 1e-12);
}

TEST(OrdinalRegression, DetectsSeparationAndRejectsAliasedPredictors) {
    const auto separated = ordinal_regression(DenseMatrix<double>(6, 1, {-2,-2,0,0,2,2}), {0,0,1,1,2,2});
    EXPECT_NE(separated.status, OrdinalRegressionStatus::Converged);
    EXPECT_EQ(separated.covariance.rows(), 0u);
    EXPECT_FALSE(separated.parallel_slopes);
    EXPECT_THROW((void)ordinal_regression(DenseMatrix<double>(4, 1, 1), {0,1,2,3}), std::invalid_argument);
    EXPECT_THROW((void)ordinal_regression(DenseMatrix<double>(4, 2, {1,2, 2,4, 3,6, 4,8}), {0,1,1,2}),
                 std::invalid_argument);
}
} // namespace
