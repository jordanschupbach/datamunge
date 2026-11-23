#include <datamunge/stats/beta_regression.hpp>

#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::BetaRegressionOptions;
using datamunge::stats::BetaRegressionStatus;
using datamunge::stats::beta_regression;

TEST(BetaRegression, MatchesWeightedRMaximumAndExpectedInformation) {
    const std::vector<double> y{.12,.26,.20,.31,.42,.34,.47,.40,.55,.51,.70,.64,.71,.79,.76,.88};
    const double levels[]{-1, -.4, .4, 1}, weights[]{1, .7, 1.3, 1.5};
    DenseMatrix<double> x(y.size(), 2, 1), z(y.size(), 2, 1);
    BetaRegressionOptions options;
    for (std::size_t i = 0; i < y.size(); ++i) {
        x(i, 1) = levels[i / 4];
        z(i, 1) = i % 4 >= 2;
        options.weights.push_back(weights[i % 4]);
    }
    const auto fit = beta_regression(x, z, y, options);
    ASSERT_EQ(fit.status, BetaRegressionStatus::Converged);
    ASSERT_EQ(fit.mean_coefficients.size(), 2u);
    ASSERT_EQ(fit.precision_coefficients.size(), 2u);
    // R dbeta() likelihood, refined using the independent digamma score.
    EXPECT_NEAR(fit.mean_coefficients[0], .1596533199031123, 2e-9);
    EXPECT_NEAR(fit.mean_coefficients[1], 1.2706730795165819, 2e-9);
    EXPECT_NEAR(fit.precision_coefficients[0], 3.1266768938603522, 8e-9);
    EXPECT_NEAR(fit.precision_coefficients[1], 1.1447350992140597, 8e-9);
    EXPECT_NEAR(fit.log_likelihood, 23.7064434412598253, 1e-11);
    // Shape-space trigamma information transformed in R to the two coefficient blocks.
    ASSERT_EQ(fit.covariance.rows(), 4u);
    EXPECT_NEAR(fit.covariance(0, 0), .005045436852444, 1e-10);
    EXPECT_NEAR(fit.covariance(1, 2), .003335780641590, 1e-10);
    EXPECT_NEAR(fit.covariance(2, 3), -.277618387013176, 3e-10);
    EXPECT_NEAR(fit.covariance(3, 3), .452837135979348, 3e-10);
    const auto prediction = fit.predict(DenseMatrix<double>(2, 2, {1,0, 1,0}),
                                        DenseMatrix<double>(2, 2, {1,0, 1,1}));
    EXPECT_DOUBLE_EQ(prediction.mean[0], prediction.mean[1]);
    EXPECT_NEAR(prediction.mean[0], 1 / (1 + std::exp(-.1596533199031123)), 1e-9);
    EXPECT_NEAR(prediction.precision[1] / prediction.precision[0], std::exp(1.1447350992140597), 3e-8);

    options.max_iterations = 1;
    const auto limited = beta_regression(x, z, y, options);
    EXPECT_EQ(limited.status, BetaRegressionStatus::IterationLimit);
    EXPECT_EQ(limited.iterations, 1u);
    EXPECT_EQ(limited.covariance.rows(), 0u);
}

TEST(BetaRegression, EstimatesLargePrecisionWithConstantMean) {
    const DenseMatrix<double> intercept(6, 1, 1);
    const auto fit = beta_regression(intercept, intercept, {.499997,.499998,.499999,.500001,.500002,.500003});
    ASSERT_EQ(fit.status, BetaRegressionStatus::Converged);
    EXPECT_NEAR(fit.fitted.mean[0], .5, 1e-13);
    EXPECT_NEAR(fit.fitted.precision[0] / 5.357142857e10, 1., 1e-6);
    EXPECT_NEAR(fit.covariance(1, 1), 1. / 3, 1e-8);
}

TEST(BetaRegression, ValidatesSupportRanksAndPerfectFits) {
    const DenseMatrix<double> intercept(4, 1, 1), rank_deficient(4, 2, 1);
    EXPECT_THROW((void)beta_regression(intercept, intercept, {0,.3,.5,.6}), std::invalid_argument);
    EXPECT_THROW((void)beta_regression(intercept, intercept, {.1,.3,.5,1}), std::invalid_argument);
    EXPECT_THROW((void)beta_regression(intercept, rank_deficient, {.1,.3,.5,.6}), std::invalid_argument);
    EXPECT_THROW((void)beta_regression(intercept, intercept, {.5,.5,.5,.5}), std::invalid_argument);
}
}
