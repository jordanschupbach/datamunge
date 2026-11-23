#include <datamunge/stats/negative_binomial_regression.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <gtest/gtest.h>

namespace {
using datamunge::linalg::DenseMatrix;
using datamunge::stats::negative_binomial_regression;
using datamunge::stats::NegativeBinomialRegressionStatus;

TEST(NegativeBinomialRegression, MatchesMassAndJointObservedInformationWithExposure) {
    DenseMatrix<double> x(16, 2, 1);
    for (std::size_t i = 0; i < x.rows(); ++i)
        x(i, 1) = i >= 8;
    std::vector<double> offset{1,.6,2,1.5,1,.7,1.1,2, 1,.8,.7,1.2,1.3,2,.9,2};
    for (auto& value : offset)
        value = std::log(value);
    const std::vector<std::size_t> y{0,3,5,0,1,6,2,8, 0,1,11,4,2,22,1,13};
    const std::vector<double> weights{1,1.2,1,.8,1,1.3,1,1, 1,1.1,1,1.2,.7,1,1,1.4};
    const auto fit = negative_binomial_regression(x, y, {.offset = offset, .weights = weights});
    ASSERT_EQ(fit.status, NegativeBinomialRegressionStatus::Converged);
    EXPECT_NEAR(fit.coefficients[0], 1.105009922891649, 1e-7);
    EXPECT_NEAR(fit.coefficients[1], .539162862709856, 1e-7);
    EXPECT_NEAR(fit.dispersion, .805917398472286, 1e-7);
    EXPECT_NEAR(fit.log_likelihood, -43.794965860914481, 1e-11);
    ASSERT_EQ(fit.covariance.rows(), 3u);
    // Independently differentiated joint log likelihood with R dnbinom() and optimHess().
    EXPECT_NEAR(fit.covariance(1, 1), .256414715810293, 5e-8);
    EXPECT_NEAR(fit.covariance(2, 0), .007239793589612, 1e-8);
    EXPECT_NEAR(fit.covariance(2, 2), .159521307479610, 5e-8);
    const auto prediction = fit.predict(DenseMatrix<double>(2, 2, {1,1,1,1}), {0, std::log(2.0)});
    EXPECT_NEAR(prediction[0], std::exp(1.644172785601505), 2e-7);
    EXPECT_NEAR(prediction[1], 2 * prediction[0], 1e-13);

    for (auto& value : offset)
        value += std::log(4.0);
    const auto rescaled = negative_binomial_regression(x, y, {.offset = offset, .weights = weights});
    ASSERT_EQ(rescaled.status, NegativeBinomialRegressionStatus::Converged);
    EXPECT_NEAR(rescaled.coefficients[0] + std::log(4.0), fit.coefficients[0], 1e-8);
    EXPECT_NEAR(rescaled.dispersion, fit.dispersion, 1e-7);
    EXPECT_NEAR(rescaled.log_likelihood, fit.log_likelihood, 1e-11);
}

TEST(NegativeBinomialRegression, ReturnsExactPoissonLimitForUnderdispersedCounts) {
    const auto fit = negative_binomial_regression(DenseMatrix<double>(6, 1, 1), {2,3,2,2,3,2},
                                                  {.tolerance = 1e-12});
    ASSERT_EQ(fit.status, NegativeBinomialRegressionStatus::PoissonBoundary);
    EXPECT_DOUBLE_EQ(fit.dispersion, 0);
    EXPECT_NEAR(fit.coefficients[0], std::log(14.0 / 6), 1e-10);
    ASSERT_EQ(fit.covariance.rows(), 1u);
    EXPECT_NEAR(fit.covariance(0, 0), 1.0 / 14, 1e-12);
    EXPECT_NEAR(fit.log_likelihood, 14 * std::log(14.0 / 6) - 14 - 4 * std::log(2.0) - 2 * std::log(6.0),
                1e-12);
}

TEST(NegativeBinomialRegression, ValidatesInputsAndDistinguishesIterationLimits) {
    const DenseMatrix<double> x(4, 1, 1);
    const auto limited = negative_binomial_regression(x, {0,0,1,11}, {.max_iterations = 1});
    EXPECT_EQ(limited.status, NegativeBinomialRegressionStatus::IterationLimit);
    EXPECT_EQ(limited.covariance.rows(), 0u);
    EXPECT_THROW((void)negative_binomial_regression(x, {0,0,0,0}), std::invalid_argument);
    EXPECT_THROW((void)negative_binomial_regression(x, {0,0,1,11}, {.offset = {0,1}}), std::invalid_argument);
    EXPECT_THROW((void)negative_binomial_regression(x, {0,0,1,11},
        {.offset = {0,0,0,std::numeric_limits<double>::infinity()}}), std::invalid_argument);
    EXPECT_THROW((void)negative_binomial_regression(DenseMatrix<double>(4, 2, 1), {0,0,1,11}), std::invalid_argument);
}
} // namespace
