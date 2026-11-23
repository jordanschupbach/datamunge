#include <datamunge/stats/conditional_logistic.hpp>

#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <gtest/gtest.h>

namespace {

using datamunge::linalg::DenseMatrix;
using datamunge::stats::conditional_logistic;
using datamunge::stats::ConditionalLogisticStatus;

TEST(ConditionalLogistic, MatchedPairsGiveTheLogDiscordantCountRatio) {
    const DenseMatrix<double> x(8, 1, {0, 1, 0, 1, 0, 1, 0, 1});
    const std::vector<double> cases{0, 1, 0, 1, 0, 1, 1, 0};
    const std::vector<std::int64_t> strata{8, 8, -9, -9, 3, 3, 0, 0};
    const auto fit = conditional_logistic(x, cases, strata);
    EXPECT_EQ(fit.status, ConditionalLogisticStatus::Converged);
    EXPECT_NEAR(fit.coefficients[0], std::log(3.0), 1e-10);
    EXPECT_NEAR(fit.covariance(0, 0), 4.0 / 3, 1e-10);
    EXPECT_NEAR(fit.fitted_probabilities[0], .25, 1e-11);
    EXPECT_NEAR(fit.fitted_probabilities[1], .75, 1e-11);
    EXPECT_NEAR(fit.log_likelihood, 3 * std::log(.75) + std::log(.25), 1e-12);
    const auto limited = conditional_logistic(x, cases, strata, {.max_iterations = 1});
    EXPECT_EQ(limited.status, ConditionalLogisticStatus::IterationLimit);
    EXPECT_EQ(limited.covariance.rows(), 0u);
    EXPECT_EQ(limited.iterations, 1u);
}

TEST(ConditionalLogistic, MatchesRAndEnumeratedProbabilitiesForMultipleCases) {
    const std::array x{-1., .5, 2., 1., 0., .3, -.2, 1.1, 2., 1., 1.5, -.5, -1., 2.,
                       -2., 1.4, .2, 3., -1., 2., -3., 0., 3., 4.};
    const std::array z{0., 1., .4, -.1, 1., 1., -1., -.4, 2., 1., 0., .5, 2., -.3,
                       1., 2., 0., -1., -2., 4., 0., .1, -1., 2.};
    const std::array y{0,1,1,0,0, 1,1,0,1, 0,1,0,0,1, 0,1,0,0, 0,0,0, 1,1,1};
    const std::array labels{31,31,31,31,31,18,18,18,18,22,22,22,22,22,100,100,100,100,
                            -4,-4,-4,1000,1000,1000};
    DenseMatrix<double> design(x.size(), 2);
    std::vector<double> cases(x.size());
    std::vector<std::int64_t> strata(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const auto row = (5 * i) % x.size();
        strata[row] = labels[i];
        cases[row] = y[i];
        design(row, 0) = x[i] + 2 * labels[i];
        design(row, 1) = z[i] - labels[i];
    }
    const auto fit = conditional_logistic(design, cases, strata);
    ASSERT_EQ(fit.status, ConditionalLogisticStatus::Converged);
    EXPECT_EQ(fit.informative_strata, 4u);
    // R survival::clogit(method = "exact"), with probability checks from combn().
    EXPECT_NEAR(fit.coefficients[0], 1.067021595599683, 1e-8);
    EXPECT_NEAR(fit.coefficients[1], .760533176349135, 1e-8);
    EXPECT_NEAR(fit.covariance(0, 0), .480689414673231, 1e-8);
    EXPECT_NEAR(fit.covariance(0, 1), .259142035091077, 1e-8);
    EXPECT_NEAR(fit.covariance(1, 1), .425869556978785, 1e-8);
    EXPECT_NEAR(fit.log_likelihood, -5.220084179481346, 1e-12);
    EXPECT_NEAR(fit.null_log_likelihood, -7.377758908227873, 1e-12);
    EXPECT_NEAR(fit.fitted_probabilities[0], .053837372033991, 1e-8);
    EXPECT_NEAR(fit.fitted_probabilities[(5 * 5) % x.size()], .901135005099700, 1e-8);
    EXPECT_NEAR(fit.fitted_probabilities[(7 * 5) % x.size()], .877894624931914, 1e-8);
    std::map<std::int64_t, double> discrepancy;
    for (std::size_t i = 0; i < x.size(); ++i)
        discrepancy[strata[i]] += fit.fitted_probabilities[i] - cases[i];
    for (const auto& entry : discrepancy)
        EXPECT_NEAR(entry.second, 0, 1e-12);
}

TEST(ConditionalLogistic, FitsAStratumWithAnIntractableNumberOfEnumeratedSubsets) {
    DenseMatrix<double> x(200, 1);
    std::vector<double> cases(200);
    for (std::size_t i = 0; i < cases.size(); ++i) {
        cases[i] = i < 100;
        x(i, 0) = i % 2 == 0 ? -1 : 1;
    }
    const auto fit = conditional_logistic(x, cases, std::vector<std::int64_t>(200, 18));
    EXPECT_EQ(fit.status, ConditionalLogisticStatus::Converged);
    EXPECT_DOUBLE_EQ(fit.coefficients[0], 0);
    EXPECT_NEAR(fit.covariance(0, 0), 199.0 / 10000, 1e-14);
    EXPECT_NEAR(fit.log_likelihood, -std::lgamma(201.0) + 2 * std::lgamma(101.0), 1e-12);
    EXPECT_NEAR(fit.fitted_probabilities[155], .5, 1e-14);
}

TEST(ConditionalLogistic, ReportsSeparationAndRejectsWithinStratumAliases) {
    const std::vector<double> cases{0, 1, 0, 1};
    const std::vector<std::int64_t> strata{1, 1, 2, 2};
    const auto separated = conditional_logistic(DenseMatrix<double>(4, 1, {0, 1, 0, 1}), cases, strata);
    EXPECT_EQ(separated.status, ConditionalLogisticStatus::Separated);
    EXPECT_EQ(separated.covariance.rows(), 0u);
    EXPECT_THROW((void)conditional_logistic(DenseMatrix<double>(4, 1, {0, 0, 1, 1}), cases, strata),
                 std::invalid_argument);
    EXPECT_THROW((void)conditional_logistic(DenseMatrix<double>(4, 2, {0, 0, 1, 2, 0, 0, 2, 4}), cases, strata),
                 std::invalid_argument);
    EXPECT_THROW((void)conditional_logistic(DenseMatrix<double>(4, 1, 1), {0, 0, 1, 1}, strata),
                 std::invalid_argument);
}

} // namespace
