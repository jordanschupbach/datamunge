#include <datamunge/stats/firth_logistic.hpp>

#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>

namespace {

using datamunge::linalg::DenseMatrix;
using datamunge::stats::firth_logistic;
using datamunge::stats::ProfileEndpointStatus;

TEST(FirthLogistic, FitsCompleteSeparationAndProfilesTheFullPenalty) {
    const DenseMatrix<double> design(2, 2, {1, 0, 1, 1});
    const auto grouped = firth_logistic(design, {0, 5}, {.trials = {6, 5}});
    DenseMatrix<double> binary(11, 2, 1);
    std::vector<double> successes(11, 1);
    for (std::size_t i = 0; i < 6; ++i)
        binary(i, 1) = successes[i] = 0;
    const auto expanded = firth_logistic(binary, successes);
    // Closed-form estimates, and independent R determinant(), optimize(), and uniroot() profiling.
    EXPECT_NEAR(grouped.coefficients[0], -std::log(13.0), 1e-7);
    EXPECT_NEAR(grouped.coefficients[1], std::log(143.0), 1e-7);
    EXPECT_NEAR(grouped.fitted[0], 1.0 / 14, 1e-8);
    EXPECT_NEAR(grouped.fitted[1], 11.0 / 12, 1e-8);
    EXPECT_NEAR(grouped.penalized_log_likelihood, -1.821647691312708, 1e-13);
    EXPECT_NEAR(grouped.covariance(0, 1), -2.512820512820513, 1e-7);
    EXPECT_NEAR(grouped.covariance(1, 1), 5.131002331002334, 1e-7);
    ASSERT_EQ(grouped.intervals.size(), 2u);
    EXPECT_EQ(grouped.intervals[1].lower.status, ProfileEndpointStatus::Crossing);
    EXPECT_EQ(grouped.intervals[1].upper.status, ProfileEndpointStatus::Crossing);
    EXPECT_NEAR(grouped.intervals[0].lower.value, -7.435664506154666, 1e-7);
    EXPECT_NEAR(grouped.intervals[0].upper.value, -.430277414924091, 1e-7);
    EXPECT_NEAR(grouped.intervals[1].lower.value, 1.711720222990715, 1e-7);
    EXPECT_NEAR(grouped.intervals[1].upper.value, 10.767205984365249, 1e-7);
    EXPECT_NEAR(expanded.coefficients[1], grouped.coefficients[1], 1e-8);
    EXPECT_NEAR(expanded.intervals[1].upper.value, grouped.intervals[1].upper.value, 1e-8);
    EXPECT_NEAR(grouped.predict(design)[1], grouped.fitted[1], 1e-15);
}

TEST(FirthLogistic, MatchesIndependentWeightedOptimizationWithSeveralCovariates) {
    const DenseMatrix<double> x(9, 3, {
        1, -2, 0, 1, -1.4, 1, 1, -1, 0, 1, -.5, 2, 1, .2, 1, 1, .5, 2, 1, 1.1, 1, 1, 2, 0, 1, 3, 2,
    });
    const auto fit = firth_logistic(x, {0, 2, 0, 2, 1, 3, 2, 3, 4}, {
        .trials = {3, 4, 3, 5, 3, 4, 6, 4, 4}, .weights = {1, .5, 1.4, 1, 1.25, .7, 1.5, .7, 1.15},
        .profile_intervals = false,
    });
    EXPECT_NEAR(fit.coefficients[0], -1.564187746804482, 1e-7);
    EXPECT_NEAR(fit.coefficients[1], .730603967189978, 1e-7);
    EXPECT_NEAR(fit.coefficients[2], .828439700963683, 1e-7);
    EXPECT_NEAR(fit.penalized_log_likelihood, -16.267598515112635, 1e-12);
    EXPECT_NEAR(fit.covariance(2, 0), -.368935390321758, 1e-7);
    EXPECT_TRUE(fit.intervals.empty());
}

TEST(FirthLogistic, IsEquivariantToPredictorUnitsAndRejectsNonidentifiableDesigns) {
    const DenseMatrix<double> design(2, 2, {1, 0, 1, 1e90});
    const auto fit = firth_logistic(design, {0, 5}, {.trials = {6, 5}});
    EXPECT_NEAR(fit.coefficients[1] * 1e90, std::log(143.0), 1e-7);
    EXPECT_NEAR(fit.intervals[1].lower.value * 1e90, 1.711720222990715, 1e-7);
    EXPECT_NEAR(fit.penalized_log_likelihood, std::log(1e90) - 1.821647691312708, 1e-13);
    EXPECT_THROW((void)firth_logistic(DenseMatrix<double>(3, 2, 1), {0, 1, 0}), std::invalid_argument);
    EXPECT_THROW((void)firth_logistic(DenseMatrix<double>(3, 1, 1), {0, 2, 0}), std::invalid_argument);
    EXPECT_THROW((void)fit.predict(DenseMatrix<double>(1, 1, 1)), std::invalid_argument);
    EXPECT_THROW((void)firth_logistic(design, {0, 5}, {.trials = {6, 5}, .max_iterations = 1}),
                 std::runtime_error);
}

TEST(FirthLogistic, RefinesTheFitWhenProfilingFindsAHigherMode) {
    const DenseMatrix<double> x(7, 3, {
        1, .338435682186, .0740426614129,
        1, .276062097918, 1.2052363184598,
        1, -.672600668507, -1.2124254259426,
        1, -.603987696116, -.7145076751335,
        1, -.460761463316, -.5107869822516,
        1, .986681130604, -2.2401108825983,
        1, -.712256088257, 1.2390627565217,
    });
    const auto fit = firth_logistic(x, {0, 1, 3, 1, 2, 0, 1}, {
        .trials = {2, 1, 3, 1, 2, 6, 1},
        .weights = {1.279933983064, 1.794427251932, 1.973571226303, 1.796447167336,
                    1.473329483788, .514154911344, 1.610514161782},
    });
    // Base R maximizes the same penalty from starts on both sides of the lower mode.
    EXPECT_NEAR(fit.penalized_log_likelihood, -2.709596267939141, 1e-11);
    EXPECT_NEAR(fit.coefficients[0], .678376984750120, 1e-6);
    EXPECT_NEAR(fit.coefficients[1], -7.947570742517225, 1e-6);
    EXPECT_NEAR(fit.coefficients[2], 2.577484189802141, 1e-6);
    ASSERT_EQ(fit.intervals.size(), 3u);
    for (std::size_t j = 0; j < fit.intervals.size(); ++j) {
        EXPECT_LT(fit.intervals[j].lower.value, fit.coefficients[j]);
        EXPECT_GT(fit.intervals[j].upper.value, fit.coefficients[j]);
    }
}

} // namespace
