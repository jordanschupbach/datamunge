#include <datamunge/stats/nonlinear_least_squares.hpp>

#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>

namespace {

using datamunge::linalg::DenseMatrix;
using datamunge::stats::NonlinearLeastSquaresOptions;
using datamunge::stats::nonlinear_least_squares;
using datamunge::stats::ProfileEndpointStatus;

TEST(NonlinearLeastSquares, MatchesWeightedRFitAndRefitsNuisanceAmplitudeForRateProfile) {
    const std::vector<double> time{0, .5, 1, 2, 3, 4, 6, 8};
    const std::vector<double> response{4.05, 3.39, 2.89, 2.07, 1.61, 1.19, .71, .46};
    const auto mean = [&](const std::vector<double>& p) {
        std::vector<double> predictions;
        for (const auto t : time)
            predictions.push_back(p[0] * std::exp(-p[1] * t));
        return predictions;
    };
    NonlinearLeastSquaresOptions options{
        .weights = {1, 2, 1, 1, 2, 3, 1, 2},
        .jacobian = [&](const std::vector<double>& p) {
            DenseMatrix<double> derivatives(time.size(), 2);
            for (std::size_t i = 0; i < time.size(); ++i) {
                derivatives(i, 0) = std::exp(-p[1] * time[i]);
                derivatives(i, 1) = -p[0] * time[i] * derivatives(i, 0);
            }
            return derivatives;
        },
        .profiles = {{1, {.lower_bound = 0, .initial_step = .01, .tolerance = 1e-11}},
                     {0, {.lower_bound = 0, .initial_step = .05, .tolerance = 1e-11}}},
    };
    const auto fit = nonlinear_least_squares(response, {3, .2}, mean, options);
    // Base R nls(), vcov() and uniroot() after independently solving for amplitude at fixed rate.
    EXPECT_NEAR(fit.parameters[0], 3.952769255609046, 1e-8);
    EXPECT_NEAR(fit.parameters[1], .298637529791873, 1e-9);
    EXPECT_NEAR(fit.residual_sum_squares, .0445464563758153, 1e-12);
    EXPECT_NEAR(fit.residual_variance, .00742440939596922, 1e-13);
    EXPECT_NEAR(fit.covariance(0, 0), .00309639605873595, 1e-9);
    EXPECT_NEAR(fit.covariance(1, 0), .000290070201024764, 1e-10);
    EXPECT_NEAR(fit.covariance(1, 1), .000068098602466383, 1e-11);
    EXPECT_DOUBLE_EQ(fit.covariance(1, 0), fit.covariance(0, 1));
    EXPECT_DOUBLE_EQ(fit.residuals[0], response[0] - fit.fitted[0]);
    ASSERT_EQ(fit.intervals.size(), 2u);
    EXPECT_EQ(fit.intervals[0].lower.status, ProfileEndpointStatus::Crossing);
    EXPECT_EQ(fit.intervals[0].upper.status, ProfileEndpointStatus::Crossing);
    EXPECT_NEAR(fit.intervals[0].lower.value, .282764334360332, 1e-10);
    EXPECT_NEAR(fit.intervals[0].upper.value, .315146021006097, 1e-10);
    EXPECT_NEAR(fit.intervals[1].lower.value, 3.845328059736434, 1e-10);
    EXPECT_NEAR(fit.intervals[1].upper.value, 4.061038242316828, 1e-10);

    options.jacobian = {};
    const auto numeric = nonlinear_least_squares(response, {3, .2}, mean, options);
    EXPECT_NEAR(numeric.parameters[0], fit.parameters[0], 1e-8);
    EXPECT_NEAR(numeric.parameters[1], fit.parameters[1], 1e-9);
    EXPECT_NEAR(numeric.intervals[0].upper.value, fit.intervals[0].upper.value, 1e-10);
}

TEST(NonlinearLeastSquares, InvertsOneParameterProfileAndDetectsUnidentifiedParameters) {
    const auto constant = [](const auto& p) { return std::vector<double>(3, p[0]); };
    const auto fit = nonlinear_least_squares({1, 2, 3}, {0}, constant,
                                            {.profiles = {{0, {.tolerance = 1e-11}}}});
    const auto half_width = std::sqrt(2.0 / 3 * std::expm1(fit.intervals[0].likelihood_ratio_cutoff / 3));
    EXPECT_NEAR(fit.parameters[0], 2, 1e-8);
    EXPECT_NEAR(fit.covariance(0, 0), 1.0 / 3, 1e-11);
    EXPECT_NEAR(fit.intervals[0].lower.value, 2 - half_width, 1e-8);
    EXPECT_NEAR(fit.intervals[0].upper.value, 2 + half_width, 1e-8);
    EXPECT_THROW((void)nonlinear_least_squares({1, 2, 3}, {2, 0}, constant), std::runtime_error);
    EXPECT_THROW((void)nonlinear_least_squares({2, 2, 2}, {0}, constant, {.profiles = {{0}}}),
                 std::domain_error);
    EXPECT_THROW((void)nonlinear_least_squares({1, 2, 3}, {0}, constant, {.weights = {0, 1, 1}}),
                 std::invalid_argument);
    EXPECT_THROW((void)nonlinear_least_squares({1, 2, 3, 4}, {0}, constant), std::invalid_argument);
}

} // namespace
