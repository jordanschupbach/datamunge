#include <datamunge/stats/segmented_regression.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>

namespace {

using datamunge::stats::ProfileEndpointStatus;
using datamunge::stats::segmented_regression;
using datamunge::stats::SegmentedRegressionOptions;

TEST(SegmentedRegression, FindsAnInteriorBreakpointAndRestoresInputOrder) {
    const std::vector<double> x{11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    const std::vector<double> y{4.54, 4.90, 5.38, 5.88, 6.28, 6.85, 7.14, 6.15, 4.90, 4.08, 2.94, 2.05};
    const auto fit = segmented_regression(x, y);
    ASSERT_TRUE(fit.converged);
    ASSERT_EQ(fit.breakpoints.size(), 1u);
    ASSERT_EQ(fit.slopes.size(), 2u);
    ASSERT_EQ(fit.intervals.size(), 1u);
    // Independent R lm.fit() with continuous hinge basis, optimize() and uniroot().
    EXPECT_NEAR(fit.breakpoints[0], 5.100595123554156, 1e-8);
    EXPECT_NEAR(fit.residual_sum_squares, .0537295238095230, 1e-13);
    EXPECT_NEAR(fit.value_at_origin, 1.979047618987979, 1e-9);
    EXPECT_NEAR(fit.slopes[0], 1.025714285757552, 1e-9);
    EXPECT_NEAR(fit.slopes[1], -.462571428514872, 1e-9);
    EXPECT_DOUBLE_EQ(fit.fitted.back(), fit.value_at_origin);
    const auto at_knot = fit.value_at_origin + fit.breakpoints[0] * fit.slopes[0];
    EXPECT_NEAR(fit.fitted.front(), at_knot + (11 - fit.breakpoints[0]) * fit.slopes[1], 1e-12);
    EXPECT_EQ(fit.intervals[0].lower.status, ProfileEndpointStatus::Crossing);
    EXPECT_EQ(fit.intervals[0].upper.status, ProfileEndpointStatus::Crossing);
    EXPECT_NEAR(fit.intervals[0].lower.value, 4.984627643856908, 1e-7);
    EXPECT_NEAR(fit.intervals[0].upper.value, 5.212625001126881, 1e-7);
}

TEST(SegmentedRegression, FitsAndProfilesTwoBreakpointsJointly) {
    std::vector<double> x, y;
    for (int i = 0; i <= 30; ++i) {
        x.push_back(i);
        y.push_back(2 + .5 * i - 1.2 * std::max(0., i - 8.3) + 1.6 * std::max(0., i - 21.6)
                    + .18 * std::sin(i * 2.4));
    }
    const SegmentedRegressionOptions options{.breakpoints = 2, .minimum_segment_size = 3};
    const auto fit = segmented_regression(x, y, options);
    ASSERT_TRUE(fit.converged);
    ASSERT_EQ(fit.intervals.size(), 2u);
    // R optimizes the two knots jointly and reoptimizes the other knot for every endpoint trial.
    EXPECT_NEAR(fit.breakpoints[0], 8.308309200046137, 1e-7);
    EXPECT_NEAR(fit.breakpoints[1], 21.595375386407820, 1e-7);
    EXPECT_NEAR(fit.slopes[0], .497832719105117, 1e-8);
    EXPECT_NEAR(fit.slopes[1], -.700880311845175, 1e-8);
    EXPECT_NEAR(fit.slopes[2], .901998914073456, 1e-8);
    EXPECT_NEAR(fit.residual_sum_squares, .481186628363373, 1e-12);
    EXPECT_NEAR(fit.intervals[0].lower.value, 8.128536593136118, 2e-7);
    EXPECT_NEAR(fit.intervals[0].upper.value, 8.491180607869511, 2e-7);
    EXPECT_NEAR(fit.intervals[1].lower.value, 21.458253277288435, 2e-7);
    EXPECT_NEAR(fit.intervals[1].upper.value, 21.730659614678611, 2e-7);

    auto limited = options;
    limited.max_sweeps = 1;
    const auto stopped = segmented_regression(x, y, limited);
    EXPECT_FALSE(stopped.converged);
    EXPECT_TRUE(stopped.intervals.empty());
}

TEST(SegmentedRegression, CountsDistinctSupportAndReportsAConstrainedBreakpoint) {
    const std::vector<double> x{0, 0, 1, 1, 2, 2};
    const std::vector<double> y{0, .1, 1, 1.2, 1.4, 1.2};
    const auto fit = segmented_regression(x, y);
    EXPECT_DOUBLE_EQ(fit.breakpoints[0], 1);
    ASSERT_EQ(fit.intervals.size(), 1u);
    EXPECT_EQ(fit.intervals[0].lower.status, ProfileEndpointStatus::ParameterBound);
    EXPECT_EQ(fit.intervals[0].upper.status, ProfileEndpointStatus::ParameterBound);
    EXPECT_DOUBLE_EQ(fit.intervals[0].lower.value, 1);
    EXPECT_DOUBLE_EQ(fit.fitted[2], fit.fitted[3]);
    EXPECT_THROW((void)segmented_regression(x, y, {.minimum_segment_size = 3}), std::invalid_argument);
    EXPECT_THROW((void)segmented_regression(x, {0, 0, 1, 1, 1.5, 1.5}), std::domain_error);
}

TEST(SegmentedRegression, ResolvesANarrowGapNearThePredictorMaximum) {
    const std::vector<double> x{0, .1, .2, .3, .4, .5, 1 - 5e-10, 1 - 4e-10, 1 - 3e-10,
                                1 - 2e-10, 1 - 1e-10, 1};
    const double knot = 1 - 3.5e-10;
    std::vector<double> y;
    for (double value : x)
        y.push_back(1 + .5 * value + 5e9 * std::max(0., value - knot));
    const auto fit = segmented_regression(x, y, {.profile_intervals = false});
    EXPECT_NEAR(fit.breakpoints[0], knot, 1e-15);
    EXPECT_NEAR(fit.slopes[0], .5, 1e-9);
    EXPECT_NEAR(fit.slopes[1], 5e9 + .5, 1);
}

} // namespace
