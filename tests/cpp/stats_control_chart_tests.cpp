#include <gtest/gtest.h>

#include <datamunge/stats/control_chart.hpp>

#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace stats = datamunge::stats;

TEST(ControlChart, AttributeLimitsUseBaselineAndSubgroupDenominators) {
    const stats::ControlChartOptions baseline{.baseline_size = 2};
    const auto c = stats::c_chart({3, 5, 10, 11}, baseline);
    EXPECT_DOUBLE_EQ(c.points[2].upper, 10);
    EXPECT_FALSE(c.points[2].outside);
    EXPECT_TRUE(c.points[3].outside);
    EXPECT_DOUBLE_EQ(c.points[0].lower, 0);

    const auto u = stats::u_chart({2, 4, 40}, {0.5, 2.5, 4}, baseline);
    EXPECT_DOUBLE_EQ(u.points[2].center, 2);
    EXPECT_DOUBLE_EQ(u.points[2].value, 10);
    EXPECT_NEAR(u.points[2].upper, 2 + 3 * std::sqrt(0.5), 1e-12);
    EXPECT_GT(u.points[0].upper, u.points[1].upper);

    const auto p = stats::p_chart({5, 10, 19}, {50, 100, 20}, baseline);
    EXPECT_DOUBLE_EQ(p.points[2].center, 0.1);
    EXPECT_DOUBLE_EQ(p.points[2].lower, 0);
    EXPECT_NEAR(p.points[2].upper, 0.1 + 3 * std::sqrt(0.09 / 20), 1e-12);
    EXPECT_TRUE(p.points[2].outside);

    const auto np = stats::np_chart({5, 5, 16}, 20, baseline);
    EXPECT_DOUBLE_EQ(np.points[2].center, 5);
    EXPECT_NEAR(np.points[2].upper, 5 + 3 * std::sqrt(3.75), 1e-12);
    EXPECT_TRUE(np.points[2].outside);
}

TEST(ControlChart, LaneyRescalesByBaselineMovingRange) {
    const stats::ControlChartOptions baseline{.baseline_size = 2};
    const double d2 = 2 / std::sqrt(std::numbers::pi);
    const auto p = stats::laney_p_chart({10, 30, 100}, {100, 100, 100}, baseline);
    EXPECT_NEAR(p.points[2].upper, 0.2 + 3 * 0.2 / d2, 1e-12);
    const auto u = stats::laney_u_chart({10, 30, 100}, {10, 10, 10}, baseline);
    EXPECT_NEAR(u.points[2].upper, 2 + 3 * 2 / d2, 1e-12);
    EXPECT_TRUE(u.points[2].outside);
    EXPECT_THROW(stats::laney_p_chart({0, 0}, {5, 5}), std::invalid_argument);
}

TEST(ControlChart, IndividualsAndMovingRangeShareRangeScale) {
    const stats::ControlChartOptions baseline{.baseline_size = 3};
    const auto individuals = stats::individuals_chart({2, 4, 2, 12}, baseline);
    const auto ranges = stats::moving_range_chart({2, 4, 2, 12}, baseline);
    EXPECT_NEAR(individuals.points.back().upper, 8. / 3 + 3 * std::sqrt(std::numbers::pi), 1e-12);
    EXPECT_EQ(ranges.points.front().index, 2U);
    EXPECT_DOUBLE_EQ(ranges.points.back().value, 10);
    EXPECT_DOUBLE_EQ(ranges.points.back().center, 2);
    EXPECT_NEAR(ranges.points.back().upper, 2 + 3 * std::sqrt(2 * std::numbers::pi - 4), 1e-12);
    EXPECT_TRUE(individuals.points.back().outside);
    EXPECT_TRUE(ranges.points.back().outside);
}

TEST(ControlChart, SubgroupChartsUseNormalRangeAndStandardDeviationMoments) {
    const std::vector<std::vector<double>> groups{{1, 3}, {2, 4}, {11, 12, 13}};
    const stats::ControlChartOptions baseline{.baseline_size = 2};
    const auto xbar = stats::xbar_chart(groups, baseline);
    const auto range = stats::r_chart(groups, baseline);
    const auto xbar_s = stats::xbar_s_chart(groups, baseline);
    const auto s = stats::s_chart(groups, baseline);

    EXPECT_DOUBLE_EQ(xbar.points.back().value, 12);
    EXPECT_DOUBLE_EQ(xbar.points.back().center, 2.5);
    EXPECT_NEAR(xbar.points.back().upper, 2.5 + 3 * std::sqrt(std::numbers::pi / 3), 1e-12);
    EXPECT_NEAR(range.points.back().center, 1.692569 * std::sqrt(std::numbers::pi), 1e-12);
    EXPECT_DOUBLE_EQ(range.points.back().value, 2);
    EXPECT_NEAR(xbar_s.points.back().upper, xbar.points.back().upper, 1e-12);
    EXPECT_DOUBLE_EQ(s.points.back().value, 1);
    EXPECT_NEAR(s.points.back().center, std::numbers::pi / 2, 1e-12);
    EXPECT_TRUE(xbar_s.points.back().outside);
}

TEST(ControlChart, SequentialChartsPreserveStartupAndDirection) {
    const auto ewma = stats::ewma_chart({2, 4}, 0, 1, 0.5, 3);
    EXPECT_DOUBLE_EQ(ewma.points[0].value, 1);
    EXPECT_DOUBLE_EQ(ewma.points[0].upper, 1.5);
    EXPECT_DOUBLE_EQ(ewma.points[1].value, 2.5);
    EXPECT_NEAR(ewma.points[1].upper, 3 * std::sqrt(0.3125), 1e-12);
    EXPECT_TRUE(ewma.points[1].outside);

    const auto moving = stats::moving_average_chart({2, 4, 6}, 2, 0, 1);
    ASSERT_EQ(moving.points.size(), 2U);
    EXPECT_EQ(moving.points[0].index, 2U);
    EXPECT_DOUBLE_EQ(moving.points[0].value, 3);
    EXPECT_DOUBLE_EQ(moving.points[1].value, 5);
    EXPECT_NEAR(moving.points[0].upper, 3 / std::sqrt(2.), 1e-12);

    const auto [up, down] = stats::cusum_chart({2, 2, -4, -4}, 0, 1, 0.5, 2);
    EXPECT_DOUBLE_EQ(up.points[1].value, 3);
    EXPECT_TRUE(up.points[1].outside);
    EXPECT_DOUBLE_EQ(up.points[2].value, 0);
    EXPECT_DOUBLE_EQ(down.points[2].value, 3.5);
    EXPECT_TRUE(down.points[2].outside);
}

TEST(ControlChart, RareEventLimitsUseDistributionTails) {
    const stats::ControlChartOptions baseline{.baseline_size = 2};
    const auto t = stats::t_chart({1, 3, 30}, baseline);
    const double tail = std::erfc(3 / std::sqrt(2.)) / 2;
    EXPECT_NEAR(t.points.back().center, 2 * std::log(2.), 1e-12);
    EXPECT_NEAR(t.points.back().upper, -2 * std::log(tail), 1e-12);
    EXPECT_TRUE(t.points.back().outside);
    const auto g = stats::g_chart({1, 3, 30}, baseline);
    EXPECT_DOUBLE_EQ(g.points.back().center, 1);
    EXPECT_DOUBLE_EQ(g.points.back().lower, 0);
    EXPECT_DOUBLE_EQ(g.points.back().upper, 16);
    EXPECT_TRUE(g.points.back().outside);
    EXPECT_TRUE(stats::g_chart({0, 0, 1}, baseline).points.back().outside);
}

TEST(ControlChart, ValidatesDataAndRendersSignalsAndSingletons) {
    EXPECT_THROW(stats::c_chart({}), std::invalid_argument);
    EXPECT_THROW(stats::c_chart({1}, {.baseline_size = 2}), std::invalid_argument);
    EXPECT_THROW(stats::u_chart({1, 2}, {1.}), std::invalid_argument);
    EXPECT_THROW(stats::u_chart({1, 2}, {0., 1.}), std::invalid_argument);
    EXPECT_THROW(stats::np_chart({4}, 3), std::invalid_argument);
    EXPECT_THROW(stats::xbar_chart({{2.}}), std::invalid_argument);
    EXPECT_THROW(stats::r_chart({std::vector<double>(11)}), std::invalid_argument);
    EXPECT_NO_THROW(stats::s_chart({std::vector<double>(11)}));
    EXPECT_THROW(stats::individuals_chart({1, std::numeric_limits<double>::infinity()}), std::invalid_argument);
    EXPECT_THROW(stats::ewma_chart({1}, 0, 1, 0), std::invalid_argument);
    EXPECT_THROW(stats::moving_average_chart({1}, 2, 0, 1), std::invalid_argument);
    EXPECT_THROW(stats::cusum_chart({1}, 0, 1, -1), std::invalid_argument);
    EXPECT_THROW(stats::t_chart({0}), std::invalid_argument);
    const auto svg = stats::c_chart({2, 3, 4, 20}, {.baseline_size = 3}).plot().to_svg();
    EXPECT_NE(svg.find("Outside limits"), std::string::npos);
    const auto single = stats::c_chart({1}).plot().to_svg();
    EXPECT_NE(single.find("Upper limit"), std::string::npos);
}

TEST(ControlChart, MovingStandardDeviationUsesCompleteWindows) {
    const auto chart = stats::moving_sd_chart({1, 3, 7}, 2, 1);
    ASSERT_EQ(chart.points.size(), 2U);
    EXPECT_EQ(chart.points[0].index, 2U);
    EXPECT_NEAR(chart.points[0].value, std::sqrt(2.), 1e-12);
    EXPECT_NEAR(chart.points[1].value, 2 * std::sqrt(2.), 1e-12);
    EXPECT_NEAR(chart.points[0].center, std::sqrt(2 / std::numbers::pi), 1e-12);
    EXPECT_TRUE(chart.points[1].outside);
    EXPECT_THROW(stats::moving_sd_chart({1, 3}, 1, 1), std::invalid_argument);
}

TEST(ControlChart, HotellingUsesCovarianceAndChiSquaredUpperLimit) {
    const datamunge::linalg::DenseMatrix<double> covariance(2, 2, {4., 1., 1., 1.});
    const auto chart = stats::hotelling_t2_chart({{1., 1.}, {3., 3.}, {9., 9.}}, {1., 1.}, covariance, 0.01);
    EXPECT_DOUBLE_EQ(chart.points[0].value, 0);
    EXPECT_DOUBLE_EQ(chart.points[1].value, 4);
    EXPECT_NEAR(chart.points[1].upper, -2 * std::log(0.01), 1e-9);
    EXPECT_TRUE(chart.points.back().outside);
    const datamunge::linalg::DenseMatrix<double> singular(2, 2, {1., 1., 1., 1.});
    EXPECT_THROW(stats::hotelling_t2_chart({{1., 1.}}, {0., 0.}, singular), std::invalid_argument);
    EXPECT_THROW(stats::hotelling_t2_chart({{1.}}, {0., 0.}, covariance), std::invalid_argument);
}
