#include <gtest/gtest.h>

#include <array>
#include <climits>
#include <cmath>
#include <datamunge/algorithms/midpoint_circle.hpp>
#include <set>
#include <stdexcept>
using namespace datamunge::algorithms;
TEST(MidpointCircleTrace, ExactPolynomialMidpointDecisionsAndAllEmissions) {
  for (int r = 0; r <= 1000; ++r) {
    auto        t = midpoint_circle_trace(-13, 27, r);
    int         x = r, y = 0;
    std::size_t k = 0;
    while (x >= y) {
      ASSERT_LT(k, t.steps.size());
      auto& s = t.steps[k];
      EXPECT_EQ(s.x, x);
      EXPECT_EQ(s.y, y);
      EXPECT_EQ(s.first_point, 8 * k);
      // The true next midpoint is (x-1/2,y+1). Four times its implicit circle value is integer.
      const long long f    = 4LL * x * x - 4LL * x + 1 + 4LL * (y + 1) * (y + 1) - 4LL * r * r;
      const bool      east = f < 0;
      EXPECT_EQ(s.east, east);
      EXPECT_EQ(s.east, s.d < 0);
      std::array<CirclePoint, 8> expected{
          {{-13 + x, 27 + y},
           {-13 - x, 27 + y},
           {-13 + x, 27 - y},
           {-13 - x, 27 - y},
           {-13 + y, 27 + x},
           {-13 - y, 27 + x},
           {-13 + y, 27 - x},
           {-13 - y, 27 - x}}};
      for (std::size_t j = 0; j < 8; ++j)
        EXPECT_EQ(t.analysis.points[8 * k + j], expected[j]);
      ++y;
      if (!east)
        --x;
      EXPECT_EQ(s.x_after, x);
      EXPECT_EQ(s.y_after, y);
      EXPECT_EQ(s.d_after, s.d + (east ? 2LL * y + 1 : 2LL * (y - x) + 1));
      ++k;
    }
    EXPECT_EQ(k, t.steps.size());
    EXPECT_EQ(t.analysis.iterations, k);
    EXPECT_EQ(t.analysis.points.size(), 8 * k);
    EXPECT_EQ(t.captured_entries, 3 + 9 * k);
    EXPECT_EQ(t.analysis.east_steps + t.analysis.southeast_steps, k);
    auto                  study = midpoint_circle_study(-13, 27, r);
    std::set<CirclePoint> unique(t.analysis.points.begin(), t.analysis.points.end());
    EXPECT_EQ(study.unique_points, (std::vector<CirclePoint>(unique.begin(), unique.end())));
    EXPECT_EQ(study.duplicate_emissions, 8 * k - unique.size());
    double maximum = 0;
    for (std::size_t j = 0; j < study.unique_points.size(); ++j) {
      auto [px, py] = study.unique_points[j];
      double error = std::abs(std::sqrt(double((px + 13) * (px + 13) + (py - 27) * (py - 27))) - r);
      EXPECT_NEAR(study.radial_errors[j], error, 1e-12);
      maximum = std::max(maximum, error);
    }
    EXPECT_NEAR(study.maximum_radial_error, maximum, 1e-12);
  }
}
TEST(MidpointCircleTrace, OriginalDuplicateCountsQuotasAndCoordinateGuards) {
  auto s = midpoint_circle_study(0, 0, midpoint_circle_demo_radius());
  EXPECT_EQ(s.analysis.points.size(), 72);
  EXPECT_EQ(s.analysis.iterations, 9);
  EXPECT_NEAR(s.maximum_radial_error, .3693168768529804, 1e-12);
  auto zero = midpoint_circle_study(INT_MAX, INT_MIN, 0);
  EXPECT_EQ(zero.unique_points.size(), 1);
  EXPECT_EQ(zero.duplicate_emissions, 7);
  EXPECT_DOUBLE_EQ(zero.maximum_radial_error, 0);
  EXPECT_NO_THROW(midpoint_circle_trace(0, 0, 0, 12));
  EXPECT_THROW(midpoint_circle_trace(0, 0, 0, 11), std::length_error);
  EXPECT_NO_THROW(midpoint_circle_trace(0, 0, 0, 0));
  EXPECT_EQ(midpoint_circle_trace(0, 0, -1, 3).captured_entries, 3);
  EXPECT_THROW(midpoint_circle_trace(0, 0, -1, 2), std::length_error);
  EXPECT_TRUE(midpoint_circle(INT_MAX, INT_MIN, -1).empty());
  EXPECT_THROW(midpoint_circle(INT_MAX, 0, 1), std::overflow_error);
  EXPECT_THROW(midpoint_circle(0, INT_MIN, 1), std::overflow_error);
  EXPECT_THROW(midpoint_circle_trace(INT_MAX, 0, 1, 1), std::overflow_error);
  auto shifted  = midpoint_circle(INT_MAX - 12, INT_MIN + 12, 12);
  auto original = midpoint_circle(0, 0, 12);
  ASSERT_EQ(shifted.size(), original.size());
  for (std::size_t i = 0; i < original.size(); ++i) {
    EXPECT_EQ(shifted[i].first, INT_MAX - 12 + original[i].first);
    EXPECT_EQ(shifted[i].second, INT_MIN + 12 + original[i].second);
  }
}
