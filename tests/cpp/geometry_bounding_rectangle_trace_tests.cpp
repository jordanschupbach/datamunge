#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/geometry/minimum_bounding_rectangle.hpp>
#include <limits>
#include <stdexcept>
using namespace datamunge::geometry;
TEST(BoundingRectangleTrace, RotationsActualExtentsAndQuota) {
  for (int w = 1; w <= 7; ++w)
    for (int h = 1; h <= 5; ++h)
      for (int angle = 0; angle < 12; ++angle) {
        auto points = bounding_rectangle_demo(w, h, angle * .17);
        points.push_back(points[0]);
        auto t = minimum_bounding_rectangle_trace(points);
        EXPECT_NEAR(t.result.area, w * h, 1e-10);
        EXPECT_EQ(t.projections, t.hull.size() * t.hull.size());
        EXPECT_EQ(t.captured_entries, points.size() + 2 * t.hull.size() + 4);
        double best = std::numeric_limits<double>::infinity();
        for (auto& c : t.candidates) {
          auto        origin = t.hull[c.edge];
          long double min_u = 1e100L, max_u = -1e100L, min_v = 1e100L, max_v = -1e100L;
          for (auto p : t.hull) {
            long double u = (static_cast<long double>(p.x) - origin.x) * c.direction.x +
                            (static_cast<long double>(p.y) - origin.y) * c.direction.y,
                        v = -(static_cast<long double>(p.x) - origin.x) * c.direction.y +
                            (static_cast<long double>(p.y) - origin.y) * c.direction.x;
            min_u = std::min(min_u, u);
            max_u = std::max(max_u, u);
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
          }
          EXPECT_NEAR(c.min_u, min_u, 1e-12);
          EXPECT_NEAR(c.max_u, max_u, 1e-12);
          EXPECT_NEAR(c.min_v, min_v, 1e-12);
          EXPECT_NEAR(c.max_v, max_v, 1e-12);
          EXPECT_NEAR(c.area, (max_u - min_u) * (max_v - min_v), 1e-10);
          EXPECT_EQ(c.selected, c.area < best);
          if (c.selected)
            best = c.area;
        }
        EXPECT_DOUBLE_EQ(best, t.result.area);
        EXPECT_DOUBLE_EQ(minimum_bounding_rectangle(points).area, best);
        EXPECT_NO_THROW(minimum_bounding_rectangle_trace(points, t.captured_entries));
        EXPECT_THROW(
            minimum_bounding_rectangle_trace(points, t.captured_entries - 1), std::length_error);
      }
}
TEST(BoundingRectangleTrace, InvalidGeometryAndOwnership) {
  EXPECT_THROW(minimum_bounding_rectangle({}), std::invalid_argument);
  EXPECT_THROW(minimum_bounding_rectangle({{0, 0}, {1, 1}, {2, 2}}), std::invalid_argument);
  EXPECT_THROW(
      minimum_bounding_rectangle({{0, 0}, {1, 0}, {0, std::nan("")}}), std::invalid_argument);
  EXPECT_THROW(bounding_rectangle_demo(0, 2, 0), std::invalid_argument);
  auto p = bounding_rectangle_demo();
  auto t = minimum_bounding_rectangle_trace(p, 0);
  p.clear();
  EXPECT_EQ(t.input.size(), 4);
  EXPECT_NEAR(t.result.area, 12, 1e-12);
  auto huge = std::numeric_limits<double>::max();
  EXPECT_THROW(minimum_bounding_rectangle({{-huge, 0}, {huge, 0}, {0, huge}}), std::overflow_error);
}
