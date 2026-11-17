#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/newell_normal.hpp>
#include <limits>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
TEST(NewellTrace, IndependentIntegerCrossSumsEveryActualEdgeAndWinding) {
  std::mt19937_64 rng(254263);
  for (int n = 0; n <= 12; ++n)
    for (int repeat = 0; repeat < 100; ++repeat) {
      std::vector<Point3D> p;
      for (int i = 0; i < n; ++i)
        p.push_back(
            {double(int(rng() % 21) - 10),
             double(int(rng() % 21) - 10),
             double(int(rng() % 21) - 10)});
      long long sums[3]{};
      for (int i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k)
          sums[k] += static_cast<long long>(p[i][(k + 1) % 3]) *
                         static_cast<long long>(p[(i + 1) % n][(k + 2) % 3]) -
                     static_cast<long long>(p[i][(k + 2) % 3]) *
                         static_cast<long long>(p[(i + 1) % n][(k + 1) % 3]);
      auto t = newell_trace(p);
      for (int k = 0; k < 3; ++k)
        EXPECT_EQ(t.analysis.raw[k], sums[k]);
      EXPECT_NEAR(
          t.analysis.vector_area,
          .5 * std::hypot(double(sums[0]), double(sums[1]), double(sums[2])),
          1e-12);
      NewellWidePoint running{};
      for (int i = 0; i < n; ++i) {
        auto& e = t.edges[i];
        EXPECT_EQ(e.index, i);
        EXPECT_EQ(e.next, (i + 1) % n);
        for (int k = 0; k < 3; ++k) {
          EXPECT_EQ(e.from[k], p[i][k] - p[0][k]);
          EXPECT_EQ(e.to[k], p[(i + 1) % n][k] - p[0][k]);
          long double cross =
              e.from[(k + 1) % 3] * e.to[(k + 2) % 3] - e.from[(k + 2) % 3] * e.to[(k + 1) % 3];
          EXPECT_EQ(e.cross[k], cross);
          running[k] += cross;
          EXPECT_EQ(e.sum_after[k], running[k]);
        }
      }
      EXPECT_EQ(t.captured_entries, 3 + 5 * n);
      EXPECT_NO_THROW(newell_trace(p, t.captured_entries));
      EXPECT_THROW(newell_trace(p, t.captured_entries - 1), std::length_error);
      auto reversed = p;
      std::reverse(reversed.begin(), reversed.end());
      auto back = newell_analyze(reversed);
      for (int k = 0; k < 3; ++k)
        EXPECT_EQ(back.raw[k], -t.analysis.raw[k]);
      EXPECT_EQ(newell_normal_raw(p), t.analysis.raw);
      EXPECT_EQ(newell_normal(p), t.analysis.unit);
      EXPECT_EQ(newell_polygon_area(p), t.analysis.vector_area);
    }
}
TEST(NewellTrace, OriginalPlanarityCorrectionTinyWideTranslationAndUndefinedControls) {
  auto fixtures = newell_demo_polygons();
  EXPECT_EQ(fixtures.size(), 4);
  auto square = newell_analyze(fixtures[0].polygon);
  EXPECT_EQ(square.unit, (Point3D{0, 0, 1}));
  EXPECT_EQ(square.vector_area, 4);
  auto tri = newell_analyze(fixtures[1].polygon);
  EXPECT_NEAR(tri.vector_area, std::sqrt(3.) / 2, 1e-15);
  EXPECT_NEAR(tri.unit[0], -1 / std::sqrt(3.), 1e-15);
  auto oldquad = newell_analyze(fixtures[2].polygon);
  EXPECT_LT(oldquad.max_plane_distance, 1e-16);
  auto warped = newell_analyze(fixtures[3].polygon);
  EXPECT_GT(warped.max_plane_distance, .09);
  EXPECT_LT(warped.max_plane_distance, .11);
  auto shifted = fixtures[0].polygon;
  for (auto& p : shifted)
    for (auto& x : p)
      x += 1e16;
  auto a = newell_analyze(shifted);
  EXPECT_EQ(a.vector_area, 4);
  EXPECT_EQ(a.unit, square.unit);
  double tiny  = std::numeric_limits<double>::denorm_min();
  auto   small = newell_analyze({{0, 0, 0}, {tiny, 0, 0}, {0, tiny, 0}});
  EXPECT_TRUE(small.has_direction);
  EXPECT_EQ(small.vector_area, 0);
  EXPECT_EQ(small.raw, (Point3D{}));
  EXPECT_EQ(small.unit, (Point3D{0, 0, 1}));
  auto bow = newell_analyze({{0, 0, 0}, {1, 1, 0}, {0, 1, 0}, {1, 0, 0}});
  EXPECT_FALSE(bow.has_direction);
  EXPECT_EQ(bow.vector_area, 0);
  EXPECT_EQ(bow.signed_plane_distances, (std::vector<double>{0, 0, 0, 0}));
  EXPECT_THROW(newell_trace({{0, 0, NAN}}, 1), std::invalid_argument);
  double m = std::numeric_limits<double>::max();
  EXPECT_THROW(newell_analyze({{0, 0, 0}, {m, 0, 0}, {0, m, 0}}), std::overflow_error);
}
