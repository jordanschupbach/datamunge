#include <gtest/gtest.h>

#include <datamunge/geometry/jump_walk_analysis.hpp>
#include <datamunge/geometry/orientation.hpp>
#include <limits>
#include <random>
using namespace datamunge::geometry;
namespace {
bool contains(const std::vector<Point2D>& p, const Triangle& t, Point2D q) {
  return orientation_sign(p[t.a], p[t.b], q) >= 0 && orientation_sign(p[t.b], p[t.c], q) >= 0 &&
         orientation_sign(p[t.c], p[t.a], q) >= 0;
}
void check(const TriangulationLocator& loc, Point2D q, std::uint64_t seed) {
  const auto  t = loc.trace(q, seed);
  const auto& a = t.analysis;
  const auto  b = loc.analyze(q, seed);
  EXPECT_EQ(a.triangle, b.triangle);
  EXPECT_EQ(a.stop, b.stop);
  EXPECT_EQ(a.triangle, loc.locate(q, seed));
  bool any = false;
  for (const auto& f : *loc.tris)
    any |= contains(*loc.points, f, q);
  EXPECT_NE(a.stop, LocateStop::StepLimit);
  EXPECT_EQ(a.stop == LocateStop::Located, any);
  if (any) {
    ASSERT_GE(a.triangle, 0);
    EXPECT_TRUE(contains(*loc.points, (*loc.tris)[a.triangle], q));
  }
  EXPECT_EQ(a.work.samples, t.samples.size());
  EXPECT_EQ(a.work.steps, t.steps.size());
  EXPECT_EQ(a.work.edge_tests, t.tests.size());
  EXPECT_EQ(a.work.random_draws, t.samples.size() + t.steps.size());
  std::size_t crossings = 0, lookups = 0;
  int         current = a.start_triangle;
  std::size_t pos     = 0;
  for (std::size_t i = 0; i < t.steps.size(); ++i) {
    const auto& s = t.steps[i];
    ASSERT_EQ(s.triangle, current);
    const auto&       f      = t.triangles[current];
    const std::size_t ids[]  = {f.a, f.b, f.c};
    int               tested = 0;
    while (pos < t.tests.size() && t.tests[pos].step == i) {
      const auto& e = t.tests[pos++];
      const int   k = (s.first_edge + tested) % 3;
      EXPECT_EQ(e.edge, k);
      EXPECT_EQ(e.triangle, current);
      EXPECT_EQ(e.u, ids[k]);
      EXPECT_EQ(e.v, ids[(k + 1) % 3]);
      EXPECT_EQ(e.sign, orientation_sign(t.points[e.u], t.points[e.v], q));
      ++tested;
      if (e.sign < 0) {
        ++lookups;
        EXPECT_EQ(e.neighbor, loc.neighbor(e.u, e.v, current));
        if (e.neighbor >= 0) {
          current = e.neighbor;
          ++crossings;
        }
        break;
      } else
        EXPECT_EQ(e.neighbor, -2);
    }
    EXPECT_GE(tested, 1);
    EXPECT_LE(tested, 3);
  }
  EXPECT_EQ(pos, t.tests.size());
  EXPECT_EQ(a.work.crossings, crossings);
  EXPECT_EQ(a.work.neighbor_lookups, lookups);
}
} // namespace
TEST(JumpWalkTrace, TinyActualCrossingAndStepLimit) {
  const std::vector<Point2D>  p = {{0, 0}, {4, 0}, {4, 4}, {0, 4}};
  const std::vector<Triangle> f = {{0, 1, 2}, {0, 2, 3}};
  const TriangulationLocator  loc(p, f);
  const auto                  t = loc.trace({3, 1}, 1);
  ASSERT_EQ(t.analysis.start_triangle, 1);
  EXPECT_EQ(t.analysis.triangle, 0);
  EXPECT_EQ(t.analysis.work.crossings, 1U);
  const auto limited = loc.trace({3, 1}, 1, {1, 100});
  EXPECT_EQ(limited.analysis.stop, LocateStop::StepLimit);
  EXPECT_EQ(limited.analysis.triangle, -1);
  check(loc, {3, 1}, 1);
  EXPECT_FALSE(jump_walk_diagram(t).to_svg().empty());
  EXPECT_FALSE(jump_walk_plot(t).to_svg().empty());
}
TEST(JumpWalkTrace, GridBoundaryAndSeedCensus) {
  for (std::size_t side = 1; side <= 6; ++side) {
    const auto                 m = jump_walk_grid(side);
    const TriangulationLocator loc(m.points, m.triangles);
    for (int y = -1; y <= int(side) * 2 + 1; ++y)
      for (int x = -1; x <= int(side) * 2 + 1; ++x)
        for (std::uint64_t seed : {0, 1, 7, 19})
          check(loc, {x * .5, y * .5}, seed);
  }
}
TEST(JumpWalkTrace, RandomDelaunayQueriesAndOriginalFixture) {
  auto                       p = jump_walk_demo_points();
  const auto                 f = delaunay_triangulation(p);
  const TriangulationLocator loc(p, f);
  for (auto q : std::vector<Point2D>{{50, 50}, {20, 80}, {75, 30}, {-10, -10}})
    check(loc, q, 1);
  std::mt19937                           rng(31);
  std::uniform_real_distribution<double> u(-20, 120);
  for (int i = 0; i < 1000; ++i)
    check(loc, {u(rng), u(rng)}, 1 + i);
}
TEST(JumpWalkTrace, ExactSubnormalAndExtremeTriangles) {
  const auto d = std::numeric_limits<double>::denorm_min(), m = std::numeric_limits<double>::max();
  const std::vector<Triangle> f    = {{0, 1, 2}};
  const std::vector<Point2D>  tiny = {{0, 0}, {d, 0}, {0, d}};
  const TriangulationLocator  small(tiny, f);
  check(small, {d, d}, 1);
  check(small, {0, 0}, 0);
  const std::vector<Point2D> huge = {{-m, -m}, {m, -m}, {0, m}};
  const TriangulationLocator big(huge, f);
  check(big, {0, 0}, 1);
  check(big, {m, m}, 1);
}
TEST(JumpWalkTrace, MeshValidationAndPackedNeighborGuards) {
  const std::vector<Point2D>  p = {{0, 0}, {4, 0}, {4, 4}, {0, 4}, {2, -2}, {2, 2}};
  const std::vector<Triangle> f = {{0, 1, 2}, {0, 2, 3}};
  EXPECT_THROW((void)TriangulationLocator(p, {{0, 2, 1}}), std::invalid_argument);
  EXPECT_THROW((void)TriangulationLocator(p, {{0, 0, 1}}), std::invalid_argument);
  EXPECT_THROW((void)TriangulationLocator(p, {{0, 1, 7}}), std::invalid_argument);
  EXPECT_THROW((void)TriangulationLocator(p, {{0, 1, 2}, {0, 1, 3}}), std::invalid_argument);
  EXPECT_THROW(
      (void)TriangulationLocator(p, {{0, 1, 2}, {1, 0, 4}, {0, 1, 5}}), std::invalid_argument);
  auto bad = p;
  bad[0].x = std::numeric_limits<double>::infinity();
  EXPECT_THROW((void)TriangulationLocator(bad, f), std::invalid_argument);
  EXPECT_THROW((void)TriangulationLocator(p, f, 7), std::length_error);
  const TriangulationLocator loc(p, f, 8);
  EXPECT_EQ(loc.neighbor(0, 2, 0), 1);
  EXPECT_EQ(loc.neighbor(1, 5, 0), -1);
  EXPECT_THROW((void)loc.neighbor(0, 1, 1), std::invalid_argument);
  EXPECT_THROW((void)loc.neighbor(0, 1, -1), std::invalid_argument);
  if (sizeof(std::size_t) > 4)
    EXPECT_THROW((void)loc.neighbor(std::size_t(UINT32_MAX) + 1, 0, 0), std::length_error);
}
TEST(JumpWalkTrace, ExactOnlineCapOwnershipAndEmptyQuery) {
  const auto                 m = jump_walk_grid(2);
  const TriangulationLocator loc(m.points, m.triangles);
  auto                       t     = loc.trace({.75, .25});
  const auto                 count = t.samples.size() + t.steps.size() + t.tests.size();
  EXPECT_NO_THROW((void)loc.trace({.75, .25}, 1, {0, count}));
  EXPECT_THROW((void)loc.trace({.75, .25}, 1, {0, count - 1}), std::length_error);
  EXPECT_NO_THROW((void)loc.analyze({.75, .25}, 1, {0, 1}));
  auto copy   = t;
  t.points[0] = {99, 99};
  EXPECT_EQ(copy.points[0].x, 0);
  LocateTrace owned;
  {
    auto local = jump_walk_grid(1);
    owned      = TriangulationLocator(local.points, local.triangles).trace({.75, .25});
  }
  EXPECT_EQ(owned.analysis.stop, LocateStop::Located);
  EXPECT_EQ(owned.points.size(), 4U);
  const std::vector<Point2D>  p;
  const std::vector<Triangle> f;
  const TriangulationLocator  empty(p, f);
  EXPECT_EQ(empty.locate({0, 0}), -1);
  EXPECT_EQ(empty.trace({0, 0}).analysis.stop, LocateStop::Empty);
  EXPECT_THROW(
      (void)empty.locate({std::numeric_limits<double>::quiet_NaN(), 0}), std::invalid_argument);
}
TEST(JumpWalkTrace, HelperCompleteAdmissionAndEditedHistoryGuards) {
  EXPECT_THROW((void)jump_walk_grid(0), std::invalid_argument);
  EXPECT_THROW((void)jump_walk_grid(2, 16), std::length_error);
  EXPECT_EQ(jump_walk_grid(2, 17).triangles.size(), 8U);
  const auto m = jump_walk_grid(1);
  auto       t = TriangulationLocator(m.points, m.triangles).trace({.75, .25});
  EXPECT_THROW((void)jump_walk_plot(t, 1), std::length_error);
  t.tests.clear();
  EXPECT_THROW((void)jump_walk_diagram(t), std::invalid_argument);
  t.steps[0].triangle = 99;
  EXPECT_THROW((void)jump_walk_plot(t), std::invalid_argument);
}
