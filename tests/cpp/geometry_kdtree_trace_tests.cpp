#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/geometry/kdtree_analysis.hpp>
#include <limits>
#include <numeric>
using namespace datamunge::geometry;
TEST(KDTrace, IntegerGridNearestKAndRadiusAgainstFullScan) {
  std::vector<Point2D> p;
  for (int x = -3; x <= 3; ++x)
    for (int y = -3; y <= 3; ++y)
      p.push_back({double(x), double(y)});
  const KDTree2D tree(p);
  for (int x = -8; x <= 8; ++x)
    for (int y = -8; y <= 8; ++y) {
      Point2D                                     q{x * .5, y * .5};
      std::vector<std::pair<double, std::size_t>> expected;
      for (std::size_t i = 0; i < p.size(); ++i)
        expected.push_back({(p[i].x - q.x) * (p[i].x - q.x) + (p[i].y - q.y) * (p[i].y - q.y), i});
      std::sort(expected.begin(), expected.end());
      const auto t = tree.nearest_trace(q);
      ASSERT_EQ(t.original_index, expected[0].second);
      EXPECT_EQ(tree.nearest(q), p[expected[0].second]);
      const auto k = tree.k_nearest(q, 7);
      ASSERT_EQ(k.size(), 7U);
      for (std::size_t i = 0; i < k.size(); ++i)
        EXPECT_EQ(k[i], p[expected[i].second]);
      auto                 radius = tree.points_in_radius(q, 1.5);
      std::vector<Point2D> want;
      for (auto [d, i] : expected)
        if (d <= 2.25)
          want.push_back(p[i]);
      const auto less = [](auto a, auto b) { return a.x < b.x || (a.x == b.x && a.y < b.y); };
      std::sort(radius.begin(), radius.end(), less);
      std::sort(want.begin(), want.end(), less);
      EXPECT_EQ(radius, want);
      std::size_t visits = 0, planes = 0, pruned = 0;
      for (const auto& e : t.probes) {
        if (e.action == KDProbeAction::Visit) {
          ++visits;
          EXPECT_EQ(
              e.measure,
              std::hypot(
                  static_cast<long double>(p[e.original_index].x) - q.x,
                  static_cast<long double>(p[e.original_index].y) - q.y));
          EXPECT_LE(e.best_distance_after, e.best_distance_before);
        } else {
          ++planes;
          pruned += !e.explore_far;
          EXPECT_EQ(e.explore_far, e.measure <= e.best_distance_before);
        }
      }
      EXPECT_EQ(t.work.visited_nodes, visits);
      EXPECT_EQ(t.work.plane_tests, planes);
      EXPECT_EQ(t.work.pruned_subtrees, pruned);
    }
}
TEST(KDTrace, DuplicateTieAndDistanceOverflowUnderflowCorrections) {
  const KDTree2D ties({{2, 0}, {0, 0}, {2, 0}});
  EXPECT_EQ(ties.nearest_trace({1, 0}).original_index, 0U);
  EXPECT_EQ(ties.k_nearest({1, 0}, 9).size(), 3U);
  const double d = std::numeric_limits<double>::denorm_min(),
               m = std::numeric_limits<double>::max();
  const KDTree2D tiny({{3 * d, 0}, {d, 0}});
  EXPECT_EQ(tiny.nearest_trace({0, 0}).original_index, 1U);
  EXPECT_EQ(tiny.points_in_radius({0, 0}, 2 * d).size(), 1U);
  const KDTree2D huge({{m, 0}, {-m / 2, 0}});
  EXPECT_EQ(huge.nearest_trace({0, 0}).original_index, 1U);
}
TEST(KDTrace, FiniteDomainQuotaOwnershipAndEmptyConventions) {
  EXPECT_THROW(
      (void)KDTree2D({{std::numeric_limits<double>::quiet_NaN(), 0}}), std::invalid_argument);
  EXPECT_THROW((void)KDTree2D({{0, 0}, {1, 1}}, 1), std::length_error);
  const KDTree2D empty({});
  EXPECT_THROW((void)empty.nearest({0, 0}), std::logic_error);
  EXPECT_TRUE(empty.k_nearest({0, 0}, 10).empty());
  EXPECT_TRUE(empty.points_in_radius({0, 0}, 1).empty());
  const KDTree2D tree({{0, 0}, {1, 0}, {0, 1}});
  EXPECT_THROW((void)tree.points_in_radius({0, 0}, -1), std::invalid_argument);
  EXPECT_EQ(tree.points_in_radius({0, 0}, std::numeric_limits<double>::infinity()).size(), 3U);
  const auto t = tree.nearest_trace({.1, .1});
  EXPECT_NO_THROW((void)tree.nearest_trace({.1, .1}, t.probes.size()));
  if (t.probes.size() > 1)
    EXPECT_THROW((void)tree.nearest_trace({.1, .1}, t.probes.size() - 1), std::length_error);
  EXPECT_FALSE(kd_trace_plot(t).to_svg().empty());
  auto copy      = t;
  copy.points[0] = {99, 99};
  EXPECT_EQ(t.points[0].x, 0);
}
