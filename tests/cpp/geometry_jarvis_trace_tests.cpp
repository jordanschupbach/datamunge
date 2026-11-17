#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/geometry/jarvis_analysis.hpp>
#include <datamunge/geometry/jarvis_march.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <set>
using namespace datamunge::geometry;
namespace {
void check(const std::vector<Point2D>& input, const JarvisTrace& t) {
  const auto& a = t.analysis;
  const auto& w = a.work;
  const auto  u = t.sorted.size(), h = a.hull.size();
  EXPECT_EQ(w.input_points, input.size());
  EXPECT_EQ(w.distinct_points, u);
  ASSERT_EQ(h, a.input_indices.size());
  for (std::size_t i = 0; i < u; ++i) {
    const auto& row = t.sorted[i];
    ASSERT_LT(row.input_index, input.size());
    EXPECT_EQ(row.point, input[row.input_index]);
    for (std::size_t j = 0; j < row.input_index; ++j)
      EXPECT_FALSE(input[j] == row.point);
    if (i) {
      const auto& prev = t.sorted[i - 1].point;
      EXPECT_TRUE(prev.x < row.point.x || (prev.x == row.point.x && prev.y < row.point.y));
    }
  }
  for (std::size_t i = 0; i < h; ++i) {
    ASSERT_LT(a.input_indices[i], input.size());
    EXPECT_EQ(a.hull[i], input[a.input_indices[i]]);
  }
  EXPECT_EQ(a.hull, convex_hull(input));
  if (u < 3) {
    EXPECT_TRUE(t.events.empty());
    EXPECT_EQ(w.orientation_tests, 0);
    EXPECT_EQ(w.wrap_steps, 0);
    return;
  }
  EXPECT_EQ(w.orientation_tests, h * (u - 2));
  EXPECT_EQ(w.wrap_steps, h);
  EXPECT_EQ(t.events.size(), h * u);
  std::size_t tests = 0, collinear = 0, updates = 0, step = 0, cur = 0, next = 0, examined = 0;
  for (const auto& e : t.events) {
    EXPECT_EQ(e.step, step);
    EXPECT_EQ(e.origin, cur);
    ASSERT_LT(e.origin, u);
    ASSERT_LT(e.before, u);
    ASSERT_LT(e.after, u);
    if (e.action == JarvisAction::Start) {
      next = cur + 1 == u ? 0 : cur + 1;
      EXPECT_EQ(e.before, next);
      EXPECT_EQ(e.after, next);
      EXPECT_EQ(a.hull[step], t.sorted[cur].point);
      examined = 0;
    } else if (e.action == JarvisAction::Test) {
      ++tests;
      ++examined;
      ASSERT_LT(e.examined, u);
      EXPECT_EQ(e.before, next);
      EXPECT_NE(e.examined, cur);
      EXPECT_NE(e.examined, cur + 1 == u ? 0 : cur + 1);
      const auto p = t.sorted[cur].point, b = t.sorted[next].point, q = t.sorted[e.examined].point;
      const auto turn = orientation_sign(p, b, q);
      EXPECT_EQ(e.turn, turn);
      bool farther = false;
      if (turn == 0) {
        ++collinear;
        farther = b.x > p.x ? q.x > b.x : b.x < p.x ? q.x < b.x : b.y > p.y ? q.y > b.y : q.y < b.y;
      }
      EXPECT_EQ(e.farther, farther);
      if (turn < 0 || (turn == 0 && farther)) {
        next = e.examined;
        ++updates;
      }
      EXPECT_EQ(e.after, next);
    } else {
      EXPECT_EQ(examined, u - 2);
      EXPECT_EQ(e.before, next);
      EXPECT_EQ(e.after, next);
      // Each actually selected edge supports all represented input points.
      for (const auto& q : t.sorted)
        EXPECT_GE(orientation_sign(t.sorted[cur].point, t.sorted[next].point, q.point), 0);
      cur = next;
      ++step;
    }
  }
  EXPECT_EQ(step, h);
  EXPECT_EQ(cur, 0);
  EXPECT_EQ(tests, w.orientation_tests);
  EXPECT_EQ(collinear, w.collinear_tests);
  EXPECT_EQ(updates, w.candidate_updates);
}
} // namespace
TEST(JarvisTrace, TinyActualCandidatesAndOriginalIdentity) {
  const std::vector<Point2D> p{{0, 0}, {4, 0}, {4, 4}, {0, 4}, {2, 2}};
  const auto                 t = jarvis_march_trace(p);
  check(p, t);
  ASSERT_EQ(t.events.size(), 20);
  const auto& e = t.events[1];
  EXPECT_EQ(e.action, JarvisAction::Test);
  EXPECT_EQ(e.origin, 0);
  EXPECT_EQ(e.before, 1);
  EXPECT_EQ(e.examined, 2);
  EXPECT_EQ(e.after, 2);
  EXPECT_EQ(e.turn, -1);
  EXPECT_EQ(t.analysis.input_indices, (std::vector<std::size_t>{0, 1, 2, 3}));
  EXPECT_EQ(jarvis_march(p), t.analysis.hull);
  EXPECT_EQ(jarvis_march_analyze(p).hull, t.analysis.hull);
}
TEST(JarvisTrace, AllGridSubsetsAndShuffledDuplicateClouds) {
  const std::vector<Point2D> grid{
      {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
  for (unsigned mask = 0; mask < 512; ++mask) {
    std::vector<Point2D> p;
    for (unsigned i = 0; i < 9; ++i)
      if (mask & (1u << i))
        p.push_back(grid[i]);
    check(p, jarvis_march_trace(p));
  }
  std::mt19937 rng(19);
  for (int trial = 0; trial < 1000; ++trial) {
    std::vector<Point2D> p;
    for (std::size_t i = 0, n = rng() % 40; i < n; ++i)
      p.push_back({double(int(rng() % 11) - 5), double(int(rng() % 11) - 5)});
    std::shuffle(p.begin(), p.end(), rng);
    check(p, jarvis_march_trace(p));
  }
}
TEST(JarvisTrace, SubnormalLargeAndCancellationProneInputs) {
  const double d = std::numeric_limits<double>::denorm_min(),
               m = std::numeric_limits<double>::max();
  for (const auto& p : std::vector<std::vector<Point2D>>{
           {{0, 0}, {d, 0}, {0, d}},
           {{0, 0}, {d, d}, {2 * d, 2 * d}, {3 * d, 3 * d}},
           {{-m, -m}, {m, -m}, {m, m}, {-m, m}, {0, 0}},
           {{-m, 0}, {0, 0}, {d, 0}, {m, 0}},
           {{0, 0}, {1, 1}, {2, std::nextafter(2., 3.)}, {2, 0}, {0, 2}}})
    check(p, jarvis_march_trace(p));
  const auto line = jarvis_march({{0, 0}, {d, d}, {2 * d, 2 * d}, {3 * d, 3 * d}});
  ASSERT_EQ(line.size(), 2);
  EXPECT_EQ(line.back(), (Point2D{3 * d, 3 * d}));
}
TEST(JarvisTrace, SignedZerosSmallShortcutsAndStrictCollinearExtremes) {
  const std::vector<Point2D> p{{-0., -0.}, {0., 0.}, {1, 0}, {0, 1}, {.5, 0}, {0, .5}};
  const auto                 t = jarvis_march_trace(p);
  check(p, t);
  EXPECT_EQ(t.sorted[0].input_index, 0);
  EXPECT_TRUE(std::signbit(t.sorted[0].point.x));
  EXPECT_TRUE(std::signbit(t.analysis.hull[0].y));
  for (const auto& input : std::vector<std::vector<Point2D>>{
           {},
           {{1, 1}},
           {{1, 1}, {1, 1}},
           {{2, 2}, {0, 0}},
           {{0, 0}, {1, 1}, {2, 2}, {-1, -1}, {2, 2}}})
    check(input, jarvis_march_trace(input));
}
TEST(JarvisTrace, CompleteInputActualEventsAndOwnedCopies) {
  const std::vector<Point2D> p{{0, 0}, {2, 0}, {0, 2}};
  EXPECT_EQ(jarvis_march_trace(p, {3, 9}).events.size(), 9);
  EXPECT_THROW(jarvis_march_trace(p, {3, 8}), std::length_error);
  EXPECT_THROW(jarvis_march_trace(p, {2, 0}), std::length_error);
  EXPECT_NO_THROW(jarvis_march_trace(p, {0, 0}));
  EXPECT_NO_THROW(jarvis_march_analyze(p, {3, 1}));
  auto       input = p;
  const auto t     = jarvis_march_trace(input);
  input.clear();
  EXPECT_EQ(t.sorted.size(), 3);
  auto edited              = t;
  edited.sorted[0].point.x = 99;
  edited.events[0].origin  = 99;
  EXPECT_NE(edited.sorted[0].point, t.sorted[0].point);
  EXPECT_NE(edited.events[0].origin, t.events[0].origin);
  const std::vector<Point2D> repeated(100, {0, 0});
  EXPECT_THROW(jarvis_march_trace(repeated, {99, 0}), std::length_error);
}
TEST(JarvisTrace, ValidationAndAtomicAccumulatingCounter) {
  const double nan = std::numeric_limits<double>::quiet_NaN(),
               inf = std::numeric_limits<double>::infinity();
  for (auto p : {Point2D{nan, 0}, Point2D{0, inf}}) {
    EXPECT_THROW(jarvis_march({p}), std::invalid_argument);
    EXPECT_THROW(jarvis_march_trace({p}), std::invalid_argument);
  }
  long long count = 7;
  EXPECT_THROW(jarvis_march({{0, 0}, {nan, 0}}, count), std::invalid_argument);
  EXPECT_EQ(count, 7);
  count            = std::numeric_limits<long long>::max() - 2;
  const auto saved = count;
  EXPECT_THROW(jarvis_march({{0, 0}, {2, 0}, {0, 2}}, count), std::overflow_error);
  EXPECT_EQ(count, saved);
  count = std::numeric_limits<long long>::max() - 3;
  EXPECT_NO_THROW(jarvis_march({{0, 0}, {2, 0}, {0, 2}}, count));
  EXPECT_EQ(count, std::numeric_limits<long long>::max());
  count = -7;
  EXPECT_NO_THROW(jarvis_march({{0, 0}, {2, 0}, {0, 2}}, count));
  EXPECT_EQ(count, -4);
}
TEST(JarvisTrace, NativeFixturesControlsAndGraphicsAdmission) {
  const auto p = jarvis_demo_points();
  EXPECT_EQ(p.size(), 20);
  const auto t = jarvis_march_trace(p);
  check(p, t);
  EXPECT_EQ(t.analysis.hull.size(), 8);
  EXPECT_EQ(t.analysis.work.orientation_tests, 144);
  EXPECT_FALSE(jarvis_hull_plot(p, t.analysis, 20).to_svg().empty());
  EXPECT_THROW(jarvis_hull_plot(p, t.analysis, 19), std::length_error);
  const auto tiny = jarvis_march_trace({{0, 0}, {4, 0}, {4, 4}, {0, 4}, {2, 2}});
  EXPECT_FALSE(jarvis_candidate_diagram(tiny).to_svg().empty());
  EXPECT_THROW(jarvis_candidate_diagram(tiny, 0), std::invalid_argument);
  EXPECT_THROW(jarvis_candidate_diagram(tiny, 20), std::out_of_range);
  auto edited            = tiny;
  edited.events[1].after = 99;
  EXPECT_THROW(jarvis_candidate_diagram(edited), std::invalid_argument);
  for (std::size_t n : {4, 8, 16, 32, 64, 128})
    for (bool full : {false, true}) {
      const auto control = jarvis_control_points(n, full);
      const auto a       = jarvis_march_analyze(control);
      EXPECT_EQ(a.work.distinct_points, n);
      EXPECT_EQ(a.hull.size(), full ? n : 4);
      EXPECT_EQ(a.work.orientation_tests, a.hull.size() * (n - 2));
    }
  EXPECT_THROW(jarvis_control_points(3), std::invalid_argument);
  EXPECT_THROW(jarvis_control_points(8, false, 7), std::length_error);
  EXPECT_THROW(jarvis_control_points((std::size_t(1) << 26) + 1, true, 0), std::length_error);
  const double               m = std::numeric_limits<double>::max();
  const std::vector<Point2D> extreme{{-m, -m}, {m, -m}, {m, m}, {-m, m}};
  EXPECT_THROW(jarvis_hull_plot(extreme, jarvis_march_analyze(extreme)), std::invalid_argument);
}
