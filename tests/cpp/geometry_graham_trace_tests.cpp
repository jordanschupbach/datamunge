#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <datamunge/geometry/graham_analysis.hpp>
#include <datamunge/geometry/graham_scan.hpp>
#include <limits>
#include <random>
#include <set>
using namespace datamunge::geometry;
namespace {
using Key = std::pair<double, double>;
std::set<Key> keys(const std::vector<Point2D>& v) {
  std::set<Key> s;
  for (auto p : v)
    s.emplace(p.x, p.y);
  return s;
}
// Independent small integer supporting-line oracle. Keep endpoints only when
// every input lies on the left and no collinear input extends the segment.
std::set<Key> supporting(const std::vector<Point2D>& v) {
  if (v.size() < 2)
    return keys(v);
  std::set<Key> out;
  for (auto a : v)
    for (auto b : v)
      if (a != b) {
        bool okay = true;
        for (auto p : v) {
          const long long ax = a.x, ay = a.y, bx = b.x, by = b.y, px = p.x, py = p.y;
          const auto      turn = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
          if (turn < 0)
            okay = false;
          if (turn == 0 && ((px - ax) * (px - bx) + (py - ay) * (py - by) > 0))
            okay = false;
        }
        if (okay) {
          out.emplace(a.x, a.y);
          out.emplace(b.x, b.y);
        }
      }
  return out;
}
void certificate(const std::vector<Point2D>& input) {
  auto t = graham_scan_trace(input);
  auto a = graham_scan_analyze(input);
  EXPECT_EQ(t.analysis.hull, a.hull);
  EXPECT_EQ(graham_scan(input), a.hull);
  EXPECT_EQ(t.analysis.work.angular_tests, a.work.angular_tests);
  EXPECT_EQ(t.analysis.work.sweep_tests, a.work.sweep_tests);
  long long legacy = 9;
  EXPECT_EQ(graham_scan(input, legacy), a.hull);
  EXPECT_EQ(legacy, 9 + a.work.angular_tests + a.work.sweep_tests);
  std::vector<std::size_t> stack;
  std::size_t              pushes = 0, pops = 0, tests = 0;
  for (auto e : t.events) {
    ASSERT_LT(e.candidate, t.ordered.size());
    if (e.action == HullAction::push) {
      EXPECT_EQ(e.point, e.candidate);
      stack.push_back(e.point);
      ++pushes;
    } else if (e.action == HullAction::pop) {
      ASSERT_GE(stack.size(), 2);
      EXPECT_EQ(stack.back(), e.point);
      stack.pop_back();
      ++pops;
    } else {
      ASSERT_GE(stack.size(), 2);
      EXPECT_EQ(e.a, stack[stack.size() - 2]);
      EXPECT_EQ(e.b, stack.back());
      EXPECT_EQ(
          e.turn,
          orientation_sign(
              t.ordered[e.a].point, t.ordered[e.b].point, t.ordered[e.candidate].point));
      ++tests;
    }
    EXPECT_EQ(e.stack_size, stack.size());
  }
  EXPECT_EQ(pushes, a.work.pushes);
  EXPECT_EQ(pops, a.work.pops);
  EXPECT_EQ(tests, a.work.sweep_tests);
  if (t.ordered.size() >= 3) {
    EXPECT_EQ(pushes, t.ordered.size());
    EXPECT_EQ(pushes - pops, a.hull.size());
    EXPECT_LE(tests, 2 * t.ordered.size() - 4);
    EXPECT_EQ(stack.size(), a.hull.size());
    for (std::size_t i = 0; i < stack.size(); ++i)
      EXPECT_EQ(t.ordered[stack[i]].point, a.hull[i]);
  }
  for (std::size_t i = 0; i < a.hull.size(); ++i) {
    EXPECT_EQ(a.hull[i], input[a.input_indices[i]]);
    for (std::size_t j = 0; j < a.input_indices[i]; ++j)
      EXPECT_NE(input[j], a.hull[i]);
  }
  if (a.hull.size() >= 3)
    for (std::size_t i = 0; i < a.hull.size(); ++i) {
      const auto p = a.hull[i], q = a.hull[(i + 1) % a.hull.size()],
                 r = a.hull[(i + 2) % a.hull.size()];
      EXPECT_GT(orientation_sign(p, q, r), 0);
      for (auto x : input)
        EXPECT_GE(orientation_sign(p, q, x), 0);
    }
}
} // namespace
TEST(GrahamTrace, AllSmallLatticeSubsetsAndIntegerSupportingOracle) {
  std::mt19937 rng(7);
  for (unsigned mask = 0; mask < 512; ++mask) {
    std::vector<Point2D> v;
    for (int i = 0; i < 9; ++i)
      if (mask & (1 << i))
        v.push_back({double(i % 3 - 1), double(i / 3 - 1)});
    auto expected = supporting(v);
    std::shuffle(v.begin(), v.end(), rng);
    EXPECT_EQ(keys(graham_scan(v)), expected);
    certificate(v);
    if (!v.empty()) {
      v.push_back(v.front());
      certificate(v);
      EXPECT_EQ(keys(graham_scan(v)), expected);
    }
  }
}
TEST(GrahamTrace, RandomIntegerCloudsAndIdentities) {
  std::mt19937 rng(99);
  for (int i = 0; i < 500; ++i) {
    std::vector<Point2D> v;
    for (int j = 0; j < 12; ++j)
      v.push_back({double(int(rng() % 31) - 15), double(int(rng() % 31) - 15)});
    EXPECT_EQ(keys(graham_scan(v)), supporting(v));
    certificate(v);
  }
}
TEST(GrahamTrace, FullRangeAndDegenerateCoordinates) {
  for (double s :
       {std::numeric_limits<double>::denorm_min(),
        std::ldexp(1., -500),
        1.,
        std::ldexp(1., 500),
        std::numeric_limits<double>::max()}) {
    std::vector<Point2D> v{{0, 0}, {s, 0}, {s, s}, {0, s}, {0, 0}};
    EXPECT_EQ(graham_scan(v).size(), 4);
    certificate(v);
  }
  const double         e = std::ldexp(1., -52);
  std::vector<Point2D> near{{0, 0}, {1, 1 + e}, {1 + e, 1 + 2 * e}};
  EXPECT_EQ(cross(near[0], near[1], near[2]), 0);
  EXPECT_LT(orientation_sign(near[0], near[1], near[2]), 0);
  EXPECT_EQ(graham_scan(near).size(), 3);
  certificate(near);
  const auto t = graham_scan_trace({{-0., 0}, {0., -0.}, {2, 0}, {0, 2}, {1, 1}});
  EXPECT_EQ(t.analysis.input_indices[0], 0);
  EXPECT_TRUE(std::signbit(t.analysis.hull[0].x));
  EXPECT_EQ(graham_scan({{2, -1}, {0, 4}}), (std::vector<Point2D>{{0, 4}, {2, -1}}));
}
TEST(GrahamTrace, CapsValidationOwnershipAndCounter) {
  const std::vector<Point2D> v{{0, 0}, {2, 0}, {1, 1}, {2, 2}, {0, 2}};
  const auto                 t   = graham_scan_trace(v);
  auto                       cap = t.events.size();
  EXPECT_NO_THROW(graham_scan_trace(v, {5, cap}));
  EXPECT_THROW(graham_scan_trace(v, {5, cap - 1}), std::length_error);
  EXPECT_NO_THROW(graham_scan_trace(v, {0, 0}));
  EXPECT_NO_THROW(graham_scan_analyze(v, {5, 1}));
  EXPECT_THROW(graham_scan_analyze(v, {4, 0}), std::length_error);
  for (double bad :
       {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
    auto invalid = v;
    invalid.push_back({bad, 0});
    EXPECT_THROW(graham_scan_trace(invalid, {1, 1}), std::invalid_argument);
  }
  auto copy               = t;
  copy.ordered[0].point.x = 99;
  copy.events[0].point    = 99;
  copy.analysis.hull[0].x = 99;
  EXPECT_NE(t.ordered[0].point.x, 99);
  EXPECT_NE(t.events[0].point, 99);
  EXPECT_NE(t.analysis.hull[0].x, 99);
  long long n = LLONG_MAX;
  EXPECT_THROW(graham_scan(v, n), std::overflow_error);
  EXPECT_EQ(n, LLONG_MAX);
  n = -5;
  EXPECT_NO_THROW(graham_scan(v, n));
  EXPECT_EQ(n, -5 + t.analysis.work.angular_tests + t.analysis.work.sweep_tests);
  EXPECT_TRUE(graham_scan_trace({}).events.empty());
}

TEST(GrahamTrace, PresentationBoundaries) {
  const auto p = graham_demo_points();
  EXPECT_EQ(p.size(), 20);
  EXPECT_EQ(graham_scan(p).size(), 8);
  EXPECT_NO_THROW(graham_hull_plot(p, graham_scan_analyze(p)));
  EXPECT_NO_THROW(graham_hull_plot({{0, 0}}, graham_scan_analyze({{0, 0}})));
  EXPECT_THROW(graham_hull_plot({}, {}), std::invalid_argument);
  const double               big = std::numeric_limits<double>::max();
  const std::vector<Point2D> v{{-big, 0}, {big, 0}};
  EXPECT_THROW(graham_hull_plot(v, graham_scan_analyze(v)), std::invalid_argument);
}
