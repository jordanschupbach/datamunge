#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/geometry/gjk.hpp>
#include <limits>
#include <random>
#include <stdexcept>

#include "../oracles/gjk_exact.hpp"

using namespace datamunge::geometry;
namespace {
std::vector<Point2D> square(double x, double y, double h) {
  return {{x - h, y - h}, {x + h, y - h}, {x + h, y + h}, {x - h, y + h}};
}
// Independent geometric certificate on a tiny retained feature: exhaustive edge
// projections, plus oriented triangle containment, in long double arithmetic.
long double feature_distance_squared(const GjkEvent& e) {
  auto cross = [](Point2D a, Point2D b) { return (long double)a.x * b.y - (long double)a.y * b.x; };
  const auto n = e.simplex_size;
  if (n == 3) {
    const auto a = e.simplex[0], b = e.simplex[1], c = e.simplex[2];
    const auto ab = cross(a, b), bc = cross(b, c), ca = cross(c, a);
    if ((ab >= 0 && bc >= 0 && ca >= 0) || (ab <= 0 && bc <= 0 && ca <= 0))
      return 0;
  }
  long double best = std::numeric_limits<long double>::infinity();
  for (std::size_t i = 0; i < n; ++i) {
    const auto        a = e.simplex[i], b = e.simplex[(i + 1) % n];
    const long double vx = (long double)b.x - a.x, vy = (long double)b.y - a.y;
    const auto        den = vx * vx + vy * vy;
    const auto        t =
        den ? std::clamp(-((long double)a.x * vx + (long double)a.y * vy) / den, 0.L, 1.L) : 0.L;
    const auto x = a.x + t * vx, y = a.y + t * vy;
    best = std::min(best, x * x + y * y);
  }
  return best;
}
} // namespace

TEST(GjkTrace, ExactIndependentMinkowskiHullFixtures) {
  for (const auto& c : gjk_exact_fixture::cases) {
    const auto*          p = gjk_exact_fixture::points + c.offset;
    std::vector<Point2D> a(p, p + c.a_size), b(p + c.a_size, p + c.a_size + c.b_size);
    const auto           expected = std::sqrt(double(c.numerator) / double(c.denominator));
    EXPECT_NEAR(gjk_distance(a, b), expected, 2e-12);
    EXPECT_NEAR(gjk_distance(b, a), expected, 2e-12);
    std::reverse(a.begin(), a.end());
    b.push_back(b.front());
    EXPECT_NEAR(gjk_distance(a, b), expected, 2e-12);
  }
}

TEST(GjkTrace, ActualSupportFeatureAndWorkCertificates) {
  std::mt19937_64 rng(135);
  for (int trial = 0; trial < 400; ++trial) {
    std::vector<Point2D> a, b;
    for (auto* p : {&a, &b}) {
      const auto n = 1 + rng() % 9;
      for (std::size_t i = 0; i < n; ++i)
        p->push_back({double(int(rng() % 33) - 16) / 4, double(int(rng() % 33) - 16) / 4});
    }
    const auto t = gjk_distance_trace(a, b);
    EXPECT_DOUBLE_EQ(t.run.distance, gjk_distance(a, b));
    std::size_t queries = 0, visits = 0, progress = 0, segments = 0, triangles = 0;
    Point2D     previous_closest{};
    for (const auto& e : t.events) {
      ASSERT_LE(e.simplex_size, 3u);
      for (std::size_t i = e.simplex_size; i < 3; ++i)
        EXPECT_EQ(e.simplex[i], (Point2D{}));
      if (e.kind == GjkEventKind::initial || e.kind == GjkEventKind::support) {
        ++queries;
        std::size_t ai = 0, bi = 0;
        auto        dot = [](Point2D p, Point2D d) { return p.x * d.x + p.y * d.y; };
        for (std::size_t i = 1; i < a.size(); ++i)
          if (dot(a[i], e.direction) > dot(a[ai], e.direction))
            ai = i;
        const Point2D negative{-e.direction.x, -e.direction.y};
        for (std::size_t i = 1; i < b.size(); ++i)
          if (dot(b[i], negative) > dot(b[bi], negative))
            bi = i;
        EXPECT_EQ(e.support_a, ai);
        EXPECT_EQ(e.support_b, bi);
        EXPECT_EQ(e.point, (Point2D{a[ai].x - b[bi].x, a[ai].y - b[bi].y}));
        if (e.kind == GjkEventKind::support) {
          ++progress;
          EXPECT_DOUBLE_EQ(e.value, dot(e.point, e.direction) - dot(previous_closest, e.direction));
        }
      }
      if (e.kind == GjkEventKind::direction) {
        ++visits;
        previous_closest = e.point;
        EXPECT_EQ(e.direction, (Point2D{-e.point.x, -e.point.y}));
        EXPECT_DOUBLE_EQ(e.value, e.direction.x * e.direction.x + e.direction.y * e.direction.y);
      }
      if (e.kind == GjkEventKind::segment)
        ++segments;
      if (e.kind == GjkEventKind::triangle || e.kind == GjkEventKind::triangle_inside)
        ++triangles;
      if (e.kind != GjkEventKind::support && e.kind != GjkEventKind::empty) {
        const auto norm = (long double)e.point.x * e.point.x + (long double)e.point.y * e.point.y;
        EXPECT_NEAR(double(norm), double(feature_distance_squared(e)), 2e-11);
      }
      if (e.kind == GjkEventKind::near)
        EXPECT_LT(e.point.x * e.point.x + e.point.y * e.point.y, 1e-20);
      if (e.kind == GjkEventKind::progress)
        EXPECT_LT((t.events[&e - t.events.data() - 1]).value, 1e-10);
    }
    const auto& w = t.run.work;
    EXPECT_EQ(w.support_calls, 2 * queries);
    EXPECT_EQ(w.support_dots, (a.size() + b.size()) * queries);
    EXPECT_EQ(w.support_comparisons, (a.size() + b.size() - 2) * queries);
    EXPECT_EQ(w.iterations, visits);
    EXPECT_EQ(w.progress_checks, progress);
    EXPECT_EQ(w.segment_reductions, segments);
    EXPECT_EQ(w.triangle_reductions, triangles);
  }
}

TEST(GjkTrace, StopReasonsAndDistinctLoopBudget) {
  const std::vector<Point2D> a{{-1, -1}, {1, -1}, {0, 1}}, b{{0, 0}};
  EXPECT_EQ(gjk_distance_run(a, b).stop, GjkStopReason::triangle_contains_origin);
  EXPECT_EQ(gjk_distance_run({{0, 0}}, {{3, 4}}).stop, GjkStopReason::progress_tolerance);
  EXPECT_EQ(gjk_distance_run({{0, 0}}, {{5e-11, 0}}).stop, GjkStopReason::near_origin);
  EXPECT_EQ(gjk_distance_run({}, b).stop, GjkStopReason::empty_input);
  const auto zero = gjk_distance_trace(a, b, 0);
  EXPECT_EQ(zero.run.stop, GjkStopReason::iteration_limit);
  EXPECT_EQ(zero.run.work.iterations, 0u);
  EXPECT_EQ(zero.events.size(), 2u);
  EXPECT_EQ(zero.events.back().iteration, 0u);
  const auto one = gjk_distance_run(a, b, 1);
  EXPECT_EQ(one.stop, GjkStopReason::iteration_limit);
  EXPECT_EQ(one.work.iterations, 1u);
  EXPECT_GT(one.distance, 0);
  EXPECT_EQ(gjk_distance(a, b), 0);
  EXPECT_STREQ(gjk_stop_name(static_cast<GjkStopReason>(999)), "unknown");
}

TEST(GjkTrace, ExactOnlineRecordAdmissionAndOwnership) {
  std::vector<Point2D> a{{0, 0}}, b{{3, 4}};
  const auto           old_a = a, old_b = b;
  EXPECT_THROW(gjk_distance_trace(a, b, 100, 3), std::length_error);
  auto t = gjk_distance_trace(a, b, 100, 4);
  ASSERT_EQ(t.events.size(), 4u);
  EXPECT_EQ(t.events.front().kind, GjkEventKind::initial);
  EXPECT_EQ(t.events.back().kind, GjkEventKind::progress);
  EXPECT_EQ(gjk_distance_trace(a, b, 100, 0).events.size(), 4u);
  EXPECT_THROW(gjk_distance_trace(a, a, 100, 2), std::length_error);
  EXPECT_EQ(gjk_distance_trace(a, a, 100, 3).events.size(), 3u);
  EXPECT_EQ(gjk_distance_trace({}, b, 100, 1).events.size(), 1u);
  EXPECT_EQ(gjk_distance_trace({}, b, 100, 0).events.size(), 1u);
  auto copied              = t;
  copied.events[0].point.x = 999;
  EXPECT_NE(t.events[0].point.x, 999);
  copied.run.simplex[0].x = 999;
  EXPECT_NE(t.run.simplex[0].x, 999);
  a[0].x = 888;
  b[0].x = 777;
  EXPECT_EQ(t.run.distance, 5);
  EXPECT_NE(t.events.front().point.x, 888);
  EXPECT_EQ(old_a, (std::vector<Point2D>{{0, 0}}));
  EXPECT_EQ(old_b, (std::vector<Point2D>{{3, 4}}));
}

TEST(GjkTrace, CompleteValidationAndFiniteIntermediateGuards) {
  const double nan = std::numeric_limits<double>::quiet_NaN(),
               inf = std::numeric_limits<double>::infinity();
  for (double bad : {nan, inf, -inf}) {
    const std::vector<Point2D> a{{0, 0}, {bad, 0}}, b{{0, 0}};
    EXPECT_THROW(gjk_distance(a, b), std::invalid_argument);
    EXPECT_THROW(gjk_distance({}, a), std::invalid_argument);
    EXPECT_THROW(gjk_distance_trace(a, b, 0, 1), std::invalid_argument);
    EXPECT_THROW(gjk_distance_trace(b, a, 0, 1), std::invalid_argument);
  }
  const std::vector<Point2D> a{{1e200, 0}}, b{{0, 0}};
  const auto                 old_a = a, old_b = b;
  EXPECT_THROW(gjk_distance(a, b), std::overflow_error);
  EXPECT_THROW(gjk_distance_run(a, b, 0), std::overflow_error);
  EXPECT_EQ(a, old_a);
  EXPECT_EQ(b, old_b);
  const double large = std::numeric_limits<double>::max() * .75;
  EXPECT_THROW(gjk_distance({{large, 0}}, {{-large, 0}}), std::overflow_error);
  EXPECT_THROW(
      gjk_distance({{-1e100, -1e100}, {1e100, -1e100}, {0, 1e100}}, {{0, 0}}), std::overflow_error);
}

TEST(GjkTrace, AbsoluteCutoffsRemainExplicitApproximationLimits) {
  EXPECT_DOUBLE_EQ(gjk_distance({{0, 0}}, {{5e-11, 0}}), 0);
  const auto a = square(0, 0, 1e-6), b = square(4e-6, 0, 1e-6);
  EXPECT_NEAR(gjk_distance(a, b), 6e-6, 1e-20); // Exact geometric gap is 2e-6.
  EXPECT_EQ(gjk_distance_run(a, b).stop, GjkStopReason::progress_tolerance);
  EXPECT_NEAR(
      gjk_distance({{-1e-6, 0}, {1e-6, 0}}, {{0, -1e-6}, {0, 1e-6}}), std::sqrt(2.) * 1e-6, 1e-20);
  EXPECT_DOUBLE_EQ(gjk_distance({}, {{0, 0}}), 0); // Empty convention, not geometric contact.
  EXPECT_NEAR(gjk_distance(square(0, 0, 1), square(5, 0, 1)), 3, 1e-12);
}

TEST(GjkTrace, RegularPolygonFormulaWholeAdmissionAndFiniteDomain) {
  for (std::size_t n : {3u, 4u, 5u, 9u, 64u})
    for (double rotation : {0., .1, .3, -2.}) {
      const auto p = regular_polygon({2, -3}, 1.5, n, rotation, n);
      ASSERT_EQ(p.size(), n);
      for (std::size_t i = 0; i < n; ++i) {
        const double angle = rotation + 2 * M_PI * i / n;
        EXPECT_DOUBLE_EQ(p[i].x, 2 + 1.5 * std::cos(angle));
        EXPECT_DOUBLE_EQ(p[i].y, -3 + 1.5 * std::sin(angle));
      }
    }
  EXPECT_THROW(regular_polygon({0, 0}, 1, 4, 0, 3), std::length_error);
  EXPECT_EQ(regular_polygon({0, 0}, 1, 4, 0, 4).size(), 4u);
  EXPECT_EQ(regular_polygon({0, 0}, 1, 4, 0, 0).size(), 4u);
  EXPECT_THROW(
      regular_polygon({0, 0}, 1, std::numeric_limits<std::size_t>::max(), 0, 0), std::length_error);
  EXPECT_THROW(regular_polygon({0, 0}, 1, 2), std::invalid_argument);
  EXPECT_THROW(regular_polygon({0, 0}, -1, 4), std::invalid_argument);
  const double nan = std::numeric_limits<double>::quiet_NaN(),
               inf = std::numeric_limits<double>::infinity();
  EXPECT_THROW(regular_polygon({nan, 0}, 1, 4, 0, 1), std::invalid_argument);
  EXPECT_THROW(regular_polygon({0, 0}, inf, 4), std::invalid_argument);
  EXPECT_THROW(regular_polygon({0, 0}, 1, 4, inf), std::invalid_argument);
  const double large = std::numeric_limits<double>::max();
  EXPECT_THROW(regular_polygon({large, 0}, large, 4), std::overflow_error);
  const auto zeros = regular_polygon({2, 3}, 0, 4);
  for (const auto& p : zeros)
    EXPECT_EQ(p, (Point2D{2, 3}));
}
