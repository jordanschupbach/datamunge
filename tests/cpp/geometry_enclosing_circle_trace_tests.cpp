#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/geometry/min_enclosing_circle.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
using namespace datamunge::geometry;
TEST(EnclosingCircleTrace, EverySmallGridSubsetAgainstIndependentBoundaryEnumeration) {
  for (unsigned mask = 1; mask < 512; ++mask) {
    std::vector<Point2D> p;
    for (unsigned i = 0; i < 9; ++i)
      if (mask & (1U << i))
        p.push_back({double(i % 3), double(i / 3)});
    long double best      = 1e100L;
    auto        candidate = [&](long double x, long double y, long double r) {
      for (auto q : p)
        if (std::hypot(x - q.x, y - q.y) > r + 1e-12L)
          return;
      best = std::min(best, r);
    };
    for (auto a : p)
      candidate(a.x, a.y, 0);
    for (auto a : p)
      for (auto b : p) {
        long double x = (a.x + b.x) / 2, y = (a.y + b.y) / 2;
        candidate(x, y, std::hypot(x - a.x, y - a.y));
      }
    for (auto a : p)
      for (auto b : p)
        for (auto c : p) {
          long double u = 2 * (b.x - a.x), v = 2 * (b.y - a.y), w = 2 * (c.x - a.x),
                      z = 2 * (c.y - a.y), det = u * z - v * w;
          if (det == 0)
            continue;
          long double rhs   = b.x * b.x + b.y * b.y - a.x * a.x - a.y * a.y,
                      other = c.x * c.x + c.y * c.y - a.x * a.x - a.y * a.y;
          auto x = (rhs * z - v * other) / det, y = (u * other - rhs * w) / det;
          candidate(x, y, std::hypot(x - a.x, y - a.y));
        }
    auto t = min_enclosing_circle_trace(p);
    EXPECT_NEAR(t.result.radius, best, 1e-10);
    EXPECT_EQ(enclosing_circle_check(t.result, p).outside, 0);
    EXPECT_EQ(t.captured_entries, 3 * p.size() + t.events.size());
    auto ordinary = min_enclosing_circle(p);
    EXPECT_DOUBLE_EQ(ordinary.radius, t.result.radius);
    EXPECT_EQ(ordinary.center, t.result.center);
    std::vector<std::size_t> order(p.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937_64 rng(42);
    std::shuffle(order.begin(), order.end(), rng);
    EXPECT_EQ(t.permutation, order);
    auto previous = t.initial;
    for (auto& e : t.events) {
      EXPECT_EQ(e.before.center, previous.center);
      EXPECT_EQ(e.before.radius, previous.radius);
      auto q = p[e.input_index];
      bool contained =
          std::hypot(
              static_cast<long double>(q.x) - previous.center.x,
              static_cast<long double>(q.y) - previous.center.y) <=
          static_cast<long double>(previous.radius) + 1e-9L * std::max(1.0, previous.radius);
      EXPECT_EQ(e.contained, contained);
      if (contained) {
        EXPECT_EQ(e.after.center, previous.center);
        EXPECT_EQ(e.after.radius, previous.radius);
      }
      previous = e.after;
    }
    EXPECT_EQ(previous.center, t.result.center);
    EXPECT_EQ(previous.radius, t.result.radius);
    EXPECT_NO_THROW(min_enclosing_circle_trace(p, t.captured_entries));
    EXPECT_THROW(min_enclosing_circle_trace(p, t.captured_entries - 1), std::length_error);
  }
}
TEST(EnclosingCircleTrace, TranslationDegeneracyGuardsAndOriginalFixture) {
  auto p = enclosing_circle_demo();
  ASSERT_EQ(p.size(), 80);
  auto t = min_enclosing_circle_trace(p);
  EXPECT_EQ(enclosing_circle_check(t.result, p).outside, 0);
  auto shape = circle_boundary(t.result);
  EXPECT_EQ(shape.size(), 101);
  EXPECT_EQ(shape.front(), shape.back());
  for (double offset : {0., 1e12}) {
    auto c = min_enclosing_circle({{offset, offset}, {offset + 4, offset}, {offset + 2, offset}});
    EXPECT_NEAR(c.center.x, offset + 2, 1e-12);
    EXPECT_NEAR(c.radius, 2, 1e-12);
  }
  auto small = min_enclosing_circle({{0, 0}, {2e-12, 0}, {1e-12, 2e-12}});
  EXPECT_TRUE(std::isfinite(small.radius)); // tolerance explicitly admits tiny deviations
  EXPECT_THROW(min_enclosing_circle({}), std::invalid_argument);
  EXPECT_THROW(min_enclosing_circle({{std::nan(""), 0}}), std::invalid_argument);
  EXPECT_THROW(circle_boundary({{0, 0}, 1}, 0), std::invalid_argument);
  EXPECT_THROW(circle_boundary({{0, 0}, 1}, 1000001), std::length_error);
  EXPECT_THROW(enclosing_circle_check({{0, 0}, -1}, {}), std::invalid_argument);
  EXPECT_NO_THROW(min_enclosing_circle_trace({{0, 0}}, 0));
  EXPECT_THROW(min_enclosing_circle_trace({{0, 0}}, 2), std::length_error);
}
