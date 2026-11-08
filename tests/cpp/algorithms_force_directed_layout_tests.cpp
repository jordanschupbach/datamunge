#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <datamunge/algorithms/force_directed_layout.hpp>
#include <limits>
#include <random>
using namespace datamunge::algorithms;
using Edges  = std::vector<std::pair<std::size_t, std::size_t>>;
using Points = std::vector<Point2D>;
namespace {
// Independent ordered-vertex force oracle in widened arithmetic.
Points oracle(const Points& p, const Edges& e, bool attract) {
  Points            out(p.size());
  const long double k = 1. / std::sqrt(double(p.size())), eps = 1e-9;
  for (std::size_t u = 0; u < p.size(); ++u) {
    long double x = 0, y = 0;
    for (std::size_t v = 0; v < p.size(); ++v)
      if (u != v) {
        const long double dx = (long double)p[u].x - p[v].x, dy = (long double)p[u].y - p[v].y;
        const auto        d = std::max(eps, std::hypot(dx, dy));
        x += dx * k * k / (d * d);
        y += dy * k * k / (d * d);
      }
    if (attract)
      for (auto [a, b] : e)
        if (a != b && (a == u || b == u)) {
          const auto        v  = a == u ? b : a;
          const long double dx = (long double)p[u].x - p[v].x, dy = (long double)p[u].y - p[v].y;
          const auto        d = std::max(eps, std::hypot(dx, dy));
          x -= dx * d / k;
          y -= dy * d / k;
        }
    out[u] = {double(x), double(y)};
  }
  return out;
}
void near_points(const Points& a, const Points& b, double tolerance = 1e-12) {
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    EXPECT_NEAR(a[i].x, b[i].x, tolerance * (1 + std::abs(b[i].x)));
    EXPECT_NEAR(a[i].y, b[i].y, tolerance * (1 + std::abs(b[i].y)));
  }
}
void same(const Points& a, const Points& b) {
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(std::bit_cast<std::uint64_t>(a[i].x), std::bit_cast<std::uint64_t>(b[i].x));
    EXPECT_EQ(std::bit_cast<std::uint64_t>(a[i].y), std::bit_cast<std::uint64_t>(b[i].y));
  }
}
} // namespace
TEST(ForceDirected, OrderedVertexOracleAndActualMoveReplay) {
  std::mt19937_64 rng(119);
  for (int fixture = 0; fixture < 300; ++fixture) {
    const std::size_t n = 2 + rng() % 9;
    Points            p(n);
    for (auto& v : p)
      v = {double(int(rng() % 101) - 50) / 32, double(int(rng() % 101) - 50) / 32};
    Edges e;
    for (std::size_t u = 0; u < n; ++u)
      for (std::size_t v = u; v < n; ++v)
        if (rng() % 3 == 0) {
          e.emplace_back(u, v);
          if (rng() % 2)
            e.emplace_back(v, u);
        }
    const auto t = force_directed_layout_trace_from(p, e, 7);
    for (const auto& f : t.frames) {
      near_points(f.repulsion, oracle(p, e, false));
      near_points(f.displacement, oracle(p, e, true));
      for (std::size_t u = 0; u < n; ++u) {
        const auto net = oracle(p, e, true)[u];
        const auto s   = std::max(1e-9, std::hypot(net.x, net.y));
        const auto h   = std::min(s, f.temperature);
        EXPECT_NEAR(f.positions[u].x, p[u].x + net.x / s * h, 2e-12);
        EXPECT_NEAR(f.positions[u].y, p[u].y + net.y / s * h, 2e-12);
        EXPECT_LE(
            std::hypot(f.positions[u].x - p[u].x, f.positions[u].y - p[u].y),
            f.temperature + 1e-14);
      }
      p = f.positions;
    }
    same(t.raw_final, p);
    const auto m = force_directed_layout_metrics(t.positions, e);
    EXPECT_NEAR(double(m.centroid_x), 0, 1e-14);
    EXPECT_NEAR(double(m.centroid_y), 0, 1e-14);
    EXPECT_NEAR(double(m.max_radius), 1, 1e-14);
  }
}
TEST(ForceDirected, AllThreeVertexGraphsAndMultiplicity) {
  const Points p{{-.4, .1}, {.2, -.3}, {.7, .8}};
  for (unsigned mask = 0; mask < 64; ++mask) {
    Edges    e;
    unsigned bit = 0;
    for (std::size_t u = 0; u < 3; ++u)
      for (std::size_t v = u; v < 3; ++v, ++bit)
        if (mask & (1u << bit))
          e.emplace_back(u, v);
    const auto f = force_directed_layout_trace_from(p, e, 1).frames[0];
    near_points(f.displacement, oracle(p, e, true));
  }
  const auto a = force_directed_layout_trace_from(p, {{0, 1}}, 1);
  const auto b = force_directed_layout_trace_from(p, {{0, 1}, {1, 0}, {0, 1}, {0, 0}}, 1);
  EXPECT_EQ(b.work.attraction_visits, 3u);
  EXPECT_EQ(b.work.edge_visits, 4u);
  const auto& r = a.frames[0].repulsion;
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(
        b.frames[0].displacement[i].x - r[i].x,
        3 * (a.frames[0].displacement[i].x - r[i].x),
        1e-14);
  }
}
TEST(ForceDirected, PotentialGradientIncludesDistanceFloor) {
  const Edges e{{0, 0}, {0, 1}, {1, 0}, {1, 2}};
  for (const Points p :
       {Points{{-.4, .1}, {.2, -.3}, {.7, .8}}, Points{{0, 0}, {5e-10, 0}, {1, 1}}}) {
    const auto t = force_directed_layout_trace_from(p, e, 1);
    for (std::size_t u = 0; u < p.size(); ++u)
      for (int axis = 0; axis < 2; ++axis) {
        const double h    = p[1].x < 1e-8 ? 1e-14 : 1e-6;
        auto         plus = p, minus = p;
        (axis ? plus[u].y : plus[u].x) += h;
        (axis ? minus[u].y : minus[u].x) -= h;
        const auto derivative = (force_directed_layout_metrics(plus, e).potential -
                                 force_directed_layout_metrics(minus, e).potential) /
                                (2 * h);
        const auto expected = axis ? t.frames[0].displacement[u].y : t.frames[0].displacement[u].x;
        EXPECT_NEAR(
            double(-derivative),
            expected,
            (p[1].x < 1e-8 ? 0.001 : 1e-8) * (1 + std::abs(expected)));
      }
  }
  // Centered finite difference exactly at the C1 boundary.
  const Points p{{0, 0}, {1e-9, 0}};
  const auto   f = force_directed_layout_trace_from(p, {{0, 1}}, 1).frames[0].displacement[0].x;
  auto         a = p, b = p;
  a[0].x += 1e-15;
  b[0].x -= 1e-15;
  const auto d = (force_directed_layout_metrics(a, {{0, 1}}).potential -
                  force_directed_layout_metrics(b, {{0, 1}}).potential) /
                 2e-15;
  EXPECT_NEAR(double(-d), f, std::abs(f) * 2e-6);
}
TEST(ForceDirected, AnalyticTwoPointCoolingIsNotDescent) {
  const Edges  e{{0, 1}};
  const auto   t = force_directed_layout_trace_from({{-.25, 0}, {.25, 0}}, e, 2);
  const double k = 1 / std::sqrt(2.);
  EXPECT_NEAR(t.frames[0].repulsion[0].x, -1, 1e-15);
  EXPECT_NEAR(t.frames[0].displacement[0].x, -1 + .25 / k, 1e-15);
  EXPECT_DOUBLE_EQ(t.frames[0].temperature, .1);
  EXPECT_DOUBLE_EQ(t.frames[1].temperature, .05);
  EXPECT_NEAR(t.frames[0].positions[0].x, -.35, 1e-15);
  const double force = -k * k / .7 + .7 * .7 / k;
  EXPECT_NEAR(t.raw_final[0].x, -.35 + force, 1e-15);
  EXPECT_GT(
      force_directed_layout_metrics(t.frames[1].positions, e).potential,
      force_directed_layout_metrics(t.frames[0].positions, e).potential);
  EXPECT_DOUBLE_EQ(t.positions[0].x, -1);
  EXPECT_DOUBLE_EQ(t.positions[1].x, 1);
  const auto original =
      force_directed_layout_trace(4, force_directed_layout_example_edges(), 500, 7);
  EXPECT_EQ(original.work.pair_visits, 3000u);
  EXPECT_EQ(original.work.attraction_visits, 1500u);
  EXPECT_EQ(original.work.vertex_moves, 2000u);
  EXPECT_EQ(original.work.uniform_values, 8u);
  EXPECT_NEAR(original.frames.back().temperature, .0002, 1e-17);
}
TEST(ForceDirected, WorkCountsAndSeededParity) {
  for (std::size_t n = 0; n < 12; ++n)
    for (std::size_t T : {0u, 1u, 2u, 17u}) {
      Edges e;
      for (std::size_t u = 0; u < n; ++u) {
        e.emplace_back(u, u);
        if (u)
          e.emplace_back(0, u);
      }
      const auto trace = force_directed_layout_trace(n, e, T, 7);
      const auto work  = force_directed_layout_work(n, e, T, 7);
      same(trace.positions, work.positions);
      same(trace.positions, force_directed_layout(n, e, T, 7));
      const auto s = n >= 2 ? T : 0;
      EXPECT_EQ(work.work.pair_visits, s * n * (n - 1) / 2);
      EXPECT_EQ(work.work.edge_visits, s * e.size());
      EXPECT_EQ(work.work.attraction_visits, s * (n ? n - 1 : 0));
      EXPECT_EQ(work.work.vertex_moves, s * n);
      EXPECT_EQ(work.work.completed_iterations, s);
      EXPECT_EQ(work.work.uniform_values, 2 * n);
      EXPECT_EQ(trace.frames.size(), s);
      EXPECT_EQ(work.work.normalized, n >= 2);
    }
}
TEST(ForceDirected, EmptySingletonZeroAndCoincident) {
  EXPECT_TRUE(force_directed_layout(0, {}, SIZE_MAX).empty());
  const auto single = force_directed_layout_trace(1, {{0, 0}}, SIZE_MAX, 7);
  same(single.positions, force_directed_layout(1, {{0, 0}}, 0, 7));
  EXPECT_NE(single.positions[0].x, 0);
  EXPECT_FALSE(single.work.normalized);
  EXPECT_TRUE(single.frames.empty());
  same(force_directed_layout_from({{3, 4}}, {{0, 0}}, SIZE_MAX), {{3, 4}});
  const auto zero = force_directed_layout_trace_from({{3, 4}, {5, 4}}, {{0, 1}}, 0);
  same(zero.positions, {{-1, 0}, {1, 0}});
  EXPECT_TRUE(zero.frames.empty());
  EXPECT_TRUE(zero.work.normalized);
  EXPECT_DOUBLE_EQ(zero.work.mean.x, 4);
  EXPECT_DOUBLE_EQ(zero.work.scale, 1);
  const auto samepos = force_directed_layout_trace_from({{1, 1}, {1, 1}}, {{0, 1}}, 10);
  same(samepos.raw_final, {{1, 1}, {1, 1}});
  same(samepos.positions, {{0, 0}, {0, 0}});
  for (auto f : samepos.frames) {
    same(f.repulsion, {{0, 0}, {0, 0}});
    same(f.displacement, {{0, 0}, {0, 0}});
  }
  EXPECT_EQ(force_directed_layout_metrics(samepos.initial, {{0, 1}}).clamped_pairs, 1u);
}
TEST(ForceDirected, TranslationRotationReflectionControls) {
  const Points p{{-.3, .1}, {.2, -.4}, {.5, .7}};
  const Edges  e{{0, 1}, {1, 2}};
  const auto   a       = force_directed_layout_trace_from(p, e, 5);
  auto         shifted = p, rotated = p, reflected = p;
  for (std::size_t i = 0; i < p.size(); ++i) {
    shifted[i]   = {p[i].x + 3, p[i].y - 4};
    rotated[i]   = {-p[i].y, p[i].x};
    reflected[i] = {-p[i].x, p[i].y};
  }
  const auto b = force_directed_layout_trace_from(shifted, e, 5),
             c = force_directed_layout_trace_from(rotated, e, 5),
             d = force_directed_layout_trace_from(reflected, e, 5);
  for (std::size_t j = 0; j < 5; ++j)
    for (std::size_t i = 0; i < p.size(); ++i) {
      EXPECT_NEAR(b.frames[j].positions[i].x, a.frames[j].positions[i].x + 3, 1e-12);
      EXPECT_NEAR(b.frames[j].positions[i].y, a.frames[j].positions[i].y - 4, 1e-12);
      EXPECT_NEAR(c.frames[j].positions[i].x, -a.frames[j].positions[i].y, 1e-12);
      EXPECT_NEAR(c.frames[j].positions[i].y, a.frames[j].positions[i].x, 1e-12);
      EXPECT_NEAR(d.frames[j].positions[i].x, -a.frames[j].positions[i].x, 1e-12);
    }
  EXPECT_NEAR(
      double(force_directed_layout_metrics(p, e).potential),
      double(force_directed_layout_metrics(rotated, e).potential),
      1e-14);
}
TEST(ForceDirected, DomainValidationPrecedesQuotaAndAllShortcuts) {
  const auto nan = std::numeric_limits<double>::quiet_NaN(),
             inf = std::numeric_limits<double>::infinity();
  EXPECT_THROW(force_directed_layout_trace(0, {{0, 0}}, 0, 7, 1), std::invalid_argument);
  EXPECT_THROW(force_directed_layout_trace(1, {{1, 1}}, SIZE_MAX, 7, 1), std::invalid_argument);
  EXPECT_THROW(force_directed_layout_trace_from({{nan, 0}}, {{0, 0}}, 0, 1), std::invalid_argument);
  EXPECT_THROW(
      force_directed_layout_trace_from({{0, 0}, {inf, 0}}, {{0, 1}}, 0, 1), std::invalid_argument);
  EXPECT_THROW(force_directed_layout_metrics({{0, nan}}, {}), std::invalid_argument);
  EXPECT_THROW(force_directed_layout_metrics({}, {{0, 0}}), std::invalid_argument);
}
TEST(ForceDirected, WholeAdmissionOwnershipAndArithmeticBounds) {
  EXPECT_EQ(force_directed_layout_trace_entries(0, 0, SIZE_MAX), 0u);
  EXPECT_EQ(force_directed_layout_trace_entries(1, 0, SIZE_MAX), 3u);
  EXPECT_EQ(force_directed_layout_trace_entries(2, 1, 1), 14u);
  EXPECT_EQ(force_directed_layout_trace_entries(2, 1, 2), 21u);
  EXPECT_EQ(force_directed_layout_trace_entries(4, 3, 500), 6515u);
  EXPECT_THROW(force_directed_layout_trace_entries(SIZE_MAX, 0, 0), std::length_error);
  EXPECT_THROW(force_directed_layout_trace_entries(2, SIZE_MAX, 0), std::length_error);
  EXPECT_THROW(force_directed_layout_trace(2, {}, SIZE_MAX, 7, 0), std::length_error);
  EXPECT_THROW(force_directed_layout_trace(2, {{0, 1}}, 2, 7, 20), std::length_error);
  EXPECT_NO_THROW(force_directed_layout_trace(2, {{0, 1}}, 2, 7, 21));
  Points p{{-.25, 0}, {.25, 0}};
  Edges  e{{0, 1}};
  auto   t = force_directed_layout_trace_from(p, e, 2, 21);
  p[0].x   = 100;
  e.clear();
  EXPECT_DOUBLE_EQ(t.initial[0].x, -.25);
  EXPECT_EQ(t.edges.size(), 1u);
  const double raw               = t.raw_final[0].x;
  t.frames.back().positions[0].x = 999;
  EXPECT_DOUBLE_EQ(t.raw_final[0].x, raw);
  t.positions[0].x = 888;
  EXPECT_DOUBLE_EQ(t.raw_final[0].x, raw);
  EXPECT_DOUBLE_EQ(t.initial[0].x, -.25);
}
TEST(ForceDirected, FiniteExtremeInputsAndActualOverflow) {
  const double max = std::numeric_limits<double>::max();
  same(force_directed_layout_from({{max, 0}}, {}), {{max, 0}});
  EXPECT_NO_THROW(force_directed_layout_from({{-1e200, 0}, {1e200, 0}}, {}, 2));
  EXPECT_THROW(
      force_directed_layout_from({{-1e200, 0}, {1e200, 0}}, {{0, 1}}, 1), std::overflow_error);
  EXPECT_THROW(force_directed_layout_from({{-max, 0}, {max, 0}}, {}, 1), std::overflow_error);
  EXPECT_THROW(force_directed_layout_from({{max, 0}, {max, 0}}, {}, 0), std::overflow_error);
  const auto m = force_directed_layout_metrics({{-1e200, 0}, {1e200, 0}}, {{0, 1}});
  EXPECT_TRUE(std::isfinite(m.potential));
  EXPECT_GT(m.attraction_potential, 1e599L);
  EXPECT_DOUBLE_EQ(double(m.mean_edge_length), 2e200);
  const auto empty = force_directed_layout_metrics({}, {});
  EXPECT_EQ(empty.potential, 0);
  EXPECT_EQ(empty.min_pair_distance, 0);
  EXPECT_EQ(empty.max_radius, 0);
}
