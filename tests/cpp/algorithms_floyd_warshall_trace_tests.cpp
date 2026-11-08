#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <datamunge/algorithms/floyd_warshall.hpp>
#include <limits>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
using Edges = std::vector<std::tuple<std::size_t, std::size_t, double>>;
namespace {
const double inf = std::numeric_limits<double>::infinity();
auto         bits(double x) {
  return std::bit_cast<std::uint64_t>(x);
}
struct Oracle {
  std::vector<double> distance;
  std::vector<bool>   affected;
};
// Independent edge-list Bellman-Ford, with copied layers then descendant propagation.
Oracle bellman(std::size_t n, const Edges& edges, std::size_t seed) {
  Oracle o{std::vector<double>(n, inf), std::vector<bool>(n)};
  o.distance[seed] = 0;
  for (std::size_t pass = 1; pass < n; ++pass) {
    auto next = o.distance;
    for (auto [u, v, w] : edges)
      if (o.distance[u] != inf)
        next[v] = std::min(next[v], o.distance[u] + w);
    o.distance = std::move(next);
  }
  for (auto [u, v, w] : edges)
    if (o.distance[u] != inf && o.distance[u] + w < o.distance[v])
      o.affected[v] = true;
  for (std::size_t pass = 0; pass < n; ++pass)
    for (auto [u, v, w] : edges) {
      (void)w;
      if (o.affected[u])
        o.affected[v] = true;
    }
  return o;
}
void path_cost(const AllPairsShortestPaths& r, const Edges& edges, std::size_t i, std::size_t j) {
  const auto path = reconstruct_path(r, i, j);
  if (r.distance[i][j] == inf) {
    EXPECT_TRUE(path.empty());
    return;
  }
  ASSERT_FALSE(path.empty());
  EXPECT_EQ(path.front(), i);
  EXPECT_EQ(path.back(), j);
  EXPECT_LE(path.size(), r.distance.size());
  double cost = 0;
  for (std::size_t k = 1; k < path.size(); ++k) {
    double weight = inf;
    for (auto [u, v, w] : edges)
      if (u == path[k - 1] && v == path[k])
        weight = std::min(weight, w);
    ASSERT_TRUE(std::isfinite(weight));
    cost += weight;
  }
  EXPECT_EQ(cost, r.distance[i][j]);
}
void compare(std::size_t n, const Edges& edges) {
  const auto r    = floyd_warshall(n, edges);
  const auto mask = floyd_warshall_unbounded_pairs(r);
  bool       any  = false;
  for (std::size_t i = 0; i < n; ++i) {
    const auto o = bellman(n, edges, i);
    for (std::size_t j = 0; j < n; ++j) {
      EXPECT_EQ(mask[i][j], o.affected[j]);
      const auto status = floyd_warshall_pair_status(r, i, j);
      if (o.affected[j]) {
        any = true;
        EXPECT_EQ(status, FloydWarshallPairStatus::Unbounded);
        EXPECT_THROW((void)reconstruct_path(r, i, j), std::domain_error);
      } else {
        EXPECT_EQ(r.distance[i][j], o.distance[j]);
        EXPECT_EQ(
            status,
            o.distance[j] == inf ? FloydWarshallPairStatus::Unreachable
                                 : FloydWarshallPairStatus::Finite);
        path_cost(r, edges, i, j);
      }
    }
  }
  EXPECT_EQ(r.has_negative_cycle, any);
}
void replay(const FloydWarshallTrace& t) {
  const auto n = t.work.vertices;
  ASSERT_EQ(t.stages.size(), n + 1);
  auto                                  d    = t.stages[0].distance;
  auto                                  hops = t.stages[0].next;
  std::vector<std::vector<double>>      seed(n, std::vector<double>(n, inf));
  std::vector<std::vector<std::size_t>> initial(n, std::vector<std::size_t>(n, kNoVertex));
  for (std::size_t i = 0; i < n; ++i)
    seed[i][i] = 0;
  std::size_t seeded = 0;
  for (auto [u, v, w] : t.edges)
    if (w < seed[u][v]) {
      seed[u][v]    = w;
      initial[u][v] = v;
      ++seeded;
    }
  EXPECT_EQ(d, seed);
  EXPECT_EQ(hops, initial);
  EXPECT_EQ(t.work.seed_improvements, seeded);
  std::size_t current = 0, finite_rows = 0, expected_columns = 0, candidates = 0,
              skipped_columns = 0;
  for (std::size_t k = 0; k < n; ++k) {
    std::size_t sources = 0, targets = 0;
    for (std::size_t i = 0; i < n; ++i)
      sources += d[i][k] != inf;
    for (std::size_t j = 0; j < n; ++j)
      targets += d[k][j] != inf;
    finite_rows += sources;
    expected_columns += sources * n;
    candidates += sources * targets;
    skipped_columns += sources * (n - targets);
    std::size_t prior = 0;
    bool        first = true;
    while (current < t.relaxations.size() && t.relaxations[current].k == k) {
      const auto& r = t.relaxations[current++];
      ASSERT_LT(r.from, n);
      ASSERT_LT(r.to, n);
      const auto index = r.from * n + r.to;
      if (!first)
        EXPECT_GT(index, prior);
      first = false;
      prior = index;
      EXPECT_EQ(bits(r.previous), bits(d[r.from][r.to]));
      EXPECT_EQ(r.previous_next, hops[r.from][r.to]);
      EXPECT_EQ(bits(r.left), bits(d[r.from][k]));
      EXPECT_EQ(bits(r.right), bits(d[k][r.to]));
      EXPECT_EQ(bits(r.candidate), bits(r.left + r.right));
      EXPECT_LT(r.candidate, r.previous);
      EXPECT_EQ(r.next, hops[r.from][k]);
      d[r.from][r.to]    = r.candidate;
      hops[r.from][r.to] = r.next;
    }
    EXPECT_EQ(t.stages[k + 1].admitted, k + 1);
    EXPECT_EQ(t.stages[k + 1].distance, d);
    EXPECT_EQ(t.stages[k + 1].next, hops);
  }
  EXPECT_EQ(current, t.relaxations.size());
  EXPECT_EQ(d, t.result.distance);
  EXPECT_EQ(hops, t.result.next);
  EXPECT_EQ(t.work.relaxations, current);
  EXPECT_EQ(t.work.row_checks, n * n);
  EXPECT_EQ(t.work.skipped_rows, n * n - finite_rows);
  EXPECT_EQ(t.work.column_checks, expected_columns);
  EXPECT_EQ(t.work.candidates, candidates);
  EXPECT_EQ(t.work.skipped_columns, skipped_columns);
  std::size_t tested = 0;
  for (std::size_t i = 0; i < n; ++i) {
    ++tested;
    if (d[i][i] < 0)
      break;
  }
  EXPECT_EQ(t.work.diagonal_checks, tested);
}
} // namespace
TEST(FloydWarshallTrace, ExhaustiveThreeVertexGraphsMatchIndependentBellmanFordAndAffectedPairs) {
  for (unsigned code = 0; code < 4096; ++code) {
    auto  z = code;
    Edges e;
    for (std::size_t i = 0; i < 3; ++i)
      for (std::size_t j = 0; j < 3; ++j)
        if (i != j) {
          auto k = z % 4;
          z /= 4;
          if (k)
            e.emplace_back(i, j, k == 1 ? -2. : k == 2 ? 0. : 3.);
        }
    compare(3, e);
    if (code % 17 == 0)
      replay(floyd_warshall_trace(3, e));
  }
}
TEST(FloydWarshallTrace, ExhaustiveFourVertexDagsHaveExactCostsAndBoundedPaths) {
  for (unsigned code = 0; code < 4096; ++code) {
    auto  z = code;
    Edges e;
    for (std::size_t i = 0; i < 4; ++i)
      for (std::size_t j = i + 1; j < 4; ++j) {
        auto k = z % 4;
        z /= 4;
        if (k)
          e.emplace_back(i, j, k == 1 ? -2. : k == 2 ? 0. : 3.);
      }
    compare(4, e);
  }
}
TEST(FloydWarshallTrace, RandomPotentialGraphsAndActualSnapshotReplay) {
  std::mt19937_64 rng(118);
  for (int fixture = 0; fixture < 400; ++fixture) {
    const std::size_t n = 1 + rng() % 9;
    std::vector<int>  p(n);
    for (auto& v : p)
      v = int(rng() % 31) - 15;
    Edges e;
    for (std::size_t u = 0; u < n; ++u)
      for (std::size_t v = 0; v < n; ++v)
        if (rng() % 3 == 0) {
          double w = double(rng() % 11) + p[v] - p[u];
          e.emplace_back(u, v, w);
          if (rng() % 4 == 0)
            e.emplace_back(u, v, w + int(rng() % 5));
        }
    compare(n, e);
    const auto t = floyd_warshall_trace(n, e);
    replay(t);
    const auto work = floyd_warshall_work(n, e);
    EXPECT_EQ(t.result.distance, work.result.distance);
    EXPECT_EQ(t.result.next, work.result.next);
    EXPECT_EQ(t.work.candidates, work.work.candidates);
  }
}
TEST(FloydWarshallTrace, NegativeCyclesArePairSpecificAndRawMatricesRemainInspectable) {
  const Edges e{{0, 1, 1}, {1, 2, -3}, {2, 1, 1}, {2, 3, 1}};
  const auto  t = floyd_warshall_trace(5, e);
  replay(t);
  compare(5, e);
  const auto& r = t.result;
  ASSERT_TRUE(r.has_negative_cycle);
  const auto mask = floyd_warshall_unbounded_pairs(r);
  for (std::size_t i = 0; i < 5; ++i)
    for (std::size_t j = 0; j < 5; ++j)
      EXPECT_EQ(mask[i][j], i <= 2 && j >= 1 && j <= 3);
  EXPECT_EQ(reconstruct_path(r, 4, 4), (std::vector<std::size_t>{4}));
  EXPECT_EQ(reconstruct_path(r, 0, 0), (std::vector<std::size_t>{0}));
  EXPECT_TRUE(reconstruct_path(r, 3, 0).empty());
  EXPECT_THROW((void)reconstruct_path(r, 1, 1), std::domain_error);
  auto untrusted               = r;
  untrusted.has_negative_cycle = false;
  EXPECT_EQ(floyd_warshall_unbounded_pairs(untrusted), mask);
  EXPECT_EQ(floyd_warshall_pair_status(untrusted, 0, 3), FloydWarshallPairStatus::Unbounded);
  const auto self = floyd_warshall(1, {{0, 0, -1}});
  EXPECT_THROW((void)reconstruct_path(self, 0, 0), std::domain_error);
}
TEST(FloydWarshallTrace, SeedOrderEqualCostsZeroCyclesAndSignedZeroArePreserved) {
  const Edges e{{0, 1, 2}, {0, 1, 1}, {0, 1, 1}, {1, 2, -1}, {0, 2, 0}, {0, 0, 1}, {2, 2, -0.}};
  const auto  t = floyd_warshall_trace(3, e);
  replay(t);
  EXPECT_EQ(reconstruct_path(t.result, 0, 2), (std::vector<std::size_t>{0, 2}));
  EXPECT_EQ(t.result.next[0][0], kNoVertex);
  const auto z = floyd_warshall(2, {{0, 1, -0.}});
  EXPECT_EQ(bits(z.distance[0][1]), bits(-0.));
  EXPECT_EQ(reconstruct_path(z, 0, 1), (std::vector<std::size_t>{0, 1}));
  compare(3, {{0, 1, 0}, {1, 0, 0}, {1, 2, 0}, {0, 2, 0}});
  const auto f = floyd_warshall(4, {{0, 1, .125}, {1, 2, -.375}, {2, 3, .5}, {0, 3, .25}});
  EXPECT_EQ(f.distance[0][2], -.25);
  EXPECT_EQ(f.distance[0][3], .25);
}
TEST(FloydWarshallTrace, WholeEntryQuotaVirtualBoundsEmptyAndOwnedMatrices) {
  const auto e = floyd_warshall_example_edges();
  EXPECT_EQ(floyd_warshall_trace_entries(5, e.size()), 560);
  EXPECT_THROW(floyd_warshall_trace(5, e, 559), std::length_error);
  auto t = floyd_warshall_trace(5, e, 560);
  EXPECT_EQ(t.stages.size(), 6);
  EXPECT_EQ(floyd_warshall_trace(5, e, 0).stages.size(), 6);
  EXPECT_EQ(floyd_warshall_trace_entries(0, 0), 1);
  EXPECT_EQ(floyd_warshall_trace(0, {}, 1).stages.size(), 1);
  EXPECT_TRUE(floyd_warshall_unbounded_pairs(floyd_warshall(0, {})).empty());
  EXPECT_THROW(floyd_warshall_trace_entries(SIZE_MAX, 0), std::length_error);
  EXPECT_THROW(floyd_warshall_trace_entries(0, SIZE_MAX), std::length_error);
  EXPECT_THROW(floyd_warshall(SIZE_MAX, {}), std::length_error);
  auto input = e;
  auto owned = floyd_warshall_trace(5, input);
  input[0]   = {4, 4, 999};
  EXPECT_EQ(owned.edges, e);
  const auto final                   = owned.result.distance;
  owned.stages.back().distance[0][2] = 99;
  EXPECT_EQ(owned.result.distance, final);
  EXPECT_EQ(owned.stages[0].distance[0][2], 8);
}
TEST(FloydWarshallTrace, ValidatesAllEdgesBeforeQuotaOrAllocationAndRejectsNonfiniteArithmetic) {
  for (double w : {inf, -inf, std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(floyd_warshall(2, {{0, 1, w}}), std::invalid_argument);
    EXPECT_THROW(floyd_warshall_trace(2, {{0, 1, 1}, {1, 0, w}}, 1), std::invalid_argument);
  }
  EXPECT_THROW(floyd_warshall_trace(2, {{0, 2, 1}}, 1), std::invalid_argument);
  EXPECT_THROW(floyd_warshall_trace(SIZE_MAX, {{SIZE_MAX, 0, 1}}, 1), std::invalid_argument);
  const auto max = std::numeric_limits<double>::max();
  for (double v : {max, -max})
    EXPECT_THROW(floyd_warshall(3, {{0, 1, v}, {1, 2, v}}), std::overflow_error);
  EXPECT_THROW(floyd_warshall(3, {{0, 1, max}, {1, 2, max}, {0, 2, 1}}), std::overflow_error);
  EXPECT_EQ(floyd_warshall(2, {{0, 1, max}}).distance[0][1], max);
}
TEST(FloydWarshallTrace, FullMatrixValidationAndBoundedExternalChains) {
  const auto valid = floyd_warshall(3, {{0, 1, 1}, {1, 2, 1}});
  for (int mode = 0; mode < 7; ++mode) {
    auto r = valid;
    if (mode == 0)
      r.next.pop_back();
    if (mode == 1)
      r.next[1].clear();
    if (mode == 2)
      r.distance[2].clear();
    if (mode == 3)
      r.next[2][0] = 99;
    if (mode == 4)
      r.distance[2][0] = -inf;
    if (mode == 5)
      r.distance[2][0] = std::numeric_limits<double>::quiet_NaN();
    if (mode == 6)
      r.next.clear();
    EXPECT_THROW((void)reconstruct_path(r, 0, 0), std::invalid_argument);
    EXPECT_THROW(floyd_warshall_unbounded_pairs(r), std::invalid_argument);
  }
  auto cyclic       = valid;
  cyclic.next[0][2] = 0;
  EXPECT_THROW((void)reconstruct_path(cyclic, 0, 2), std::invalid_argument);
  cyclic.next[0][2] = 1;
  cyclic.next[1][2] = 0;
  EXPECT_THROW((void)reconstruct_path(cyclic, 0, 2), std::invalid_argument);
  cyclic.next[1][2] = kNoVertex;
  EXPECT_TRUE(reconstruct_path(cyclic, 0, 2).empty());
  EXPECT_THROW((void)reconstruct_path(valid, 3, 0), std::invalid_argument);
  EXPECT_THROW(floyd_warshall_pair_status(valid, 0, 3), std::invalid_argument);
  auto boolean               = valid;
  boolean.has_negative_cycle = true;
  EXPECT_EQ(reconstruct_path(boolean, 0, 2), (std::vector<std::size_t>{0, 1, 2}));
}
TEST(FloydWarshallTrace, DenseAndDisconnectedGuardCountsAreActual) {
  for (std::size_t n : {1u, 2u, 4u, 8u, 16u, 32u}) {
    Edges dense;
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        if (i != j)
          dense.emplace_back(i, j, 1);
    const auto d = floyd_warshall_work(n, dense);
    EXPECT_EQ(d.work.candidates, n * n * n);
    EXPECT_EQ(d.work.row_checks, n * n);
    EXPECT_EQ(d.work.skipped_rows, 0);
    EXPECT_EQ(d.work.skipped_columns, 0);
    const auto sparse = floyd_warshall_work(n, {});
    EXPECT_EQ(sparse.work.candidates, n);
    EXPECT_EQ(sparse.work.skipped_rows, n * n - n);
    EXPECT_EQ(sparse.work.column_checks, n * n);
    EXPECT_EQ(sparse.work.skipped_columns, n * n - n);
  }
}
TEST(FloydWarshallTrace, OriginalMatrixAllPathsAndLegacySignatures) {
  using Detector = AllPairsShortestPaths (*)(std::size_t, const Edges&);
  Detector fn    = &floyd_warshall;
  using Path = std::vector<std::size_t> (*)(const AllPairsShortestPaths&, std::size_t, std::size_t);
  Path       path = &reconstruct_path;
  const auto e    = floyd_warshall_example_edges();
  const auto r    = fn(5, e);
  EXPECT_EQ(
      r.distance,
      (std::vector<std::vector<double>>{
          {0, 1, -3, 2, -4},
          {3, 0, -4, 1, -1},
          {7, 4, 0, 5, 3},
          {2, -1, -5, 0, -2},
          {8, 5, 1, 6, 0}}));
  EXPECT_EQ(path(r, 0, 2), (std::vector<std::size_t>{0, 4, 3, 2}));
  for (std::size_t i = 0; i < 5; ++i)
    for (std::size_t j = 0; j < 5; ++j)
      path_cost(r, e, i, j);
  replay(floyd_warshall_trace(5, e));
}
