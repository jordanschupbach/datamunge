#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/page_rank.hpp>
#include <numeric>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
namespace {
std::vector<long double> stationary(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    long double                                             damping) {
  std::vector<std::size_t> degree(n);
  for (auto [u, v] : edges)
    ++degree[u];
  std::vector<std::vector<long double>> a(n, std::vector<long double>(n + 1));
  for (std::size_t v = 0; v < n; ++v) {
    a[v][v] = 1;
    a[v][n] = (1 - damping) / n;
  }
  for (std::size_t u = 0; u < n; ++u)
    if (!degree[u])
      for (std::size_t v = 0; v < n; ++v)
        a[v][u] -= damping / n;
  for (auto [u, v] : edges)
    a[v][u] -= damping / degree[u];
  for (std::size_t k = 0; k < n; ++k) {
    std::size_t pivot = k;
    for (std::size_t i = k + 1; i < n; ++i)
      if (std::abs(a[i][k]) > std::abs(a[pivot][k]))
        pivot = i;
    std::swap(a[k], a[pivot]);
    auto d = a[k][k];
    for (std::size_t j = k; j <= n; ++j)
      a[k][j] /= d;
    for (std::size_t i = 0; i < n; ++i)
      if (i != k) {
        auto scale = a[i][k];
        for (std::size_t j = k; j <= n; ++j)
          a[i][j] -= scale * a[k][j];
      }
  }
  std::vector<long double> x(n);
  for (std::size_t i = 0; i < n; ++i)
    x[i] = a[i][n];
  return x;
}
} // namespace
TEST(PageRankTrace, IndependentDenseStationarySolveAndEveryActualEdgeTransfer) {
  std::mt19937_64 rng(264273);
  for (std::size_t n = 1; n <= 9; ++n)
    for (int repeat = 0; repeat < 20; ++repeat) {
      std::vector<std::pair<std::size_t, std::size_t>> edges;
      for (int k = 0; k < repeat; ++k)
        edges.push_back({rng() % n, rng() % n});
      double damping   = repeat % 3 == 0 ? .99 : repeat % 3 == 1 ? .85 : 0.;
      auto   t         = page_rank_trace(n, edges, {damping, 1e-14, 10000, 0});
      auto   reference = stationary(n, edges, damping);
      EXPECT_TRUE(t.run.converged);
      for (std::size_t i = 0; i < n; ++i)
        EXPECT_NEAR(t.run.scores[i], double(reference[i]), 5e-12);
      EXPECT_NEAR(std::accumulate(t.run.scores.begin(), t.run.scores.end(), 0.), 1., 2e-13);
      EXPECT_EQ(t.run.edge_transfers, edges.size() * t.run.iterations);
      EXPECT_EQ(t.run.dangling_checks, n * t.run.iterations);
      EXPECT_EQ(
          t.captured_entries,
          4 + 2 * edges.size() + 2 * n + t.iterations.size() * (1 + 2 * n) + t.transfers.size());
      std::vector<std::size_t> degree(n);
      for (auto [u, v] : edges)
        ++degree[u];
      auto previous = t.initial;
      for (auto& s : t.iterations) {
        EXPECT_EQ(s.before, previous);
        double dangling = 0;
        for (std::size_t u = 0; u < n; ++u)
          if (!degree[u])
            dangling += previous[u];
        EXPECT_EQ(s.dangling_mass, dangling);
        EXPECT_EQ(s.uniform_base, (1 - damping) / n + damping * dangling / n);
        std::vector<double> next(n, s.uniform_base);
        ASSERT_EQ(s.transfer_end - s.transfer_begin, edges.size());
        std::size_t k = s.transfer_begin;
        for (std::size_t u = 0; u < n; ++u)
          for (auto [from, v] : edges)
            if (from == u) {
              auto& tr = t.transfers[k++];
              EXPECT_EQ(tr.source, u);
              EXPECT_EQ(tr.destination, v);
              EXPECT_EQ(tr.iteration, s.ordinal);
              EXPECT_EQ(tr.source_rank, previous[u]);
              EXPECT_EQ(tr.out_degree, degree[u]);
              EXPECT_EQ(tr.destination_before, next[v]);
              auto amount = damping * previous[u] / degree[u];
              EXPECT_EQ(tr.contribution, amount);
              next[v] += amount;
              EXPECT_EQ(tr.destination_after, next[v]);
            }
        EXPECT_EQ(s.after, next);
        double change = 0;
        for (std::size_t i = 0; i < n; ++i)
          change += std::abs(next[i] - previous[i]);
        EXPECT_EQ(s.l1_change, change);
        previous = next;
      }
      EXPECT_EQ(t.run.scores, previous);
      EXPECT_EQ(page_rank(n, edges, damping, 1e-14, 10000), t.run.scores);
      auto o              = t.options;
      o.max_trace_entries = t.captured_entries;
      EXPECT_NO_THROW(page_rank_trace(n, edges, o));
      --o.max_trace_entries;
      EXPECT_THROW(page_rank_trace(n, edges, o), std::length_error);
    }
}
TEST(PageRankTrace, OriginalGraphDanglingAnalyticBudgetAndCompleteGuards) {
  auto r = page_rank_run(3, page_rank_demo_edges());
  EXPECT_NEAR(r.scores[0], .05, 1e-14);
  EXPECT_NEAR(r.scores[1], .05, 1e-14);
  EXPECT_NEAR(r.scores[2], .9, 1e-14);
  auto d = page_rank_run(2, {{0, 1}});
  EXPECT_NEAR(d.scores[0], 1 / 2.85, 1e-12);
  EXPECT_NEAR(d.scores[1], 1 - 1 / 2.85, 1e-12);
  auto zero = page_rank_trace(3, page_rank_demo_edges(), {.85, 1e-12, 0, 100});
  EXPECT_FALSE(zero.run.converged);
  EXPECT_EQ(zero.run.iterations, 0);
  EXPECT_EQ(zero.run.scores, zero.initial);
  EXPECT_EQ(zero.captured_entries, 16);
  auto empty = page_rank_trace(0, {});
  EXPECT_TRUE(empty.run.converged);
  EXPECT_EQ(empty.captured_entries, 4);
  EXPECT_THROW(page_rank(0, {{0, 0}}), std::invalid_argument);
  EXPECT_THROW(page_rank(1, {}, NAN), std::invalid_argument);
  EXPECT_THROW(page_rank(1, {}, 1), std::invalid_argument);
  EXPECT_THROW(page_rank(1, {}, .85, INFINITY), std::invalid_argument);
  EXPECT_THROW(page_rank(1, {}, .85, 0), std::invalid_argument);
  EXPECT_THROW(page_rank(SIZE_MAX, {}), std::length_error);
  EXPECT_THROW(page_rank_trace(0, {}, {.85, 1e-12, 0, 3}), std::length_error);
}
