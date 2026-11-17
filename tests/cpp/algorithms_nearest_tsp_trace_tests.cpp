#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/nearest_neighbor_tsp.hpp>
#include <limits>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
TEST(NearestTSPTrace, IndependentSubsetOptimalReferenceAndActualGreedyReplay) {
  std::mt19937_64 rng(244253);
  for (int n = 1; n <= 8; ++n)
    for (int repeat = 0; repeat < 20; ++repeat) {
      std::vector<std::vector<double>> d(n, std::vector<double>(n));
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          d[i][j] = i == j ? 0 : rng() % 30;
      std::size_t       start = rng() % n;
      auto              t     = nearest_neighbor_tsp_trace(d, start);
      std::vector<bool> used(n);
      used[start]         = true;
      std::size_t current = start, trial = 0;
      for (auto& choice : t.choices) {
        std::vector<std::size_t> candidates;
        for (int i = 0; i < n; ++i)
          if (!used[i])
            candidates.push_back(i);
        auto best = *std::min_element(candidates.begin(), candidates.end(), [&](auto a, auto b) {
          return d[current][a] < d[current][b];
        });
        EXPECT_EQ(choice.from, current);
        EXPECT_EQ(choice.to, best);
        EXPECT_EQ(choice.first_trial, trial);
        EXPECT_EQ(choice.trial_count, candidates.size());
        std::size_t running = n;
        for (auto v : candidates) {
          auto& r = t.trials[trial++];
          EXPECT_EQ(r.from, current);
          EXPECT_EQ(r.candidate, v);
          EXPECT_EQ(r.distance, d[current][v]);
          EXPECT_EQ(r.first, running == std::size_t(n));
          bool improve = running == std::size_t(n) || d[current][v] < d[current][running];
          EXPECT_EQ(r.improved, improve);
          if (improve)
            running = v;
        }
        EXPECT_EQ(running, best);
        used[best] = true;
        current    = best;
      }
      EXPECT_EQ(t.analysis.candidate_checks, n * (n - 1));
      EXPECT_EQ(t.analysis.distance_comparisons, (n - 1) * (n - 2) / 2);
      EXPECT_EQ(t.analysis.edges_summed, n);
      double total = 0;
      for (auto& e : t.edges) {
        total += d[e.from][e.to];
        EXPECT_EQ(e.distance, d[e.from][e.to]);
        EXPECT_EQ(e.cumulative, total);
      }
      EXPECT_EQ(total, t.analysis.result.cost);
      EXPECT_EQ(nearest_neighbor_tsp(d, start).tour, t.analysis.result.tour);
      const auto                       states = 1U << n;
      std::vector<std::vector<double>> dp(
          states, std::vector<double>(n, std::numeric_limits<double>::infinity()));
      dp[1U << start][start] = 0;
      for (unsigned mask = 0; mask < states; ++mask)
        for (int last = 0; last < n; ++last)
          if (std::isfinite(dp[mask][last]))
            for (int next = 0; next < n; ++next)
              if (!(mask & (1U << next)))
                dp[mask | (1U << next)][next] =
                    std::min(dp[mask | (1U << next)][next], dp[mask][last] + d[last][next]);
      double optimal = std::numeric_limits<double>::infinity();
      for (int last = 0; last < n; ++last)
        optimal = std::min(optimal, dp.back()[last] + d[last][start]);
      auto exact = enumerate_tsp(d, start);
      EXPECT_EQ(exact.cost, optimal);
      EXPECT_GE(t.analysis.result.cost, optimal);
      EXPECT_NO_THROW(nearest_neighbor_tsp_trace(d, start, t.captured_entries));
      EXPECT_THROW(nearest_neighbor_tsp_trace(d, start, t.captured_entries - 1), std::length_error);
    }
}
TEST(NearestTSPTrace, OriginalAdversaryEmptySingletonGuardsAndUnusedOverflow) {
  auto original = nearest_neighbor_tsp_trace({{0, 1, 2}, {1, 0, 1}, {2, 1, 0}});
  EXPECT_EQ(original.analysis.result.cost, 4);
  EXPECT_EQ(original.analysis.result.tour, (std::vector<std::size_t>{0, 1, 2, 0}));
  EXPECT_EQ(original.captured_entries, 25);
  auto bad = nearest_neighbor_tsp_adversary();
  EXPECT_EQ(nearest_neighbor_tsp(bad).cost, 102);
  EXPECT_EQ(enumerate_tsp(bad).cost, 4);
  EXPECT_THROW(enumerate_tsp(bad, 0, 2), std::length_error);
  EXPECT_NO_THROW(enumerate_tsp(bad, 0, 0));
  EXPECT_EQ(nearest_neighbor_tsp({}, 999).cost, 0);
  EXPECT_EQ(nearest_neighbor_tsp({{7}}).cost, 7);
  EXPECT_THROW(nearest_neighbor_tsp({{0, 1}, {1, 0}}, 2), std::invalid_argument);
  EXPECT_THROW(nearest_neighbor_tsp({{0}, {1, 0}}), std::invalid_argument);
  EXPECT_THROW(nearest_neighbor_tsp({{0, -1}, {1, 0}}), std::invalid_argument);
  EXPECT_THROW(nearest_neighbor_tsp({{std::nan("")}}), std::invalid_argument);
  auto max = std::numeric_limits<double>::max();
  EXPECT_THROW(nearest_neighbor_tsp({{0, max}, {max, 0}}), std::overflow_error);
  EXPECT_THROW(enumerate_tsp({{0, max}, {max, 0}}), std::overflow_error);
  EXPECT_EQ(enumerate_tsp({{0, max, 1}, {1, 0, max}, {1, 1, 0}}).cost, 3);
  EXPECT_THROW(nearest_neighbor_tsp_adversary(1), std::invalid_argument);
}
