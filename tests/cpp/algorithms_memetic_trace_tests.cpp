#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/memetic.hpp>
#include <limits>
#include <stdexcept>
using namespace datamunge::algorithms;
namespace {
double score(const MemeticGraph& g, const std::vector<int>& side) {
  double sum = 0;
  for (std::size_t u = 0; u < g.size(); ++u)
    for (auto [v, w] : g[u])
      if (u < static_cast<std::size_t>(v) && side[u] != side[v])
        sum += w;
  return sum;
}
double brute(const MemeticGraph& g) {
  double best = 0;
  for (unsigned mask = 0; mask < (1u << g.size()); ++mask) {
    std::vector<int> side(g.size());
    for (std::size_t u = 0; u < side.size(); ++u)
      side[u] = (mask >> u) & 1;
    best = std::max(best, score(g, side));
  }
  return best;
}
void verify(const MemeticTrace& t) {
  auto state = t.seed ? t.seed : 0x2545F4914F6CDD1DULL;
  for (auto& d : t.draws) {
    EXPECT_EQ(d.before, state);
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    EXPECT_EQ(d.after, state);
    if (d.bound)
      EXPECT_EQ(d.choice, state % d.bound);
    if (d.kind == 3)
      EXPECT_DOUBLE_EQ(d.unit, (state >> 11) * (1.0 / 9007199254740992.0));
  }
  std::size_t entries = 5 + 2 * t.n + t.draws.size() + t.local_steps.size(), flips = 0, refs = 0;
  for (auto& row : t.graph)
    entries += 2 * row.size();
  bool             hasbest = false, refined = false;
  double           best = 0;
  std::vector<int> bestside;
  for (auto& i : t.individuals) {
    entries += 1 + 2 * t.n;
    auto side = i.before_local;
    ASSERT_EQ(side.size(), t.n);
    if (i.generation < 0) {
      ASSERT_EQ(i.past_draw - i.first_draw, t.n);
      for (int u = 0; u < t.n; ++u) {
        auto& d = t.draws[i.first_draw + u];
        EXPECT_EQ(d.kind, 0);
        EXPECT_EQ(side[u], d.choice);
      }
    } else {
      ASSERT_LT(i.generation, t.generations.size());
      auto& prior = t.generations[i.generation];
      ASSERT_EQ(i.past_draw - i.first_draw, 6 + 2 * t.n);
      auto tournament = [&](std::size_t start) {
        auto b = t.draws[start].choice;
        for (std::size_t j = 1; j < 3; ++j) {
          auto c = t.draws[start + j].choice;
          if (prior.fitness[c] > prior.fitness[b])
            b = c;
        }
        return b;
      };
      auto a = tournament(i.first_draw), b = tournament(i.first_draw + 3);
      EXPECT_EQ(i.parent_a, a);
      EXPECT_EQ(i.parent_b, b);
      for (int u = 0; u < t.n; ++u) {
        auto& cross    = t.draws[i.first_draw + 6 + u];
        auto& mutation = t.draws[i.first_draw + 6 + t.n + u];
        EXPECT_EQ(cross.kind, 2);
        EXPECT_EQ(mutation.kind, 3);
        int bit = cross.choice ? prior.population[a][u] : prior.population[b][u];
        if (mutation.unit < 1.0 / t.n)
          bit ^= 1;
        EXPECT_EQ(side[u], bit);
      }
    }
    for (std::size_t k = i.first_local; k < i.past_local; ++k) {
      auto& s = t.local_steps[k];
      EXPECT_EQ(s.generation, i.generation);
      EXPECT_EQ(s.individual, i.slot);
      EXPECT_EQ(s.before, side[s.vertex]);
      double delta = 0;
      for (auto [v, w] : t.graph[s.vertex]) {
        delta += side[v] != side[s.vertex] ? -w : w;
        ++refs;
      }
      EXPECT_DOUBLE_EQ(s.delta, delta);
      EXPECT_EQ(s.flipped, delta > 1e-12);
      if (s.flipped) {
        side[s.vertex] ^= 1;
        ++flips;
      }
      EXPECT_EQ(s.after, side[s.vertex]);
    }
    EXPECT_EQ(side, i.after_local);
    EXPECT_DOUBLE_EQ(i.fitness, score(t.graph, side));
    if (i.refined)
      for (int u = 0; u < t.n; ++u) {
        double delta = 0;
        for (auto [v, w] : t.graph[u])
          delta += side[v] != side[u] ? -w : w;
        EXPECT_LE(delta, 1e-12);
      }
    if (!hasbest || i.fitness > best) {
      hasbest  = true;
      best     = i.fitness;
      bestside = side;
      refined  = i.refined;
    }
  }
  for (std::size_t k = 0; k < t.generations.size(); ++k) {
    auto& g = t.generations[k];
    entries += 1 + t.population_size * t.n + t.population_size;
    EXPECT_EQ(g.index, k);
    ASSERT_EQ(g.population.size(), t.population_size);
    ASSERT_EQ(g.fitness.size(), t.population_size);
    std::size_t winner = 0;
    for (std::size_t p = 0; p < g.population.size(); ++p) {
      EXPECT_DOUBLE_EQ(g.fitness[p], score(t.graph, g.population[p]));
      if (g.fitness[p] > g.fitness[winner])
        winner = p;
    }
    EXPECT_EQ(g.best_index, winner);
    EXPECT_DOUBLE_EQ(g.best_value, g.fitness[winner]);
    if (k) {
      auto& prior = t.generations[k - 1];
      EXPECT_EQ(g.population[0], prior.population[prior.best_index]);
      EXPECT_DOUBLE_EQ(g.fitness[0], prior.best_value);
      EXPECT_GE(g.best_value, prior.best_value);
    }
  }
  EXPECT_EQ(t.captured_entries, entries);
  EXPECT_EQ(t.analysis.result.side, bestside);
  EXPECT_DOUBLE_EQ(t.analysis.result.value, best);
  EXPECT_EQ(t.analysis.locally_refined, refined);
  EXPECT_EQ(t.analysis.work.rng_draws, t.draws.size());
  EXPECT_EQ(t.analysis.work.local_checks, t.local_steps.size());
  EXPECT_EQ(t.analysis.work.flips, flips);
  EXPECT_EQ(t.analysis.work.delta_references, refs);
  EXPECT_EQ(t.analysis.work.individuals, t.individuals.size());
  if (t.analysis.complete) {
    EXPECT_TRUE(t.analysis.initialized);
    EXPECT_EQ(t.generations.size(), t.requested_generations + 1);
    EXPECT_EQ(
        t.individuals.size(),
        t.population_size + t.requested_generations * (t.population_size - 1));
    EXPECT_EQ(
        t.draws.size(),
        t.n * t.population_size +
            t.requested_generations * (t.population_size - 1) * (6 + 2 * t.n));
    EXPECT_EQ(
        t.analysis.work.best_comparisons, (t.population_size - 1) * (t.requested_generations + 1));
  }
}
} // namespace
TEST(
    MemeticTrace,
    EverySmallIntegerWeightedGraphFeasibilityRefinementAndIndependentExactEnumeration) {
  for (int n = 0; n <= 4; ++n) {
    std::size_t edges = n * (n - 1) / 2, count = 1;
    for (std::size_t i = 0; i < edges; ++i)
      count *= 3;
    for (std::size_t encoded = 0; encoded < count; ++encoded) {
      auto         bits = encoded;
      MemeticGraph graph(n);
      for (int u = 0; u < n; ++u)
        for (int v = u + 1; v < n; ++v) {
          auto weight = bits % 3;
          bits /= 3;
          if (weight) {
            graph[u].push_back({v, double(weight)});
            graph[v].push_back({u, double(weight)});
          }
        }
      auto t = memetic_max_cut_trace(n, graph, 4, 2, encoded + 1);
      verify(t);
      EXPECT_TRUE(t.analysis.complete);
      EXPECT_TRUE(t.analysis.locally_refined);
      double exact = brute(graph);
      EXPECT_LE(t.analysis.result.value, exact);
      auto e = memetic_max_cut_exact(n, graph);
      EXPECT_DOUBLE_EQ(e.value, exact);
      EXPECT_DOUBLE_EQ(score(graph, e.side), exact);
    }
  }
}
TEST(MemeticTrace, OriginalGraphDrawsPopulationHistoryAndLegacyPrefixResults) {
  auto          graph = memetic_demo_graph();
  std::uint64_t state = 999;
  auto          draw  = [&]() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  };
  MemeticGraph expected(16);
  std::size_t  edges = 0;
  for (int u = 0; u < 16; ++u)
    for (int v = u + 1; v < 16; ++v)
      if (draw() % 2) {
        double w = 1 + draw() % 9;
        expected[u].push_back({v, w});
        expected[v].push_back({u, w});
        ++edges;
      }
  EXPECT_EQ(graph, expected);
  EXPECT_EQ(edges, 56);
  auto t = memetic_max_cut_trace(16, graph, 24, 40, 1);
  verify(t);
  EXPECT_DOUBLE_EQ(t.analysis.result.value, 195);
  EXPECT_DOUBLE_EQ(memetic_max_cut_exact(16, graph).value, 195);
  for (int generation : {0, 1, 5, 30, 40}) {
    auto r = memetic_max_cut(16, graph, 24, generation, 1);
    EXPECT_DOUBLE_EQ(r.value, t.generations[generation].best_value);
    EXPECT_EQ(r.side, t.generations[generation].population[t.generations[generation].best_index]);
  }
}
TEST(MemeticTrace, WorkCapsCaptureCapsEmptySignedParallelCompleteGuardsAndOwnership) {
  MemeticGraph graph{{{1, 2}}, {{0, 2}}};
  auto         full = memetic_max_cut_trace(2, graph, 4, 2, 1);
  for (std::size_t cap : {1, 2, 3, 5}) {
    auto t = memetic_max_cut_trace(2, graph, 4, 2, 1, cap);
    verify(t);
    EXPECT_LE(t.analysis.work.local_checks, cap);
    EXPECT_DOUBLE_EQ(t.analysis.result.value, score(graph, t.analysis.result.side));
  }
  EXPECT_NO_THROW(memetic_max_cut_trace(2, graph, 4, 2, 1, 0, full.captured_entries));
  EXPECT_THROW(
      memetic_max_cut_trace(2, graph, 4, 2, 1, 0, full.captured_entries - 1), std::length_error);
  EXPECT_NO_THROW(memetic_max_cut_trace(2, graph, 4, 2, 1, 0, 0));
  EXPECT_EQ(memetic_max_cut_trace(0, {}, 1, 0, 1).captured_entries, 8);
  auto zero = memetic_max_cut_trace(1, {{}}, 1, 2, 0);
  verify(zero);
  EXPECT_TRUE(zero.analysis.complete);
  EXPECT_EQ(zero.analysis.result.side.size(), 1);
  EXPECT_DOUBLE_EQ(memetic_max_cut_exact(2, {{{1, -3}}, {{0, -3}}}).value, 0);
  EXPECT_DOUBLE_EQ(memetic_max_cut_exact(2, {{{1, 1}, {1, 2}}, {{0, 2}, {0, 1}}}).value, 3);
  EXPECT_THROW(memetic_max_cut_exact(2, graph, 1), std::length_error);
  EXPECT_THROW(memetic_max_cut(-1, {}), std::invalid_argument);
  EXPECT_THROW(memetic_max_cut(2, {{}}), std::invalid_argument);
  EXPECT_THROW(memetic_max_cut(2, {{{1, 2}}, {{0, 3}}}), std::invalid_argument);
  EXPECT_THROW(memetic_max_cut(1, {{{0, 1}}}), std::invalid_argument);
  EXPECT_THROW(memetic_max_cut(1, {{{-1, 1}}}), std::invalid_argument);
  EXPECT_THROW(
      memetic_max_cut(2, {{{1, std::numeric_limits<double>::infinity()}}, {{0, 1}}}),
      std::invalid_argument);
  EXPECT_THROW(memetic_max_cut(2, graph, 0), std::invalid_argument);
  EXPECT_THROW(memetic_max_cut(2, graph, 1, -1), std::invalid_argument);
  EXPECT_THROW(memetic_cut_value(2, graph, {0, 2}), std::invalid_argument);
  EXPECT_THROW(memetic_cut_value(2, graph, {0}), std::invalid_argument);
  double max = std::numeric_limits<double>::max();
  EXPECT_THROW(
      memetic_max_cut(3, {{{1, max}, {2, max}}, {{0, max}}, {{0, max}}}), std::overflow_error);
  auto owned = memetic_max_cut_trace(2, graph, 1, 0, 1);
  graph[0].clear();
  EXPECT_FALSE(owned.graph[0].empty());
}
