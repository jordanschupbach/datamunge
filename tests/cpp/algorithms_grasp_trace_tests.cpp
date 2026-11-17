#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/grasp_analysis.hpp>
#include <limits>
#include <random>
#include <tuple>
using namespace datamunge::algorithms;
namespace {
GraspAdjacency graph(int n, const std::vector<std::tuple<int, int, double>>& edges) {
  GraspAdjacency a(n);
  for (auto [u, v, w] : edges) {
    a[u].push_back({v, w});
    a[v].push_back({u, w});
  }
  return a;
}
void certificate(const GraspAdjacency& adj, int iterations, double alpha, std::uint64_t seed) {
  const int  n = adj.size();
  const auto t = grasp_max_cut_trace(n, adj, iterations, alpha, seed);
  const auto a = grasp_max_cut_run(n, adj, iterations, alpha, seed);
  const auto r = grasp_max_cut(n, adj, iterations, alpha, seed);
  EXPECT_EQ(r.side, a.result.side);
  EXPECT_EQ(r.value, a.result.value);
  EXPECT_EQ(t.run.result.side, r.side);
  EXPECT_EQ(t.run.result.value, r.value);
  std::size_t stepindex = 0, flipindex = 0, entries = 0, neighbors = 0, rngwords = 0, passes = 0,
              accepted = 0;
  double        best   = -std::numeric_limits<double>::infinity();
  std::uint64_t state  = seed ? seed : 0x9E3779B97F4A7C15ULL;
  for (int restart = 0; restart < iterations; ++restart) {
    std::vector<int> side(n, -1);
    entries += 1 + 2 * n;
    for (int step = 0; step < n; ++step) {
      ASSERT_LT(stepindex, t.construction.size());
      const auto& s = t.construction[stepindex++];
      EXPECT_EQ(s.restart, restart);
      EXPECT_EQ(s.step, step);
      entries += 1 + s.candidates.size();
      std::size_t              ci = 0;
      std::vector<std::size_t> rcl;
      double                   hi = -INFINITY, lo = INFINITY;
      for (int v = 0; v < n; ++v)
        if (side[v] < 0) {
          ASSERT_LT(ci, s.candidates.size());
          const auto& c  = s.candidates[ci];
          double      g0 = 0, g1 = 0;
          for (auto [u, w] : adj[v]) {
            ++neighbors;
            if (side[u] >= 0) {
              if (side[u] == 1)
                g0 += w;
              else
                g1 += w;
            }
          }
          EXPECT_EQ(c.vertex, v);
          EXPECT_EQ(c.gain0, g0);
          EXPECT_EQ(c.gain1, g1);
          EXPECT_EQ(c.side, g0 >= g1 ? 0 : 1);
          EXPECT_EQ(c.gain, std::max(g0, g1));
          hi = std::max(hi, c.gain);
          lo = std::min(lo, c.gain);
          ++ci;
        }
      EXPECT_EQ(ci, s.candidates.size());
      EXPECT_EQ(s.best_gain, hi);
      EXPECT_EQ(s.worst_gain, lo);
      EXPECT_EQ(s.threshold, hi - alpha * (hi - lo));
      for (std::size_t i = 0; i < ci; ++i) {
        bool member = s.candidates[i].gain >= s.threshold - 1e-12;
        EXPECT_EQ(s.candidates[i].in_rcl, member);
        if (member)
          rcl.push_back(i);
      }
      EXPECT_EQ(s.rng_before, state);
      state ^= state << 13;
      state ^= state >> 7;
      state ^= state << 17;
      ++rngwords;
      EXPECT_EQ(s.rng_after, state);
      ASSERT_FALSE(rcl.empty());
      EXPECT_EQ(s.pick, rcl[state % rcl.size()]);
      const auto& pick  = s.candidates[s.pick];
      side[pick.vertex] = pick.side;
    }
    const auto& summary = t.restarts[restart];
    EXPECT_EQ(summary.constructed_side, side);
    bool        improved = true;
    std::size_t pass     = 0;
    while (improved) {
      improved = false;
      ++pass;
      ++passes;
      for (int v = 0; v < n; ++v) {
        ASSERT_LT(flipindex, t.flips.size());
        const auto& f = t.flips[flipindex++];
        ++entries;
        double delta = 0;
        for (auto [u, w] : adj[v])
          delta += (side[u] != side[v]) ? -w : w;
        EXPECT_EQ(f.restart, restart);
        EXPECT_EQ(f.pass, pass);
        EXPECT_EQ(f.vertex, v);
        EXPECT_EQ(f.before_side, side[v]);
        EXPECT_EQ(f.delta, delta);
        EXPECT_EQ(f.accepted, delta > 1e-12);
        if (f.accepted) {
          side[v] ^= 1;
          improved = true;
          ++accepted;
        }
      }
    }
    double val = 0;
    for (int u = 0; u < n; ++u)
      for (auto [v, w] : adj[u])
        if (u < v && side[u] != side[v])
          val += w;
    EXPECT_EQ(summary.final_side, side);
    EXPECT_EQ(summary.final_value, val);
    EXPECT_EQ(summary.improved_best, val > best);
    best = std::max(best, val);
    EXPECT_EQ(summary.best_value, best);
  }
  EXPECT_EQ(stepindex, t.construction.size());
  EXPECT_EQ(flipindex, t.flips.size());
  EXPECT_EQ(entries, t.captured_entries);
  EXPECT_EQ(a.work.candidate_scores, std::size_t(iterations) * n * (n + 1) / 2);
  EXPECT_EQ(a.work.rcl_tests, a.work.candidate_scores);
  EXPECT_EQ(a.work.rng_words, rngwords);
  EXPECT_EQ(a.work.construction_neighbors, neighbors);
  EXPECT_EQ(a.work.flip_tests, flipindex);
  EXPECT_EQ(a.work.accepted_flips, accepted);
  EXPECT_EQ(a.work.local_passes, passes);
  std::size_t directed = 0;
  for (auto& row : adj)
    directed += row.size();
  EXPECT_EQ(a.work.local_neighbors, directed * passes);
  EXPECT_EQ(a.work.objective_calls, iterations);
  EXPECT_EQ(a.work.objective_neighbors, directed * iterations);
  GraspOptions o;
  o.max_trace_entries = entries;
  EXPECT_NO_THROW(grasp_max_cut_trace(n, adj, iterations, alpha, seed, o));
  if (entries > 1) {
    o.max_trace_entries = entries - 1;
    EXPECT_THROW(grasp_max_cut_trace(n, adj, iterations, alpha, seed, o), std::length_error);
  }
}
} // namespace
TEST(GraspTrace, TinyActualRclAndImmediateFlip) {
  const auto adj = graph(3, {{0, 1, 1}, {0, 2, 2}, {1, 2, 3}});
  const auto t   = grasp_max_cut_trace(3, adj, 1, 1, 3);
  EXPECT_EQ(t.restarts[0].constructed_side, (std::vector<int>{0, 1, 0}));
  EXPECT_EQ(t.restarts[0].final_side, (std::vector<int>{1, 1, 0}));
  EXPECT_EQ(t.run.result.value, 5);
  EXPECT_EQ(t.run.work.accepted_flips, 1);
  EXPECT_EQ(t.flips[0].delta, 1);
  certificate(adj, 3, 1, 3);
}
TEST(GraspTrace, AllFourVertexSignedGraphsAgainstIndependentMasks) {
  const std::vector<std::pair<int, int>> pairs{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
  for (unsigned code = 0; code < 4096; ++code) {
    auto                                      bits = code;
    std::vector<std::tuple<int, int, double>> edges;
    for (auto [u, v] : pairs) {
      int digit = bits % 4;
      bits /= 4;
      if (digit)
        edges.emplace_back(u, v, digit == 3 ? -1 : digit);
    }
    const auto adj = graph(4, edges);
    double     opt = 0;
    for (unsigned mask = 0; mask < 16; ++mask) {
      double val = 0;
      for (auto [u, v, w] : edges)
        if (((mask >> u) & 1) != ((mask >> v) & 1))
          val += w;
      opt = std::max(opt, val);
    }
    const auto exact = grasp_exact_max_cut(adj);
    EXPECT_EQ(exact.partitions, 8);
    EXPECT_EQ(exact.result.value, opt);
    const auto r   = grasp_max_cut(4, adj, 5, .3, code + 1);
    double     val = 0;
    for (auto [u, v, w] : edges)
      if (r.side[u] != r.side[v])
        val += w;
    EXPECT_EQ(val, r.value);
    EXPECT_LE(val, opt);
    for (int v = 0; v < 4; ++v) {
      auto side = r.side;
      side[v] ^= 1;
      double changed = 0;
      for (auto [u, w, weight] : edges)
        if (side[u] != side[w])
          changed += weight;
      EXPECT_LE(changed - val, 1e-12);
    }
    if (code % 64 == 0)
      certificate(adj, 2, .3, code);
  }
}
TEST(GraspTrace, PrefixesOriginalFixtureAndLocalTrap) {
  const auto a = grasp_demo_graph();
  auto       t = grasp_max_cut_trace(14, a, 40, .3, 1);
  EXPECT_EQ(grasp_exact_max_cut(a).result.value, 186);
  EXPECT_EQ(t.run.result.value, 186);
  for (int budget : {1, 2, 7, 20, 40})
    EXPECT_EQ(grasp_max_cut(14, a, budget, .3, 1).value, t.restarts[budget - 1].best_value);
  certificate(graph(4, {{0, 1, 1}, {1, 2, 1}, {2, 3, 1}, {3, 0, 1}}), 3, 0, 0);
  const auto cycle = graph(4, {{0, 1, 1}, {1, 2, 1}, {2, 3, 1}, {3, 0, 1}});
  auto       d     = grasp_cut_check(cycle, {0, 0, 1, 1});
  EXPECT_EQ(d.value, 2);
  EXPECT_TRUE(d.tolerance_local);
  EXPECT_EQ(grasp_exact_max_cut(cycle).result.value, 4);
}
TEST(GraspTrace, DomainsCapsOverflowOwnershipAndZeroCases) {
  EXPECT_THROW(grasp_max_cut(-1, {}), std::invalid_argument);
  EXPECT_THROW(grasp_max_cut(2, {{}}), std::invalid_argument);
  EXPECT_THROW(grasp_max_cut(1, {{}}, 0), std::invalid_argument);
  EXPECT_THROW(grasp_max_cut(1, {{{0, 1}}}), std::invalid_argument);
  EXPECT_THROW(grasp_max_cut(1, {{{-1, 1}}}), std::invalid_argument);
  EXPECT_THROW(grasp_max_cut(1, {{{1, 1}}}), std::invalid_argument);
  EXPECT_THROW(grasp_max_cut(2, {{{1, 1}}, {}}), std::invalid_argument);
  EXPECT_THROW(grasp_max_cut(2, {{{1, 1}}, {{0, 2}}}), std::invalid_argument);
  EXPECT_THROW(grasp_max_cut(2, {{{1, 1}, {1, 1}}, {{0, 1}}}), std::invalid_argument);
  for (double x :
       {-1.,
        2.,
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(grasp_max_cut(1, {{}}, 1, x), std::invalid_argument);
  }
  const auto   tiny = graph(3, {{0, 1, 1}, {0, 2, 2}, {1, 2, 3}});
  GraspOptions o;
  o.max_local_passes = 1;
  EXPECT_THROW(grasp_max_cut_run(3, tiny, 1, 1, 3, o), std::length_error);
  o.max_local_passes = 2;
  EXPECT_NO_THROW(grasp_max_cut_run(3, tiny, 1, 1, 3, o));
  o.max_vertices = 2;
  EXPECT_THROW(grasp_max_cut_run(3, tiny, 1, 1, 3, o), std::length_error);
  EXPECT_THROW(grasp_max_cut_run(3, {{{0, 1}}, {}, {}}, 1, 1, 3, o), std::invalid_argument);
  auto t                                  = grasp_max_cut_trace(3, tiny, 1, 1, 3);
  auto copy                               = t;
  copy.construction[0].candidates[0].gain = 99;
  copy.restarts[0].final_side[0]          = 99;
  EXPECT_NE(t.construction[0].candidates[0].gain, 99);
  EXPECT_NE(t.restarts[0].final_side[0], 99);
  EXPECT_EQ(grasp_max_cut(0, {}, 2).value, 0);
  certificate({}, 2, 0, 0);
  certificate({{}}, 2, 1, 1);
  EXPECT_THROW(grasp_exact_max_cut(tiny, 3), std::length_error);
  EXPECT_NO_THROW(grasp_exact_max_cut(tiny, 4));
  EXPECT_THROW(grasp_cut_check(tiny, {0, 1}), std::invalid_argument);
  EXPECT_THROW(grasp_cut_check(tiny, {0, 1, 2}), std::invalid_argument);
  auto huge = graph(3, {{0, 1, 1e308}, {0, 2, 1e308}, {1, 2, 1e308}});
  EXPECT_THROW(grasp_max_cut(3, huge, 1), std::overflow_error);
  detail::GraspRng rng(1);
  EXPECT_THROW(rng.below(0), std::invalid_argument);
  EXPECT_EQ(rng.s, 1);
}

TEST(GraspTrace, AbsoluteSlackCanStopBeforeAnExactImprovement) {
  const auto a = graph(3, {{0, 1, 1e-13}, {0, 2, 2e-13}, {1, 2, 3e-13}});
  const auto t = grasp_max_cut_trace(3, a, 1, 1, 3);
  const auto d = grasp_cut_check(a, t.run.result.side);
  EXPECT_EQ(t.run.work.accepted_flips, 0);
  EXPECT_GT(d.flip_gains[0], 0);
  EXPECT_LE(d.flip_gains[0], 1e-12);
  EXPECT_TRUE(d.tolerance_local);
  EXPECT_GT(grasp_exact_max_cut(a).result.value, t.run.result.value);
}

TEST(GraspTrace, ParallelZeroWeightsAndDisabledCaps) {
  const auto a = graph(3, {{0, 1, 2}, {0, 1, -1}, {1, 2, 0}, {0, 2, 3}});
  certificate(a, 2, .3, 7);
  EXPECT_THROW(
      grasp_max_cut(2, graph(2, {{0, 1, std::numeric_limits<double>::infinity()}}), 1),
      std::invalid_argument);
  EXPECT_THROW(
      grasp_max_cut(2, graph(2, {{0, 1, std::numeric_limits<double>::quiet_NaN()}}), 1),
      std::invalid_argument);
  const auto   triangle = graph(3, {{0, 1, 1}, {0, 2, 2}, {1, 2, 3}});
  GraspOptions o;
  o.max_adjacency_entries = 5;
  EXPECT_THROW(grasp_max_cut_run(3, triangle, 1, 1, 3, o), std::length_error);
  o.max_adjacency_entries = 0;
  o.max_vertices          = 0;
  o.max_trace_entries     = 0;
  o.max_local_passes      = 0;
  EXPECT_NO_THROW(grasp_max_cut_trace(3, triangle, 1, 1, 3, o));
  EXPECT_THROW(grasp_exact_max_cut(GraspAdjacency(25)), std::length_error);
}
