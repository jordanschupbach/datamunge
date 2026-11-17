#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/subset_sum.hpp>
#include <datamunge/algorithms/summed_area_table.hpp>
#include <datamunge/algorithms/sundaram_sieve.hpp>
#include <datamunge/algorithms/sutherland_hodgman.hpp>
#include <datamunge/algorithms/symbolic_cholesky.hpp>
#include <datamunge/algorithms/tarjan_lca.hpp>
#include <datamunge/algorithms/tarjan_scc.hpp>
#include <datamunge/algorithms/td_prediction.hpp>
#include <datamunge/algorithms/tea.hpp>
#include <datamunge/algorithms/ternary_search.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <set>
namespace a = datamunge::algorithms;
TEST(Batch384, SubsetExactAndBrute) {
  std::mt19937_64 rng(384);
  for (int trial = 0; trial < 100; ++trial) {
    std::vector<long long> v(rng() % 15);
    for (auto& x : v)
      x = int(rng() % 21) - 10;
    long long target = int(rng() % 101) - 50;
    bool      found  = false;
    for (std::size_t mask = 0; mask < (std::size_t{1} << v.size()); ++mask) {
      long long sum = 0;
      for (std::size_t j = 0; j < v.size(); ++j)
        if (mask & (std::size_t{1} << j))
          sum += v[j];
      found |= sum == target;
    }
    auto r = a::subset_sum_trace(v, target);
    EXPECT_EQ(r.result.found, found);
    if (found) {
      long long sum = 0;
      for (auto j : r.result.indices)
        sum += v[j];
      EXPECT_EQ(sum, target);
    }
    EXPECT_EQ(r.right.size(), r.right_subsets);
    EXPECT_EQ(r.probes.size(), r.left_probes);
  }
}
TEST(Batch384, SubsetFullWidthAndLimits) {
  long long              m = std::numeric_limits<long long>::max();
  std::vector<long long> v{m, m, -m, std::numeric_limits<long long>::min()};
  auto                   r = a::subset_sum_trace(v, m);
  ASSERT_TRUE(r.result.found);
  __int128 sum = 0;
  for (auto i : r.result.indices)
    sum += v[i];
  EXPECT_TRUE(sum == m);
  EXPECT_TRUE(a::subset_sum({}, 0).found);
  EXPECT_FALSE(a::subset_sum({}, 1).found);
  EXPECT_THROW(a::subset_sum(std::vector<long long>(63), 0), std::length_error);
  a::SubsetSumOptions o;
  o.max_half_subsets = 2;
  EXPECT_THROW(a::subset_sum_run({1, 2, 3, 4}, 1, o), std::length_error);
}
TEST(Batch384, SatEveryRectangle) {
  auto               g = a::summed_area_example_grid();
  a::SummedAreaTable s(g, {.capture = true});
  EXPECT_EQ(s.steps().size(), 36);
  for (std::size_t r0 = 0; r0 < 6; ++r0)
    for (std::size_t c0 = 0; c0 < 6; ++c0)
      for (std::size_t r1 = r0; r1 < 6; ++r1)
        for (std::size_t c1 = c0; c1 < 6; ++c1) {
          double sum = 0;
          for (std::size_t r = r0; r <= r1; ++r)
            for (std::size_t c = c0; c <= c1; ++c)
              sum += g[r][c];
          auto q = s.query(r0, c0, r1, c1);
          EXPECT_EQ(q.value, sum);
          EXPECT_EQ(q.lookups, 4);
          EXPECT_EQ(s.rect_mean(r0, c0, r1, c1), sum / ((r1 - r0 + 1) * (c1 - c0 + 1)));
        }
}
TEST(Batch384, SatCompleteValidation) {
  EXPECT_THROW(
      a::SummedAreaTable((std::vector<std::vector<double>>{{1}, {}})), std::invalid_argument);
  EXPECT_THROW(
      a::SummedAreaTable(
          (std::vector<std::vector<double>>{{std::numeric_limits<double>::infinity()}})),
      std::invalid_argument);
  a::SummedAreaTable s({{1, 2}});
  EXPECT_THROW(s.rect_sum(0, 0, 0, 2), std::out_of_range);
  EXPECT_THROW(s.rect_sum(1, 0, 0, 0), std::out_of_range);
  a::SummedAreaTable e({});
  EXPECT_EQ(e.rows(), 0);
  EXPECT_THROW(e.rect_sum(0, 0, 0, 0), std::out_of_range);
  EXPECT_THROW(
      a::SummedAreaTable((std::vector<std::vector<double>>{{1, 2}}), {.max_cells = 2}),
      std::length_error);
}
TEST(Batch384, SundaramInclusiveAndStrikes) {
  for (std::uint64_t n = 0; n <= 1000; ++n) {
    auto                       r = a::sundaram_trace(n);
    std::vector<std::uint64_t> ref;
    for (std::uint64_t k = 2; k <= n; ++k) {
      bool prime = true;
      for (std::uint64_t d = 2; d <= k / d; ++d)
        if (k % d == 0) {
          prime = false;
          break;
        }
      if (prime)
        ref.push_back(k);
    }
    EXPECT_EQ(r.primes, ref);
    EXPECT_EQ(r.strikes, r.steps.size());
    std::set<std::uint64_t> hits;
    for (const auto& s : r.steps) {
      EXPECT_EQ(s.index, s.i + s.j + 2 * s.i * s.j);
      EXPECT_LE(s.i, s.j);
      EXPECT_EQ(s.newly_marked, hits.insert(s.index).second);
    }
    EXPECT_EQ(hits.size(), r.first_marks);
  }
}
TEST(Batch384, SundaramPolicies) {
  EXPECT_EQ(a::sundaram_primes(100).size(), 25);
  EXPECT_THROW(a::sundaram_run(std::numeric_limits<std::uint64_t>::max()), std::length_error);
  a::SundaramOptions o;
  o.max_strikes = 1;
  EXPECT_THROW(a::sundaram_run(100, o), std::length_error);
  o             = {};
  o.max_records = 1;
  EXPECT_THROW(a::sundaram_trace(100, o), std::length_error);
  EXPECT_EQ(a::sundaram_run(2).primes, (std::vector<std::uint64_t>{2}));
}
TEST(Batch384, ClipActualPhases) {
  std::vector<a::Point2> p{{-4, 2}, {14, 4}, {5, 16}};
  auto                   r = a::sutherland_hodgman_trace(p, 0, 0, 10, 10);
  ASSERT_FALSE(r.polygon.empty());
  EXPECT_EQ(r.history.back().after, r.polygon);
  EXPECT_EQ(r.phases, r.history.size());
  EXPECT_EQ(r.edge_visits, r.steps.size());
  std::size_t emitted = 0, cross = 0;
  for (const auto& s : r.steps) {
    emitted += s.emitted.size();
    cross += s.previous_inside != s.current_inside;
    EXPECT_EQ(
        s.emitted.size(), std::size_t(s.previous_inside != s.current_inside) + s.current_inside);
  }
  EXPECT_EQ(emitted, r.emissions);
  EXPECT_EQ(cross, r.intersections);
  for (auto [x, y] : r.polygon) {
    EXPECT_GE(x, 0);
    EXPECT_LE(x, 10);
    EXPECT_GE(y, 0);
    EXPECT_LE(y, 10);
  }
}
TEST(Batch384, ClipDegenerateAndExtreme) {
  EXPECT_TRUE(a::sutherland_hodgman_clip({}, 0, 0, 1, 1).empty());
  EXPECT_THROW(a::sutherland_hodgman_clip({}, 1, 0, 0, 1), std::invalid_argument);
  EXPECT_THROW(
      a::sutherland_hodgman_clip({{std::numeric_limits<double>::infinity(), 0}}, 0, 0, 1, 1),
      std::invalid_argument);
  auto z = a::sutherland_hodgman_run({{-1, -1}, {1, -1}, {1, 1}, {-1, 1}}, 0, -2, 0, 2);
  for (auto [x, y] : z.polygon)
    EXPECT_EQ(x, 0);
  double m = std::numeric_limits<double>::max();
  auto   e = a::sutherland_hodgman_run({{-m, -m}, {m, -m}, {m, m}, {-m, m}}, -1, -1, 1, 1);
  EXPECT_FALSE(e.polygon.empty());
  for (auto [x, y] : e.polygon) {
    EXPECT_GE(x, -1);
    EXPECT_LE(x, 1);
    EXPECT_GE(y, -1);
    EXPECT_LE(y, 1);
  }
}
TEST(Batch384, SymbolicActualFill) {
  auto r = a::symbolic_cholesky_trace({{1, 2}, {0}, {0}});
  EXPECT_EQ(r.fill, (std::vector<std::pair<int, int>>{{1, 2}}));
  EXPECT_EQ(r.lower, (std::vector<std::vector<int>>{{}, {0}, {0, 1}}));
  EXPECT_EQ(r.factor_entries, 3);
  EXPECT_EQ(r.pair_tests, 1);
  EXPECT_EQ(r.steps[0].fill, r.fill);
  auto grid = a::symbolic_cholesky_run(a::symbolic_cholesky_grid());
  EXPECT_EQ(grid.input_edges, 40);
  EXPECT_EQ(grid.factor_entries, grid.input_edges + grid.fill.size());
  EXPECT_EQ(grid.fill.size(), 64);
}
TEST(Batch384, SymbolicNormalizationAndPolicies) {
  EXPECT_EQ(a::symbolic_cholesky({{0, 1, 1}, {}}), (std::vector<std::vector<int>>{{}, {0}}));
  EXPECT_THROW(a::symbolic_cholesky({{2}, {}}), std::invalid_argument);
  EXPECT_TRUE(a::symbolic_cholesky({}).empty());
  a::SymbolicCholeskyOptions o;
  o.max_pair_tests = 1;
  EXPECT_THROW(a::symbolic_cholesky_run(a::symbolic_cholesky_grid(), o), std::length_error);
  o           = {};
  o.max_edges = 1;
  EXPECT_THROW(a::symbolic_cholesky_run({{1, 2}, {0}, {0}}, o), std::length_error);
}
TEST(Batch384, LcaOriginalAndActualEvents) {
  std::vector<std::pair<std::size_t, std::size_t>> e{
      {0, 1}, {0, 2}, {1, 3}, {1, 4}, {3, 6}, {2, 5}, {5, 7}},
      q{{6, 7}, {3, 4}, {6, 4}, {5, 7}, {6, 1}, {4, 4}, {7, 2}};
  auto r = a::tarjan_lca_trace(8, 0, e, q);
  EXPECT_EQ(r.lca, (std::vector<std::size_t>{0, 1, 1, 5, 1, 4, 2}));
  EXPECT_EQ(r.unions, 7);
  EXPECT_EQ(r.entered.size(), 8);
  EXPECT_EQ(r.finished.size(), 8);
  EXPECT_EQ(r.edge_visits, 14);
  EXPECT_EQ(r.preflight_edges, 14);
  EXPECT_EQ(r.query_visits, 14);
  EXPECT_EQ(r.answer_writes, 8);
  for (const auto& s : r.steps)
    if (s.action == "answer")
      EXPECT_EQ(s.ancestor, r.lca[s.query]);
}
TEST(Batch384, LcaRejectsNonTreesAndHandlesDepth) {
  EXPECT_THROW(a::tarjan_lca_run(4, 0, {{0, 1}, {1, 2}, {2, 0}}, {}), std::invalid_argument);
  EXPECT_THROW(a::tarjan_lca_run(3, 0, {{0, 1}, {0, 1}}, {}), std::invalid_argument);
  EXPECT_THROW(a::tarjan_lca_run(2, 0, {{0, 0}}, {}), std::invalid_argument);
  EXPECT_THROW(a::tarjan_lca_run(1, 0, {}, {{0, 1}}), std::invalid_argument);
  std::vector<std::pair<std::size_t, std::size_t>> e;
  for (std::size_t i = 1; i < 10000; ++i)
    e.emplace_back(i - 1, i);
  EXPECT_EQ(
      a::tarjan_offline_lca(10000, 0, e, {{9999, 1234}, {9999, 9999}}),
      (std::vector<std::size_t>{1234, 9999}));
}
TEST(Batch384, SccAgainstReachability) {
  std::mt19937_64 rng(384);
  for (int trial = 0; trial < 100; ++trial) {
    int                            n = rng() % 12;
    std::vector<std::vector<int>>  adj(n);
    std::vector<std::vector<bool>> reach(n, std::vector<bool>(n));
    std::size_t                    edges = 0;
    for (int i = 0; i < n; ++i) {
      reach[i][i] = true;
      for (int j = 0; j < n; ++j)
        if (rng() % 4 == 0) {
          adj[i].push_back(j);
          reach[i][j] = true;
          ++edges;
        }
    }
    for (int k = 0; k < n; ++k)
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          reach[i][j] = reach[i][j] || (reach[i][k] && reach[k][j]);
    auto r = a::tarjan_scc_trace(n, adj);
    EXPECT_EQ(r.discoveries, std::size_t(n));
    EXPECT_EQ(r.edge_visits, edges);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        EXPECT_EQ(r.components[i] == r.components[j], reach[i][j] && reach[j][i]);
  }
}
TEST(Batch384, SccCompleteValidation) {
  EXPECT_THROW(a::tarjan_scc(-1, {}), std::invalid_argument);
  EXPECT_THROW(a::tarjan_scc(1, {}), std::invalid_argument);
  EXPECT_THROW(a::tarjan_scc(1, {{1}}), std::invalid_argument);
  EXPECT_TRUE(a::tarjan_scc(0, {}).empty());
  EXPECT_EQ(a::tarjan_scc(2, {{1}, {0}}), (std::vector<int>{0, 0}));
  a::TarjanSCCOptions o;
  o.max_records = 1;
  EXPECT_THROW(a::tarjan_scc_trace(2, {{1}, {0}}, o), std::length_error);
}
TEST(Batch384, TdTerminalBootstrapAndCaps) {
  a::TabularEnvironment e{
      1,
      1,
      [] { return std::size_t{0}; },
      [](std::size_t, std::size_t) { return a::StepResult{0, 2, true}; }};
  a::RLParameters p;
  p.episodes      = 3;
  p.learning_rate = .5;
  auto r          = a::td_prediction_trace(e, {0}, p);
  EXPECT_EQ(r.result.value[0], 1.75);
  EXPECT_EQ(r.transitions, 3);
  EXPECT_EQ(r.terminated, 3);
  EXPECT_EQ(r.truncated, 0);
  EXPECT_EQ(r.episodes.size(), 4);
  for (const auto& s : r.steps) {
    EXPECT_EQ(s.successor, 0);
    EXPECT_EQ(s.target, 2);
    EXPECT_EQ(s.after, s.before + .5 * s.error);
  }
  p.max_steps = 0;
  auto z      = a::td_prediction_run(e, {0}, p);
  EXPECT_EQ(z.resets, 3);
  EXPECT_EQ(z.transitions, 0);
  EXPECT_EQ(z.truncated, 3);
}
TEST(Batch384, TdOriginalSeedAndValidation) {
  a::RLParameters p;
  p.discount = 1;
  p.episodes = 300;
  auto r = a::td_prediction_trace(a::td_random_walk_environment(), std::vector<std::size_t>(7), p);
  EXPECT_EQ(r.terminated, 300);
  EXPECT_EQ(r.truncated, 0);
  EXPECT_LT(a::td_random_walk_rmse(r.result.value), .2);
  auto s = a::td_prediction(a::td_random_walk_environment(), std::vector<std::size_t>(7), p);
  EXPECT_EQ(r.result.value, s.value);
  EXPECT_EQ(a::td_random_walk_rmse(a::td_random_walk_truth()), 0);
  a::TabularEnvironment e{
      1,
      1,
      [] { return std::size_t{1}; },
      [](std::size_t, std::size_t) { return a::StepResult{0, 0, true}; }};
  EXPECT_THROW(a::td_prediction(e, {0}, p), std::out_of_range);
  p.episodes = 0;
  EXPECT_NO_THROW(a::td_prediction(e, {0}, p));
  EXPECT_THROW(a::td_prediction(e, {1}, p), std::invalid_argument);
  p.learning_rate = 0;
  EXPECT_THROW(a::td_prediction(e, {0}, p), std::invalid_argument);
}
TEST(Batch384, TeaKnownVectorAndActualInverse) {
  auto r = a::tea_encrypt_trace({0, 0}, {0, 0, 0, 0});
  EXPECT_EQ(r.value, (std::array<std::uint32_t, 2>{0x41ea3a0au, 0x94baa940u}));
  EXPECT_EQ(r.steps.size(), 32);
  EXPECT_EQ(r.half_updates, 64);
  auto d = a::tea_decrypt_trace(r.value, {0, 0, 0, 0});
  EXPECT_EQ(d.value, (std::array<std::uint32_t, 2>{0, 0}));
  for (std::size_t i = 0; i < 32; ++i) {
    EXPECT_EQ(d.steps[i].before, r.steps[31 - i].after);
    EXPECT_EQ(d.steps[i].after, r.steps[31 - i].before);
    EXPECT_EQ(d.steps[i].sum, r.steps[31 - i].sum);
  }
}
TEST(Batch384, TeaFullWidthRoundTrips) {
  std::mt19937_64 rng(384);
  for (int i = 0; i < 1000; ++i) {
    std::array<std::uint32_t, 2> v{std::uint32_t(rng()), std::uint32_t(rng())};
    std::array<std::uint32_t, 4> k{
        std::uint32_t(rng()), std::uint32_t(rng()), std::uint32_t(rng()), std::uint32_t(rng())};
    EXPECT_EQ(a::tea_decrypt(a::tea_encrypt(v, k), k), v);
  }
}
TEST(Batch384, TernaryActualBracketsAndCalls) {
  std::vector<double> args;
  auto                r = a::ternary_search_max_trace(
      [&](double x) {
        args.push_back(x);
        return -(x - 1) * (x - 1);
      },
      -2,
      4);
  EXPECT_TRUE(r.converged);
  EXPECT_NEAR(r.result.x, 1, 1e-6);
  EXPECT_EQ(args.size(), r.evaluations);
  EXPECT_EQ(r.evaluations, 2 * r.steps.size() + 1);
  for (std::size_t i = 0; i < r.steps.size(); ++i) {
    const auto& s = r.steps[i];
    EXPECT_EQ(args[2 * i], s.first);
    EXPECT_EQ(args[2 * i + 1], s.second);
    EXPECT_EQ(s.kept_left, s.first_value >= s.second_value);
    EXPECT_LT(s.hi_after - s.lo_after, s.hi_before - s.lo_before);
  }
  EXPECT_EQ(args.back(), r.result.x);
}
TEST(Batch384, TernaryControlsAndExtremes) {
  a::TernaryOptions o;
  o.max_iterations = 0;
  auto z           = a::ternary_search_min_run([](double x) { return x * x; }, -2, 4, o);
  EXPECT_EQ(z.result.iterations, 0);
  EXPECT_EQ(z.evaluations, 1);
  EXPECT_FALSE(z.converged);
  o           = {};
  o.tolerance = 0;
  auto s      = a::ternary_search_max_run([](double x) { return x; }, 1, std::nextafter(1., 2.), o);
  EXPECT_TRUE(s.precision_limited);
  EXPECT_FALSE(s.converged);
  double m = std::numeric_limits<double>::max();
  auto   x = a::ternary_search_max_run([m](double v) { return v / m; }, -m, m, o);
  EXPECT_TRUE(std::isfinite(x.result.x));
  EXPECT_TRUE(x.precision_limited);
  EXPECT_THROW(a::ternary_search_max_run({}, 0, 1), std::invalid_argument);
  EXPECT_THROW(
      a::ternary_search_max_run(
          [](double) { return std::numeric_limits<double>::quiet_NaN(); }, 0, 1),
      std::domain_error);
  EXPECT_THROW(a::ternary_search_min_run([](double x) { return x; }, 1, 0), std::invalid_argument);
}
