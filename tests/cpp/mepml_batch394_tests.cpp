#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/thomas.hpp>
#include <datamunge/algorithms/timsort.hpp>
#include <datamunge/algorithms/todd_coxeter.hpp>
#include <datamunge/algorithms/token_bucket.hpp>
#include <datamunge/algorithms/tonelli_shanks.hpp>
#include <datamunge/algorithms/toom3.hpp>
#include <datamunge/algorithms/topological_sort.hpp>
#include <datamunge/algorithms/tournament_selection.hpp>
#include <datamunge/algorithms/transitive_closure.hpp>
#include <datamunge/algorithms/trapezoidal_ode.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <set>
namespace a = datamunge::algorithms;
TEST(Batch394, ThomasActualSolutionAndResidual) {
  auto r = a::thomas_trace({0, -1, -1}, {2, 2, 2}, {-1, -1, 0}, {1, 0, 1});
  for (double x : r.x)
    EXPECT_NEAR(x, 1, 1e-14);
  EXPECT_EQ(r.eliminations, 2);
  EXPECT_EQ(r.substitutions, 3);
  EXPECT_EQ(r.divisions, 5);
  EXPECT_EQ(r.steps.size(), 5);
  EXPECT_LT(r.residual_norm, 1e-14);
  auto g = a::thomas_poisson_example();
  auto z = a::thomas_run(g.sub, g.diagonal, g.super, g.rhs);
  EXPECT_LT(z.residual_norm, 1e-10);
  double err = 0;
  for (std::size_t i = 0; i < 49; ++i)
    err = std::max(err, std::abs(z.x[i] - g.exact[i]));
  EXPECT_GT(err, 1e-4);
  EXPECT_LT(err, 4e-4);
  EXPECT_EQ(g.sub.front(), 0);
  EXPECT_EQ(g.super.back(), 0);
}
TEST(Batch394, ThomasCompleteValidationAndPivots) {
  EXPECT_TRUE(a::thomas_solve({}, {}, {}, {}).empty());
  EXPECT_THROW(a::thomas_solve({}, {1}, {0}, {1}), std::invalid_argument);
  EXPECT_THROW(a::thomas_solve({0, 1}, {0, 1}, {1, 0}, {1, 2}), std::domain_error);
  EXPECT_THROW(
      a::thomas_solve({std::numeric_limits<double>::quiet_NaN()}, {1}, {0}, {1}),
      std::invalid_argument);
  a::ThomasOptions o;
  o.max_records = 1;
  EXPECT_THROW(a::thomas_trace({0, -1}, {2, 2}, {-1, 0}, {1, 1}, o), std::length_error);
}
struct StableTimsortKey {
  int  key, id;
  bool operator<(const StableTimsortKey& other) const { return key < other.key; }
};
TEST(Batch394, TimsortStableAndProfiles) {
  std::mt19937_64 rng(394);
  for (int trial = 0; trial < 100; ++trial) {
    std::vector<StableTimsortKey> keys;
    for (int i = 0, n = rng() % 400; i < n; ++i)
      keys.push_back({int(rng() % 12), i});
    auto expected = keys;
    std::stable_sort(expected.begin(), expected.end());
    auto r = a::timsort_run(keys);
    ASSERT_EQ(r.values.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(r.values[i].key, expected[i].key);
      EXPECT_EQ(r.values[i].id, expected[i].id);
    }
  }
  auto p = a::timsort_input_profile(std::vector<int>{3, 2, 1, 4, 5});
  EXPECT_EQ(p.lengths, (std::vector<std::size_t>{3, 2}));
  EXPECT_TRUE(p.descending[0]);
  EXPECT_FALSE(p.descending[1]);
}
TEST(Batch394, TimsortAdaptiveAndLimits) {
  std::vector<int> sorted(1000);
  std::iota(sorted.begin(), sorted.end(), 0);
  auto r = a::timsort_trace(sorted);
  EXPECT_EQ(r.natural_runs, 1);
  EXPECT_EQ(r.comparisons, 999);
  EXPECT_EQ(r.merges, 0);
  EXPECT_EQ(r.values, sorted);
  for (const auto& v : a::timsort_example_arrays()) {
    auto s = a::timsort_run(v);
    EXPECT_EQ(s.values, sorted);
  }
  a::TimsortOptions o;
  o.max_comparisons = 1;
  EXPECT_THROW(a::timsort_run(std::vector<int>{3, 2, 1}, o), std::length_error);
  EXPECT_THROW(
      a::timsort_run(std::vector<double>{std::numeric_limits<double>::infinity()}),
      std::invalid_argument);
}
TEST(Batch394, ToddFiniteGroupsAndActions) {
  for (int n = 1; n <= 20; ++n) {
    auto r = a::todd_coxeter_trace(1, {std::vector<int>(n, 1)}, {});
    ASSERT_TRUE(r.closed);
    ASSERT_EQ(r.index, n);
    ASSERT_EQ(r.compact_table.size(), std::size_t(n));
    for (int row = 0; row < n; ++row) {
      int next = row;
      for (int i = 0; i < n; ++i)
        next = r.compact_table[next][0];
      EXPECT_EQ(next, row);
      EXPECT_EQ(r.compact_table[r.compact_table[row][0]][1], row);
    }
  }
  EXPECT_EQ(a::todd_coxeter_index(1, {std::vector<int>(12, 1)}, {std::vector<int>(4, 1)}), 4);
  EXPECT_EQ(a::todd_coxeter_index(2, {{1, 1}, {2, 2, 2}, {1, 2, 1, 2}}, {}), 6);
  EXPECT_EQ(a::todd_coxeter_index(2, {{1, 1}, {2, 2, 2}, {1, 2, 1, 2, 1, 2, 1, 2, 1, 2}}, {}), 60);
  EXPECT_EQ(a::todd_coxeter_index(0, {}, {}), 1);
}
TEST(Batch394, ToddExplicitStopsAndValidation) {
  a::ToddOptions o;
  o.max_cosets = 1;
  auto r       = a::todd_coxeter_run(1, {{1, 1, 1}}, {}, o);
  EXPECT_FALSE(r.closed);
  EXPECT_EQ(r.index, -1);
  EXPECT_EQ(r.stop, "coset limit");
  o                = {};
  o.max_operations = 1;
  r                = a::todd_coxeter_run(1, {{1, 1, 1}}, {}, o);
  EXPECT_FALSE(r.closed);
  EXPECT_EQ(r.stop, "operation limit");
  EXPECT_THROW(a::todd_coxeter_index(1, {{0}}, {}), std::invalid_argument);
  EXPECT_THROW(
      a::todd_coxeter_index(1, {{std::numeric_limits<int>::min()}}, {}), std::invalid_argument);
  EXPECT_THROW(a::todd_coxeter_index(1, {}, {}, 0), std::invalid_argument);
}
TEST(Batch394, BucketActualBurstAndRefill) {
  a::TokenBucket b(5, 1);
  for (int i = 0; i < 8; ++i) {
    auto s = b.consume(0);
    EXPECT_EQ(s.consumed, i < 5);
    EXPECT_EQ(s.after, std::max(0, 4 - i));
    EXPECT_EQ(s.refilled, 0);
  }
  for (int t = 1; t <= 6; ++t) {
    auto s = b.consume(t);
    EXPECT_TRUE(s.consumed);
    EXPECT_EQ(s.before, 0);
    EXPECT_EQ(s.refilled, 1);
    EXPECT_EQ(s.after, 0);
  }
  EXPECT_EQ(b.available(100), 5);
  EXPECT_EQ(b.last_time(), 100);
}
TEST(Batch394, BucketValidationZeroAndSaturation) {
  EXPECT_THROW(a::TokenBucket(-1, 1), std::invalid_argument);
  a::TokenBucket b(3, 0);
  EXPECT_TRUE(b.allow(0, 0));
  EXPECT_FALSE(b.allow(0, 4));
  EXPECT_EQ(b.stored_tokens(), 3);
  EXPECT_THROW(b.allow(-1), std::invalid_argument);
  EXPECT_THROW(b.allow(1, -1), std::invalid_argument);
  EXPECT_EQ(b.last_time(), 0);
  EXPECT_EQ(b.available(100), 3);
  a::TokenBucket huge(std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
  EXPECT_TRUE(huge.allow(0, std::numeric_limits<double>::max()));
  EXPECT_EQ(huge.available(std::numeric_limits<double>::max()), std::numeric_limits<double>::max());
}
TEST(Batch394, TonelliExhaustivePrimeRoots) {
  for (std::uint64_t p : {2, 3, 5, 7, 13, 17, 41, 101, 257})
    for (std::uint64_t n = 0; n < p; ++n) {
      std::uint64_t expected = 0;
      bool          exists   = false;
      for (std::uint64_t x = 0; x < p; ++x)
        if (x * x % p == n) {
          expected = x;
          exists   = true;
          break;
        }
      auto r = a::tonelli_shanks_trace(n, p);
      EXPECT_EQ(r.result.exists, exists);
      if (exists)
        EXPECT_EQ(r.result.root, expected);
      for (const auto& s : r.steps) {
        EXPECT_LT(s.i, s.M_before);
        EXPECT_EQ(s.squares.back(), 1);
        EXPECT_EQ(
            static_cast<std::uint64_t>((__uint128_t(s.root_after) * s.root_after) % p),
            static_cast<std::uint64_t>((__uint128_t(n) * s.t_after) % p));
      }
    }
}
TEST(Batch394, TonelliFullWidthAndInvalidModuli) {
  constexpr std::uint64_t p = 18446744073709551557ULL;
  auto                    r = a::tonelli_shanks(4, p);
  EXPECT_TRUE(r.exists);
  EXPECT_EQ(r.root, 2);
  for (std::uint64_t m : {0, 1, 4, 9, 15, 21})
    EXPECT_THROW(a::tonelli_shanks(0, m), std::invalid_argument);
  a::TonelliOptions o;
  o.max_products = 1;
  EXPECT_THROW(a::tonelli_shanks_run(10, 13, o), std::length_error);
}
TEST(Batch394, ToomActualFiveProductsAndExactInterpolation) {
  a::ToomOptions o;
  o.toom_leaf      = 3;
  o.karatsuba_leaf = 3;
  auto r           = a::toom3_trace("123456", "789012", o);
  EXPECT_EQ(r.value, "97408265472");
  EXPECT_EQ(r.toom_calls, 1);
  EXPECT_EQ(r.schoolbook_calls, 5);
  EXPECT_EQ(r.scalar_products, 20);
  ASSERT_EQ(r.steps.back().kind, "toom3");
  EXPECT_EQ(r.steps.back().products.size(), 5);
  EXPECT_EQ(r.steps.back().interpolated.size(), 5);
  EXPECT_EQ(r.steps.back().products[0], r.steps.back().interpolated[0]);
  EXPECT_EQ(r.steps.back().products[4], r.steps.back().interpolated[4]);
}
TEST(Batch394, ToomCarryCrossoverAndMalformedInput) {
  for (std::size_t n : {1, 2, 8, 32, 48, 49, 64, 128, 256, 512, 1024}) {
    auto        r        = a::toom3_run(std::string(n, '9'), std::string(n, '9'));
    std::string expected = std::string(n - 1, '9') + '8' + std::string(n - 1, '0') + '1';
    EXPECT_EQ(r.value, expected);
  }
  EXPECT_EQ(a::toom3_mul("", "123"), "0");
  EXPECT_EQ(a::toom3_mul("00012", "003"), "36");
  EXPECT_THROW(a::toom3_mul("-1", "2"), std::invalid_argument);
  EXPECT_THROW(a::toom3_mul("12x", "2"), std::invalid_argument);
  a::ToomOptions o;
  o.max_operations = 1;
  EXPECT_THROW(a::toom3_run("123", "456", o), std::length_error);
}
TEST(Batch394, TopologicalActualFifoAndCycleRemainder) {
  auto r = a::topological_sort_trace(6, {{0, 2}, {0, 3}, {1, 2}, {1, 4}, {2, 4}, {2, 5}, {3, 5}});
  EXPECT_EQ(r.result.order, (std::vector<std::size_t>{0, 1, 3, 2, 4, 5}));
  EXPECT_TRUE(r.result.is_dag);
  EXPECT_EQ(r.edge_visits, 7);
  EXPECT_EQ(r.enqueues, 6);
  for (std::size_t i = 0; i < 6; ++i)
    EXPECT_EQ(r.positions[r.emitted[i]], i);
  auto c = a::topological_sort_run(4, {{0, 1}, {1, 0}, {1, 2}});
  EXPECT_FALSE(c.result.is_dag);
  EXPECT_TRUE(c.result.order.empty());
  EXPECT_EQ(c.emitted, (std::vector<std::size_t>{3}));
  EXPECT_EQ(c.positions[2], 4);
  EXPECT_EQ(c.remaining_indegree[2], 1);
}
TEST(Batch394, TopologicalWitnessAndCompleteValidation) {
  EXPECT_TRUE(a::is_topological_order(3, {{0, 1}, {1, 2}}, {0, 1, 2}));
  EXPECT_FALSE(a::is_topological_order(3, {{0, 1}}, {0, 0, 2}));
  EXPECT_FALSE(a::is_topological_order(1, {{0, 1}}, {0}));
  EXPECT_TRUE(a::topological_sort(0, {}).is_dag);
  EXPECT_THROW(a::topological_sort(1, {{0, 1}}), std::invalid_argument);
  EXPECT_EQ(
      a::topological_sort(3, {{0, 1}, {0, 1}, {1, 2}}).order, (std::vector<std::size_t>{0, 1, 2}));
}
TEST(Batch394, TournamentActualContestantsAndMonotoneTransform) {
  std::vector<double> fitness{-1, 3, 3};
  auto                r = a::tournament_trace(fitness, 100, 4, 1);
  EXPECT_EQ(r.random_words, 400);
  EXPECT_EQ(r.comparisons, 300);
  EXPECT_EQ(r.steps.size(), 400);
  for (const auto& s : r.steps) {
    EXPECT_EQ(s.candidate, s.raw_word % 3);
    auto expected = s.round == 0                               ? s.candidate
                    : fitness[s.candidate] > fitness[s.before] ? s.candidate
                                                               : s.before;
    EXPECT_EQ(s.after, expected);
    if (s.round == 3)
      EXPECT_EQ(r.indices[s.pick], s.after);
  }
  EXPECT_EQ(
      a::tournament_sample({-5, 0, 8}, 100, 4, 1), a::tournament_sample({-49, 1, 81}, 100, 4, 1));
  EXPECT_EQ(std::accumulate(r.counts.begin(), r.counts.end(), std::size_t{}), 100);
}
TEST(Batch394, TournamentCompletePolicies) {
  EXPECT_TRUE(a::tournament_sample({}, 0, 1).empty());
  EXPECT_THROW(a::tournament_sample({}, 1, 1), std::invalid_argument);
  EXPECT_THROW(a::tournament_sample({1}, -1, 1), std::invalid_argument);
  EXPECT_THROW(a::tournament_sample({1}, 1, 0), std::invalid_argument);
  EXPECT_THROW(
      a::tournament_sample({std::numeric_limits<double>::quiet_NaN()}, 0, 1),
      std::invalid_argument);
  a::TournamentOptions o;
  o.max_contestants = 3;
  EXPECT_THROW(a::tournament_run({1, 2}, 2, 2, 1, o), std::length_error);
}
TEST(Batch394, ClosureActualPhasesAndPositiveCycles) {
  auto r = a::transitive_closure_trace(3, {{0, 1}, {1, 2}}, false);
  EXPECT_TRUE(r.reachable[0][2]);
  EXPECT_FALSE(r.reachable[0][0]);
  EXPECT_EQ(r.row_tests, 9);
  EXPECT_EQ(r.candidate_updates, 9);
  EXPECT_EQ(r.new_paths, 1);
  EXPECT_EQ(r.phases.back().after, r.reachable);
  auto c = a::transitive_closure_run(2, {{0, 1}, {1, 0}}, false);
  EXPECT_TRUE(c.reachable[0][0]);
  EXPECT_TRUE(c.reachable[1][1]);
}
TEST(Batch394, ClosureCompleteGeometryAndLimits) {
  EXPECT_TRUE(a::transitive_closure(0, {}).empty());
  EXPECT_THROW(a::transitive_closure(1, {{0, 1}}), std::invalid_argument);
  a::ClosureOptions o;
  o.max_cells = 3;
  EXPECT_THROW(a::transitive_closure_run(2, {}, true, o), std::length_error);
  o             = {};
  o.max_updates = 1;
  EXPECT_THROW(a::transitive_closure_run(3, {{0, 1}, {1, 2}}, true, o), std::length_error);
}
TEST(Batch394, TrapezoidSolvedLinearAndStiffEquations) {
  for (int n : {4, 8, 16, 32, 64}) {
    auto r = a::trapezoidal_ode_trace([](double, double y) { return -y; }, 1, 0, 1, n);
    ASSERT_TRUE(r.converged);
    EXPECT_EQ(r.completed_steps, std::size_t(n));
    EXPECT_EQ(r.trajectory.back().first, 1);
    double h = 1. / n, expected = std::pow((1 - h / 2) / (1 + h / 2), n);
    EXPECT_NEAR(r.trajectory.back().second, expected, 1e-11);
    for (const auto& s : r.steps)
      EXPECT_LE(
          std::abs(s.residual),
          1e-14 + 1e-12 * std::max(std::abs(s.previous), std::abs(s.candidate)));
  }
  auto stiff = a::trapezoidal_ode_run([](double, double y) { return -50 * y; }, 1, 0, 1, 10);
  ASSERT_TRUE(stiff.converged);
  EXPECT_NEAR(stiff.trajectory.back().second, std::pow(-3. / 7, 10), 1e-11);
  EXPECT_LT(stiff.trajectory[1].second, 0);
}
TEST(Batch394, TrapezoidHonestStopsAndValidation) {
  a::TrapezoidOptions o;
  o.max_newton_iterations = 0;
  auto r = a::trapezoidal_ode_run([](double, double y) { return -y; }, 1, 0, 1, 10, o);
  EXPECT_FALSE(r.converged);
  EXPECT_EQ(r.completed_steps, 0);
  EXPECT_EQ(r.trajectory.size(), 1);
  auto impossible = a::trapezoidal_ode_run([](double, double y) { return 2 * y; }, 1, 0, 1, 1);
  EXPECT_FALSE(impossible.converged);
  EXPECT_EQ(impossible.stop, "zero Newton derivative");
  EXPECT_THROW(
      a::trapezoidal_ode([](double, double y) { return 2 * y; }, 1, 0, 1, 1), std::runtime_error);
  EXPECT_THROW(
      a::trapezoidal_ode([](double, double) { return 0.; }, 1, 0, 1, 0), std::invalid_argument);
  int  calls = 0;
  auto z     = a::trapezoidal_ode_run(
      [&](double, double) {
        ++calls;
        return 0.;
      },
      2,
      1,
      1,
      3);
  EXPECT_TRUE(z.converged);
  EXPECT_EQ(calls, 0);
  EXPECT_EQ(z.trajectory.size(), 4);
  auto back = a::trapezoidal_ode_run([](double, double) { return 0.; }, 2, 1, 0, 3);
  EXPECT_TRUE(back.converged);
  EXPECT_EQ(back.trajectory.back().first, 0);
  EXPECT_EQ(back.trajectory.back().second, 2);
}
