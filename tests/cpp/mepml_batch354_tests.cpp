#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <datamunge/algorithms/shell_sort.hpp>
#include <datamunge/algorithms/shortest_first.hpp>
#include <datamunge/algorithms/shunting_yard.hpp>
#include <datamunge/algorithms/sieve_of_eratosthenes.hpp>
#include <datamunge/algorithms/signed_reversals.hpp>
#include <datamunge/algorithms/simon.hpp>
#include <datamunge/algorithms/simplex.hpp>
#include <datamunge/algorithms/siphash.hpp>
#include <datamunge/geometry/shoelace.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
namespace a   = datamunge::algorithms;
namespace geo = datamunge::geometry;
TEST(Mepml354Shell, ActualPassesAndIndependentSort) {
  std::mt19937 rng(19);
  for (int n = 0; n < 100; ++n) {
    std::vector<int> v(n);
    for (auto& x : v)
      x = int(rng() % 15) - 7;
    auto expected = v;
    std::sort(expected.begin(), expected.end());
    auto r = a::shell_sort_trace(v);
    EXPECT_EQ(r.values, expected);
    std::size_t comparisons = 0, shifts = 0;
    for (const auto& p : r.passes) {
      for (std::size_t i = p.gap; i < p.after.size(); ++i)
        EXPECT_LE(p.after[i - p.gap], p.after[i]);
      comparisons += p.comparisons;
      shifts += p.shifts;
    }
    EXPECT_EQ(comparisons, r.comparisons);
    EXPECT_EQ(shifts, r.shifts);
  }
  auto r = a::shell_sort_trace({3, 1, 2});
  EXPECT_EQ(r.comparisons, 3);
  EXPECT_EQ(r.shifts, 2);
  EXPECT_EQ(a::shell_sort_example_arrays().size(), 40);
}
TEST(Mepml354Shell, PoliciesAndOrdinaryInsertion) {
  a::ShellSortOptions o;
  o.max_comparisons = 1;
  EXPECT_THROW(a::shell_sort_run({3, 2, 1}, o), std::length_error);
  o                    = {};
  o.max_history_values = 3;
  EXPECT_THROW(a::shell_sort_trace({3, 2, 1}, o), std::length_error);
  EXPECT_EQ(a::insertion_sort_work({4, 3, 2, 1}).comparisons, 6);
  EXPECT_EQ(a::insertion_sort_work({1, 2, 3, 4}).shifts, 0);
}
TEST(Mepml354Shoelace, AnalyticAndTranslation) {
  auto r = geo::shoelace_trace({{0, 0}, {4, 0}, {0, 3}});
  EXPECT_EQ(r.signed_area, 6);
  EXPECT_EQ(r.edges.size(), 3);
  EXPECT_EQ(r.edges[1].determinant, 0);
  EXPECT_EQ(r.edges.back().signed_partial_area, 6);
  EXPECT_EQ(geo::signed_polygon_area({{1e15, 1e15}, {1e15 + 4, 1e15}, {1e15, 1e15 + 3}}), 6);
  EXPECT_EQ(geo::signed_polygon_area({{0, 3}, {4, 0}, {0, 0}}), -6);
  EXPECT_EQ(geo::polygon_area({{0, 0}, {2, 2}, {0, 2}, {2, 0}}), 0);
  for (std::size_t n : {3, 4, 8, 32, 256})
    EXPECT_NEAR(
        geo::polygon_area(geo::shoelace_regular_polygon(n)),
        n * std::sin(2 * std::acos(-1) / n) / 2,
        1e-13);
}
TEST(Mepml354Shoelace, DegeneracyAndBounds) {
  EXPECT_EQ(geo::polygon_area({}), 0);
  EXPECT_THROW(geo::shoelace_run({{NAN, 0}}), std::invalid_argument);
  EXPECT_THROW(geo::shoelace_run({{0, 0}, {1e308, 0}, {0, 1e308}}), std::overflow_error);
  geo::ShoelaceOptions o;
  o.max_records = 2;
  EXPECT_THROW(geo::shoelace_trace({{0, 0}, {4, 0}, {0, 3}}, o), std::length_error);
}
TEST(Mepml354SJN, OriginalAndTieOrder) {
  auto r = a::shortest_job_next_trace({{0, 0, 6}, {1, 1, 3}, {2, 2, 1}, {3, 3, 4}});
  EXPECT_EQ(r.schedule.metrics[0].completion, 6);
  EXPECT_EQ(r.schedule.metrics[1].completion, 10);
  EXPECT_EQ(r.schedule.metrics[2].completion, 7);
  EXPECT_EQ(r.schedule.metrics[3].completion, 14);
  EXPECT_DOUBLE_EQ(r.schedule.avg_waiting, 4.25);
  EXPECT_EQ(r.preemptions, 0);
  EXPECT_EQ(a::shortest_job_next({{9, 0, 2}, {4, 0, 2}}).gantt.front().id, 9);
}
TEST(Mepml354SJN, MalformedIdleAndOptimalityScope) {
  EXPECT_THROW(a::shortest_job_next({{1, 0, 0}}), std::invalid_argument);
  EXPECT_THROW(a::shortest_job_next({{1, -1, 2}}), std::invalid_argument);
  EXPECT_THROW(a::shortest_job_next({{1, 0, 2}, {1, 1, 1}}), std::invalid_argument);
  EXPECT_THROW(a::shortest_job_next({{1, std::numeric_limits<int>::max(), 1}}), std::length_error);
  auto r = a::shortest_job_next_trace({{99, 1000000, 2}});
  EXPECT_EQ(r.decisions.size(), 2);
  EXPECT_EQ(r.scans, 2);
  EXPECT_EQ(r.schedule.metrics[0].waiting, 0);
  EXPECT_DOUBLE_EQ(a::shortest_job_next({{0, 0, 10}, {1, 1, 1}}).avg_waiting, 4.5);
}
TEST(Mepml354SRT, OriginalAndActualPreemption) {
  auto r = a::shortest_remaining_time_trace({{0, 0, 8}, {1, 1, 4}, {2, 2, 9}, {3, 3, 5}});
  EXPECT_EQ(r.schedule.metrics[0].completion, 17);
  EXPECT_EQ(r.schedule.metrics[1].completion, 5);
  EXPECT_EQ(r.schedule.metrics[2].completion, 26);
  EXPECT_EQ(r.schedule.metrics[3].completion, 10);
  EXPECT_EQ(r.preemptions, 1);
  EXPECT_DOUBLE_EQ(r.schedule.avg_waiting, 6.5);
  EXPECT_EQ(r.decisions[1].remaining[0], 7);
}
TEST(Mepml354SRT, TicksMatchIndependentDiscreteSimulation) {
  std::mt19937 rng(21);
  for (int trial = 0; trial < 300; ++trial) {
    const std::size_t            n = 1 + rng() % 8;
    std::vector<a::SchedProcess> p;
    std::vector<int>             rem, completion(n);
    for (std::size_t i = 0; i < n; ++i) {
      p.push_back({100 - i, int(rng() % 12), 1 + int(rng() % 7)});
      rem.push_back(p.back().burst);
    }
    int         t    = 0;
    std::size_t done = 0;
    while (done < n) {
      std::size_t best = n;
      for (std::size_t i = 0; i < n; ++i)
        if (rem[i] && p[i].arrival <= t &&
            (best == n || rem[i] < rem[best] ||
             (rem[i] == rem[best] && p[i].arrival < p[best].arrival)))
          best = i;
      if (best < n && !--rem[best]) {
        completion[best] = t + 1;
        ++done;
      }
      ++t;
    }
    auto r = a::shortest_remaining_time(p);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_EQ(r.metrics[i].completion, completion[i]);
  }
  a::ShortestFirstOptions o;
  o.max_comparisons = 1;
  EXPECT_THROW(a::shortest_remaining_time_run({{0, 0, 3}, {1, 1, 1}}, o), std::length_error);
}
TEST(Mepml354Parser, ActualPrecedenceAndStacks) {
  auto r = a::shunting_yard_trace("3+4*2");
  EXPECT_EQ(r.postfix, "3 4 2 * +");
  EXPECT_EQ(r.value, 11);
  EXPECT_EQ(r.tokens, 5);
  EXPECT_EQ(r.operator_pops, 2);
  EXPECT_EQ(r.evaluation.back().stack, std::vector<double>{11});
  EXPECT_EQ(a::shunting_yard_evaluate("2^3^2"), 512);
  EXPECT_EQ(a::shunting_yard_evaluate("(2^3)^2"), 64);
  EXPECT_EQ(a::shunting_yard_evaluate(".5+2."), 2.5);
  EXPECT_EQ(a::evaluate_rpn("-2 3 *"), -6);
}
TEST(Mepml354Parser, RejectMalformedAndArithmetic) {
  for (const std::string t :
       {"", "3+", "2(3)", "1..2", "(1+2", "1+2)", "2@3", "-1", "1 2", "()", ".", "1e3"})
    EXPECT_THROW(a::shunting_yard_run(t), std::invalid_argument) << t;
  for (const std::string t : {"+", "1 2", "3foo", "1 0 /"}) {
    if (t == "1 0 /")
      EXPECT_THROW(a::evaluate_rpn(t), std::domain_error);
    else
      EXPECT_THROW(a::evaluate_rpn(t), std::invalid_argument);
  }
  EXPECT_THROW(a::shunting_yard_run("1/0"), std::domain_error);
  EXPECT_THROW(a::shunting_yard_run("0^(1-2)"), std::domain_error);
  a::ShuntingYardOptions o;
  o.max_history_values = 3;
  EXPECT_THROW(a::shunting_yard_trace("1+2", o), std::length_error);
  EXPECT_EQ(a::shunting_yard_to_postfix("1/0"), "1 0 /");
}
TEST(Mepml354Sieve, ActualSquaresAndTrialDivision) {
  for (std::uint64_t n = 0; n <= 300; ++n) {
    auto                       r = a::prime_sieve_trace(n);
    std::vector<std::uint64_t> expected;
    for (std::uint64_t x = 2; x <= n; ++x) {
      bool prime = true;
      for (std::uint64_t d = 2; d <= x / d; ++d)
        if (x % d == 0) {
          prime = false;
          break;
        }
      if (prime)
        expected.push_back(x);
    }
    EXPECT_EQ(r.primes, expected);
    std::size_t strikes = 0;
    for (const auto& s : r.stages) {
      EXPECT_EQ(s.strikes.front(), s.prime * s.prime);
      strikes += s.strikes.size();
    }
    EXPECT_EQ(strikes, r.strikes);
  }
  auto r = a::prime_sieve_trace(30);
  EXPECT_EQ(r.primes.size(), 10);
  EXPECT_EQ(r.strikes, 24);
  EXPECT_EQ(r.stages.back().prime, 5);
  EXPECT_EQ(a::primes_up_to(10000).size(), 1229);
}
TEST(Mepml354Sieve, PoliciesAndCapacity) {
  EXPECT_TRUE(a::prime_sieve(1).empty());
  a::PrimeSieveOptions o;
  o.max_n = 0;
  EXPECT_THROW(a::prime_sieve_run(std::numeric_limits<std::uint64_t>::max(), o), std::length_error);
  o             = {};
  o.max_strikes = 1;
  EXPECT_THROW(a::prime_sieve_run(10, o), std::length_error);
  o             = {};
  o.max_records = 1;
  EXPECT_THROW(a::prime_sieve_trace(10, o), std::length_error);
}
TEST(Mepml354Reversals, ActualOptimalPathAndBound) {
  auto tiny = a::signed_reversal_trace({-2, -1});
  EXPECT_EQ(tiny.result.distance, 1);
  EXPECT_EQ(tiny.path.front().breakpoints, 2);
  auto r = a::signed_reversal_trace({3, 1, 2, -5, 4});
  EXPECT_EQ(r.path.size(), r.result.reversals.size() + 1);
  auto cur = r.path.front().permutation;
  for (std::size_t i = 0; i < r.result.reversals.size(); ++i) {
    auto [first, last] = r.result.reversals[i];
    cur                = a::signed_apply_reversal(cur, first, last);
    EXPECT_EQ(cur, r.path[i + 1].permutation);
  }
  EXPECT_EQ(cur, (std::vector<int>{1, 2, 3, 4, 5}));
  EXPECT_LE((a::signed_breakpoints({3, 1, 2, -5, 4}) + 1) / 2, r.result.distance);
  EXPECT_EQ(a::signed_reversal_run({}).result.distance, 0);
}
TEST(Mepml354Reversals, ValidationAndExhaustion) {
  for (const auto& p :
       std::vector<std::vector<int>>{{0}, {1, 1}, {3, 1}, {std::numeric_limits<int>::min()}})
    EXPECT_THROW(a::signed_reversal_run(p), std::invalid_argument);
  a::SignedReversalOptions o;
  o.max_states = 1;
  EXPECT_THROW(a::signed_reversal_run({-1}, o), std::length_error);
  o           = {};
  o.max_edges = 1;
  EXPECT_THROW(a::signed_reversal_run({-1, -2, -3}, o), std::length_error);
  EXPECT_THROW(a::signed_apply_reversal({1}, 1, 1), std::invalid_argument);
}
TEST(Mepml354Simon, AllSmallPeriodsAndFullProbability) {
  for (int n = 1; n <= 5; ++n)
    for (std::size_t p = 1; p < (std::size_t{1} << n); ++p) {
      auto r = a::simon_trace(n, p);
      EXPECT_EQ(r.result.recovered_period, p);
      EXPECT_EQ(r.rank, n - 1);
      double total = 0;
      for (std::size_t y = 0; y < r.probabilities.size(); ++y) {
        const double expected = std::popcount(y & p) % 2 ? 0 : std::ldexp(1., 1 - n);
        EXPECT_NEAR(r.probabilities[y], expected, 1e-13);
        total += r.probabilities[y];
      }
      EXPECT_NEAR(total, 1, 1e-13);
      EXPECT_EQ(r.hadamard_calls, 2 * n);
      EXPECT_EQ(r.oracle_writes, std::size_t{1} << n);
    }
}
TEST(Mepml354Simon, PromiseAndAllocationBounds) {
  EXPECT_THROW(a::simon(0, 1), std::invalid_argument);
  EXPECT_THROW(a::simon(4, 0), std::invalid_argument);
  EXPECT_THROW(a::simon(4, 16), std::invalid_argument);
  EXPECT_THROW(a::simon(11, 1), std::length_error);
  a::SimonOptions o;
  o.max_history_values = 1;
  EXPECT_THROW(a::simon_trace(4, 11, o), std::length_error);
  EXPECT_EQ(a::simon(4, 11).observed_y.size(), 7);
}
TEST(Mepml354Simplex, ActualTableauAndExplicitLimit) {
  auto r = a::simplex_trace({5, 4}, {{6, 4}, {1, 2}}, {24, 6});
  EXPECT_EQ(r.result.status, a::SimplexStatus::Optimal);
  EXPECT_NEAR(r.result.value, 21, 1e-12);
  EXPECT_NEAR(r.result.x[0], 3, 1e-12);
  EXPECT_NEAR(r.result.x[1], 1.5, 1e-12);
  EXPECT_EQ(r.history.size(), 2);
  EXPECT_EQ(r.history.back().tableau.back().back(), r.result.value);
  a::SimplexOptions o;
  o.max_pivots = 1;
  auto limited = a::simplex_run({5, 4}, {{6, 4}, {1, 2}}, {24, 6}, o);
  EXPECT_EQ(limited.result.status, a::SimplexStatus::IterationLimit);
  EXPECT_EQ(limited.pivots, 1);
  o.max_pivots = 0;
  EXPECT_EQ(a::simplex_run({-1}, {{1}}, {2}, o).result.status, a::SimplexStatus::Optimal);
  EXPECT_EQ(a::simplex_run({1}, {}, {}).result.status, a::SimplexStatus::Unbounded);
}
TEST(Mepml354Simplex, ShapeFiniteAndPolicies) {
  EXPECT_THROW(a::simplex_run({1}, {{1, 2}}, {2}), std::invalid_argument);
  EXPECT_THROW(a::simplex_run({1}, {{1}}, {-1}), std::invalid_argument);
  EXPECT_THROW(a::simplex_run({NAN}, {}, {}), std::invalid_argument);
  a::SimplexOptions o;
  o.max_cells = 1;
  EXPECT_THROW(a::simplex_run({1}, {{1}}, {2}, o), std::length_error);
  o                    = {};
  o.max_history_values = 1;
  EXPECT_THROW(a::simplex_trace({1}, {{1}}, {2}, o), std::length_error);
  EXPECT_TRUE(a::simplex_run({}, {{}}, {1}).result.x.empty());
}
TEST(Mepml354SipHash, ReferenceVectorsAndActualRounds) {
  const auto key = a::siphash_example_key();
  EXPECT_EQ(a::siphash24(key, ""), 0x726fdb47dd0e0e31ULL);
  EXPECT_EQ(a::siphash24(key, a::siphash_example_message(15)), 0xa129ca6149be45e5ULL);
  for (std::size_t n = 0; n < 80; ++n) {
    auto r = a::siphash24_trace(key, a::siphash_example_message(n));
    EXPECT_EQ(r.hash, a::siphash24(key, a::siphash_example_message(n)));
    EXPECT_EQ(r.blocks, n / 8 + 1);
    EXPECT_EQ(r.rounds, 2 * r.blocks + 4);
    EXPECT_EQ(r.history.size(), r.rounds);
    EXPECT_TRUE(r.history.back().finalization);
  }
  auto counts = a::siphash_bucket_counts(key, a::siphash_example_words(), 16);
  EXPECT_EQ(std::accumulate(counts.begin(), counts.end(), std::size_t{0}), 32);
}
TEST(Mepml354SipHash, RawNulAndPolicies) {
  const auto key = a::siphash_example_key();
  EXPECT_NE(a::siphash24(key, std::string(1, '\0')), a::siphash24(key, ""));
  a::SipHashOptions o;
  o.max_bytes = 1;
  EXPECT_THROW(a::siphash24_run(key, "ab", o), std::length_error);
  o                   = {};
  o.max_round_records = 5;
  EXPECT_THROW(a::siphash24_trace(key, "", o), std::length_error);
  EXPECT_THROW(a::siphash_bucket_counts(key, {}, 0), std::invalid_argument);
}
