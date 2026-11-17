#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/reverse_delete.hpp>
#include <datamunge/algorithms/rice_coding.hpp>
#include <datamunge/algorithms/ridders.hpp>
#include <datamunge/algorithms/roulette_wheel.hpp>
#include <datamunge/algorithms/round_robin.hpp>
#include <datamunge/algorithms/rounding_functions.hpp>
#include <datamunge/algorithms/run_length_encoding.hpp>
#include <datamunge/algorithms/runge_kutta4.hpp>
#include <datamunge/algorithms/ruzzo_tompa.hpp>
#include <datamunge/geometry/rotating_calipers.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <tuple>
namespace a = datamunge::algorithms;
namespace z = datamunge::geometry;
TEST(Batch324ReverseDelete, RealMasksAndDisconnectedForest) {
  const std::vector<a::WeightedEdge> e{{0, 1, 1}, {1, 2, 2}, {0, 2, 8}, {3, 3, -100}};
  auto                               t = a::reverse_delete_trace(4, e);
  EXPECT_EQ(t.initial_components, 2);
  EXPECT_FALSE(t.forest.is_connected);
  EXPECT_EQ(t.forest.total_weight, 3);
  EXPECT_EQ(t.edge_visits, e.size() * (e.size() + 1));
  ASSERT_EQ(t.steps.size(), 4);
  EXPECT_EQ(t.steps[0].edge_index, 2);
  EXPECT_TRUE(t.steps[0].removed);
  EXPECT_EQ(t.kept, (std::vector<char>{1, 1, 0, 0}));
  for (const auto& s : t.steps)
    EXPECT_EQ(s.removed, s.components_without == t.initial_components);
}
TEST(Batch324ReverseDelete, ParallelEdgesAndPolicies) {
  EXPECT_EQ(a::reverse_delete(2, {{0, 1, 2}, {0, 1, 1}}).total_weight, 1);
  EXPECT_TRUE(a::reverse_delete(0, {}).is_connected);
  EXPECT_THROW(a::reverse_delete(1, {{0, 1, 1}}), std::invalid_argument);
  a::ReverseDeleteOptions o;
  o.max_edge_visits = 11;
  EXPECT_THROW(a::reverse_delete_run(3, a::reverse_delete_cycle(3), o), std::length_error);
  o                   = {};
  o.max_trace_entries = 11;
  EXPECT_THROW(a::reverse_delete_trace(3, a::reverse_delete_cycle(3), o), std::length_error);
}
TEST(Batch324Rice, ActualFormulaAndRawBits) {
  for (int k = 0; k <= 6; ++k) {
    std::vector<std::uint64_t> x(64);
    std::iota(x.begin(), x.end(), 0);
    auto t = a::rice_encode_trace(x, k);
    auto d = a::rice_decode_trace(t.bits, k, x.size());
    ASSERT_EQ(t.words.size(), x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
      const auto& w = t.words[i];
      EXPECT_EQ(w.end - w.begin, (x[i] >> k) + 1 + k);
      EXPECT_EQ(w.quotient, x[i] >> k);
      EXPECT_EQ(w.remainder, x[i] & ((std::uint64_t{1} << k) - 1));
      EXPECT_EQ(d.words[i].value, x[i]);
      for (std::size_t j = w.begin; j < w.remainder_begin - 1; ++j)
        EXPECT_TRUE(t.bits[j]);
      EXPECT_FALSE(t.bits[w.remainder_begin - 1]);
    }
  }
}
TEST(Batch324Rice, FiniteSupportAndInvalidControls) {
  auto rows = a::rice_length_study();
  ASSERT_EQ(rows.size(), 7);
  for (const auto& r : rows) {
    long double sum = 0, norm = 0;
    for (int i = 0; i < 300; ++i) {
      auto w = std::exp2(-static_cast<long double>(i) / 8);
      sum += w * ((i >> r.k) + 1 + r.k);
      norm += w;
    }
    EXPECT_NEAR(r.mean_bits, static_cast<double>(sum / norm), 1e-12);
  }
  EXPECT_THROW(a::rice_encode_trace({}, 31), std::invalid_argument);
  EXPECT_THROW(a::rice_decode_trace({true}, 2, 1), std::invalid_argument);
  EXPECT_THROW(a::rice_encode_trace({100}, 0, 10), std::length_error);
}
TEST(Batch324Ridders, QueryCountsAndNormalizedExtremeValues) {
  std::size_t calls = 0;
  auto        f     = [&](double x) {
    ++calls;
    return (x * x - 2) * 1e300;
  };
  a::RiddersOptions o;
  o.residual_tolerance = 0;
  o.x_tolerance        = 1e-14;
  auto t               = a::ridders_trace(f, 1, 2, o);
  EXPECT_TRUE(t.result.converged);
  EXPECT_EQ(calls, t.function_evaluations);
  EXPECT_EQ(calls, t.evaluations.size());
  EXPECT_NEAR(t.result.root, std::sqrt(2.), 1e-13);
  EXPECT_EQ(t.steps.size(), t.result.iterations);
  for (const auto& s : t.steps) {
    EXPECT_LE(s.a, s.b);
    EXPECT_GE(s.candidate, s.a);
    EXPECT_LE(s.candidate, s.b);
    EXPECT_LE(s.a_after, s.b_after);
    EXPECT_GE(s.normalized_radicand, 0);
  }
  auto exact = a::ridders_trace([](double x) { return x - 2; }, 0, 4);
  EXPECT_EQ(exact.function_evaluations, 3);
  EXPECT_FALSE(exact.steps[0].correction_evaluated);
}
TEST(Batch324Ridders, StopReasonsAndCallbackAdmission) {
  auto no = a::ridders_run([](double x) { return x * x + 1; }, 0, 1);
  EXPECT_FALSE(no.bracketed);
  EXPECT_EQ(no.result.root, 0);
  EXPECT_EQ(no.result.residual, 1);
  a::RiddersOptions o;
  o.max_iterations = 0;
  auto t           = a::ridders_run([](double x) { return x * x - 2; }, 1, 2, o);
  EXPECT_EQ(t.stop, a::RiddersStop::Budget);
  EXPECT_EQ(t.function_evaluations, 2);
  std::size_t calls   = 0;
  o                   = {};
  o.max_trace_entries = 2;
  EXPECT_THROW(
      a::ridders_trace(
          [&](double x) {
            ++calls;
            return x * x - 2;
          },
          1,
          2,
          o),
      std::length_error);
  EXPECT_EQ(calls, 2);
  EXPECT_THROW(
      a::ridders([](double) { return std::numeric_limits<double>::infinity(); }, 0, 1),
      std::invalid_argument);
}
TEST(Batch324Diameter, ActualCalipersAgainstExhaustiveDistances) {
  std::mt19937 rng(4);
  for (int trial = 0; trial < 150; ++trial) {
    std::vector<z::Point2D> p;
    for (int i = 0; i < 3 + trial % 40; ++i)
      p.push_back({double(int(rng() % 101) - 50), double(int(rng() % 101) - 50)});
    auto   t        = z::polygon_diameter_trace(p);
    double expected = 0;
    for (auto u : p)
      for (auto v : p)
        expected = std::max(expected, std::hypot(u.x - v.x, u.y - v.y));
    EXPECT_NEAR(t.result.distance, expected, 1e-12);
    EXPECT_EQ(t.events.size(), t.area_tests + t.pair_tests);
    EXPECT_LE(t.pointer_advances, 2 * t.hull.size());
    EXPECT_LE(t.pair_tests, 4 * t.hull.size());
  }
}
TEST(Batch324Diameter, DegenerateTiesAndFullRangePredicate) {
  EXPECT_DOUBLE_EQ(z::polygon_diameter({{0, 0}, {3, 4}}).distance, 5);
  EXPECT_DOUBLE_EQ(z::polygon_diameter({{0, 0}, {1, 0}, {2, 0}, {1, 0}}).distance, 2);
  EXPECT_NEAR(
      z::polygon_diameter({{0, 0}, {3e-200, 0}, {3e-200, 4e-200}, {0, 4e-200}}).distance,
      5e-200,
      1e-214);
  EXPECT_THROW(z::polygon_diameter({{0, 0}, {0, 0}}), std::invalid_argument);
  EXPECT_THROW(z::polygon_diameter({{-1e308, 0}, {1e308, 0}}), std::overflow_error);
  z::DiameterOptions o;
  o.max_trace_entries = 1;
  EXPECT_THROW(z::polygon_diameter_trace({{0, 0}, {1, 0}, {0, 1}}, o), std::length_error);
}
TEST(Batch324Roulette, RawStreamAndStrictUpperBuckets) {
  auto t = a::roulette_wheel_trace({0, 2, 0, 3}, 100, 1);
  EXPECT_EQ(t.counts[0], 0);
  EXPECT_EQ(t.counts[2], 0);
  EXPECT_EQ(t.picks, a::roulette_wheel_sample({0, 2, 0, 3}, 100, 1));
  std::uint64_t state       = 1;
  std::size_t   comparisons = 0;
  for (const auto& s : t.draws) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    EXPECT_EQ(s.raw, state);
    EXPECT_DOUBLE_EQ(s.uniform, (state >> 11) * (1. / 9007199254740992.));
    EXPECT_GE(s.target, s.lower);
    EXPECT_LT(s.target, s.upper);
    comparisons += s.comparisons;
  }
  EXPECT_EQ(comparisons, t.comparisons);
}
TEST(Batch324Roulette, InvalidModelsAndOverflowSafeScaling) {
  for (const auto& f : std::vector<std::vector<double>>{
           {}, {0, 0}, {1, -1}, {std::numeric_limits<double>::quiet_NaN()}})
    EXPECT_THROW(a::roulette_wheel_run(f, 0), std::invalid_argument);
  auto t = a::roulette_wheel_run({1e308, 1e308}, 100);
  EXPECT_TRUE(t.scaled);
  EXPECT_DOUBLE_EQ(t.total, 2);
  EXPECT_EQ(std::accumulate(t.counts.begin(), t.counts.end(), std::size_t{}), 100);
  EXPECT_THROW(a::roulette_wheel_sample({1}, -1), std::invalid_argument);
}
TEST(Batch324Rounding, BothSidesOfPositiveAndNegativeHalf) {
  EXPECT_EQ(a::round_half_up(std::nextafter(.5, 0.)), 0);
  EXPECT_EQ(a::round_half_up(.5), 1);
  EXPECT_EQ(a::round_half_down(std::nextafter(-.5, 0.)), 0);
  EXPECT_EQ(a::round_half_down(-.5), -1);
  auto t = a::rounding_trace(
      {std::nextafter(-.5, 0.), -.5, std::nextafter(-.5, -1.)}, a::RoundMode::HalfDown);
  EXPECT_FALSE(t.steps[0].tie);
  EXPECT_TRUE(t.steps[1].tie);
  EXPECT_FALSE(t.steps[2].tie);
  EXPECT_EQ(t.rounded, (std::vector<double>{0, -1, -1}));
  EXPECT_TRUE(std::signbit(a::round_half_to_even(-0.)));
  EXPECT_TRUE(std::isinf(a::round_half_to_even(std::numeric_limits<double>::infinity())));
}
TEST(Batch324Rounding, ActualRunningMeansAndValidation) {
  auto values = a::rounding_example_ties();
  auto t      = a::rounding_trace(values, a::RoundMode::HalfToEven);
  EXPECT_EQ(t.ties, 60);
  EXPECT_DOUBLE_EQ(t.mean_error, 0);
  EXPECT_DOUBLE_EQ(a::rounding_bias(values, a::RoundMode::HalfUp), .5);
  long double sum = 0;
  for (std::size_t i = 0; i < t.steps.size(); ++i) {
    sum += t.steps[i].error;
    EXPECT_DOUBLE_EQ(t.steps[i].mean_error, static_cast<double>(sum / (i + 1)));
  }
  EXPECT_THROW(
      a::rounding_run({std::numeric_limits<double>::quiet_NaN()}, a::RoundMode::Floor),
      std::invalid_argument);
  EXPECT_THROW(a::round_with(1, static_cast<a::RoundMode>(99)), std::invalid_argument);
}
TEST(Batch324RoundRobin, OriginalQueueAndArrivalsBeforeRequeue) {
  auto t = a::round_robin_trace({{7, 0, 3}, {91, 1, 1}}, 2);
  ASSERT_EQ(t.steps.size(), 3);
  EXPECT_EQ(t.steps[0].ready_after, (std::vector<std::size_t>{1, 0}));
  EXPECT_EQ(t.result.metrics[0].completion, 4);
  EXPECT_EQ(t.result.metrics[1].completion, 3);
  EXPECT_EQ(t.busy_ticks, 4);
  EXPECT_EQ(t.preemptions, 1);
  auto r = a::round_robin({{0, 0, 5}, {1, 0, 3}, {2, 0, 1}, {3, 0, 2}}, 2);
  EXPECT_EQ(r.metrics[0].completion, 11);
  EXPECT_EQ(r.metrics[1].completion, 10);
  EXPECT_EQ(r.metrics[2].completion, 5);
  EXPECT_EQ(r.metrics[3].completion, 7);
}
TEST(Batch324RoundRobin, ZeroBurstsIdleJumpsAndValidation) {
  auto t = a::round_robin_trace({{7, 5, 0}, {91, 1000000000, 2}}, 1);
  EXPECT_EQ(t.result.metrics[0].completion, 5);
  EXPECT_EQ(t.result.metrics[1].completion, 1000000002);
  EXPECT_EQ(t.idle_jumps, 1);
  EXPECT_EQ(t.busy_ticks, 2);
  EXPECT_EQ(a::round_robin({}, 1).avg_waiting, 0);
  EXPECT_THROW(a::round_robin({{7, 0, 1}}, 0), std::invalid_argument);
  EXPECT_THROW(a::round_robin({{7, 0, 1}, {7, 0, 1}}, 1), std::invalid_argument);
  EXPECT_THROW(a::round_robin({{7, std::numeric_limits<int>::max(), 1}}, 1), std::overflow_error);
}
TEST(Batch324RK4, ActualStageQueriesAndKnownScalarStep) {
  std::vector<std::pair<double, std::vector<double>>> calls;
  auto                                                t = a::runge_kutta4_trace(
      [&](double time, const auto& y) {
        calls.push_back({time, y});
        return y;
      },
      {1},
      0,
      1,
      1);
  ASSERT_EQ(t.steps.size(), 1);
  const auto& s = t.steps[0];
  EXPECT_EQ(t.rhs_calls, 4);
  ASSERT_EQ(calls.size(), 4);
  EXPECT_EQ(calls[0].second, s.before);
  EXPECT_EQ(calls[1].second, s.second);
  EXPECT_EQ(calls[2].second, s.third);
  EXPECT_EQ(calls[3].second, s.fourth);
  EXPECT_DOUBLE_EQ(s.k1[0], 1);
  EXPECT_DOUBLE_EQ(s.k2[0], 1.5);
  EXPECT_DOUBLE_EQ(s.k3[0], 1.75);
  EXPECT_DOUBLE_EQ(s.k4[0], 2.75);
  EXPECT_NEAR(s.after[0], 65. / 24, 1e-15);
}
TEST(Batch324RK4, ConvergenceReverseAndCheckedHistoryAdmission) {
  auto   f = [](double, const auto& y) { return std::vector<double>{-y[0]}; };
  auto   x = a::runge_kutta4_run(f, {1}, 0, 1, 16), y = a::runge_kutta4_run(f, {1}, 0, 1, 32);
  double e1 = std::abs(x.solution.y.back()[0] - std::exp(-1.)),
         e2 = std::abs(y.solution.y.back()[0] - std::exp(-1.));
  EXPECT_GT(e1 / e2, 15);
  EXPECT_LT(e1 / e2, 17);
  EXPECT_NEAR(a::runge_kutta4(f, {std::exp(-1.)}, 1, 0, 64).y.back()[0], 1, 1e-8);
  std::size_t   calls = 0;
  a::RK4Options o;
  o.max_trace_entries = 9;
  EXPECT_THROW(
      a::runge_kutta4_trace(
          [&](double, const auto& y) {
            ++calls;
            return y;
          },
          {1},
          0,
          1,
          1,
          o),
      std::length_error);
  EXPECT_EQ(calls, 0);
  EXPECT_THROW(
      a::runge_kutta4([](double, const auto&) { return std::vector<double>{}; }, {1}, 0, 1, 2),
      std::invalid_argument);
}
TEST(Batch324RLE, ExhaustiveRawByteRoundTripsAndActualCounts) {
  for (int length = 0; length <= 7; ++length) {
    std::size_t count = 1;
    for (int i = 0; i < length; ++i)
      count *= 3;
    for (std::size_t code = 0; code < count; ++code) {
      auto        value = code;
      std::string text;
      for (int i = 0; i < length; ++i) {
        text.push_back(static_cast<char>(value % 3));
        value /= 3;
      }
      auto t = a::rle_encode_trace(text);
      auto d = a::rle_decode_trace(t.runs);
      EXPECT_EQ(d.text, text);
      EXPECT_EQ(d.emitted_bytes, text.size());
      EXPECT_EQ(t.comparisons, text.empty() ? 0 : text.size() + t.runs.size() - 1);
      EXPECT_EQ(t.steps.size(), t.runs.size());
    }
  }
}
TEST(Batch324RLE, CompleteDecodeAdmissionAndNoncanonicalRecords) {
  EXPECT_EQ(a::rle_decode({{'A', 0}, {'A', 2}, {'A', 1}}), "AAA");
  a::RLEOptions o;
  o.max_bytes = 3;
  EXPECT_THROW(a::rle_decode_run({{'X', 4}}, o), std::length_error);
  o.max_bytes = 0;
  EXPECT_THROW(
      a::rle_decode_run({{'X', std::numeric_limits<std::size_t>::max()}, {'Y', 1}}, o),
      std::length_error);
  EXPECT_EQ(a::rle_encoded_cells({{'A', 100}}), 1);
}
namespace {
std::vector<std::tuple<std::size_t, std::size_t, double>> maximal_intervals(
    const std::vector<double>& x) {
  std::vector<std::tuple<std::size_t, std::size_t, double>> strict, out;
  for (std::size_t a = 0; a < x.size(); ++a)
    for (std::size_t b = a + 1; b <= x.size(); ++b) {
      double score = std::accumulate(x.begin() + a, x.begin() + b, 0.);
      bool   good  = score > 0;
      for (std::size_t c = a; c < b && good; ++c)
        for (std::size_t d = c + 1; d <= b; ++d)
          if (c != a || d != b)
            if (std::accumulate(x.begin() + c, x.begin() + d, 0.) >= score)
              good = false;
      if (good)
        strict.emplace_back(a, b, score);
    }
  for (auto candidate : strict) {
    auto [a, b, s] = candidate;
    bool contained = false;
    for (auto [c, d, t] : strict)
      if (c <= a && d >= b && (c < a || d > b))
        contained = true;
    if (!contained)
      out.push_back(candidate);
  }
  return out;
}
} // namespace
TEST(Batch324Ruzzo, ExhaustiveIndependentStrictIntervalDefinition) {
  for (int length = 0; length <= 7; ++length) {
    std::size_t count = 1;
    for (int i = 0; i < length; ++i)
      count *= 3;
    for (std::size_t code = 0; code < count; ++code) {
      std::vector<double> x;
      auto                c = code;
      for (int i = 0; i < length; ++i) {
        x.push_back(double(int(c % 3) - 1));
        c /= 3;
      }
      auto r        = a::ruzzo_tompa_run(x);
      auto expected = maximal_intervals(x);
      ASSERT_EQ(r.segments.size(), expected.size()) << "length " << length << " code " << code;
      for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(r.segments[i].begin, std::get<0>(expected[i]));
        EXPECT_EQ(r.segments[i].end, std::get<1>(expected[i]));
        EXPECT_EQ(r.segments[i].score, std::get<2>(expected[i]));
      }
    }
  }
}
TEST(Batch324Ruzzo, HonestQuadraticWorkAndPolicies) {
  for (std::size_t m : {1, 2, 4, 16, 64}) {
    auto t = a::ruzzo_tompa_run(a::ruzzo_alternating_scores(m));
    EXPECT_EQ(t.comparisons, m * (m - 1) / 2);
    EXPECT_EQ(t.segments.size(), m);
    EXPECT_EQ(t.absorbed, 0);
  }
  auto t = a::ruzzo_tompa_trace({3, -2, 5});
  EXPECT_EQ(t.steps.size(), 3);
  EXPECT_EQ(t.segments[0].score, 6);
  EXPECT_EQ(t.absorbed, 1);
  EXPECT_THROW(a::ruzzo_tompa({std::numeric_limits<double>::quiet_NaN()}), std::invalid_argument);
}
