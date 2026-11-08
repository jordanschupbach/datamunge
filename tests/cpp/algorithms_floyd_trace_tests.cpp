#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <climits>
#include <datamunge/algorithms/cycle_detection.hpp>
#include <map>
#include <random>
#include <stdexcept>
#include <tuple>
using namespace datamunge::algorithms;
namespace {
struct Orbit {
  std::size_t                mu, lambda;
  std::vector<std::uint64_t> unique;
};
Orbit first_repeat(std::uint64_t x, const CycleFunction& f) {
  std::map<std::uint64_t, std::size_t> seen;
  Orbit                                o{};
  while (!seen.contains(x)) {
    seen[x] = o.unique.size();
    o.unique.push_back(x);
    x = f(x);
  }
  o.mu     = seen.at(x);
  o.lambda = o.unique.size() - o.mu;
  return o;
}
std::uint64_t at(const Orbit& o, std::size_t i) {
  return o.unique[i < o.unique.size() ? i : o.mu + (i - o.mu) % o.lambda];
}
void check(const CycleFunction& f, std::uint64_t seed) {
  const auto                                           orbit = first_repeat(seed, f);
  std::vector<std::pair<std::uint64_t, std::uint64_t>> ordinary_calls, trace_calls;
  const auto                                           w = floyd_cycle_work(seed, [&](auto x) {
    const auto y = f(x);
    ordinary_calls.emplace_back(x, y);
    return y;
  });
  const auto                                           t = floyd_cycle_trace(seed, [&](auto x) {
    const auto y = f(x);
    trace_calls.emplace_back(x, y);
    return y;
  });
  ASSERT_EQ(ordinary_calls, trace_calls);
  EXPECT_EQ(w.evaluations, t.work.evaluations);
  EXPECT_EQ(t.work.result.mu, orbit.mu);
  EXPECT_EQ(t.work.result.lambda, orbit.lambda);
  EXPECT_EQ(t.work.entry, at(orbit, orbit.mu));
  const auto rounds =
      orbit.lambda * std::max(std::size_t(1), (orbit.mu + orbit.lambda - 1) / orbit.lambda);
  EXPECT_EQ(w.detection_evaluations, 3 * rounds);
  EXPECT_EQ(w.entry_evaluations, 2 * orbit.mu);
  EXPECT_EQ(w.period_evaluations, orbit.lambda);
  EXPECT_EQ(w.evaluations, 3 * rounds + 2 * orbit.mu + orbit.lambda);
  EXPECT_EQ(t.steps.size(), rounds + orbit.mu + 1 + orbit.lambda);
  EXPECT_EQ(t.steps.size(), w.comparisons);
  std::size_t detection = 0, locate = 0, measure = 0;
  for (const auto& s : t.steps) {
    EXPECT_EQ(s.tortoise, at(orbit, s.tortoise_index));
    EXPECT_EQ(s.hare, at(orbit, s.hare_index));
    if (s.phase == CyclePhase::detect) {
      ++detection;
      EXPECT_EQ(s.progress, detection);
      EXPECT_EQ(s.tortoise_index, detection);
      EXPECT_EQ(s.hare_index, 2 * detection);
      EXPECT_EQ(s.evaluations, 3 * detection);
      EXPECT_EQ(s.phase_entry, detection == 1);
      EXPECT_EQ(s.tortoise == s.hare, detection == rounds);
    } else if (s.phase == CyclePhase::locate) {
      EXPECT_EQ(s.progress, locate);
      EXPECT_EQ(s.tortoise_index, locate);
      EXPECT_EQ(s.hare_index, 2 * rounds + locate);
      EXPECT_EQ(s.evaluations, 3 * rounds + 2 * locate);
      EXPECT_EQ(s.phase_entry, locate == 0);
      EXPECT_EQ(s.tortoise == s.hare, locate == orbit.mu);
      ++locate;
    } else {
      ASSERT_EQ(s.phase, CyclePhase::measure);
      ++measure;
      EXPECT_EQ(s.progress, measure);
      EXPECT_EQ(s.tortoise_index, orbit.mu);
      EXPECT_EQ(s.hare_index, orbit.mu + measure);
      EXPECT_EQ(s.evaluations, 3 * rounds + 2 * orbit.mu + measure);
      EXPECT_EQ(s.phase_entry, measure == 1);
      EXPECT_EQ(s.tortoise == s.hare, measure == orbit.lambda);
    }
  }
  EXPECT_EQ(detection, rounds);
  EXPECT_EQ(locate, orbit.mu + 1);
  EXPECT_EQ(measure, orbit.lambda);
}
} // namespace
TEST(FloydTrace, ExhaustiveAllMapsThroughFiveStatesWithIndependentFirstRepeat) {
  for (unsigned n = 1; n <= 5; ++n) {
    unsigned count = 1;
    for (unsigned i = 0; i < n; ++i)
      count *= n;
    for (unsigned code = 0; code < count; ++code) {
      std::vector<std::uint64_t> next(n);
      auto                       z = code;
      for (auto& v : next) {
        v = z % n;
        z /= n;
      }
      for (unsigned seed = 0; seed < n; ++seed)
        check([&](auto x) { return next[x]; }, seed);
    }
  }
}
TEST(FloydTrace, TailPeriodBoundariesAndRandomMapsHaveActualIndices) {
  for (std::size_t mu = 0; mu <= 64; ++mu)
    for (std::size_t lambda = 1; lambda <= 64; ++lambda) {
      const auto next = rho_successors(mu, lambda);
      check([&](auto x) { return next[x]; }, 0);
    }
  std::mt19937_64 rng(116);
  for (int k = 0; k < 500; ++k) {
    std::vector<std::uint64_t> next(1 + rng() % 80);
    for (auto& v : next)
      v = rng() % next.size();
    check([&](auto x) { return next[x]; }, rng() % next.size());
  }
}
TEST(FloydTrace, EveryRowQuotaStopsBeforeNextMovementWithoutUndoingCalls) {
  for (auto [mu, lambda] : std::vector<std::pair<std::size_t, std::size_t>>{
           {0, 1}, {0, 3}, {1, 1}, {3, 4}, {8, 7}, {7, 8}}) {
    auto       next = rho_successors(mu, lambda);
    const auto full = floyd_cycle_trace(0, [&](auto x) { return next[x]; });
    for (std::size_t cap = 1; cap < full.steps.size(); ++cap) {
      std::size_t calls = 0;
      EXPECT_THROW(
          floyd_cycle_trace(
              0,
              [&](auto x) {
                ++calls;
                return next[x];
              },
              {},
              cap),
          std::length_error);
      EXPECT_EQ(calls, full.steps[cap - 1].evaluations);
    }
    EXPECT_EQ(
        floyd_cycle_trace(
            0, [&](auto x) { return next[x]; }, {}, full.steps.size())
            .steps.size(),
        full.steps.size());
    EXPECT_EQ(
        floyd_cycle_trace(
            0, [&](auto x) { return next[x]; }, {}, 0)
            .steps.size(),
        full.steps.size());
  }
}
TEST(FloydTrace, EveryGlobalEvaluationBudgetAndCallbackFailurePropagates) {
  const auto next = rho_successors(3, 4);
  const auto full = floyd_cycle_trace(0, [&](auto x) { return next[x]; });
  for (std::size_t cap = 1; cap < full.work.evaluations; ++cap) {
    std::size_t calls = 0;
    EXPECT_THROW(
        floyd_cycle_trace(
            0,
            [&](auto x) {
              ++calls;
              return next[x];
            },
            {cap},
            0),
        std::runtime_error);
    EXPECT_EQ(calls, cap);
  }
  EXPECT_EQ(
      floyd_cycle_trace(
          0, [&](auto x) { return next[x]; }, {full.work.evaluations}, 0)
          .work.evaluations,
      full.work.evaluations);
  struct Failure {};
  for (std::size_t fail = 1; fail <= full.work.evaluations; ++fail) {
    std::size_t calls = 0;
    EXPECT_THROW(
        floyd_cycle_trace(
            0,
            [&](auto x) {
              if (++calls == fail)
                throw Failure{};
              return next[x];
            }),
        Failure);
    EXPECT_EQ(calls, fail);
  }
  EXPECT_THROW(floyd_cycle_trace(0, {}), std::bad_function_call);
  std::size_t increasing_calls = 0;
  EXPECT_THROW(
      floyd_cycle_trace(
          0,
          [&](auto x) {
            ++increasing_calls;
            return x + 1;
          },
          {1000},
          0),
      std::runtime_error);
  EXPECT_EQ(increasing_calls, 1000);
}
TEST(FloydTrace, HighStateLabelsAndOwnedRecordsNeedNoSentinel) {
  const std::array<std::uint64_t, 7> labels{UINT64_MAX, 0, UINT64_MAX - 1, 7, 1, UINT64_MAX - 2, 9};
  auto                               next = rho_successors(3, 4);
  std::map<std::uint64_t, std::uint64_t> map;
  for (std::size_t i = 0; i < 7; ++i)
    map[labels[i]] = labels[next[i]];
  check([&](auto x) { return map.at(x); }, UINT64_MAX);
  const auto t     = floyd_cycle_trace(UINT64_MAX, [&](auto x) { return map.at(x); });
  const auto steps = t.steps;
  map.clear();
  ASSERT_EQ(t.steps.size(), steps.size());
  EXPECT_EQ(t.steps.front().tortoise, 0);
  EXPECT_EQ(t.work.entry, 7);
  EXPECT_EQ(t.work.result.lambda, 4);
}
TEST(FloydTrace, FixedPointStillMakesHistoricalThreeInitialCalls) {
  std::vector<std::uint64_t> inputs;
  const auto                 t = floyd_cycle_trace(42, [&](auto x) {
    inputs.push_back(x);
    return x;
  });
  EXPECT_EQ(inputs, (std::vector<std::uint64_t>{42, 42, 42, 42}));
  ASSERT_EQ(t.steps.size(), 3);
  EXPECT_EQ(t.steps[0].evaluations, 3);
  EXPECT_EQ(t.steps[1].evaluations, 3);
  EXPECT_EQ(t.steps[2].evaluations, 4);
  EXPECT_EQ(t.work.result.mu, 0);
  EXPECT_EQ(t.work.result.lambda, 1);
}
TEST(QuadraticCycle, FullUint64ValuesMatchIndependentBinaryModularArithmetic) {
  // Independent add-and-double oracle: no multiplication or widened integers.
  const auto add_mod = [](std::uint64_t a, std::uint64_t b, std::uint64_t m) {
    return a >= m - b ? a - (m - b) : a + b;
  };
  const auto oracle = [&](std::uint64_t x, std::uint64_t m, std::uint64_t c) {
    auto base = x % m, bits = x, sum = std::uint64_t(0);
    while (bits) {
      if (bits & 1)
        sum = add_mod(sum, base, m);
      bits >>= 1;
      if (bits)
        base = add_mod(base, base, m);
    }
    return add_mod(sum, c % m, m);
  };
  std::mt19937_64                  rng(116);
  const std::vector<std::uint64_t> edge{0, 1, 2, UINT64_MAX / 2, UINT64_MAX - 1, UINT64_MAX};
  for (auto x : edge)
    for (auto m : edge)
      if (m)
        for (auto c : edge) {
          const auto expected = oracle(x, m, c);
          EXPECT_EQ(quadratic_cycle_step(x, m, c), expected);
        }
  for (int i = 0; i < 10000; ++i) {
    const auto x = rng(), m = rng() | 1, c = rng();
    const auto expected = oracle(x, m, c);
    EXPECT_EQ(quadratic_cycle_step(x, m, c), expected);
  }
  EXPECT_THROW(quadratic_cycle_step(0, 0), std::invalid_argument);
  EXPECT_EQ(quadratic_cycle_step(UINT64_MAX, 1, UINT64_MAX), 0);
  EXPECT_EQ(quadratic_cycle_step(2, 1009), 5);
}
TEST(FloydTrace, OriginalQuadraticModuliPreserveResultsAndCounts) {
  const std::vector<std::size_t> fe{196, 352, 203, 1514, 2620, 4473, 6290},
      be{161, 289, 151, 1353, 1651, 3201, 4625}, length{49, 113, 44, 421, 570, 1089, 1289};
  std::size_t i = 0;
  for (std::uint64_t m : {1009u, 5003u, 10007u, 50021u, 100003u, 500009u, 1000003u}) {
    const CycleFunction f = [=](auto x) { return quadratic_cycle_step(x, m); };
    const auto          a = floyd_cycle_trace(2, f);
    const auto          b = brent_cycle_work(2, f);
    EXPECT_EQ(a.work.result.mu, b.result.mu);
    EXPECT_EQ(a.work.result.lambda, b.result.lambda);
    EXPECT_EQ(a.work.result.mu + a.work.result.lambda, length[i]);
    EXPECT_EQ(a.work.evaluations, fe[i]);
    EXPECT_EQ(b.evaluations, be[i]);
    ++i;
  }
}
