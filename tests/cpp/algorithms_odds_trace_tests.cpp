#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/odds_algorithm.hpp>
#include <limits>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
TEST(OddsTrace, IndependentEnumeratedOutcomesAndOptimalAdaptiveStopping) {
  std::mt19937_64 rng(264273);
  for (std::size_t n = 0; n <= 12; ++n)
    for (int repeat = 0; repeat < 20; ++repeat) {
      std::vector<double> p(n);
      for (auto& x : p)
        x = double(rng() % 5) / 4;
      auto t     = odds_algorithm_trace(p);
      auto curve = odds_threshold_curve(p);
      ASSERT_EQ(curve.size(), n + 1);
      std::vector<long double> wins(n + 1);
      for (std::size_t mask = 0; mask < (std::size_t(1) << n); ++mask) {
        long double probability = 1;
        for (std::size_t i = 0; i < n; ++i)
          probability *= mask & (std::size_t(1) << i) ? p[i] : 1 - p[i];
        for (std::size_t first = 0; first < n; ++first) {
          int successes = 0;
          for (std::size_t i = first; i < n; ++i)
            successes += (mask >> i) & 1;
          if (successes == 1)
            wins[first] += probability;
        }
      }
      for (std::size_t i = 0; i <= n; ++i)
        EXPECT_NEAR(curve[i].win_probability, double(wins[i]), 2e-15);
      long double optimal = 0, none = 1;
      for (std::size_t i = n; i-- > 0;) {
        optimal = (1 - p[i]) * optimal + p[i] * std::max(none, optimal);
        none *= 1 - p[i];
      }
      EXPECT_NEAR(t.run.result.win_probability, double(optimal), 2e-15);
      if (n) {
        ASSERT_GE(t.run.result.threshold, 1);
        ASSERT_LE(t.run.result.threshold, n);
        EXPECT_NEAR(t.run.result.win_probability, double(wins[t.run.result.threshold - 1]), 2e-15);
      } else
        EXPECT_EQ(t.run.result.threshold, 0);
      EXPECT_EQ(t.run.visits, t.steps.size());
      EXPECT_EQ(t.captured_entries, n + 2 + t.steps.size());
      long double q = 1, w = 0, sum = 0;
      std::size_t index = n;
      for (auto& s : t.steps) {
        EXPECT_EQ(s.event, index--);
        EXPECT_EQ(s.none_before, q);
        EXPECT_EQ(s.one_before, w);
        EXPECT_EQ(s.sum_before, sum);
        auto fail = 1 - static_cast<long double>(p[s.event - 1]);
        w         = static_cast<long double>(p[s.event - 1]) * q + fail * w;
        q *= fail;
        if (!s.certain)
          sum += p[s.event - 1] / fail;
        EXPECT_EQ(s.none_after, q);
        EXPECT_EQ(s.one_after, w);
        EXPECT_EQ(s.sum_after, sum);
        EXPECT_EQ(s.certain, p[s.event - 1] == 1);
        EXPECT_EQ(s.reached, s.certain || sum >= 1);
      }
      EXPECT_EQ(odds_algorithm(p).win_probability, t.run.result.win_probability);
      EXPECT_NO_THROW(odds_algorithm_trace(p, t.captured_entries));
      EXPECT_THROW(odds_algorithm_trace(p, t.captured_entries - 1), std::length_error);
    }
}
TEST(OddsTrace, CertainEventsCompleteGuardsAndOriginalStudies) {
  EXPECT_EQ(odds_algorithm({1.}).win_probability, 1);
  EXPECT_EQ(odds_algorithm({1., 1.}).threshold, 2);
  EXPECT_EQ(odds_algorithm({1., 0., 0.}).win_probability, 1);
  EXPECT_EQ(odds_algorithm({0., 0.}).threshold, 1);
  EXPECT_EQ(odds_algorithm({0., 0.}).win_probability, 0);
  auto t = odds_algorithm_trace({1.});
  EXPECT_TRUE(t.steps[0].certain);
  EXPECT_EQ(t.steps[0].odds, 0);
  EXPECT_THROW(odds_algorithm({NAN, .9}), std::invalid_argument);
  EXPECT_THROW(odds_algorithm({-1., .9}), std::invalid_argument);
  EXPECT_THROW(odds_threshold_curve({INFINITY}), std::invalid_argument);
  EXPECT_THROW(odds_threshold_curve({.5}, 1), std::length_error);
  EXPECT_THROW(odds_secretary_probabilities(10, 9), std::length_error);
  EXPECT_EQ(odds_secretary_probabilities(3), (std::vector<double>{1, .5, 1. / 3}));
  auto r = odds_algorithm(odds_algorithm_demo_probabilities());
  EXPECT_EQ(r.threshold, 8);
  EXPECT_NEAR(r.win_probability, .43808, 1e-12);
  for (std::size_t n : {10, 100, 1000}) {
    auto   p       = odds_secretary_probabilities(n);
    auto   a       = odds_algorithm(p);
    auto   c       = odds_threshold_curve(p);
    double maximum = 0;
    for (auto& x : c)
      maximum = std::max(maximum, x.win_probability);
    EXPECT_NEAR(a.win_probability, maximum, 1e-15);
  }
}
