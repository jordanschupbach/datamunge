#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/algorithms/miser.hpp>
#include <limits>
#include <stdexcept>
using namespace datamunge::algorithms;
TEST(MiserTrace, ExactBudgetsActualSamplesAndLeafEstimator) {
  for (long calls : {2L, 3L, 16L, 32L, 63L, 64L, 65L, 79L, 80L, 127L, 128L, 256L, 1024L})
    for (std::uint64_t seed : {0ULL, 1ULL, 7ULL}) {
      long actual = 0;
      auto f      = [&](const std::vector<double>& p) {
        ++actual;
        return p[0] + 2 * p[1];
      };
      auto t = miser_trace(f, {0, 0}, {1, 1}, calls, seed);
      EXPECT_EQ(actual, calls);
      EXPECT_EQ(t.samples.size(), calls);
      EXPECT_EQ(t.evaluations, calls);
      std::uint64_t       state = seed ? seed : 0x2545F4914F6CDD1DULL;
      std::vector<double> sum(t.nodes.size()), sum2(t.nodes.size());
      std::vector<long>   count(t.nodes.size()), pilot(t.nodes.size());
      std::size_t         units = 4;
      for (auto& node : t.nodes)
        units += 1 + node.lower.size() + node.upper.size();
      for (auto& s : t.samples) {
        auto& node = t.nodes[s.node];
        for (std::size_t i = 0; i < s.point.size(); ++i) {
          state ^= state << 13;
          state ^= state >> 7;
          state ^= state << 17;
          double unit = (state >> 11) / 9007199254740992.0;
          EXPECT_DOUBLE_EQ(s.point[i], node.lower[i] + (node.upper[i] - node.lower[i]) * unit);
        }
        EXPECT_DOUBLE_EQ(s.value, s.point[0] + 2 * s.point[1]);
        if (s.pilot)
          ++pilot[s.node];
        else {
          ++count[s.node];
          sum[s.node] += s.value;
          sum2[s.node] += s.value * s.value;
        }
        units += 1 + s.point.size();
      }
      double value = 0, variance = 0;
      for (std::size_t i = 0; i < t.nodes.size(); ++i) {
        auto& n = t.nodes[i];
        EXPECT_EQ(pilot[i], n.pilot_calls);
        if (!n.pilot_calls) {
          EXPECT_EQ(count[i], n.calls);
          double vol = 1;
          for (std::size_t d = 0; d < n.lower.size(); ++d)
            vol *= n.upper[d] - n.lower[d];
          double mean = sum[i] / count[i], var = (sum2[i] / count[i] - mean * mean) / count[i];
          EXPECT_DOUBLE_EQ(n.result.value, vol * mean);
          EXPECT_DOUBLE_EQ(n.result.variance, vol * vol * std::max(0., var));
          value += n.result.value;
          variance += n.result.variance;
        } else {
          EXPECT_EQ(n.left_calls + n.right_calls + n.pilot_calls, n.calls);
          EXPECT_GE(n.left_calls, 2);
          EXPECT_GE(n.right_calls, 2);
        }
      }
      EXPECT_NEAR(t.result.value, value, 1e-12);
      EXPECT_NEAR(t.result.variance, variance, 1e-12);
      EXPECT_EQ(t.captured_entries, units);
      auto ordinary = miser_integrate(
          [](const auto& p) { return p[0] + 2 * p[1]; }, {0, 0}, {1, 1}, calls, seed);
      EXPECT_DOUBLE_EQ(t.result.value, ordinary.value);
      EXPECT_DOUBLE_EQ(t.result.variance, ordinary.variance);
      EXPECT_NO_THROW(miser_trace(
          [](const auto& p) { return p[0] + 2 * p[1]; }, {0, 0}, {1, 1}, calls, seed, units));
      EXPECT_THROW(
          miser_trace(
              [](const auto& p) { return p[0] + 2 * p[1]; },
              {0, 0},
              {1, 1},
              calls,
              seed,
              units - 1),
          std::length_error);
    }
}
TEST(MiserTrace, GuardsBeforeCallbacksConstantAndOriginalGaussian) {
  int  callbacks = 0;
  auto f         = [&](const auto&) {
    ++callbacks;
    return 2.;
  };
  EXPECT_THROW(miser_integrate(f, {0}, {1}, 1), std::invalid_argument);
  EXPECT_THROW(miser_integrate(f, {}, {}, 20), std::invalid_argument);
  EXPECT_THROW(miser_integrate(f, {0, 0}, {1}, 20), std::invalid_argument);
  EXPECT_THROW(miser_integrate(f, {1}, {0}, 20), std::invalid_argument);
  EXPECT_THROW(
      miser_integrate(f, {0}, {std::numeric_limits<double>::infinity()}, 20),
      std::invalid_argument);
  EXPECT_EQ(callbacks, 0);
  auto c = miser_integrate(f, {0}, {3}, 128);
  EXPECT_DOUBLE_EQ(c.value, 6);
  EXPECT_DOUBLE_EQ(c.variance, 0);
  EXPECT_EQ(callbacks, 128);
  EXPECT_THROW(
      miser_integrate([](const auto&) { return std::nan(""); }, {0}, {1}, 2), std::domain_error);
  EXPECT_THROW(
      miser_integrate([](const auto&) { return 1e300; }, {0}, {1}, 2), std::overflow_error);
  auto r = miser_integrate(miser_gaussian, {0, 0}, {1, 1}, 200000, 1);
  EXPECT_NEAR(r.value, miser_gaussian_integral(), 1e-4);
  EXPECT_GT(r.variance, 0);
  EXPECT_EQ(miser_gaussian({.5, .5}), 1);
  EXPECT_THROW(miser_gaussian({0}), std::invalid_argument);
  EXPECT_THROW(miser_gaussian_comparison({1}, 1), std::invalid_argument);
}
