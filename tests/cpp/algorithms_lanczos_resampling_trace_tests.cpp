#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/algorithms/lanczos_resampling.hpp>
#include <limits>
using namespace datamunge::algorithms;
TEST(LanczosResampleTrace, ActualOriginalIdentitiesWeightsAndSequentialSums) {
  const std::vector<double> s = {0, 1, 4, 9, 16, 25, 36};
  const auto                t = lanczos_resample_trace(s, 2.5, 3);
  ASSERT_EQ(t.taps.size(), 6U);
  double      sum = 0, weights = 0;
  std::size_t negatives = 0;
  for (std::size_t i = 0; i < t.taps.size(); ++i) {
    const auto& e = t.taps[i];
    EXPECT_EQ(e.virtual_index, static_cast<std::ptrdiff_t>(i));
    EXPECT_EQ(e.sample_index, i);
    EXPECT_EQ(e.sample, s[i]);
    EXPECT_EQ(e.weight, lanczos_kernel_weight(2.5 - i, 3));
    EXPECT_EQ(e.weighted_value, e.sample * e.weight);
    EXPECT_EQ(e.sum_before, sum);
    sum += e.weighted_value;
    weights += e.weight;
    EXPECT_EQ(e.sum_after, sum);
    EXPECT_EQ(e.weight_sum_after, weights);
    negatives += e.weight < 0;
  }
  EXPECT_EQ(t.analysis.raw_sum, sum);
  EXPECT_EQ(t.analysis.weight_sum, weights);
  EXPECT_EQ(t.analysis.value, sum);
  EXPECT_EQ(t.analysis.work.taps, 6U);
  EXPECT_EQ(t.analysis.work.clamped_taps, 0U);
  EXPECT_EQ(t.analysis.work.negative_weights, negatives);
  EXPECT_EQ(t.captured_entries, 13U);
}
TEST(LanczosResampleTrace, ExactNumericIntegerSamplesAndHugeFinitePositionClamp) {
  const std::vector<double> s = {-2, 1, 4, -9, 16, 25, -36};
  for (int a : {1, 2, 3, 8}) {
    for (std::size_t i = 0; i < s.size(); ++i)
      EXPECT_EQ(lanczos_resample_at(s, double(i), a), s[i]);
    EXPECT_EQ(lanczos_resample_at(s, -std::numeric_limits<double>::max(), a), s.front());
    EXPECT_EQ(lanczos_resample_at(s, std::numeric_limits<double>::max(), a), s.back());
  }
  EXPECT_EQ(lanczos_resample_at({5}, .5), 5);
  EXPECT_EQ(lanczos_resample({0, 1, 4}, {-1, 0, 1, 2, 100}), (std::vector<double>{0, 0, 1, 4, 4}));
}
TEST(LanczosResampleTrace, ExplicitDcNormalizationAndNegativeLobeRinging) {
  const std::vector<double> ones(9, 1);
  const auto                raw        = lanczos_resample_analyze(ones, 4.5, 3, false);
  const auto                normalized = lanczos_resample_analyze(ones, 4.5, 3, true);
  EXPECT_GT(std::abs(raw.value - 1), 1e-5);
  EXPECT_EQ(raw.value, raw.weight_sum);
  EXPECT_EQ(normalized.raw_sum, raw.raw_sum);
  EXPECT_EQ(normalized.weight_sum, raw.weight_sum);
  EXPECT_EQ(normalized.value, 1);
  EXPECT_GT(raw.work.negative_weights, 0U);
  const std::vector<double> step = {0, 0, 0, 0, 1, 1, 1, 1, 1};
  EXPECT_LT(lanczos_resample_analyze(step, 2.5, 3, true).value, 0);
  EXPECT_GT(lanczos_resample_analyze(step, 4.5, 3, true).value, 1);
  for (int a = 1; a <= 8; ++a) {
    EXPECT_EQ(lanczos_kernel_weight(0, a), 1);
    for (int k = -a; k <= a; ++k)
      if (k)
        EXPECT_EQ(lanczos_kernel_weight(k, a), 0);
  }
}
TEST(LanczosResampleTrace, CompleteInputQuotaAndFiniteIntermediateGuards) {
  EXPECT_THROW((void)lanczos_resample_at({}, 0), std::invalid_argument);
  EXPECT_THROW((void)lanczos_resample_at({1}, 0, 0), std::invalid_argument);
  EXPECT_THROW((void)lanczos_resample_at({1}, 0, -1), std::invalid_argument);
  EXPECT_THROW(
      (void)lanczos_resample_at({1}, std::numeric_limits<double>::infinity()),
      std::invalid_argument);
  EXPECT_THROW(
      (void)lanczos_resample({1}, {0, std::numeric_limits<double>::quiet_NaN()}),
      std::invalid_argument);
  EXPECT_THROW(
      (void)lanczos_resample_trace({1, std::numeric_limits<double>::quiet_NaN()}, 0, 3, false, 1),
      std::invalid_argument);
  const auto t = lanczos_resample_trace({0, 1, 4}, .5, 3);
  EXPECT_NO_THROW((void)lanczos_resample_trace(t.samples, .5, 3, false, 9));
  EXPECT_THROW((void)lanczos_resample_trace(t.samples, .5, 3, false, 8), std::length_error);
  EXPECT_NO_THROW((void)lanczos_resample_trace(t.samples, .5, 3, false, 0));
  EXPECT_THROW(
      (void)lanczos_resample_trace({1}, 0, std::numeric_limits<int>::max(), false, 100),
      std::length_error);
  EXPECT_THROW(
      (void)lanczos_resample_at(std::vector<double>(9, std::numeric_limits<double>::max()), 4.5),
      std::overflow_error);
  auto c           = t;
  c.samples[0]     = 99;
  c.taps[0].weight = 99;
  c.analysis.value = 99;
  EXPECT_EQ(t.samples[0], 0);
  EXPECT_NE(t.taps[0].weight, 99);
}
