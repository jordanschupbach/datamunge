#include <gtest/gtest.h>

#include <bit>
#include <climits>
#include <cmath>
#include <cstdint>
#include <datamunge/algorithms/gauss_legendre_pi.hpp>
#include <datamunge/algorithms/pi_and_cordic.hpp>
#include <limits>
#include <numbers>

#include "../oracles/gauss_legendre_machin.hpp"
using namespace datamunge::algorithms;
namespace {
void same(const GaussLegendreState& a, const GaussLegendreState& b) {
  EXPECT_EQ(a.a, b.a);
  EXPECT_EQ(a.b, b.b);
  EXPECT_EQ(a.t, b.t);
  EXPECT_EQ(a.p, b.p);
}
void certificate(const GaussLegendreTrace& t) {
  const auto& run = t.run;
  EXPECT_EQ(run.initial.a, 1);
  EXPECT_EQ(run.initial.b, 1 / std::sqrt(2.));
  EXPECT_EQ(run.initial.t, .25);
  EXPECT_EQ(run.initial.p, 1);
  auto   previous       = run.initial;
  double previous_value = (previous.a + previous.b) * (previous.a + previous.b) / (4 * previous.t);
  int    first_value = -1, first_state = -1, index = 0;
  for (const auto& e : t.steps) {
    EXPECT_EQ(e.iteration, ++index);
    same(e.before, previous);
    const long double a = e.before.a, b = e.before.b;
    const auto        delta = a - e.after.a;
    EXPECT_NEAR(e.after.a, (a + b) / 2, 2e-16L);
    EXPECT_NEAR(static_cast<long double>(e.after.b) * e.after.b, a * b, 3e-16L);
    const long double correction = static_cast<long double>(e.before.p) * delta * delta;
    EXPECT_NEAR(e.correction, correction, std::fabs(correction) * 5e-16L + 1e-300L);
    EXPECT_NEAR(e.after.t, static_cast<long double>(e.before.t) - correction, 1e-16L);
    EXPECT_GT(e.after.t, 0);
    EXPECT_TRUE(std::isfinite(e.before.p));
    if (e.iteration < gauss_legendre_pi_limit())
      EXPECT_EQ(e.after.p, 2 * e.before.p);
    else
      EXPECT_TRUE(std::isinf(e.after.p));
    EXPECT_LE(e.after.a, e.before.a);
    EXPECT_GE(e.after.b, e.before.b);
    EXPECT_GE(e.after.a, e.after.b);
    const auto gap_identity =
        (a - b) * (a - b) / (2 * (std::sqrt(a) + std::sqrt(b)) * (std::sqrt(a) + std::sqrt(b)));
    EXPECT_NEAR(static_cast<long double>(e.after.a) - e.after.b, gap_identity, 3e-16L);
    EXPECT_EQ(e.estimate_changed, e.estimate != previous_value);
    const bool changed =
        e.after.a != e.before.a || e.after.b != e.before.b || e.after.t != e.before.t;
    EXPECT_EQ(e.mean_state_changed, changed);
    if (!e.estimate_changed && first_value < 0)
      first_value = index;
    if (!changed && first_state < 0)
      first_state = index;
    EXPECT_EQ(e.estimate, gauss_legendre_pi(index));
    previous       = e.after;
    previous_value = e.estimate;
  }
  same(run.final, previous);
  EXPECT_EQ(run.value, previous_value);
  EXPECT_EQ(run.first_unchanged_estimate, first_value);
  EXPECT_EQ(run.first_unchanged_mean_state, first_state);
  EXPECT_EQ(run.iterations, index);
  EXPECT_EQ(run.work.square_roots, index + 1);
  EXPECT_EQ(run.work.geometric_means, index);
  EXPECT_EQ(run.work.arithmetic_means, index);
  EXPECT_EQ(run.work.correction_updates, index);
  EXPECT_EQ(run.work.power_updates, index);
  EXPECT_EQ(run.work.estimate_evaluations, index + 1);
  EXPECT_EQ(t.captured_records, std::size_t(index + 1));
}
} // namespace
TEST(LegendreChecked, IndependentMachinOracleAndUncertifiedRoundingPlateau) {
  EXPECT_EQ(gauss_legendre_oracle::rounded, std::numbers::pi);
  EXPECT_NEAR(gauss_legendre_oracle::pi, std::numbers::pi_v<long double>, 2e-19L);
  for (int n = 1; n <= 3; ++n) {
    const auto value  = gauss_legendre_pi(n);
    const auto d      = pi_reference_accuracy(value);
    const auto actual = std::fabs(static_cast<long double>(value) - gauss_legendre_oracle::pi);
    const auto reference_quantum =
        std::nextafter(gauss_legendre_oracle::pi, std::numeric_limits<long double>::infinity()) -
        gauss_legendre_oracle::pi;
    EXPECT_NEAR(
        d.wider_absolute_error,
        gauss_legendre_oracle::true_error[n - 1],
        reference_quantum + 2e-19L);
    EXPECT_EQ(d.wider_absolute_error, actual);
  }
  EXPECT_LT(
      gauss_legendre_oracle::true_error[1],
      gauss_legendre_oracle::true_error[0] * gauss_legendre_oracle::true_error[0]);
  EXPECT_EQ(
      std::bit_cast<std::int64_t>(gauss_legendre_pi(3)) -
          std::bit_cast<std::int64_t>(std::numbers::pi),
      2);
}
TEST(LegendreChecked, AllActualStagesAndSameKernelFinitePrefix) {
  const auto t = gauss_legendre_pi_trace(gauss_legendre_pi_limit(), 0);
  certificate(t);
  for (int n = 0; n <= gauss_legendre_pi_limit(); ++n) {
    const auto s = gauss_legendre_pi_run(n);
    EXPECT_EQ(s.value, gauss_legendre_pi(n));
    if (n) {
      same(s.final, t.steps[n - 1].after);
      EXPECT_EQ(s.value, t.steps[n - 1].estimate);
    } else
      same(s.initial, s.final);
  }
}
TEST(LegendreChecked, ZeroInitializationAndLegacyNegativeCompatibility) {
  const auto t = gauss_legendre_pi_trace(0, 1);
  certificate(t);
  EXPECT_TRUE(t.steps.empty());
  EXPECT_EQ(t.captured_records, 1u);
  EXPECT_NEAR(t.run.value, 1.5 + std::sqrt(2.), 1e-15);
  EXPECT_EQ(gauss_legendre_pi(-1), gauss_legendre_pi(0));
  EXPECT_EQ(gauss_legendre_pi(INT_MIN), gauss_legendre_pi(0));
}
TEST(LegendreChecked, UnchangedEstimateIsNotEqualMeansOrCorrectRounding) {
  const auto t = gauss_legendre_pi_trace(6);
  EXPECT_EQ(t.run.first_unchanged_estimate, 4);
  EXPECT_EQ(t.run.first_unchanged_mean_state, 5);
  EXPECT_FALSE(t.steps[3].estimate_changed);
  EXPECT_TRUE(t.steps[3].mean_state_changed);
  EXPECT_FALSE(t.steps[4].mean_state_changed);
  EXPECT_NE(t.steps[4].after.a, t.steps[4].after.b);
  EXPECT_EQ(
      std::nextafter(t.steps[4].after.b, std::numeric_limits<double>::infinity()),
      t.steps[4].after.a);
  EXPECT_EQ(t.run.value, t.steps[2].estimate);
  EXPECT_NE(t.run.value, std::numbers::pi);
}
TEST(LegendreChecked, UsedCorrectionCanDisappearInRoundedTAssignment) {
  const auto  t = gauss_legendre_pi_trace(5);
  const auto& e = t.steps[3];
  EXPECT_GT(e.correction, 0);
  EXPECT_EQ(e.before.t, e.after.t);
  EXPECT_NE(e.correction, e.before.t - e.after.t);
  EXPECT_EQ(t.steps[4].correction, 0);
}
TEST(LegendreChecked, UnusedWeightBoundaryIsRetainedAndFutureUseRejected) {
  const auto limit = gauss_legendre_pi_limit();
  const auto s     = gauss_legendre_pi_run(limit);
  EXPECT_TRUE(std::isfinite(s.value));
  EXPECT_TRUE(std::isinf(s.final.p));
  EXPECT_TRUE(std::isfinite(gauss_legendre_pi_run(limit - 1).final.p));
  EXPECT_EQ(gauss_legendre_pi(limit - 1), s.value);
  EXPECT_THROW((void)gauss_legendre_pi(limit + 1), std::invalid_argument);
  EXPECT_THROW((void)gauss_legendre_pi(INT_MAX), std::invalid_argument);
  EXPECT_THROW((void)gauss_legendre_pi_run(INT_MAX), std::invalid_argument);
  EXPECT_THROW((void)gauss_legendre_pi_trace(INT_MAX, 1), std::invalid_argument);
}
TEST(LegendreChecked, WholeRecordQuotaDomainOrderAndOwnedCopies) {
  EXPECT_EQ(gauss_legendre_pi_trace_records(6), 7u);
  EXPECT_THROW((void)gauss_legendre_pi_trace(6, 6), std::length_error);
  auto t = gauss_legendre_pi_trace(6, 7);
  EXPECT_EQ(t.captured_records, 7u);
  EXPECT_EQ(gauss_legendre_pi_trace(6, 0).captured_records, 7u);
  t.steps[0].after.a  = 99;
  t.steps[0].before.a = 98;
  EXPECT_EQ(t.run.initial.a, 1);
  EXPECT_NE(t.steps[1].before.a, 99);
  EXPECT_NE(t.run.final.a, 99);
  for (int n : {-1, INT_MIN, INT_MAX, gauss_legendre_pi_limit() + 1}) {
    EXPECT_THROW((void)gauss_legendre_pi_trace_records(n), std::invalid_argument);
    EXPECT_THROW((void)gauss_legendre_pi_trace(n, 1), std::invalid_argument);
    EXPECT_THROW((void)gauss_legendre_pi_run(n), std::invalid_argument);
  }
}
TEST(LegendreChecked, ReferenceDifferencesHaveSeparateUnitsAndLogAvailability) {
  const auto exact_reference = pi_reference_accuracy(std::numbers::pi);
  EXPECT_EQ(exact_reference.rounded_absolute_error, 0);
  EXPECT_EQ(exact_reference.rounded_ulp_offset, 0);
  EXPECT_FALSE(exact_reference.has_rounded_log_error);
  EXPECT_EQ(exact_reference.rounded_log10_error, 0);
  EXPECT_EQ(
      exact_reference.wider_has_more_bits,
      std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits);
  if (exact_reference.wider_has_more_bits) {
    EXPECT_TRUE(exact_reference.has_wider_log_error);
    EXPECT_GT(exact_reference.wider_absolute_error, 0);
    EXPECT_NEAR(exact_reference.wider_absolute_error, 1.2246467991473531772e-16L, 2e-19L);
  }
  const auto plateau = pi_reference_accuracy(gauss_legendre_pi(6));
  EXPECT_EQ(plateau.rounded_ulp_offset, 2);
  EXPECT_TRUE(plateau.has_rounded_log_error);
  EXPECT_NEAR(plateau.rounded_log10_error, std::log10(plateau.rounded_absolute_error), 1e-14);
  EXPECT_NEAR(plateau.wider_log10_error, std::log10(plateau.wider_absolute_error), 1e-16L);
}
TEST(LegendreChecked, ReferenceDomainAndAdjacentCandidates) {
  const double p     = std::numbers::pi;
  const auto   above = pi_reference_accuracy(std::nextafter(p, INFINITY)),
             below   = pi_reference_accuracy(std::nextafter(p, -INFINITY));
  EXPECT_EQ(above.rounded_ulp_offset, 1);
  EXPECT_EQ(below.rounded_ulp_offset, -1);
  const auto neg = pi_reference_accuracy(-1);
  EXPECT_GT(neg.rounded_absolute_error, 4);
  EXPECT_LT(neg.rounded_ulp_offset, 0);
  EXPECT_THROW((void)pi_reference_accuracy(INFINITY), std::invalid_argument);
  EXPECT_THROW((void)pi_reference_accuracy(-INFINITY), std::invalid_argument);
  EXPECT_THROW(
      (void)pi_reference_accuracy(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
}
TEST(LegendreChecked, PerStageWorkAndUnchangedSuffixAreStillExecuted) {
  for (int n : {0, 1, 2, 3, 4, 5, 6, 20, 100, 1000}) {
    const auto s = gauss_legendre_pi_run(n);
    EXPECT_EQ(s.work.square_roots, n + 1);
    EXPECT_EQ(s.work.correction_updates, n);
    EXPECT_EQ(s.work.power_updates, n);
    EXPECT_EQ(s.work.estimate_evaluations, n + 1);
    EXPECT_TRUE(std::isfinite(s.value));
  }
  EXPECT_EQ(gauss_legendre_pi_run(100).first_unchanged_mean_state, 5);
  EXPECT_EQ(gauss_legendre_pi_run(100).work.power_updates, 100);
}
