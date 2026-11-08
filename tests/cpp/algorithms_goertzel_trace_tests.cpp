#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <datamunge/algorithms/goertzel.hpp>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>

#include "../oracles/goertzel_exact.hpp"

using namespace datamunge::algorithms;
namespace {
// Independent long-double direct Fourier sum, with no state recurrence or FFT.
long double direct_power(const std::vector<double>& x, long double omega) {
  std::complex<long double> sum{0, 0};
  for (std::size_t i = 0; i < x.size(); ++i) {
    const auto angle = -omega * static_cast<long double>(i);
    sum += static_cast<long double>(x[i]) *
           std::complex<long double>{std::cos(angle), std::sin(angle)};
  }
  return std::norm(sum);
}
} // namespace
TEST(GoertzelTrace, ExactAlgebraicDirectFourierFixtures) {
  for (const auto& c : goertzel_exact_fixture::cases) {
    const auto*               p = goertzel_exact_fixture::values + c.offset;
    const std::vector<double> x(p, p + c.size);
    const long double         truth =
        (long double)c.a_num / c.a_den + (long double)c.b_num / c.b_den * std::sqrt(2.L);
    EXPECT_NEAR(goertzel_power_bin(x, c.bin), double(truth), 2e-11);
    const auto r = goertzel_bin_run(x, c.bin);
    EXPECT_DOUBLE_EQ(r.power, goertzel_power_bin(x, c.bin));
    EXPECT_EQ(r.sample_updates, x.size());
  }
}
TEST(GoertzelTrace, DirectFrequencyAndLegacySelectionAgainstIndependentComplexSums) {
  std::mt19937_64 rng(137);
  for (int trial = 0; trial < 400; ++trial) {
    std::vector<double> x(1 + rng() % 128);
    double              l1 = 0;
    for (auto& v : x) {
      v = double(int(rng() % 65) - 32) / 8;
      l1 += std::abs(v);
    }
    const double fs = 128, f = double(int(rng() % 513) - 256) / 8;
    for (auto mode : {GoertzelFrequencyMode::legacy_bin, GoertzelFrequencyMode::unrounded}) {
      const auto r = goertzel_run(x, f, fs, mode);
      EXPECT_NEAR(r.power, double(direct_power(x, r.omega)), 2e-10 * std::max(1., l1 * l1));
      EXPECT_EQ(r.sample_updates, x.size());
      EXPECT_TRUE(r.has_coefficient);
      if (mode == GoertzelFrequencyMode::legacy_bin) {
        EXPECT_EQ(r.selected_bin, int(.5 + double(x.size()) * f / fs));
        EXPECT_DOUBLE_EQ(r.power, goertzel_power(x, f, fs));
      } else
        EXPECT_DOUBLE_EQ(r.power, goertzel_power_at_frequency(x, f, fs));
    }
  }
  EXPECT_DOUBLE_EQ(goertzel_power({1, 2, 3, 4}, -1, 4), 100); // Legacy offset-truncation bias.
  EXPECT_NEAR(goertzel_power_at_frequency({1, 2, 3, 4}, -1, 4), 8, 1e-12);
  EXPECT_DOUBLE_EQ(
      goertzel_power_at_frequency({1, 2, 3, 4}, -1, 4),
      goertzel_power_at_frequency({1, 2, 3, 4}, 1, 4));
  const std::vector<double> x{1, 2, 3, 4};
  EXPECT_EQ(
      goertzel_run(x, double(std::numeric_limits<int>::max()) + .25, 4).selected_bin,
      std::numeric_limits<int>::max());
  EXPECT_EQ(
      goertzel_run(x, double(std::numeric_limits<int>::min()) - .75, 4).selected_bin,
      std::numeric_limits<int>::min());
  const auto large = goertzel_bin_run(x, std::numeric_limits<std::size_t>::max());
  EXPECT_EQ(large.bin, std::numeric_limits<std::size_t>::max());
  EXPECT_TRUE(std::isfinite(large.power));
  EXPECT_DOUBLE_EQ(large.omega, 2 * std::numbers::pi * double(large.bin) / 4);
}
TEST(GoertzelTrace, ActualStatesMetadataAndSampleCountCertificates) {
  std::mt19937_64 rng(9137);
  for (int trial = 0; trial < 200; ++trial) {
    std::vector<double> x(rng() % 33);
    for (auto& v : x)
      v = double(int(rng() % 17) - 8) / 4;
    const auto k = rng() % 35;
    const auto t = goertzel_bin_trace(x, k, 0);
    ASSERT_EQ(t.events.size(), x.size() + 2);
    EXPECT_EQ(t.events.front().kind, GoertzelEventKind::initial);
    EXPECT_EQ(t.events.back().kind, GoertzelEventKind::stop);
    EXPECT_EQ(t.events.front().index, 0u);
    EXPECT_DOUBLE_EQ(t.events.front().value, t.run.coefficient);
    double s1 = 0, s2 = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
      const auto& e = t.events[i + 1];
      EXPECT_EQ(e.kind, GoertzelEventKind::sample);
      EXPECT_EQ(e.index, i);
      EXPECT_DOUBLE_EQ(e.input, x[i]);
      EXPECT_DOUBLE_EQ(e.before_s1, s1);
      EXPECT_DOUBLE_EQ(e.before_s2, s2);
      EXPECT_DOUBLE_EQ(e.value, x[i] + t.run.coefficient * s1 - s2);
      s2 = s1;
      s1 = e.value;
    }
    EXPECT_DOUBLE_EQ(t.run.s1, s1);
    EXPECT_DOUBLE_EQ(t.run.s2, s2);
    EXPECT_EQ(t.events.back().index, x.size());
    EXPECT_DOUBLE_EQ(t.events.back().before_s1, s1);
    EXPECT_DOUBLE_EQ(t.events.back().before_s2, s2);
    EXPECT_DOUBLE_EQ(t.events.back().value, t.run.power);
    EXPECT_DOUBLE_EQ(t.run.power, goertzel_power_bin(x, k));
    EXPECT_EQ(t.run.sample_updates, x.size());
    EXPECT_EQ(t.run.has_coefficient, !x.empty());
  }
  const auto t = goertzel_bin_trace({1, 2, 3, 4}, 0);
  EXPECT_DOUBLE_EQ(t.run.s1, 20);
  EXPECT_DOUBLE_EQ(t.run.s2, 10);
  EXPECT_DOUBLE_EQ(t.run.power, 100);
  const auto empty = goertzel_run({}, 0, 4);
  EXPECT_FALSE(empty.has_coefficient);
  EXPECT_DOUBLE_EQ(empty.omega, 0);
  EXPECT_EQ(empty.sample_updates, 0u);
}
TEST(GoertzelTrace, CompleteValidationBeforeEmptyQuotaAndSafeIntegerConversion) {
  const double nan = std::numeric_limits<double>::quiet_NaN(),
               inf = std::numeric_limits<double>::infinity();
  for (double bad : {nan, inf, -inf}) {
    EXPECT_THROW(goertzel_power({1, 2}, bad, 8), std::invalid_argument);
    EXPECT_THROW(goertzel_power({}, bad, 8), std::invalid_argument);
    EXPECT_THROW(goertzel_bin_trace({1, 2, bad}, 0, 1), std::invalid_argument);
    EXPECT_THROW(
        goertzel_trace({1, 2, bad}, 0, 8, GoertzelFrequencyMode::legacy_bin, 1),
        std::invalid_argument);
  }
  for (double fs : {0., -1., nan, inf})
    EXPECT_THROW(goertzel_power({}, 0, fs), std::invalid_argument);
  EXPECT_THROW(
      goertzel_run({}, 0, 4, static_cast<GoertzelFrequencyMode>(999)), std::invalid_argument);
  EXPECT_THROW(goertzel_power({1, 2}, 1e300, 1), std::overflow_error);
  EXPECT_THROW(
      goertzel_power({1}, double(std::numeric_limits<int>::max()) + .5, 1), std::overflow_error);
  EXPECT_THROW(
      goertzel_power({1}, double(std::numeric_limits<int>::min()) - 1.5, 1), std::overflow_error);
  EXPECT_THROW(goertzel_power_at_frequency({1}, 1e308, 1e-308), std::overflow_error);
  const std::vector<double> x{1e308, 1e308};
  const auto                original = x;
  EXPECT_THROW(goertzel_power_bin(x, 0), std::overflow_error);
  EXPECT_EQ(x, original);
  EXPECT_THROW(goertzel_power_bin({1e200, -1e200}, 0), std::overflow_error);
}
TEST(GoertzelTrace, RawNegativeCancellationAndLongDcLimitRemainVisible) {
  const std::vector<double> x{0x1.59497b6052b08p+0, -0x1.59497b6052b07p+0};
  EXPECT_DOUBLE_EQ(goertzel_power_bin(x, 0), -0x1p-51);
  EXPECT_DOUBLE_EQ((x[0] + x[1]) * (x[0] + x[1]), 0x1p-104);
  const auto r = goertzel_bin_run(std::vector<double>(100000, 1), 0);
  EXPECT_DOUBLE_EQ(r.power, 10000007168.);
  EXPECT_NE(r.power, 10000000000.);
  EXPECT_EQ(r.sample_updates, 100000u);
}
TEST(GoertzelTrace, OnlineWholeRecordQuotaAndIndependentOwnership) {
  const std::vector<double> x{1, 2, 3, 4};
  const auto                original = x;
  EXPECT_THROW(goertzel_bin_trace(x, 0, 5), std::length_error);
  EXPECT_EQ(goertzel_bin_trace(x, 0, 6).events.size(), 6u);
  EXPECT_EQ(goertzel_bin_trace(x, 0, 0).events.size(), 6u);
  EXPECT_THROW(goertzel_bin_trace({}, 0, 1), std::length_error);
  EXPECT_EQ(goertzel_bin_trace({}, 0, 2).events.size(), 2u);
  EXPECT_EQ(goertzel_bin_trace({}, 0, 0).events.size(), 2u);
  auto t               = goertzel_trace(x, 0, 4, GoertzelFrequencyMode::unrounded, 6);
  auto copy            = t;
  copy.events[1].value = 999;
  copy.run.s1          = 999;
  EXPECT_NE(t.events[1].value, 999);
  EXPECT_NE(t.run.s1, 999);
  EXPECT_EQ(x, original);
}
TEST(GoertzelTrace, NativeSignalFormulaWholeAdmissionAndRealInputChecks) {
  const auto x = dtmf_signal('5', 410);
  ASSERT_EQ(x.size(), 410u);
  for (std::size_t n = 0; n < x.size(); ++n) {
    const auto expected =
        std::sin(2 * M_PI * 770 * n / 8000.) + std::sin(2 * M_PI * 1336 * n / 8000.);
    EXPECT_DOUBLE_EQ(x[n], expected);
  }
  EXPECT_FALSE(std::signbit(x[0]));
  const auto silent = sinusoid_signal(4, 8, {});
  EXPECT_EQ(silent, (std::vector<double>{0, 0, 0, 0}));
  EXPECT_THROW(sinusoid_signal(4, 8, {}, 3), std::length_error);
  EXPECT_EQ(sinusoid_signal(4, 8, {}, 4).size(), 4u);
  EXPECT_EQ(sinusoid_signal(4, 8, {}, 0).size(), 4u);
  EXPECT_THROW(
      sinusoid_signal(std::numeric_limits<std::size_t>::max(), 8, {}, 0), std::length_error);
  const double nan = std::numeric_limits<double>::quiet_NaN(),
               inf = std::numeric_limits<double>::infinity();
  EXPECT_THROW(sinusoid_signal(0, 8, {{1, 1, 0}, {nan, 1, 0}}, 1), std::invalid_argument);
  EXPECT_THROW(sinusoid_signal(4, 8, {{1, inf, 0}}, 1), std::invalid_argument);
  EXPECT_THROW(sinusoid_signal(0, 0, {}), std::invalid_argument);
  EXPECT_THROW(sinusoid_signal(4, 8, {{1e308, 1, 0}}), std::overflow_error);
  EXPECT_THROW(dtmf_signal('X', 0, 8000, 1), std::invalid_argument);
  EXPECT_THROW(dtmf_signal('a', 4), std::invalid_argument);
  auto copy = x;
  copy[0]   = 999;
  EXPECT_EQ(x[0], 0);
}
TEST(GoertzelTrace, NativeDtmfCandidatesAllKeysAndCounterexamples) {
  for (char key : std::string("123A456B789C*0#D")) {
    const auto x = dtmf_signal(key, 410);
    for (auto mode : {GoertzelFrequencyMode::legacy_bin, GoertzelFrequencyMode::unrounded}) {
      const auto p = dtmf_peaks(x, 8000, mode);
      EXPECT_EQ(p.candidate, key);
      EXPECT_EQ(p.sample_updates, 3280u);
      for (std::size_t i = 0; i < 8; ++i) {
        EXPECT_DOUBLE_EQ(p.responses[i].power, goertzel_run(x, p.target_hz[i], 8000, mode).power);
        EXPECT_EQ(p.responses[i].sample_updates, 410u);
      }
    }
  }
  const auto p = dtmf_peaks(dtmf_signal('5', 410));
  EXPECT_EQ(p.row, 1u);
  EXPECT_EQ(p.column, 1u);
  EXPECT_EQ(p.responses[1].selected_bin, 39);
  EXPECT_EQ(p.responses[5].selected_bin, 68);
  EXPECT_NEAR(
      p.responses[1].power,
      double(direct_power(dtmf_signal('5', 410), p.responses[1].omega)),
      1e-7);
  EXPECT_EQ(dtmf_peaks({}).candidate, '1');
  EXPECT_EQ(dtmf_peaks(std::vector<double>(410, 0)).candidate, '1');
  EXPECT_EQ(dtmf_peaks({}).sample_updates, 0u);
  const auto single = dtmf_peaks(sinusoid_signal(410, 8000, {{770, 1, 0}}));
  EXPECT_EQ(single.row, 1u);
  EXPECT_NE(single.candidate, '\0'); // Candidate exists despite no valid second tone.
  auto copy               = p;
  copy.responses[0].power = 999;
  EXPECT_NE(p.responses[0].power, 999);
}
