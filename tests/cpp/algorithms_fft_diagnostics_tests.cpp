#include <gtest/gtest.h>

#include <datamunge/algorithms/fft_diagnostics.hpp>
#include <limits>
#include <numeric>
#include <random>
namespace a = datamunge::algorithms;
TEST(FftDiagnostics, FourPointSignsAndInverseNormalization) {
  const std::vector<std::complex<double>> x        = {{1, 0}, {2, 0}, {3, 0}, {4, 0}},
                                          expected = {{10, 0}, {-2, 2}, {-2, 0}, {-2, -2}};
  const auto actual                                = a::fft(x);
  for (std::size_t k = 0; k < 4; ++k)
    EXPECT_NEAR(std::abs(actual[k] - expected[k]), 0, 1e-14);
  EXPECT_EQ(x[1], std::complex<double>(2, 0));
  const auto back = a::fft(actual, true);
  for (std::size_t i = 0; i < 4; ++i)
    EXPECT_NEAR(std::abs(back[i] - x[i]), 0, 1e-14);
  EXPECT_TRUE(a::fft({}).empty());
  EXPECT_EQ(a::fft({{3, -2}})[0], std::complex<double>(3, -2));
  EXPECT_THROW((void)a::fft_real({1, 2, 3}), std::invalid_argument);
}
TEST(FftDiagnostics, DirectReferenceAcceptsNonRadixLengthsAndRandomComplexSignals) {
  const auto three = a::dft_reference({{1, 0}, {1, 0}, {1, 0}});
  EXPECT_NEAR(three[0].real(), 3, 1e-14);
  EXPECT_LT(std::abs(three[1]), 1e-14);
  std::mt19937_64                        rng(17);
  std::uniform_real_distribution<double> sample(-1, 1);
  for (std::size_t n : {1, 2, 4, 8, 16, 32, 64, 128, 256}) {
    std::vector<std::complex<double>> x(n);
    for (auto& v : x)
      v = {sample(rng), sample(rng)};
    const auto stats = a::fft_diagnostics(x);
    EXPECT_LT(stats.double_at("relative_dft_error", 0), 2e-13);
    EXPECT_LT(stats.double_at("relative_inverse_error", 0), 2e-13);
    EXPECT_LT(stats.double_at("relative_energy_error", 0), 2e-13);
    const auto inverse = a::dft_reference(a::fft(x), true);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_NEAR(std::abs(inverse[i] - x[i]), 0, 1e-12);
  }
}
TEST(FftDiagnostics, OneSidedAmplitudesIncludeUndoubledDcAndNyquist) {
  std::vector<double> x(8);
  for (int i = 0; i < 8; ++i)
    x[i] = 2 + 3 * std::cos(2 * std::numbers::pi_v<double> * i / 8) + .5 * (i % 2 ? -1 : 1);
  const auto s = a::fft_spectrum(x, 8);
  ASSERT_EQ(s.nrows(), 5);
  EXPECT_NEAR(s.double_at("amplitude", 0), 2, 1e-14);
  EXPECT_NEAR(s.double_at("amplitude", 1), 3, 1e-14);
  EXPECT_NEAR(s.double_at("amplitude", 4), .5, 1e-14);
  double power = 0;
  for (auto v : s.numeric_column("mean_square").values())
    power += v;
  EXPECT_NEAR(power, 8.75, 1e-13);
  EXPECT_EQ(s.double_at("frequency", 4), 4);
  const auto singleton = a::fft_spectrum({-3}, 2);
  EXPECT_EQ(singleton.nrows(), 1);
  EXPECT_EQ(singleton.double_at("amplitude", 0), 3);
  EXPECT_EQ(singleton.double_at("mean_square", 0), 9);
}
TEST(FftDiagnostics, PaddingPreservesCoherentAmplitudeAndPeriodicHannEnergy) {
  std::vector<double> x(64);
  for (int i = 0; i < 64; ++i)
    x[i] = 2 * std::cos(2 * std::numbers::pi_v<double> * 5 * i / 64);
  for (auto w : {a::FftWindow::Rectangular, a::FftWindow::PeriodicHann}) {
    const auto basic = a::fft_spectrum(x, 64, 0, w), padded = a::fft_spectrum(x, 64, 512, w);
    EXPECT_NEAR(basic.double_at("amplitude", 5), 2, 1e-13);
    EXPECT_NEAR(padded.double_at("amplitude", 40), 2, 1e-13);
    for (std::size_t k = 0; k < basic.nrows(); ++k) {
      EXPECT_NEAR(basic.double_at("real", k), padded.double_at("real", 8 * k), 1e-12);
      EXPECT_NEAR(basic.double_at("imaginary", k), padded.double_at("imaginary", 8 * k), 1e-12);
    }
    long double sum = 0, energy = 0;
    for (int i = 0; i < 64; ++i) {
      long double weight = w == a::FftWindow::Rectangular
                               ? 1
                               : .5 - .5 * std::cos(2 * std::numbers::pi_v<double> * i / 64);
      sum += weight * weight;
      energy += weight * weight * x[i] * x[i];
    }
    double measured = 0;
    for (auto v : padded.numeric_column("mean_square").values())
      measured += v;
    EXPECT_NEAR(measured, double(energy / sum), 1e-13);
  }
}
TEST(FftDiagnostics, LinearConvolutionMatchesIndependentDirectSums) {
  EXPECT_EQ(a::fft_linear_convolution({}, {1, 2}).size(), 0);
  const auto                simple = a::fft_linear_convolution({1, 2, 3, 4}, {1, 1});
  const std::vector<double> exact  = {1, 3, 5, 7, 4};
  for (std::size_t i = 0; i < 5; ++i)
    EXPECT_NEAR(simple[i], exact[i], 1e-13);
  for (int na : {1, 3, 8, 17})
    for (int nb : {1, 2, 5, 11}) {
      std::vector<double> x(na), y(nb), expected(na + nb - 1, 0);
      for (int i = 0; i < na; ++i)
        x[i] = std::sin(i + .3);
      for (int j = 0; j < nb; ++j)
        y[j] = std::cos(j + .7);
      for (int i = 0; i < na; ++i)
        for (int j = 0; j < nb; ++j)
          expected[i + j] += x[i] * y[j];
      const auto actual = a::fft_linear_convolution(x, y);
      ASSERT_EQ(actual.size(), expected.size());
      for (std::size_t i = 0; i < actual.size(); ++i)
        EXPECT_NEAR(actual[i], expected[i], 2e-13);
    }
}
TEST(FftDiagnostics, DomainNonfiniteAndEmptyCasesAreExplicit) {
  EXPECT_EQ(a::fft_diagnostics({}).double_at("length", 0), 0);
  EXPECT_EQ(
      a::fft_diagnostics(std::vector<std::complex<double>>(8))
          .double_at("relative_energy_error", 0),
      0);
  EXPECT_THROW((void)a::fft_spectrum({}), std::invalid_argument);
  EXPECT_THROW((void)a::fft_spectrum({1}, 0), std::invalid_argument);
  EXPECT_THROW((void)a::fft_spectrum({1, 2, 3}), std::invalid_argument);
  EXPECT_THROW((void)a::fft_spectrum({1, 2, 3}, 1, 2), std::invalid_argument);
  EXPECT_THROW((void)a::fft_spectrum({1}, 1, 1, a::FftWindow::PeriodicHann), std::invalid_argument);
  EXPECT_THROW(
      (void)a::fft_spectrum({1}, 1, 1, static_cast<a::FftWindow>(99)), std::invalid_argument);
  const double inf = std::numeric_limits<double>::infinity();
  EXPECT_THROW((void)a::fft_spectrum({inf}), std::invalid_argument);
  EXPECT_THROW((void)a::dft_reference({{0, inf}}), std::invalid_argument);
  EXPECT_THROW((void)a::fft_linear_convolution({1}, {inf}), std::invalid_argument);
  EXPECT_THROW((void)a::fft_diagnostics({{1e308, 0}, {1e308, 0}}), std::invalid_argument);
  EXPECT_THROW((void)a::fft_spectrum({1e308, 1e308}), std::invalid_argument);
}
