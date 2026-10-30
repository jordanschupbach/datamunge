#include <datamunge/algorithms/bernstein_vazirani.hpp>
#include <datamunge/algorithms/fourier_sampling.hpp>

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <random>
#include <vector>

namespace alg = datamunge::algorithms;

TEST(FourierSampling, ParityOracleGivesTheSecretWithCertainty) {
  for (int n = 1; n <= 8; ++n)
    for (std::size_t s = 0; s < (std::size_t{1} << n); s += 7) {
      const auto psi = alg::fourier_sampling_state(alg::parity_truth_table(n, s));
      EXPECT_NEAR(std::norm(psi[s]), 1.0, 1e-12);
      EXPECT_EQ(alg::bernstein_vazirani(n, s), s);
      EXPECT_EQ(alg::bernstein_vazirani_classical(n, [&](std::size_t x) { return std::popcount(x & s) & 1; }), s);
    }
}

TEST(FourierSampling, CircuitAmplitudesEqualTheWalshSpectrum) {
  std::mt19937 rng(9);
  std::vector<int> f(256);
  for (auto& v : f) v = static_cast<int>(rng() & 1);
  const auto psi = alg::fourier_sampling_state(f);
  const auto F = alg::walsh_spectrum(f);
  double parseval = 0.0;
  for (std::size_t s = 0; s < F.size(); ++s) {
    EXPECT_NEAR(psi[s].real(), F[s], 1e-12);
    EXPECT_NEAR(psi[s].imag(), 0.0, 1e-12);
    parseval += F[s] * F[s];
    // Direct definition.
    double direct = 0.0;
    for (std::size_t x = 0; x < f.size(); ++x) direct += ((f[x] + std::popcount(x & s)) & 1) ? -1.0 : 1.0;
    EXPECT_NEAR(F[s], direct / 256.0, 1e-12);
  }
  EXPECT_NEAR(parseval, 1.0, 1e-12);
  EXPECT_EQ(alg::walsh_spectrum({0, 0, 0, 1}), (std::vector<double>{0.5, 0.5, 0.5, -0.5}));
  EXPECT_THROW((void)alg::walsh_spectrum({0, 1, 1}), std::invalid_argument);
}

TEST(FourierSampling, ShotFrequenciesMatchProbabilities) {
  const auto psi = alg::fourier_sampling_state({0, 0, 0, 1});  // each outcome 1/4
  const auto shots = alg::measure_shots(psi, 40000, 3);
  std::vector<double> count(4, 0.0);
  for (auto o : shots) count[o] += 1.0;
  for (double c : count) EXPECT_NEAR(c / 40000.0, 0.25, 0.01);
  EXPECT_EQ(shots, alg::measure_shots(psi, 40000, 3));
}
