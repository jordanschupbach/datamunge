#include <datamunge/algorithms/aks.hpp>
#include <datamunge/algorithms/number_theory_advanced.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

namespace alg = datamunge::algorithms;

TEST(Aks, OrderPerfectPowerAndR) {
  EXPECT_EQ(alg::multiplicative_order(2, 7), 3U);
  EXPECT_EQ(alg::multiplicative_order(10, 7), 6U);
  EXPECT_EQ(alg::multiplicative_order(4, 6), 0U);
  EXPECT_EQ(alg::perfect_power(729), (std::pair<std::uint64_t, int>{3, 6}));
  EXPECT_EQ(alg::perfect_power(1ULL << 62), (std::pair<std::uint64_t, int>{2, 62}));
  EXPECT_EQ(alg::perfect_power(4294967311ULL * 3), (std::pair<std::uint64_t, int>{4294967311ULL * 3, 1}));
  EXPECT_EQ(alg::perfect_power(4294967291ULL * 4294967291ULL).second, 2);
  for (std::uint64_t n : {31ULL, 1000003ULL, 4294967291ULL}) {
    const auto r = alg::aks_r(n);
    const double bound = std::pow(std::log2(static_cast<double>(n)), 2);
    EXPECT_GT(static_cast<double>(alg::multiplicative_order(n, r)), bound);
    for (std::uint64_t s = 2; s < r; ++s)  // r is the least such modulus
      EXPECT_FALSE(std::gcd(n % s, s) == 1 && alg::multiplicative_order(n, s) > bound);
    EXPECT_LE(static_cast<double>(r), std::max(3.0, std::ceil(std::pow(std::log2(static_cast<double>(n)), 5))));
  }
}

TEST(Aks, RingPowerMatchesBinomialExpansion) {
  // (x+1)^7 mod (x^5 - 1, 7) = x^7 + 1 = x^2 + 1.
  EXPECT_EQ(alg::aks_power(1, 7, 5), (std::vector<std::uint64_t>{1, 0, 1, 0, 0}));
  // n = 6, r large: coefficients are C(6,k) * 2^(6-k) mod 6.
  const auto c = alg::aks_power(2, 6, 10);
  const std::vector<std::uint64_t> binom{1, 6, 15, 20, 15, 6, 1};
  for (int k = 0; k <= 6; ++k)
    EXPECT_EQ(c[k], binom[k] * (1ULL << (6 - k)) % 6) << k;
  EXPECT_TRUE(alg::aks_congruence_holds(13, 7, 3));
  EXPECT_FALSE(alg::aks_congruence_holds(561, 7, 1));
}

TEST(Aks, AgreesWithTrialDivisionAndTracesSteps) {
  auto prime = [](std::uint64_t n) {
    if (n < 2) return false;
    for (std::uint64_t d = 2; d * d <= n; ++d)
      if (n % d == 0) return false;
    return true;
  };
  for (std::uint64_t n = 0; n <= 1500; ++n) EXPECT_EQ(alg::aks_is_prime(n), prime(n)) << n;
  for (std::uint64_t n : {561ULL, 1105ULL, 1729ULL, 2465ULL, 2821ULL, 6601ULL, 8911ULL, 1000003ULL})
    EXPECT_EQ(alg::aks_is_prime(n), prime(n)) << n;
  const auto big = alg::aks_trace(1000003);
  EXPECT_EQ(big.decided_by, "all congruences hold");
  EXPECT_EQ(big.congruences_checked, big.a_max);
  EXPECT_EQ(alg::aks_trace(343).decided_by, "perfect power");
  EXPECT_EQ(alg::aks_trace(7).decided_by, "n <= r");
}

TEST(Aks, TotientAndPlanOnly) {
  EXPECT_EQ(alg::euler_totient(1), 1U);
  EXPECT_EQ(alg::euler_totient(36), 12U);
  EXPECT_EQ(alg::euler_totient(97), 96U);
  const auto plan = alg::aks_trace(1000003, false);
  EXPECT_EQ(plan.decided_by, "congruences not checked");
  EXPECT_EQ(plan.congruences_checked, 0U);
  EXPECT_EQ(plan.a_max, alg::aks_trace(1000003).a_max);
}
