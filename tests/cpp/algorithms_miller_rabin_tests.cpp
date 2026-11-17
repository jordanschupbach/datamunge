#include <gtest/gtest.h>

#include <cstdint>
#include <datamunge/algorithms/miller_rabin.hpp>
#include <vector>

using datamunge::algorithms::is_probable_prime;
using datamunge::algorithms::is_strong_probable_prime_base;

namespace {

// A dead-simple, obviously-correct primality oracle to cross-check against.
bool trial_division_prime(std::uint64_t n) {
  if (n < 2)
    return false;
  for (std::uint64_t i = 2; i * i <= n; ++i)
    if (n % i == 0)
      return false;
  return true;
}

} // namespace

TEST(MillerRabin, KnownSmallPrimesAndComposites) {
  for (std::uint64_t p : {2ULL, 3ULL, 5ULL, 7ULL, 97ULL, 7919ULL})
    EXPECT_TRUE(is_probable_prime(p)) << p;
  for (std::uint64_t c : {0ULL, 1ULL, 4ULL, 9ULL, 15ULL, 561ULL, 1105ULL})
    EXPECT_FALSE(is_probable_prime(c)) << c;
}

TEST(MillerRabin, CarmichaelNumbersAreReportedComposite) {
  // Carmichael numbers satisfy Fermat's congruence a^(n-1) == 1 for every coprime base, fooling the
  // Fermat test -- but Miller-Rabin exposes them.
  for (std::uint64_t c : {561ULL, 1105ULL, 1729ULL, 2465ULL, 6601ULL, 8911ULL, 10585ULL, 15841ULL})
    EXPECT_FALSE(is_probable_prime(c)) << c;
}

TEST(MillerRabin, LargePrimesTrueNearbyCompositesFalse) {
  EXPECT_TRUE(is_probable_prime(1000000007ULL));
  EXPECT_TRUE(is_probable_prime(2147483647ULL));     // 2^31 - 1, a Mersenne prime
  EXPECT_TRUE(is_probable_prime(67280421310721ULL)); // a known 14-digit prime

  EXPECT_FALSE(is_probable_prime(1000000005ULL));     // divisible by 5
  EXPECT_FALSE(is_probable_prime(1000000008ULL));     // even
  EXPECT_FALSE(is_probable_prime(2147483645ULL));     // divisible by 5
  EXPECT_FALSE(is_probable_prime(2147483649ULL));     // 3 * 715827883
  EXPECT_FALSE(is_probable_prime(67280421310725ULL)); // divisible by 5
}

TEST(MillerRabin, VeryLarge64BitValuesAreOverflowSafe) {
  // Bases squared near these magnitudes overflow 64 bits; the __int128 mulmod must stay exact.
  EXPECT_TRUE(is_probable_prime(9223372036854775783ULL));   // largest prime below 2^63
  EXPECT_FALSE(is_probable_prime(9223372036854775807ULL));  // 2^63 - 1 = 7^2 * 73 * 127 * 337 * ...
  EXPECT_TRUE(is_probable_prime(18446744073709551557ULL));  // largest prime below 2^64
  EXPECT_FALSE(is_probable_prime(18446744073709551615ULL)); // 2^64 - 1 = 3 * 5 * 17 * 257 * ...
}

TEST(MillerRabin, AgreesWithTrialDivisionUpTo100000) {
  for (std::uint64_t n = 0; n <= 100000ULL; ++n)
    ASSERT_EQ(is_probable_prime(n), trial_division_prime(n)) << n;
}

TEST(MillerRabin, SingleBaseRoundCanBeFooledButFullTestCannot) {
  // 2047 = 23 * 89 is the smallest strong pseudoprime to base 2: one round with base 2 passes...
  EXPECT_TRUE(is_strong_probable_prime_base(2047ULL, 2ULL));
  // ...but base 3 is a witness, so the deterministic multi-base test rejects it.
  EXPECT_FALSE(is_strong_probable_prime_base(2047ULL, 3ULL));
  EXPECT_FALSE(is_probable_prime(2047ULL));

  // Base 2 already exposes the Carmichael number 561.
  EXPECT_FALSE(is_strong_probable_prime_base(561ULL, 2ULL));
  // A genuine prime is a strong probable prime to every base.
  for (std::uint64_t a : {2ULL, 3ULL, 5ULL, 7ULL, 11ULL})
    EXPECT_TRUE(is_strong_probable_prime_base(97ULL, a)) << a;
}

TEST(MillerRabin, EdgeCasesAroundZeroOneAndTwo) {
  EXPECT_FALSE(is_probable_prime(0ULL));
  EXPECT_FALSE(is_probable_prime(1ULL));
  EXPECT_TRUE(is_probable_prime(2ULL));
  EXPECT_TRUE(is_probable_prime(3ULL));
  EXPECT_FALSE(is_probable_prime(4ULL));
  EXPECT_FALSE(is_strong_probable_prime_base(1ULL, 2ULL));
  EXPECT_FALSE(is_strong_probable_prime_base(0ULL, 2ULL));
}

TEST(MillerRabin, ActualHistoriesMatchIndependentSmallIntegerPowering) {
  using namespace datamunge::algorithms;
  for (std::uint64_t n = 0; n <= 400; ++n)
    for (std::uint64_t a = 0; a <= 50; ++a) {
      auto t = miller_rabin_round_trace(n, a);
      EXPECT_EQ(t.passed, is_strong_probable_prime_base(n, a));
      EXPECT_EQ(t.n, n);
      EXPECT_EQ(t.base, a);
      if (n < 3 || n % 2 == 0 || a % n == 0) {
        EXPECT_TRUE(t.residues.empty());
        continue;
      }
      EXPECT_EQ(t.d << t.s, n - 1);
      EXPECT_EQ(t.d % 2, 1);
      std::uint64_t x = 1;
      for (std::uint64_t k = 0; k < t.d; ++k)
        x = x * (a % n) % n;
      ASSERT_FALSE(t.residues.empty());
      EXPECT_EQ(t.residues[0], x);
      bool        pass  = x == 1 || x == n - 1;
      std::size_t count = 1;
      for (unsigned r = 1; r < t.s && !pass; ++r) {
        x = x * x % n;
        ASSERT_LT(count, t.residues.size());
        EXPECT_EQ(t.residues[count++], x);
        pass = x == n - 1;
      }
      EXPECT_EQ(count, t.residues.size());
      EXPECT_EQ(pass, t.passed);
      EXPECT_EQ(t.powering.result, t.residues.front());
      auto cap = t.powering.steps.size() + t.s;
      EXPECT_NO_THROW(miller_rabin_round_trace(n, a, cap));
      if (cap > 1)
        EXPECT_THROW(miller_rabin_round_trace(n, a, cap - 1), std::length_error);
    }
}
TEST(MillerRabin, FullWidthCaptureGuardsAndOwnership) {
  using namespace datamunge::algorithms;
  auto t = miller_rabin_round_trace(18446744073709551557ULL, 2);
  EXPECT_TRUE(t.passed);
  EXPECT_LE(t.powering.steps.size() + t.residues.size(), 64U);
  auto owned   = t.residues;
  auto another = miller_rabin_round_trace(561, 2);
  EXPECT_EQ(t.residues, owned);
  EXPECT_FALSE(another.passed);
  EXPECT_TRUE(miller_rabin_round_trace(2047, 2).passed);
  EXPECT_FALSE(miller_rabin_round_trace(2047, 3).passed);
  EXPECT_TRUE(miller_rabin_round_trace(9, 18, 1).passed);
  EXPECT_TRUE(miller_rabin_round_trace(2, 0, 1).passed);
  EXPECT_THROW(miller_rabin_round_trace(561, 2, 1), std::length_error);
  EXPECT_NO_THROW(miller_rabin_round_trace(561, 2, 0));
}
