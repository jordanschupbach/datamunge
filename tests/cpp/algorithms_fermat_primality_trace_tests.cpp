#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <datamunge/algorithms/primality.hpp>
#include <numeric>
#include <random>
#include <stdexcept>
namespace {
using namespace datamunge::algorithms;
// Slow repeated multiplication, independent of the binary powering implementation.
std::uint64_t naive(std::uint64_t n, std::uint64_t a) {
  std::uint64_t r = 1;
  for (std::uint64_t i = 0; i < n - 1; ++i)
    r = (r * a) % n;
  return r;
}
void same(const FermatPrimalityOutcome& a, const FermatPrimalityOutcome& b) {
  EXPECT_EQ(a.n, b.n);
  EXPECT_EQ(a.seed, b.seed);
  EXPECT_EQ(a.requested_rounds, b.requested_rounds);
  EXPECT_EQ(a.completed_rounds, b.completed_rounds);
  EXPECT_EQ(a.probable_prime, b.probable_prime);
  EXPECT_EQ(a.stop, b.stop);
  EXPECT_EQ(a.last_base, b.last_base);
  EXPECT_EQ(a.last_gcd, b.last_gcd);
  EXPECT_EQ(a.last_residue, b.last_residue);
}
TEST(FermatPrimalityTrace, TinyActualBases) {
  const auto r = fermat_base_test(15, 2);
  EXPECT_EQ(r.residue, 4);
  EXPECT_EQ(r.common_divisor, 1);
  EXPECT_FALSE(r.passes);
  EXPECT_TRUE(fermat_base_test(15, 4).passes);
  EXPECT_EQ(fermat_base_test(561, 3).common_divisor, 3);
  EXPECT_FALSE(fermat_base_test(561, 3).passes);
  EXPECT_TRUE(fermat_base_test(561, 2).passes);
  const std::uint64_t residues[] = {1, 4, 9, 1, 10, 6, 4, 4, 6, 10, 1, 9, 4, 1};
  for (std::uint64_t a = 1; a < 15; ++a)
    EXPECT_EQ(fermat_base_test(15, a).residue, residues[a - 1]);
}
TEST(FermatPrimalityTrace, ExhaustiveNaivePowers) {
  for (std::uint64_t n = 2; n <= 256; ++n)
    for (std::uint64_t a = 1; a < n; ++a) {
      const auto r = fermat_base_test(n, a);
      EXPECT_EQ(r.n, n);
      EXPECT_EQ(r.base, a);
      EXPECT_EQ(r.residue, naive(n, a));
      EXPECT_EQ(r.common_divisor, std::gcd(n, a));
      EXPECT_EQ(r.passes, r.residue == 1);
      if (r.common_divisor > 1)
        EXPECT_FALSE(r.passes);
    }
}
TEST(FermatPrimalityTrace, CensusIntervalsAndPartitions) {
  for (std::uint64_t n = 2; n <= 512; ++n) {
    const auto all = fermat_base_census(n, 1, n), left = fermat_base_census(n, 1, n / 2),
               right = fermat_base_census(n, n / 2, n);
    EXPECT_EQ(all.n, n);
    EXPECT_EQ(all.first, 1);
    EXPECT_EQ(all.last, n);
    EXPECT_EQ(all.total, n - 1);
    EXPECT_EQ(all.total, left.total + right.total);
    EXPECT_EQ(all.coprime, left.coprime + right.coprime);
    EXPECT_EQ(all.passes, left.passes + right.passes);
    std::uint64_t units = 0, passes = 0;
    for (std::uint64_t a = 1; a < n; ++a) {
      units += std::gcd(a, n) == 1;
      passes += naive(n, a) == 1;
    }
    EXPECT_EQ(all.coprime, units);
    EXPECT_EQ(all.passes, passes);
    const auto empty = fermat_base_census(n, n, n);
    EXPECT_EQ(empty.total, 0);
    EXPECT_EQ(empty.coprime, 0);
    EXPECT_EQ(empty.passes, 0);
  }
}
TEST(FermatPrimalityTrace, KnownCarmichaelAndDenominators) {
  for (auto n : {561ULL, 1105ULL, 1729ULL, 2465ULL, 2821ULL, 6601ULL}) {
    const auto c = fermat_base_census(n, 1, n);
    EXPECT_EQ(c.passes, c.coprime);
    EXPECT_LT(c.passes, c.total);
  }
  const auto original = fermat_base_census(561, 2, 561), sampler = fermat_base_census(561, 2, 560);
  EXPECT_EQ(original.total, 559);
  EXPECT_EQ(original.coprime, 319);
  EXPECT_EQ(original.passes, 319);
  EXPECT_EQ(sampler.total, 558);
  EXPECT_EQ(sampler.coprime, 318);
  EXPECT_EQ(sampler.passes, 318);
  const auto c = fermat_base_census(91, 2, 91);
  EXPECT_EQ(c.total, 89);
  EXPECT_EQ(c.coprime, 71);
  EXPECT_EQ(c.passes, 35);
}
TEST(FermatPrimalityTrace, SeededActualPrefixes) {
  const std::uint64_t words[] = {
      0xe220a8397b1dcdafULL, 0x6e789e6aa1b965f4ULL, 0x06c45d188009454fULL};
  const auto t = fermat_probable_prime_trace(97, 3, 0, 3);
  ASSERT_EQ(t.rows.size(), 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(t.rows[i].base, 2 + words[i] % 94);
    EXPECT_EQ(t.rows[i].residue, 1);
    EXPECT_EQ(t.rows[i].common_divisor, 1);
  }
  for (int cap = 1; cap <= 3; ++cap) {
    const auto p = fermat_probable_prime_trace(97, cap, 0);
    for (int j = 0; j < cap; ++j)
      EXPECT_EQ(p.rows[j].base, t.rows[j].base);
  }
  EXPECT_EQ(t.outcome.completed_rounds, 3);
  EXPECT_EQ(t.outcome.stop, FermatPrimalityStop::RoundLimit);
}
TEST(FermatPrimalityTrace, SeededWorkTraceParity) {
  std::mt19937_64 rng(107);
  for (int i = 0; i < 3000; ++i) {
    const auto n = i < 1000 ? std::uint64_t(i) : rng(), seed = rng();
    const int  rounds = 1 + int(rng() % 30);
    const auto t      = fermat_probable_prime_trace(n, rounds, seed, 0);
    same(t.outcome, fermat_probable_prime_work(n, rounds, seed));
    EXPECT_EQ(t.outcome.probable_prime, fermat_probable_prime(n, rounds, seed));
    EXPECT_EQ(t.rows.size(), t.captured_entries);
    EXPECT_EQ(t.rows.size(), t.outcome.completed_rounds);
    for (std::size_t j = 0; j < t.rows.size(); ++j) {
      const auto& r = t.rows[j];
      EXPECT_GE(r.base, 2);
      EXPECT_LE(r.base, n - 2);
      const auto check = fermat_base_test(n, r.base);
      EXPECT_EQ(check.residue, r.residue);
      EXPECT_EQ(check.common_divisor, r.common_divisor);
      EXPECT_EQ(check.passes, r.passes);
      if (j + 1 < t.rows.size())
        EXPECT_TRUE(r.passes);
    }
    if (!t.rows.empty()) {
      const auto& r = t.rows.back();
      EXPECT_EQ(t.outcome.last_base, r.base);
      EXPECT_EQ(t.outcome.last_residue, r.residue);
      EXPECT_EQ(t.outcome.last_gcd, r.common_divisor);
      EXPECT_EQ(r.passes, t.outcome.probable_prime);
    }
  }
}
TEST(FermatPrimalityTrace, FullWidthAndSmallFilters) {
  for (std::uint64_t n : {0, 1, 2, 3, 4, 6}) {
    auto t = fermat_probable_prime_trace(n, INT_MAX, 0, 1);
    EXPECT_TRUE(t.rows.empty());
    EXPECT_EQ(t.outcome.completed_rounds, 0);
    EXPECT_EQ(t.outcome.last_base, 0);
    EXPECT_EQ(t.outcome.last_residue, 0);
    EXPECT_EQ(t.outcome.last_gcd, 0);
    EXPECT_EQ(t.outcome.probable_prime, n == 2 || n == 3);
  }
  const auto p = fermat_probable_prime_work(UINT64_MAX - 58, 64, UINT64_MAX);
  EXPECT_TRUE(p.probable_prime);
  EXPECT_EQ(p.completed_rounds, 64);
  const auto c = fermat_probable_prime_work(UINT64_MAX, INT_MAX);
  EXPECT_FALSE(c.probable_prime);
  EXPECT_GT(c.completed_rounds, 0);
  const auto last = fermat_base_test(UINT64_MAX, UINT64_MAX - 1);
  EXPECT_EQ(last.residue, 1);
  EXPECT_TRUE(last.passes);
  const auto census = fermat_base_census(UINT64_MAX, UINT64_MAX - 1, UINT64_MAX, 1);
  EXPECT_EQ(census.total, 1);
  EXPECT_EQ(census.passes, 1);
  EXPECT_EQ(census.coprime, 1);
}
TEST(FermatPrimalityTrace, QuotasAndOwnership) {
  EXPECT_THROW(fermat_probable_prime_trace(97, 3, 0, 2), std::length_error);
  EXPECT_NO_THROW(fermat_probable_prime_trace(97, 3, 0, 3));
  EXPECT_NO_THROW(fermat_probable_prime_trace(97, 3, 0, 0));
  EXPECT_NO_THROW(fermat_probable_prime_work(97, 3, 0));
  const auto early = fermat_probable_prime_trace(15, 20, 0, 0);
  ASSERT_GT(early.rows.size(), 0);
  EXPECT_NO_THROW(fermat_probable_prime_trace(15, 20, 0, early.rows.size()));
  if (early.rows.size() > 1)
    EXPECT_THROW(fermat_probable_prime_trace(15, 20, 0, early.rows.size() - 1), std::length_error);
  auto       a        = fermat_probable_prime_trace(97, 3, 0);
  const auto copy     = a;
  a.rows.front().base = 0;
  a.outcome.n         = 0;
  EXPECT_NE(copy.rows.front().base, 0);
  EXPECT_EQ(copy.outcome.n, 97);
  EXPECT_EQ(copy.outcome.completed_rounds, 3);
  EXPECT_THROW(fermat_base_census(561, 1, 561, 559), std::length_error);
  EXPECT_NO_THROW(fermat_base_census(561, 1, 561, 560));
  EXPECT_NO_THROW(fermat_base_census(561, 1, 561, 0));
  EXPECT_THROW(fermat_base_census(UINT64_MAX, 1, UINT64_MAX), std::length_error);
}
TEST(FermatPrimalityTrace, DomainNamesAndSignature) {
  for (int rounds : {INT_MIN, -1, 0})
    for (std::uint64_t n : {0, 2, 4, 9, 97}) {
      EXPECT_THROW(fermat_probable_prime(n, rounds), std::invalid_argument);
      EXPECT_THROW(fermat_probable_prime_work(n, rounds), std::invalid_argument);
      EXPECT_THROW(fermat_probable_prime_trace(n, rounds), std::invalid_argument);
    }
  for (auto n : {0ULL, 1ULL})
    EXPECT_THROW(fermat_base_test(n, 1), std::invalid_argument);
  EXPECT_THROW(fermat_base_test(7, 0), std::invalid_argument);
  EXPECT_THROW(fermat_base_test(7, 7), std::invalid_argument);
  EXPECT_THROW(fermat_base_census(1, 1, 1), std::invalid_argument);
  EXPECT_THROW(fermat_base_census(7, 0, 1), std::invalid_argument);
  EXPECT_THROW(fermat_base_census(7, 4, 3), std::invalid_argument);
  EXPECT_THROW(fermat_base_census(7, 1, 8), std::invalid_argument);
  EXPECT_EQ(fermat_primality_stop_name(FermatPrimalityStop::BelowTwo), "below two");
  EXPECT_EQ(fermat_primality_stop_name(FermatPrimalityStop::SmallPrime), "small prime");
  EXPECT_EQ(fermat_primality_stop_name(FermatPrimalityStop::Even), "even");
  EXPECT_EQ(fermat_primality_stop_name(FermatPrimalityStop::Witness), "witness");
  EXPECT_EQ(fermat_primality_stop_name(FermatPrimalityStop::RoundLimit), "rounds passed");
  EXPECT_EQ(fermat_primality_stop_name(static_cast<FermatPrimalityStop>(99)), "unknown");
  bool (*legacy)(std::uint64_t, int, std::uint64_t) = &fermat_probable_prime;
  EXPECT_TRUE(legacy(97, 20, 1));
}
TEST(FermatPrimalityTrace, FirstWitnessAndDefaultFixtures) {
  for (std::uint64_t n : {97, 91, 561}) {
    const auto t = fermat_probable_prime_trace(n);
    EXPECT_EQ(t.outcome.probable_prime, n == 97);
    if (n != 97) {
      EXPECT_EQ(t.outcome.stop, FermatPrimalityStop::Witness);
      ASSERT_FALSE(t.rows.empty());
      EXPECT_FALSE(t.rows.back().passes);
    }
  }
  const auto all = fermat_base_census(45, 2, 45);
  EXPECT_EQ(all.coprime, 23);
  EXPECT_EQ(all.passes, 7);
}
} // namespace
