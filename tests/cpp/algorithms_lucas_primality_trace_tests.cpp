#include <gtest/gtest.h>

#include <datamunge/algorithms/lucas_primality.hpp>
#include <datamunge/algorithms/modular_exponentiation.hpp>
#include <limits>
using namespace datamunge::algorithms;
namespace {
bool trial_prime(std::uint64_t n) {
  if (n < 2)
    return false;
  for (std::uint64_t d = 2; d <= n / d; ++d)
    if (n % d == 0)
      return false;
  return true;
}
} // namespace
TEST(LucasPrimalityTrace, AllSmallInputsAndActualCertificateRecords) {
  for (std::uint64_t n = 0; n <= 3000; ++n) {
    auto  t = lucas_primality_trace(n);
    auto& a = t.analysis;
    ASSERT_TRUE(a.conclusive);
    EXPECT_EQ(a.prime, trial_prime(n));
    EXPECT_EQ(a.prime, lucas_primality_test(n));
    EXPECT_EQ(t.captured_entries, 1 + t.factor_steps.size() + t.powers.size() + t.bases.size());
    EXPECT_EQ(a.work.trial_divisors, t.factor_steps.size());
    EXPECT_EQ(a.work.powers, t.powers.size());
    EXPECT_EQ(a.work.bases, t.bases.size());
    if (a.factorization_complete && n > 2) {
      unsigned __int128 product = 1;
      for (auto& f : a.factors) {
        EXPECT_TRUE(trial_prime(f.prime));
        for (unsigned k = 0; k < f.exponent; ++k)
          product *= f.prime;
      }
      EXPECT_EQ(product, n - 1);
    }
    for (auto& s : t.factor_steps) {
      auto remaining = s.before;
      for (unsigned k = 0; k < s.multiplicity; ++k) {
        EXPECT_EQ(remaining % s.divisor, 0);
        remaining /= s.divisor;
      }
      EXPECT_EQ(remaining, s.after);
      EXPECT_NE(s.after % s.divisor, 0);
    }
    for (auto& p : t.powers) {
      EXPECT_EQ(p.exponent, p.factor ? (n - 1) / p.factor : n - 1);
      EXPECT_EQ(p.residue, mod_pow(p.base, p.exponent, n));
      EXPECT_EQ(p.passed, p.factor ? p.residue != 1 : p.residue == 1);
    }
    if (a.prime && n > 2) {
      EXPECT_EQ(mod_pow(a.witness, n - 1, n), 1);
      for (auto& f : a.factors)
        EXPECT_NE(mod_pow(a.witness, (n - 1) / f.prime, n), 1);
      for (std::uint64_t base = 2; base < a.witness; ++base) {
        bool passes = mod_pow(base, n - 1, n) == 1;
        for (auto& f : a.factors)
          passes = passes && mod_pow(base, (n - 1) / f.prime, n) != 1;
        EXPECT_FALSE(passes);
      }
    }
    if (a.stop == LucasPrimalityStop::CompositeFermat)
      EXPECT_NE(mod_pow(a.composite_base, n - 1, n), 1);
  }
}
TEST(LucasPrimalityTrace, TinyOriginalAndFullWidthArithmetic) {
  auto tiny = lucas_primality_trace(7);
  EXPECT_EQ(tiny.analysis.witness, 3);
  ASSERT_EQ(tiny.bases.size(), 2);
  EXPECT_EQ(tiny.bases[0].rejected_factor, 2);
  EXPECT_TRUE(tiny.bases[1].accepted);
  EXPECT_EQ(tiny.powers[tiny.powers.size() - 2].residue, 6);
  EXPECT_EQ(tiny.powers.back().residue, 2);
  auto original = lucas_primality_trace(1009);
  EXPECT_EQ(original.analysis.witness, 11);
  EXPECT_EQ(original.analysis.factors.size(), 3);
  EXPECT_EQ(original.powers[original.powers.size() - 3].residue, 1008);
  EXPECT_EQ(original.powers[original.powers.size() - 2].residue, 374);
  EXPECT_EQ(original.powers.back().residue, 935);
  EXPECT_FALSE(lucas_primality_test(1001));
  const auto large = (std::uint64_t{1} << 61) - 1;
  auto       a     = lucas_primality_analyze(large);
  EXPECT_TRUE(a.conclusive);
  EXPECT_TRUE(a.prime);
  EXPECT_EQ(mod_pow(a.witness, large - 1, large), 1);
}
TEST(LucasPrimalityTrace, BudgetIsNotCompositeAndCaptureOwnership) {
  auto factor = lucas_primality_trace(7, 0, 100);
  EXPECT_EQ(factor.analysis.stop, LucasPrimalityStop::FactorBudget);
  EXPECT_FALSE(factor.analysis.conclusive);
  EXPECT_EQ(factor.analysis.remaining, 6);
  EXPECT_TRUE(factor.powers.empty());
  auto bases = lucas_primality_trace(7, 100, 0);
  EXPECT_TRUE(bases.analysis.factorization_complete);
  EXPECT_EQ(bases.analysis.stop, LucasPrimalityStop::BaseBudget);
  EXPECT_FALSE(bases.analysis.conclusive);
  auto limited = lucas_primality_trace(7, 100, 1);
  EXPECT_FALSE(limited.analysis.conclusive);
  EXPECT_EQ(limited.analysis.work.bases, 1);
  auto t    = lucas_primality_trace(7);
  auto size = t.captured_entries;
  EXPECT_THROW(lucas_primality_trace(7, 100, 100, size - 1), std::length_error);
  EXPECT_EQ(lucas_primality_trace(7, 100, 100, size).captured_entries, size);
  EXPECT_EQ(lucas_primality_trace(7, 100, 100, 0).analysis.witness, 3);
  EXPECT_EQ(lucas_primality_analyze(2, 0, 0).witness, 1);
  EXPECT_TRUE(lucas_primality_analyze(4, 0, 0).conclusive);
  EXPECT_THROW(lucas_primality_census(10, 9), std::invalid_argument);
  EXPECT_THROW(lucas_primality_census(0, 10000), std::invalid_argument);
  EXPECT_EQ(lucas_primality_census(0, 10).size(), 11);
}
