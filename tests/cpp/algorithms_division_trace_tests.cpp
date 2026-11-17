#include <gtest/gtest.h>

#include <bit>
#include <datamunge/algorithms/division.hpp>
#include <limits>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
namespace {
unsigned __int128 magnitude(DivisionWideValue v) {
  return (static_cast<unsigned __int128>(v.high) << 64) | v.low;
}
__int128 signed_value(DivisionWideValue v) {
  return v.negative ? -static_cast<__int128>(magnitude(v)) : static_cast<__int128>(magnitude(v));
}
} // namespace
TEST(DivisionTrace, FullWidthIndependentBuiltinAndActualNonrestoringRegisters) {
  std::mt19937_64 rng(254263);
  for (int i = 0; i < 3000; ++i) {
    std::uint64_t n = rng(), d = rng();
    if (!d)
      d = 1;
    if (i < 64) {
      n = std::numeric_limits<std::uint64_t>::max();
      d = i + 1;
    }
    auto t = non_restoring_division_trace(n, d);
    EXPECT_EQ(t.analysis.result.quotient, n / d);
    EXPECT_EQ(t.analysis.result.remainder, n % d);
    EXPECT_EQ(t.steps.size(), 64);
    EXPECT_EQ(t.captured_entries, 70);
    __int128      previous = 0;
    std::uint64_t q        = n;
    std::size_t   adds = 0, subs = 0;
    for (auto& s : t.steps) {
      EXPECT_EQ(signed_value(s.before), previous);
      EXPECT_EQ(s.quotient_before, q);
      EXPECT_EQ(s.incoming, (q >> 63) != 0);
      auto shifted = previous * 2 + int(s.incoming);
      EXPECT_EQ(signed_value(s.shifted), shifted);
      EXPECT_EQ(s.subtracted, shifted >= 0);
      auto after = shifted + (s.subtracted ? -static_cast<__int128>(d) : static_cast<__int128>(d));
      EXPECT_EQ(signed_value(s.after), after);
      EXPECT_GE(after, -static_cast<__int128>(d));
      EXPECT_LT(after, static_cast<__int128>(d));
      EXPECT_EQ(s.output_bit, after >= 0);
      q = (q << 1) | std::uint64_t(after >= 0);
      EXPECT_EQ(s.quotient_after, q);
      previous = after;
      if (s.subtracted)
        ++subs;
      else
        ++adds;
    }
    EXPECT_EQ(t.final_corrected, previous < 0);
    EXPECT_EQ(
        signed_value(t.final_after), previous + (previous < 0 ? static_cast<__int128>(d) : 0));
    EXPECT_EQ(t.analysis.additions, adds + t.final_corrected);
    EXPECT_EQ(t.analysis.subtractions, subs);
    EXPECT_EQ(t.analysis.additions + t.analysis.subtractions, 64 + t.final_corrected);
    EXPECT_EQ(t.analysis.comparison_tests, 129);
    auto r = restoring_division_trace(n, d);
    EXPECT_EQ(r.analysis.result.quotient, n / d);
    EXPECT_EQ(r.analysis.result.remainder, n % d);
    EXPECT_EQ(r.analysis.subtractions, std::popcount(n / d));
    EXPECT_EQ(r.analysis.additions, 0);
    EXPECT_EQ(r.analysis.hypothetical_restores, 64 - std::popcount(n / d));
    EXPECT_EQ(r.captured_entries, 69);
    auto nr = integer_newton_division_trace(n, d);
    EXPECT_EQ(nr.analysis.result.quotient, n / d);
    EXPECT_EQ(nr.analysis.result.remainder, n % d);
    EXPECT_EQ(nr.captured_entries, 10 + nr.steps.size() + nr.checks.size());
    if (n >= d) {
      EXPECT_EQ(nr.steps.size(), 5);
      EXPECT_GE(nr.checks.size(), 2);
      EXPECT_EQ(nr.checks.size(), nr.analysis.decrements + nr.analysis.increments + 2);
      std::uint64_t candidate = nr.initial_quotient;
      bool          upward    = false;
      for (auto& c : nr.checks) {
        EXPECT_EQ(c.quotient_before, candidate);
        if (!c.downward)
          upward = true;
        else
          EXPECT_FALSE(upward);
        auto product = (static_cast<unsigned __int128>(candidate) + (c.downward ? 0 : 1)) * d;
        EXPECT_EQ(magnitude(c.tested_product), product);
        EXPECT_EQ(c.changed, c.downward ? product > n : product <= n);
        if (c.changed) {
          if (c.downward)
            --candidate;
          else
            ++candidate;
        }
        EXPECT_EQ(c.quotient_after, candidate);
      }
      EXPECT_EQ(candidate, n / d);
    } else
      EXPECT_TRUE(nr.steps.empty());
  }
}
TEST(DivisionTrace, ExhaustiveSmallQuotientsZeroSentinelExactQuotaAndFullWidthConversion) {
  for (std::uint64_t n = 0; n <= 255; ++n)
    for (std::uint64_t d = 1; d <= 255; ++d) {
      auto a = non_restoring_divide(n, d), b = newton_raphson_divide(n, d);
      EXPECT_EQ(a.quotient, n / d);
      EXPECT_EQ(a.remainder, n % d);
      EXPECT_EQ(b.quotient, n / d);
      EXPECT_EQ(b.remainder, n % d);
    }
  auto max = std::numeric_limits<std::uint64_t>::max();
  auto t   = integer_newton_division_trace(max, 1);
  EXPECT_TRUE(t.estimate_clamped);
  EXPECT_EQ(t.initial_quotient, max);
  EXPECT_EQ(t.analysis.result.quotient, max);
  EXPECT_EQ(t.checks.back().tested_product.high, 1);
  EXPECT_EQ(t.checks.back().tested_product.low, 0);
  for (auto a :
       {restoring_division_analyze(5, 0),
        non_restoring_division_analyze(5, 0),
        integer_newton_division_analyze(5, 0)}) {
    EXPECT_FALSE(a.valid_divisor);
    EXPECT_EQ(a.result.quotient, 0);
    EXPECT_EQ(a.result.remainder, 0);
    EXPECT_EQ(a.result.iterations, 0);
  }
  EXPECT_NO_THROW(non_restoring_division_trace(217, 13, 70));
  EXPECT_THROW(non_restoring_division_trace(217, 13, 69), std::length_error);
  EXPECT_NO_THROW(restoring_division_trace(217, 13, 69));
  EXPECT_THROW(restoring_division_trace(217, 13, 68), std::length_error);
  EXPECT_NO_THROW(integer_newton_division_trace(max, 1, t.captured_entries));
  EXPECT_THROW(integer_newton_division_trace(max, 1, t.captured_entries - 1), std::length_error);
  EXPECT_EQ(division_wide_decimal({1, 0, false}), "18446744073709551616");
  EXPECT_EQ(division_wide_decimal({max, max, true}), "-340282366920938463463374607431768211455");
  EXPECT_EQ(division_wide_decimal({0, 0, true}), "0");
}
