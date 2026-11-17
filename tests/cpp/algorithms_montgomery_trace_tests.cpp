#include <gtest/gtest.h>

#include <bit>
#include <datamunge/algorithms/montgomery.hpp>
#include <limits>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
namespace {
std::uint64_t slow_product(std::uint64_t a, std::uint64_t b, std::uint64_t n) {
  a %= n;
  std::uint64_t value = 0;
  while (b) {
    if (b & 1)
      value = value >= n - a ? value - (n - a) : value + a;
    a = a >= n - a ? a - (n - a) : a + a;
    b >>= 1;
  }
  return value;
}
std::uint64_t slow_power(std::uint64_t a, std::uint64_t e, std::uint64_t n) {
  std::uint64_t value = 1 % n;
  for (int bit = 63; bit >= 0; --bit) {
    value = slow_product(value, value, n);
    if ((e >> bit) & 1)
      value = slow_product(value, a, n);
  }
  return value;
}
} // namespace
TEST(MontgomeryTrace, IndependentBitSerialArithmeticAndSplitLimbCarry) {
  std::mt19937_64 rng(814);
  bool            saw_carry = false;
  for (int k = 0; k < 3000; ++k) {
    auto              n = rng() | 1ULL;
    MontgomeryContext c(n);
    auto              a = rng() % n, b = rng() % n;
    auto              t          = c.reduce_trace(a, b);
    const auto        product    = static_cast<__uint128_t>(a) * b;
    const auto        correction = static_cast<__uint128_t>(t.correction) * n;
    const auto        low_sum    = static_cast<__uint128_t>(static_cast<std::uint64_t>(product)) +
                         static_cast<std::uint64_t>(correction);
    const auto quotient = (product >> 64) + (correction >> 64) + (low_sum >> 64);
    EXPECT_EQ(static_cast<std::uint64_t>(n * c.nprime() + 1), 0);
    EXPECT_EQ(static_cast<std::uint64_t>(low_sum), 0);
    EXPECT_EQ(t.product_low, static_cast<std::uint64_t>(product));
    EXPECT_EQ(t.product_high, static_cast<std::uint64_t>(product >> 64));
    EXPECT_EQ(t.sum_carry, bool(quotient >> 64));
    EXPECT_EQ(t.quotient_low, static_cast<std::uint64_t>(quotient));
    EXPECT_EQ(t.subtracted, quotient >= n);
    saw_carry |= t.sum_carry;
    const auto expected = quotient >= n ? quotient - n : quotient;
    EXPECT_EQ(t.result, static_cast<std::uint64_t>(expected));
    EXPECT_LT(t.result, n);
    const auto rmod = (std::numeric_limits<std::uint64_t>::max() % n + 1) % n;
    EXPECT_EQ(slow_product(t.result, rmod, n), slow_product(a, b, n));
    EXPECT_EQ(c.reduce(a, b), t.result);
    auto raw_a = rng(), raw_b = rng();
    EXPECT_EQ(c.multiply(raw_a, raw_b), slow_product(raw_a, raw_b, n));
    auto e = rng();
    auto p = c.power_trace(raw_a, e);
    EXPECT_EQ(p.result, slow_power(raw_a, e, n));
    EXPECT_EQ(c.power(raw_a, e), p.result);
    EXPECT_EQ(p.reductions, 2U + std::bit_width(e) + std::popcount(e));
    auto acc = p.initial_accumulator, running = p.initial_running, remaining = e;
    for (auto& step : p.steps) {
      EXPECT_EQ(step.remaining, remaining);
      EXPECT_EQ(step.running, running);
      EXPECT_EQ(step.accumulator_before, acc);
      EXPECT_EQ(step.multiplied, bool(remaining & 1));
      if (remaining & 1)
        acc = c.reduce(acc, running);
      EXPECT_EQ(step.accumulator_after, acc);
      running = c.reduce(running, running);
      EXPECT_EQ(step.squared, running);
      remaining >>= 1;
    }
    EXPECT_EQ(remaining, 0);
    EXPECT_EQ(c.reduce(acc, 1), p.result);
  }
  EXPECT_TRUE(saw_carry);
}
TEST(MontgomeryTrace, GuardsIdentityAndToyRSA) {
  EXPECT_THROW(MontgomeryContext(0), std::invalid_argument);
  EXPECT_THROW(MontgomeryContext(2), std::invalid_argument);
  MontgomeryContext one(1);
  EXPECT_EQ(one.multiply(~0ULL, ~0ULL), 0);
  EXPECT_EQ(one.power(0, 0), 0);
  EXPECT_TRUE(one.power_trace(4, ~0ULL, 1).steps.empty());
  EXPECT_EQ(one.power_trace(4, 9).reductions, 0);
  MontgomeryContext c(3233);
  EXPECT_EQ(c.power(123, 17), 855);
  EXPECT_EQ(c.power(855, 2753), 123);
  EXPECT_THROW(c.reduce(3233, 0), std::invalid_argument);
  EXPECT_THROW(c.reduce_trace(0, 3233), std::invalid_argument);
  EXPECT_THROW(c.power_trace(2, 16, 4), std::length_error);
  EXPECT_NO_THROW(c.power_trace(2, 16, 5));
  EXPECT_NO_THROW(c.power_trace(2, 16, 0));
  EXPECT_EQ(c.power(0, 0), 1);
  auto t = c.power_trace(0, 0);
  EXPECT_EQ(t.reductions, 2);
  EXPECT_TRUE(t.steps.empty());
}
