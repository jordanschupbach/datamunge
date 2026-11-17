#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/algorithms/newton_division.hpp>
#include <limits>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
TEST(NewtonDivisionTrace, ActualSharedReciprocalOperandsAndBoundedReferenceCensus) {
  std::mt19937_64 rng(254263);
  for (int i = 0; i < 2000; ++i) {
    double n = std::ldexp(.5 + double(rng() >> 11) * 0x1p-54, int(rng() % 1001) - 500),
           d = std::ldexp(.5 + double(rng() >> 11) * 0x1p-54, int(rng() % 1001) - 500);
    if (i % 2)
      n = -n;
    if (i % 3)
      d = -d;
    auto   t = newton_division_trace(n, d);
    auto&  r = t.run;
    double x = r.seed;
    for (auto& s : t.steps) {
      EXPECT_EQ(s.before, x);
      EXPECT_EQ(s.product, r.denominator_mantissa * x);
      EXPECT_EQ(s.factor, 2 - s.product);
      EXPECT_EQ(s.after, x * s.factor);
      EXPECT_EQ(s.residual_before, std::fma(-r.denominator_mantissa, x, 1.));
      EXPECT_EQ(s.residual_after, std::fma(-r.denominator_mantissa, s.after, 1.));
      x = s.after;
    }
    EXPECT_EQ(r.reciprocal, x);
    EXPECT_EQ(r.updates, t.steps.size());
    EXPECT_LE(r.updates, 60);
    EXPECT_EQ(t.captured_entries, 3 + r.updates);
    EXPECT_EQ(newton_raphson_division(n, d), r.quotient);
    long double reference = static_cast<long double>(n) / d;
    EXPECT_LT(std::abs((static_cast<long double>(r.quotient) - reference) / reference), 2e-15L);
    EXPECT_NO_THROW(newton_division_trace(n, d, t.captured_entries));
    EXPECT_THROW(newton_division_trace(n, d, t.captured_entries - 1), std::length_error);
  }
  auto check = newton_division_check();
  EXPECT_EQ(check.checked, 100000);
  EXPECT_LT(check.max_relative_error, 2e-15);
  EXPECT_LT(check.worst_index, check.checked);
}
TEST(NewtonDivisionTrace, FullExponentSubnormalSignedZeroAndCompleteDomain) {
  double tiny = std::numeric_limits<double>::denorm_min(), min = std::numeric_limits<double>::min(),
         max = std::numeric_limits<double>::max();
  EXPECT_EQ(newton_raphson_division(tiny, tiny), 1);
  EXPECT_EQ(newton_raphson_division(min, tiny), 0x1p52);
  EXPECT_NEAR(newton_raphson_division(max, max), 1, 1e-15);
  auto under = newton_division_run(-tiny, max);
  EXPECT_EQ(under.quotient, 0);
  EXPECT_TRUE(std::signbit(under.quotient));
  EXPECT_TRUE(under.underflowed);
  auto z = newton_division_trace(0, -2);
  EXPECT_TRUE(z.run.zero_shortcut);
  EXPECT_TRUE(std::signbit(z.run.quotient));
  EXPECT_EQ(z.run.updates, 0);
  EXPECT_EQ(z.captured_entries, 3);
  EXPECT_THROW(newton_raphson_division(1, 0), std::invalid_argument);
  EXPECT_THROW(newton_division_trace(NAN, 0, 1), std::invalid_argument);
  EXPECT_THROW(newton_raphson_division(1, INFINITY), std::invalid_argument);
  EXPECT_THROW(newton_raphson_division(max, tiny), std::overflow_error);
  EXPECT_THROW(newton_division_trace(1, 2, 2), std::length_error);
  for (auto [n, d] : newton_division_demo_pairs())
    EXPECT_NEAR(newton_raphson_division(n, d), n / d, std::abs(n / d) * 2e-15);
}
TEST(NewtonReciprocalTrace, ExplicitFixedSeedStatesAndCompleteQuota) {
  auto   t        = newton_reciprocal_trace(2, .25, 4);
  double residual = .5;
  for (auto& s : t.steps) {
    EXPECT_EQ(s.residual_before, residual);
    residual *= residual;
    EXPECT_EQ(s.residual_after, residual);
  }
  EXPECT_EQ(t.captured_entries, 8);
  EXPECT_NO_THROW(newton_reciprocal_trace(2, .25, 4, 8));
  EXPECT_THROW(newton_reciprocal_trace(2, .25, 4, 7), std::length_error);
  EXPECT_EQ(newton_reciprocal_trace(2, .25, 0).final, .25);
  EXPECT_THROW(newton_reciprocal_trace(0, .25, 0, 1), std::invalid_argument);
  EXPECT_THROW(newton_reciprocal_step(2, INFINITY), std::invalid_argument);
  EXPECT_THROW(newton_reciprocal_step(std::numeric_limits<double>::max(), 2), std::overflow_error);
  auto diagnostic = newton_reciprocal_trace(7, 1. / 14., 6);
  EXPECT_EQ(diagnostic.steps.size(), 6);
  EXPECT_NEAR(diagnostic.final, 1. / 7., 1e-15);
}
