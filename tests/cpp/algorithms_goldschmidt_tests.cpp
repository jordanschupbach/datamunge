#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <datamunge/algorithms/goldschmidt_division.hpp>
#include <limits>
#include <random>
#include <stdexcept>

namespace {
double frozen(double n, double d) {
  const double s = d < 0 ? -1 : 1;
  int          e = 0;
  double       D = std::frexp(std::fabs(d), &e), N = std::ldexp(n, -e);
  for (int i = 0; i < 60; ++i) {
    const double f = 2 - D;
    N *= f;
    D *= f;
    if (std::fabs(D - 1) < 1e-16)
      break;
  }
  return s * N;
}
auto bits(double x) {
  return std::bit_cast<std::uint64_t>(x);
}
} // namespace
using namespace datamunge::algorithms;
TEST(Goldschmidt, FrozenFiniteInputsAndActualRecords) {
  std::mt19937_64 rng(73119);
  for (int i = 0; i < 20000; ++i) {
    double n = std::bit_cast<double>(rng()), d = std::bit_cast<double>(rng());
    if (!std::isfinite(n) || !std::isfinite(d) || d == 0)
      continue;
    const auto t = goldschmidt_trace(n, d);
    EXPECT_EQ(bits(t.run.quotient), bits(frozen(n, d)));
    EXPECT_EQ(bits(goldschmidt_division(n, d)), bits(t.run.quotient));
    EXPECT_EQ(bits(goldschmidt_run(n, d).quotient), bits(t.run.quotient));
    EXPECT_EQ(t.steps.size(), t.run.updates);
    EXPECT_EQ(t.run.loop_multiplications, 2 * t.steps.size());
    auto state = t.run.initial;
    for (const auto& step : t.steps) {
      EXPECT_EQ(bits(step.before.numerator), bits(state.numerator));
      EXPECT_EQ(bits(step.before.denominator), bits(state.denominator));
      EXPECT_EQ(step.factor, 2 - state.denominator);
      EXPECT_EQ(bits(step.after.numerator), bits(state.numerator * step.factor));
      EXPECT_EQ(bits(step.after.denominator), bits(state.denominator * step.factor));
      state = step.after;
    }
  }
}
TEST(Goldschmidt, ExactDyadicStepsAndOwnedHistory) {
  auto t = goldschmidt_trace(3, 4, 2, 2);
  EXPECT_EQ(t.run.initial.numerator, .375);
  EXPECT_EQ(t.run.initial.denominator, .5);
  EXPECT_EQ(t.steps[0].factor, 1.5);
  EXPECT_EQ(t.steps[0].after.numerator, .5625);
  EXPECT_EQ(t.steps[1].factor, 1.25);
  EXPECT_EQ(t.steps[1].after.numerator, .703125);
  EXPECT_EQ(t.steps[1].after.denominator, .9375);
  auto copy         = t;
  t.steps[0].factor = 0;
  EXPECT_EQ(copy.steps[0].factor, 1.5);
}
TEST(Goldschmidt, RoundingLimitsAreObservable) {
  auto t = goldschmidt_trace(1, std::nextafter(1., 0.));
  EXPECT_EQ(t.run.updates, 60);
  EXPECT_EQ(t.run.stop, GoldschmidtStop::iteration_limit);
  EXPECT_EQ(t.steps.back().factor, 1);
  EXPECT_EQ(t.run.quotient, 1);
  EXPECT_NE(t.run.quotient, 1 / std::nextafter(1., 0.));
  double tiny = std::numeric_limits<double>::denorm_min();
  auto   r    = goldschmidt_run(tiny, 1);
  EXPECT_EQ(r.initial.numerator, 0);
  EXPECT_EQ(r.quotient, 0);
  EXPECT_EQ(r.final_state.denominator, 1);
  EXPECT_NE(r.quotient, tiny / 1);
  for (double n : {0., -0.})
    for (double d : {1., -1.})
      EXPECT_EQ(bits(goldschmidt_division(n, d)), bits(n / d));
  EXPECT_TRUE(std::isinf(goldschmidt_division(std::numeric_limits<double>::max(), tiny)));
}
TEST(Goldschmidt, BudgetsQuotaAndValidation) {
  EXPECT_EQ(goldschmidt_run(3, 4, 0).quotient, .375);
  EXPECT_TRUE(goldschmidt_trace(3, 4, 0, 1).steps.empty());
  EXPECT_EQ(goldschmidt_trace(22, 7, 60, 5).steps.size(), 5);
  EXPECT_THROW(goldschmidt_trace(22, 7, 60, 4), std::length_error);
  EXPECT_EQ(goldschmidt_trace(22, 7, 60, 0).steps.size(), 5);
  EXPECT_THROW(goldschmidt_trace(1, 0, 0, 1), std::invalid_argument);
  EXPECT_THROW(goldschmidt_run(1, -0.), std::invalid_argument);
  EXPECT_THROW(goldschmidt_trace(1, 1, -1, 1), std::invalid_argument);
  for (double x :
       {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(goldschmidt_division(x, 1), std::invalid_argument);
    EXPECT_THROW(goldschmidt_division(1, x), std::invalid_argument);
  }
}
