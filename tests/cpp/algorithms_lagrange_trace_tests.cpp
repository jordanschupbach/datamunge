#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/lagrange.hpp>
#include <limits>
#include <numbers>
using namespace datamunge::algorithms;
TEST(LagrangeTrace, ExactQuadraticAndActualProductSumBoundaries) {
  const auto t = lagrange_trace({-1, 0, 1}, {2, 1, 2}, .5);
  EXPECT_DOUBLE_EQ(t.value, 1.25);
  ASSERT_EQ(t.terms.size(), 3U);
  EXPECT_DOUBLE_EQ(t.terms[0].value, -.25);
  EXPECT_DOUBLE_EQ(t.terms[1].value, .75);
  EXPECT_DOUBLE_EQ(t.terms[2].value, .75);
  std::size_t at  = 0;
  double      sum = 0;
  for (std::size_t j = 0; j < t.xs.size(); ++j) {
    double term = t.ys[j];
    for (std::size_t k = 0; k < t.xs.size(); ++k)
      if (k != j) {
        const auto& e = t.factors[at++];
        EXPECT_EQ(e.term_index, j);
        EXPECT_EQ(e.excluded_node, k);
        EXPECT_EQ(e.term_before, term);
        EXPECT_EQ(e.ratio, (t.query - t.xs[k]) / (t.xs[j] - t.xs[k]));
        term *= e.ratio;
        EXPECT_EQ(e.term_after, term);
      }
    EXPECT_EQ(t.terms[j].value, term);
    EXPECT_EQ(t.terms[j].sum_before, sum);
    sum += term;
    EXPECT_EQ(t.terms[j].sum_after, sum);
  }
  EXPECT_EQ(sum, t.value);
  EXPECT_EQ(t.work.terms, 3U);
  EXPECT_EQ(t.work.ratio_evaluations, 6U);
  EXPECT_EQ(t.work.term_multiplications, 6U);
  EXPECT_EQ(t.work.sum_additions, 3U);
  EXPECT_EQ(t.captured_entries, 15U);
}
TEST(LagrangeTrace, PolynomialReproductionNodesPermutationAndExtrapolation) {
  const std::vector<double> xs = {-2, -1, 0, 1, 2};
  for (int degree = 0; degree <= 4; ++degree) {
    std::vector<double> ys;
    for (double x : xs)
      ys.push_back(std::pow(x, degree));
    for (int i = -12; i <= 12; ++i) {
      double x = i * .25;
      EXPECT_NEAR(lagrange_interpolate(xs, ys, x), std::pow(x, degree), 1e-10);
      auto xr = xs, yr = ys;
      std::reverse(xr.begin(), xr.end());
      std::reverse(yr.begin(), yr.end());
      EXPECT_NEAR(lagrange_interpolate(xr, yr, x), std::pow(x, degree), 1e-10);
    }
    for (std::size_t i = 0; i < xs.size(); ++i)
      EXPECT_EQ(lagrange_interpolate(xs, ys, xs[i]), ys[i]);
  }
  EXPECT_EQ(lagrange_interpolate({}, {}, 0), 0);
  EXPECT_EQ(lagrange_interpolate({2}, {3}, 100), 3);
}
TEST(LagrangeTrace, CompleteInputQuotaAndOwnedSnapshots) {
  EXPECT_THROW((void)lagrange_interpolate({0}, {}, 0), std::invalid_argument);
  EXPECT_THROW((void)lagrange_interpolate({0, 0}, {1, 2}, 0), std::invalid_argument);
  EXPECT_THROW((void)lagrange_interpolate({-0., 0.}, {1, 2}, 0), std::invalid_argument);
  EXPECT_THROW(
      (void)lagrange_interpolate({}, {}, std::numeric_limits<double>::infinity()),
      std::invalid_argument);
  EXPECT_THROW(
      (void)lagrange_interpolate({0}, {std::numeric_limits<double>::quiet_NaN()}, 0),
      std::invalid_argument);
  const auto t = lagrange_trace({0, 1}, {0, 1}, .5);
  EXPECT_NO_THROW((void)lagrange_trace(t.xs, t.ys, t.query, 8));
  EXPECT_THROW((void)lagrange_trace(t.xs, t.ys, t.query, 7), std::length_error);
  EXPECT_NO_THROW((void)lagrange_trace(t.xs, t.ys, t.query, 0));
  auto c                  = t;
  c.xs[0]                 = 99;
  c.ys[0]                 = 99;
  c.factors[0].term_after = 99;
  c.terms[0].sum_after    = 99;
  EXPECT_EQ(t.xs[0], 0);
  EXPECT_EQ(t.ys[0], 0);
  EXPECT_NE(t.factors[0].term_after, 99);
}
TEST(LagrangeTrace, WidenedDifferenceFallbackAndConservativeFiniteIntermediateGuard) {
  const double m = std::numeric_limits<double>::max();
  const auto   t = lagrange_trace({-m, 0, m}, {1, 0, 1}, m / 2);
  EXPECT_NEAR(t.value, .25, 1e-15);
  EXPECT_TRUE(std::any_of(t.factors.begin(), t.factors.end(), [](const auto& e) {
    return e.widened_differences;
  }));
  EXPECT_THROW((void)lagrange_interpolate({0, 1}, {m, m}, 2), std::overflow_error);
}
TEST(LagrangeTrace, OriginalRungeOvershootAndChebyshevLobattoControl) {
  std::vector<double> x, y, c, cy;
  for (int i = 0; i < 11; ++i) {
    const double a = -1. + 2. * i / 10, b = std::cos(std::numbers::pi * i / 10.);
    x.push_back(a);
    y.push_back(1 / (1 + 25 * a * a));
    c.push_back(b);
    cy.push_back(1 / (1 + 25 * b * b));
  }
  double eq = 0, ch = 0, peak = 0;
  for (int i = 0; i <= 2000; ++i) {
    const double q = -1. + 2. * i / 2000, f = 1 / (1 + 25 * q * q),
                 a = lagrange_interpolate(x, y, q), b = lagrange_interpolate(c, cy, q);
    eq   = std::max(eq, std::abs(a - f));
    ch   = std::max(ch, std::abs(b - f));
    peak = std::max(peak, a);
  }
  EXPECT_GT(eq, 1.8);
  EXPECT_LT(eq, 2);
  EXPECT_LT(ch, .2);
  EXPECT_GT(peak, 1.8);
}
