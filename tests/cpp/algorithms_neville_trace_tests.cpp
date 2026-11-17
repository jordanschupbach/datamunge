#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/lagrange.hpp>
#include <datamunge/algorithms/neville.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
TEST(NevilleTrace, IndependentPolynomialValuesEveryActualUpdateAndQuotas) {
  std::mt19937_64 rng(254263);
  for (int n = 1; n <= 10; ++n)
    for (int repeat = 0; repeat < 20; ++repeat) {
      std::vector<double> x(n), y(n);
      std::iota(x.begin(), x.end(), -n / 2.);
      std::shuffle(x.begin(), x.end(), rng);
      for (int i = 0; i < n; ++i)
        y[i] = 2 * x[i] * x[i] - 3 * x[i] + 1;
      double q = double(int(rng() % 100) - 50) / 10.;
      auto   t = neville_trace(x, y, q);
      if (n >= 3)
        EXPECT_NEAR(t.analysis.value, 2 * q * q - 3 * q + 1, 1e-5);
      EXPECT_NEAR(t.analysis.value, lagrange_interpolate(x, y, q), 1e-5);
      EXPECT_EQ(t.steps.size(), std::size_t(n * (n - 1) / 2));
      std::vector<double> p     = y;
      std::size_t         index = 0;
      for (int degree = 1; degree < n; ++degree)
        for (int j = n - 1; j >= degree; --j) {
          auto& r = t.steps[index++];
          EXPECT_EQ(r.degree, degree);
          EXPECT_EQ(r.first, j - degree);
          EXPECT_EQ(r.last, j);
          EXPECT_EQ(r.right_before, p[j]);
          EXPECT_EQ(r.left_before, p[j - 1]);
          long double a   = (static_cast<long double>(q) - x[j - degree]) * p[j],
                      b   = (static_cast<long double>(q) - x[j]) * p[j - 1];
          double expected = double((a - b) / (static_cast<long double>(x[j]) - x[j - degree]));
          EXPECT_NEAR(r.value, expected, 1e-7);
          p[j] = r.value;
        }
      EXPECT_EQ(t.final_working, p);
      EXPECT_EQ(t.captured_entries, 1 + 3 * n + n * (n - 1) / 2);
      EXPECT_NO_THROW(neville_trace(x, y, q, {0, t.captured_entries}));
      EXPECT_THROW(neville_trace(x, y, q, {0, t.captured_entries - 1}), std::length_error);
      EXPECT_EQ(neville_interpolate(x, y, q), t.analysis.value);
    }
}
TEST(NevilleTrace, CompleteGuardsWideFallbackOwnershipAndOriginalStudies) {
  EXPECT_EQ(neville_interpolate({}, {}, 1), 0);
  EXPECT_EQ(neville_trace({}, {}, 1).captured_entries, 1);
  EXPECT_THROW(neville_interpolate({}, {1}, 0), std::invalid_argument);
  EXPECT_THROW(neville_interpolate({}, {}, NAN), std::invalid_argument);
  EXPECT_THROW(neville_trace({0, 0}, {1, 2}, 0, {1, 1}), std::invalid_argument);
  EXPECT_THROW(neville_interpolate({0, 1}, {1, NAN}, 0), std::invalid_argument);
  EXPECT_THROW(neville_analyze({0, 1}, {0, 1}, .5, {1, 0}), std::length_error);
  double m    = std::numeric_limits<double>::max();
  auto   wide = neville_trace({-m, m}, {-1, 1}, 0);
  EXPECT_EQ(wide.analysis.value, 0);
  EXPECT_EQ(wide.analysis.widened_updates, 1);
  EXPECT_THROW(neville_interpolate({0, 1}, {-m, m}, m), std::overflow_error);
  std::vector<double> x{0, 1}, y{0, 2};
  auto                t = neville_trace(x, y, .25);
  x[0]                  = 4;
  y[0]                  = 8;
  EXPECT_EQ(t.xs[0], 0);
  EXPECT_EQ(t.ys[0], 0);
  EXPECT_EQ(t.analysis.value, .5);
  auto original = neville_cosine_study();
  EXPECT_EQ(original.xs.size(), 7);
  EXPECT_EQ(original.degrees.size(), 6);
  EXPECT_EQ(original.query, 1.25);
  for (int i = 0; i < 7; ++i) {
    EXPECT_EQ(original.xs[i], .5 * i);
    EXPECT_EQ(original.ys[i], std::cos(.5 * i));
  }
  auto counter = neville_zero_sample_study();
  EXPECT_EQ(counter.estimates, (std::vector<double>{0, 0}));
  EXPECT_EQ(counter.successive_changes.back(), 0);
  EXPECT_EQ(counter.absolute_errors.back(), .375);
}
