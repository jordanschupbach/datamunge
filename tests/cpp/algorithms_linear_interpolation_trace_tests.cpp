#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/algorithms/interpolation_extra.hpp>
#include <limits>
#include <random>
using namespace datamunge::algorithms;
TEST(LinearInterpolationTrace, ActualSelectedIntervalSearchAndBlend) {
  LinearInterpolator f({0, 1, 3, 6}, {1, 4, 10, 19});
  auto               t = f.trace(2.5);
  EXPECT_EQ(t.evaluation.interval, 1);
  EXPECT_EQ(t.evaluation.left_x, 1);
  EXPECT_EQ(t.evaluation.right_x, 3);
  EXPECT_EQ(t.evaluation.fraction, .75);
  EXPECT_EQ(t.evaluation.value, 8.5);
  EXPECT_EQ(t.evaluation.endpoint_tests, 2);
  EXPECT_EQ(t.evaluation.binary_tests, t.probes.size() - 2);
  for (auto& p : t.probes) {
    EXPECT_EQ(p.abscissa, t.xs[p.knot]);
    EXPECT_EQ(
        p.predicate,
        p.kind == 0   ? 2.5 <= p.abscissa
        : p.kind == 1 ? 2.5 >= p.abscissa
                      : 2.5 < p.abscissa);
  }
  EXPECT_EQ(t.captured_entries, 2 * t.xs.size() + t.probes.size());
  EXPECT_EQ(f(-1), -2);
  EXPECT_EQ(f(7), 22);
  EXPECT_TRUE(f.evaluate(-1).extrapolated);
  for (std::size_t i = 0; i < f.xs().size(); ++i) {
    EXPECT_TRUE(f.evaluate(f.xs()[i]).exact_node);
    EXPECT_EQ(f(f.xs()[i]), f.ys()[i]);
  }
}
TEST(LinearInterpolationTrace, IndependentLongDoubleLineAndAffineReproduction) {
  std::mt19937 rng(204213);
  for (int fixture = 0; fixture < 100; ++fixture) {
    std::vector<double> x, y;
    for (int i = 0; i < 10; ++i) {
      x.push_back(i * 1.25);
      y.push_back((int(rng() % 101) - 50) * .25);
    }
    LinearInterpolator f(x, y);
    for (int j = -10; j <= 110; ++j) {
      double      q = j * .125;
      std::size_t k = 0;
      while (k + 2 < x.size() && q >= x[k + 1])
        ++k;
      const long double expected = y[k] + (static_cast<long double>(q) - x[k]) /
                                              (static_cast<long double>(x[k + 1]) - x[k]) *
                                              (static_cast<long double>(y[k + 1]) - y[k]);
      EXPECT_NEAR(f(q), double(expected), 1e-12);
      EXPECT_EQ(f(q), linear_interpolate(x, y, q));
    }
  }
}
TEST(LinearInterpolationTrace, OriginalSineGridsAndSmoothReference) {
  auto original = linear_interpolation_sine_study(7, 300);
  EXPECT_EQ(original.queries.size(), 301);
  EXPECT_EQ(original.xs.size(), 7);
  double previous = 0;
  for (std::size_t n : {3, 5, 9, 17, 33, 65}) {
    auto s = linear_interpolation_sine_study(n, 1000);
    EXPECT_LE(s.max_grid_error, s.error_reference + 1e-15);
    if (n >= 33)
      EXPECT_GT(previous / s.max_grid_error, 3.9);
    previous = s.max_grid_error;
  }
}
TEST(LinearInterpolationTrace, CompleteGuardsConservativeQuotaWideRangeAndOwnership) {
  double q = std::numeric_limits<double>::quiet_NaN(), m = std::numeric_limits<double>::max();
  EXPECT_THROW(LinearInterpolator({}, {}), std::invalid_argument);
  EXPECT_THROW(LinearInterpolator({0}, {1}), std::invalid_argument);
  EXPECT_THROW(LinearInterpolator({0, 1}, {1}), std::invalid_argument);
  EXPECT_THROW(LinearInterpolator({0, 0}, {1, 2}), std::invalid_argument);
  EXPECT_THROW(LinearInterpolator({1, 0}, {1, 2}), std::invalid_argument);
  EXPECT_THROW(LinearInterpolator({0, 1}, {1, q}), std::invalid_argument);
  LinearInterpolator f({0, 1, 3, 6}, {1, 4, 10, 19});
  EXPECT_THROW(f(q), std::invalid_argument);
  EXPECT_THROW(f.trace(2.5, 12), std::length_error);
  EXPECT_EQ(f.trace(2.5, 13).evaluation.value, 8.5);
  EXPECT_EQ(f.trace(2.5, 0).evaluation.value, 8.5);
  LinearInterpolator wide({-m, m}, {1, 3});
  EXPECT_TRUE(wide.evaluate(0).widened);
  EXPECT_EQ(wide(0), 2);
  LinearInterpolator opposite({-1, 1}, {-m, m});
  EXPECT_EQ(opposite(0), 0);
  EXPECT_EQ(opposite(-1), -m);
  EXPECT_THROW(opposite(2), std::overflow_error);
  std::vector<double> x{0, 1}, y{1, 3};
  LinearInterpolator  own(x, y);
  x[0] = -1;
  y[0] = 9;
  EXPECT_EQ(own(.5), 2);
  auto t = own.trace(.5);
  EXPECT_EQ(t.xs[0], 0);
  EXPECT_THROW(linear_interpolation_sine_study(1), std::invalid_argument);
  EXPECT_THROW(linear_interpolation_sine_study(7, 0), std::invalid_argument);
}
