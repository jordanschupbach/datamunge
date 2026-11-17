#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/algorithms/hermite.hpp>
#include <limits>
using namespace datamunge::algorithms;
TEST(Hermite, ActualTraceAndCubicReproductionOnUnequalIntervals) {
  const std::vector<double> xs{-2, -.5, 1, 3.5};
  for (int a = -2; a <= 2; ++a)
    for (int b = -2; b <= 2; ++b)
      for (int c = -2; c <= 2; ++c)
        for (int d = -2; d <= 2; ++d) {
          auto                f  = [&](double x) { return ((a * x + b) * x + c) * x + d; };
          auto                df = [&](double x) { return (3 * a * x + 2 * b) * x + c; };
          std::vector<double> y, m, q;
          for (auto x : xs) {
            y.push_back(f(x));
            m.push_back(df(x));
          }
          for (int j = 0; j <= 64; ++j)
            q.push_back(-3 + j / 8.);
          auto batch = hermite_evaluate_many(xs, y, m, q);
          for (std::size_t j = 0; j < q.size(); ++j) {
            const auto e = hermite_evaluate(xs, y, m, q[j]);
            EXPECT_EQ(e.value, batch[j]);
            EXPECT_EQ(e.value, hermite_interpolate(xs, y, m, q[j]));
            EXPECT_NEAR(e.value, f(q[j]), 5e-12);
            EXPECT_EQ(e.value, e.terms[0] + e.terms[1] + e.terms[2] + e.terms[3]);
            EXPECT_EQ(e.t2, e.t * e.t);
            EXPECT_EQ(e.t3, e.t2 * e.t);
            EXPECT_EQ(e.extrapolated, q[j] < xs.front() || q[j] > xs.back());
          }
        }
  const auto e = hermite_evaluate({2, 4}, {1, 3}, {2, -1}, 3);
  EXPECT_EQ(e.value, 2.75);
  EXPECT_EQ(e.terms, (std::array<double, 4>{.5, .5, 1.5, .25}));
  EXPECT_EQ(hermite_evaluate({0, 1, 2}, {0, 1, 0}, {0, 0, 0}, 1).interval, 1);
}
TEST(Hermite, ExactQuarticErrorSlopesAndOvershoot) {
  for (int k = 0; k <= 20; ++k) {
    const double h = std::ldexp(1., -k);
    const auto   v = hermite_interpolate({0, h}, {0, h * h * h * h}, {0, 4 * h * h * h}, h / 2);
    EXPECT_EQ(std::pow(h / 2, 4) - v, std::pow(h, 4) / 16);
  }
  EXPECT_NEAR(hermite_interpolate({0, 1}, {0, 1}, {8, 0}, 4. / 9), 368. / 243, 1e-14);
  const std::vector<double> xs{0, 1, 2, 3}, ys{0, 1, 0, 1}, ms{1, 0, 0, 1};
  for (int i = 0; i <= 300; ++i) {
    const auto v = hermite_interpolate(xs, ys, ms, i / 100.);
    EXPECT_GE(v, -1e-14);
    EXPECT_LE(v, 1 + 1e-14);
  }
  for (std::size_t i = 0; i < xs.size(); ++i) {
    double h = 1e-5;
    auto   derivative =
        (hermite_interpolate(xs, ys, ms, xs[i] + h) - hermite_interpolate(xs, ys, ms, xs[i] - h)) /
        (2 * h);
    EXPECT_NEAR(derivative, ms[i], 4e-5);
  }
}
TEST(Hermite, CheckedDomainsCapsExtrapolationAndArithmetic) {
  EXPECT_THROW(hermite_interpolate({}, {}, {}, 0), std::invalid_argument);
  EXPECT_THROW(hermite_interpolate({0, 1}, {0}, {0, 0}, 0), std::invalid_argument);
  EXPECT_THROW(hermite_interpolate({0, 0}, {0, 1}, {0, 0}, 0), std::invalid_argument);
  EXPECT_THROW(hermite_interpolate({1, 0}, {0, 1}, {0, 0}, 0), std::invalid_argument);
  EXPECT_THROW(
      hermite_interpolate({0, 1}, {0, 1}, {0, 0}, std::numeric_limits<double>::infinity()),
      std::invalid_argument);
  EXPECT_THROW(hermite_interpolate({-1e308, 1e308}, {0, 1}, {0, 0}, 0), std::overflow_error);
  EXPECT_THROW(hermite_interpolate({0, 1e-300}, {0, 1}, {0, 0}, 1e300), std::overflow_error);
  EXPECT_THROW(hermite_evaluate_many({0, 1}, {0, 1}, {0, 0}, {0, .5, 1}, 2), std::length_error);
  EXPECT_NO_THROW(hermite_evaluate_many({0, 1}, {0, 1}, {0, 0}, {0, .5, 1}, 3));
  EXPECT_NO_THROW(hermite_evaluate_many({0, 1}, {0, 1}, {0, 0}, {0, .5, 1}, 0));
  EXPECT_THROW(
      hermite_evaluate_many(
          {0, 1}, {0, 1}, {0, 0}, {0, std::numeric_limits<double>::quiet_NaN()}, 1),
      std::invalid_argument);
  EXPECT_TRUE(hermite_evaluate_many({0, 1}, {0, 1}, {0, 0}, {}).empty());
  EXPECT_EQ(hermite_interpolate({0, 1}, {0, 1}, {0, 0}, 2), -4);
  auto e        = hermite_evaluate({0, 1}, {0, 1}, {0, 0}, .5);
  auto copy     = e;
  copy.terms[0] = 999;
  EXPECT_NE(e.terms[0], 999);
}
