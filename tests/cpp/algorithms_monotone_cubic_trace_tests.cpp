#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/monotone_cubic.hpp>
#include <limits>
#include <stdexcept>
using namespace datamunge::algorithms;
TEST(MonotoneCubicTrace, ExhaustiveIntervalShapeNodesAndActualTangentHistory) {
  for (int code = 0; code < 625; ++code) {
    int                 q = code;
    std::vector<double> x{0, .5, 2, 4}, y(4);
    for (auto& v : y) {
      v = q % 5 - 2;
      q /= 5;
    }
    MonotoneCubic       c(x, y);
    std::vector<double> d(3), m(4);
    for (int i = 0; i < 3; ++i)
      d[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
    m[0] = d[0];
    m[3] = d[2];
    for (int i = 1; i < 3; ++i)
      m[i] = .5 * d[i - 1] + .5 * d[i];
    for (int i = 0; i < 3; ++i) {
      auto& r = c.decisions()[i];
      EXPECT_EQ(r.interval, i);
      EXPECT_EQ(r.secant, d[i]);
      EXPECT_EQ(r.left_before, m[i]);
      EXPECT_EQ(r.right_before, m[i + 1]);
      if (d[i] == 0) {
        m[i] = m[i + 1] = 0;
        EXPECT_TRUE(r.flat);
      } else {
        if (m[i] * d[i] < 0)
          m[i] = 0;
        if (m[i + 1] * d[i] < 0)
          m[i + 1] = 0;
        long double a = static_cast<long double>(m[i]) / d[i],
                    b = static_cast<long double>(m[i + 1]) / d[i], norm = std::sqrt(a * a + b * b);
        if (norm > 3) {
          m[i]     = double(m[i] * 3 / norm);
          m[i + 1] = double(m[i + 1] * 3 / norm);
        }
      }
      EXPECT_NEAR(r.left_after, m[i], 1e-14);
      EXPECT_NEAR(r.right_after, m[i + 1], 1e-14);
    }
    for (int i = 0; i < 4; ++i) {
      EXPECT_EQ(c.evaluate(x[i]), y[i]);
      EXPECT_NEAR(c.tangents()[i], m[i], 1e-14);
    }
    for (int i = 0; i < 3; ++i) {
      double prev = y[i];
      for (int k = 1; k <= 100; ++k) {
        double t = k / 100.0, v = c.evaluate(x[i] + t * (x[i + 1] - x[i]));
        EXPECT_GE(v, std::min(y[i], y[i + 1]) - 1e-12);
        EXPECT_LE(v, std::max(y[i], y[i + 1]) + 1e-12);
        if (y[i + 1] >= y[i])
          EXPECT_GE(v, prev - 1e-12);
        else
          EXPECT_LE(v, prev + 1e-12);
        prev = v;
      }
    }
  }
}
TEST(MonotoneCubicTrace, WrongSignRegressionClampsGuardsOwnershipAndCaps) {
  MonotoneCubic c({0, 1, 2}, {0, 1, -100});
  EXPECT_EQ(c.tangents()[1], 0);
  EXPECT_TRUE(c.decisions()[0].right_clamped);
  EXPECT_EQ(c.evaluate(-4), 0);
  EXPECT_EQ(c.evaluate(10), -100);
  EXPECT_THROW(MonotoneCubic({}, {}), std::invalid_argument);
  EXPECT_THROW(MonotoneCubic({0, 0}, {0, 1}), std::invalid_argument);
  EXPECT_THROW(MonotoneCubic({1, 0}, {0, 1}), std::invalid_argument);
  EXPECT_THROW(MonotoneCubic({0, 1}, {0}), std::invalid_argument);
  auto inf = std::numeric_limits<double>::infinity();
  EXPECT_THROW(MonotoneCubic({0, inf}, {0, 1}), std::invalid_argument);
  EXPECT_THROW(c.evaluate(inf), std::invalid_argument);
  EXPECT_THROW(c.evaluate(std::nan("")), std::invalid_argument);
  EXPECT_THROW(MonotoneCubic({0, 1}, {0, 1}, 1), std::length_error);
  EXPECT_NO_THROW(MonotoneCubic({0, 1}, {0, 1}, 2));
  EXPECT_NO_THROW(MonotoneCubic({0, 1}, {0, 1}, 0));
  std::vector<double> x{0, 1}, y{0, 2};
  MonotoneCubic       owned(x, y);
  x.clear();
  y.clear();
  EXPECT_EQ(owned.evaluate(.5), 1);
  EXPECT_EQ(monotone_cubic_interpolate({0, 1}, {0, 2}, 2), 2);
}
