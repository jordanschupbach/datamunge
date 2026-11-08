#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <climits>
#include <cmath>
#include <cstdint>
#include <datamunge/algorithms/pde_solvers.hpp>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>
#include <type_traits>
namespace {
using namespace datamunge::algorithms;
constexpr double pi = std::numbers::pi_v<double>;
TEST(FiniteDifferenceNative, TinyAssemblyAndBothBoundaryFolds) {
  auto r = finite_difference_poisson_solve([](double) { return 1.; }, 0, 0, 1);
  EXPECT_EQ(r.x, (std::vector<double>{0, .5, 1}));
  EXPECT_EQ(r.u, (std::vector<double>{0, .125, 0}));
  EXPECT_EQ(r.source, (std::vector<double>{1}));
  EXPECT_EQ(r.rhs, r.source);
  EXPECT_EQ(r.diagonal, 8);
  EXPECT_EQ(r.off_diagonal, -4);
  EXPECT_EQ(r.max_abs_residual, 0);
  r = finite_difference_poisson_solve([](double) { return 2.; }, 2, 5, 1);
  EXPECT_EQ(r.source[0], 2);
  EXPECT_EQ(r.rhs[0], 30);
  EXPECT_EQ(r.u[1], 3.75);
  r = finite_difference_poisson_solve([](double) { return 1.; }, 2, 5, 3);
  EXPECT_EQ(r.rhs, (std::vector<double>{33, 1, 81}));
  EXPECT_EQ(r.diagonal, 32);
  EXPECT_EQ(r.off_diagonal, -16);
  EXPECT_EQ(r.u, (std::vector<double>{2, 2.84375, 3.625, 4.34375, 5}));
}
TEST(FiniteDifferenceNative, IndependentDiscreteGreenFunction) {
  std::mt19937 rng(111);
  for (int fixture = 0; fixture < 400; ++fixture) {
    const int           n = 1 + rng() % 40;
    const double        a = int(rng() % 31) - 15, b = int(rng() % 31) - 15;
    std::vector<double> f(n);
    for (auto& v : f)
      v = (int(rng() % 101) - 50) / 8.;
    int        calls = 0;
    const auto r = finite_difference_poisson_solve([&](double) { return f.at(calls++); }, a, b, n);
    EXPECT_EQ(calls, n);
    const long double m = n + 1, h = 1 / m;
    for (int i = 1; i <= n; ++i) {
      long double oracle = a + (static_cast<long double>(b) - a) * i / m;
      for (int j = 1; j <= n; ++j)
        oracle += h * h * std::min(i, j) * (m - std::max(i, j)) / m * f[j - 1];
      EXPECT_NEAR(
          r.u[i], static_cast<double>(oracle), 2e-11 * (1 + std::abs(static_cast<double>(oracle))));
    }
  }
}
TEST(FiniteDifferenceNative, IndependentDensePivotedElimination) {
  std::mt19937 rng(911);
  for (int fixture = 0; fixture < 160; ++fixture) {
    const int           n = 1 + rng() % 14;
    const double        a = int(rng() % 11) - 5, b = int(rng() % 11) - 5;
    std::vector<double> f(n);
    for (auto& v : f)
      v = int(rng() % 17) - 8;
    int        calls = 0;
    const auto r     = finite_difference_poisson_solve([&](double) { return f[calls++]; }, a, b, n);
    std::vector<std::vector<long double>> m(n, std::vector<long double>(n + 1));
    for (int i = 0; i < n; ++i) {
      m[i][i] = 2;
      if (i)
        m[i][i - 1] = -1;
      if (i + 1 < n)
        m[i][i + 1] = -1;
      m[i][n] = static_cast<long double>(f[i]) / (static_cast<long double>(n + 1) * (n + 1));
    }
    m[0][n] += a;
    m[n - 1][n] += b;
    for (int i = 0; i < n; ++i) {
      int pivot = i;
      for (int j = i + 1; j < n; ++j)
        if (std::abs(m[j][i]) > std::abs(m[pivot][i]))
          pivot = j;
      std::swap(m[i], m[pivot]);
      const auto d = m[i][i];
      for (int j = i; j <= n; ++j)
        m[i][j] /= d;
      for (int k = 0; k < n; ++k)
        if (k != i) {
          auto w = m[k][i];
          for (int j = i; j <= n; ++j)
            m[k][j] -= w * m[i][j];
        }
    }
    for (int i = 0; i < n; ++i)
      EXPECT_NEAR(r.u[i + 1], static_cast<double>(m[i][n]), 2e-12);
  }
}
TEST(FiniteDifferenceNative, PolynomialNodalExactnessAndMaximumPrinciple) {
  for (int n = 1; n <= 256; ++n) {
    const auto r = finite_difference_poisson_solve([](double x) { return -6 * x + 4; }, 2, 4, n);
    for (std::size_t i = 0; i < r.x.size(); ++i) {
      double x = r.x[i];
      EXPECT_NEAR(r.u[i], x * x * x - 2 * x * x + 3 * x + 2, 4e-10);
    }
    const auto z = finite_difference_poisson([](double) { return 0.; }, -3, 5, n);
    for (double x : z) {
      EXPECT_GE(x, -3 - 1e-10);
      EXPECT_LE(x, 5 + 1e-10);
    }
  }
}
TEST(FiniteDifferenceNative, SineDiscreteEigenmodeAndOriginalRefinements) {
  double old = 0;
  for (int n : {9, 19, 39, 79, 159, 319}) {
    const auto r = finite_difference_poisson_solve(
        [](double x) { return pi * pi * std::sin(pi * x); }, 0, 0, n);
    const double lam = 4 * std::pow(std::sin(pi * r.h / 2), 2) / (r.h * r.h), gain = pi * pi / lam;
    double       error = 0;
    for (int i = 1; i <= n; ++i) {
      EXPECT_NEAR(r.u[i], gain * std::sin(pi * i * r.h), 2e-11);
      error = std::max(error, std::abs(r.u[i] - std::sin(pi * i * r.h)));
    }
    EXPECT_NEAR(error, gain - 1, 2e-11);
    if (old)
      EXPECT_NEAR(old / error, 4., .025);
    old = error;
  }
}
TEST(FiniteDifferenceNative, CallbacksOnceOrderedAndOrdinaryBitParity) {
  for (int n : {1, 2, 3, 7, 19, 128, 1001}) {
    std::vector<double> x, y;
    const auto          r = finite_difference_poisson_solve(
        [&](double t) {
          x.push_back(t);
          return t * t - 2 * t;
        },
        -1,
        2,
        n);
    const auto u = finite_difference_poisson(
        [&](double t) {
          y.push_back(t);
          return t * t - 2 * t;
        },
        -1,
        2,
        n);
    ASSERT_EQ(x.size(), n);
    EXPECT_EQ(x, y);
    EXPECT_EQ(r.u, u);
    for (int i = 0; i < n; ++i) {
      EXPECT_EQ(x[i], (i + 1) * r.h);
      EXPECT_EQ(x[i], r.x[i + 1]);
      EXPECT_EQ(r.source[i], x[i] * x[i] - 2 * x[i]);
    }
  }
  struct Failure {};
  int calls = 0;
  try {
    finite_difference_poisson(
        [&](double) {
          if (++calls == 3)
            throw Failure{};
          return 1.;
        },
        0,
        0,
        9);
    FAIL();
  } catch (const Failure&) {
  }
  EXPECT_EQ(calls, 3);
}
TEST(FiniteDifferenceNative, ResidualDefinitionAndOwnedSnapshots) {
  auto   r       = finite_difference_poisson_solve([](double x) { return std::exp(x); }, 1, 2, 97);
  auto   copy    = r;
  double largest = 0;
  for (std::size_t i = 0; i < r.residual.size(); ++i) {
    long double q = static_cast<long double>(r.diagonal) * r.u[i + 1];
    if (i)
      q += static_cast<long double>(r.off_diagonal) * r.u[i];
    if (i + 1 < r.residual.size())
      q += static_cast<long double>(r.off_diagonal) * r.u[i + 2];
    q -= r.rhs[i];
    EXPECT_EQ(r.residual[i], static_cast<double>(q));
    largest = std::max(largest, std::abs(r.residual[i]));
  }
  EXPECT_EQ(largest, r.max_abs_residual);
  r.x[1]        = 99;
  r.source[0]   = 99;
  r.rhs[0]      = 99;
  r.u[1]        = 99;
  r.residual[0] = 99;
  EXPECT_NE(copy.x[1], 99);
  EXPECT_NE(copy.source[0], 99);
  EXPECT_NE(copy.rhs[0], 99);
  EXPECT_NE(copy.u[1], 99);
  EXPECT_NE(copy.residual[0], 99);
}
TEST(FiniteDifferenceNative, DomainAndCapacityBeforeCallbacks) {
  int  calls = 0;
  auto f     = [&](double) {
    ++calls;
    return 0.;
  };
  const double nan = std::numeric_limits<double>::quiet_NaN(),
               inf = std::numeric_limits<double>::infinity();
  for (int n : {INT_MIN, -1, 0}) {
    EXPECT_THROW(finite_difference_poisson(f, 0, 0, n), std::invalid_argument);
    EXPECT_THROW(finite_difference_poisson_solve(f, 0, 0, n), std::invalid_argument);
  }
  for (int n : {INT_MAX - 1, INT_MAX}) {
    EXPECT_THROW(finite_difference_poisson(f, 0, 0, n), std::length_error);
    EXPECT_THROW(finite_difference_poisson_solve(f, 0, 0, n), std::length_error);
  }
  EXPECT_THROW(finite_difference_poisson({}, 0, 0, 1), std::invalid_argument);
  for (double z : {nan, inf, -inf}) {
    EXPECT_THROW(finite_difference_poisson(f, z, 0, 1), std::invalid_argument);
    EXPECT_THROW(finite_difference_poisson(f, 0, z, 1), std::invalid_argument);
  }
  EXPECT_EQ(calls, 0);
}
TEST(FiniteDifferenceNative, NonfiniteSamplesAndFiniteArithmeticOverflow) {
  const double nan = std::numeric_limits<double>::quiet_NaN(),
               inf = std::numeric_limits<double>::infinity(),
               max = std::numeric_limits<double>::max();
  for (double x : {nan, inf, -inf}) {
    int calls = 0;
    EXPECT_THROW(
        finite_difference_poisson_solve(
            [&](double) {
              ++calls;
              return x;
            },
            0,
            0,
            3),
        std::invalid_argument);
    EXPECT_EQ(calls, 1);
  }
  int calls = 0;
  EXPECT_THROW(
      finite_difference_poisson(
          [&](double) {
            ++calls;
            return 0.;
          },
          max,
          0,
          1),
      std::overflow_error);
  EXPECT_EQ(calls, 0);
  EXPECT_THROW(
      finite_difference_poisson([&](double) { return max; }, 0, 0, 3), std::overflow_error);
  // Finite boundary contributions separately, but adding both to one RHS overflows.
  EXPECT_THROW(
      finite_difference_poisson([](double) { return 0.; }, max / 5, max / 5, 1),
      std::overflow_error);
}
TEST(FiniteDifferenceNative, LegacySignatureAndBoundaryBits) {
  using Old = std::vector<double> (*)(const std::function<double(double)>&, double, double, int);
  static_assert(std::is_same_v<decltype(&finite_difference_poisson), Old>);
  const auto r = finite_difference_poisson_solve([](double) { return 0.; }, -0., 0., 1);
  EXPECT_EQ(std::bit_cast<std::uint64_t>(r.u[0]), std::bit_cast<std::uint64_t>(-0.));
  EXPECT_EQ(std::bit_cast<std::uint64_t>(r.u[2]), 0u);
}
} // namespace
