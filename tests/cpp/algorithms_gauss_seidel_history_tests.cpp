#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <datamunge/algorithms/gauss_seidel.hpp>
#include <limits>
#include <random>
#include <stdexcept>

namespace {
using namespace datamunge::algorithms;
using M = std::vector<std::vector<double>>;
using V = std::vector<double>;

TEST(GaussSeidelHistory, HandComputedMixedReadStatesAndNumerators) {
  const auto t = gauss_seidel_history({{2, -1}, {-1, 2}}, {1, 1}, 1, 0, 16);
  ASSERT_EQ(t.coordinates.size(), 2);
  EXPECT_EQ(t.initial, (V{0, 0}));
  EXPECT_DOUBLE_EQ(t.initial_residual, std::sqrt(2.));
  const auto& first  = t.coordinates[0];
  const auto& second = t.coordinates[1];
  EXPECT_EQ(first.sweep, 1);
  EXPECT_EQ(first.row, 0);
  EXPECT_EQ(first.numerator, 1);
  EXPECT_DOUBLE_EQ(first.previous, 0);
  EXPECT_DOUBLE_EQ(first.value, .5);
  EXPECT_EQ(first.before, (V{0, 0}));
  EXPECT_EQ(first.after, (V{.5, 0}));
  EXPECT_EQ(second.sweep, 1);
  EXPECT_EQ(second.row, 1);
  EXPECT_EQ(second.numerator, 1.5L);
  EXPECT_EQ(second.before, (V{.5, 0}));
  EXPECT_EQ(second.after, (V{.5, .75}));
  ASSERT_EQ(t.sweeps.size(), 1);
  EXPECT_EQ(t.sweeps[0].x, (V{.5, .75}));
  EXPECT_DOUBLE_EQ(t.sweeps[0].residual, .75);
  EXPECT_FALSE(t.run.solution.converged);
  EXPECT_EQ(t.captured_entries, 16);
  EXPECT_EQ(t.run.work.residual_evaluations, 2);
}

TEST(GaussSeidelHistory, ClosedFormBinaryTwoByTwoFamilyIncludesDivergence) {
  // Independent exact formula: e_k=(-q^(2k-1),-q^(2k)); no solver replay.
  for (double q : {.25, .5, .75, 1.25}) {
    const auto t = gauss_seidel_history({{1, -q}, {-q, 1}}, {1 - q, 1 - q}, 8, 0, 0);
    ASSERT_EQ(t.sweeps.size(), 8);
    for (int k = 1; k <= 8; ++k) {
      const auto&       s = t.sweeps[k - 1];
      const long double x = 1 - std::pow(static_cast<long double>(q), 2 * k - 1);
      const long double y = 1 - std::pow(static_cast<long double>(q), 2 * k);
      EXPECT_DOUBLE_EQ(s.x[0], static_cast<double>(x));
      EXPECT_DOUBLE_EQ(s.x[1], static_cast<double>(y));
      const long double residual =
          std::abs(std::pow(static_cast<long double>(q), 2 * k - 1) * (1 - q * q));
      EXPECT_NEAR(s.residual, static_cast<double>(residual), 3e-15 * std::max(1.L, residual));
      EXPECT_EQ(s.work.coordinate_updates, 2 * k);
      EXPECT_EQ(s.work.off_diagonal_products, 2 * k);
      EXPECT_EQ(s.work.residual_products, 4 * k);
      EXPECT_EQ(s.work.residual_evaluations, k + 1);
    }
    EXPECT_EQ(t.run.solution.x, t.sweeps.back().x);
    EXPECT_EQ(t.captured_entries, 3 + 8 * 13);
    if (q > 1)
      EXPECT_GT(t.sweeps.back().residual, t.initial_residual);
  }
}

TEST(GaussSeidelHistory, DominantSystemCanGrowResidualAndOrderingChangesCoordinates) {
  // Strictly row dominant, nilpotent error iteration; exact binary arithmetic.
  const auto t = gauss_seidel_history({{1, .5}, {0, 1. / 128}}, {0, 1}, 10, 0, 0);
  ASSERT_EQ(t.sweeps.size(), 2);
  EXPECT_DOUBLE_EQ(t.initial_residual, 1);
  EXPECT_DOUBLE_EQ(t.sweeps[0].residual, 64);
  EXPECT_DOUBLE_EQ(t.sweeps[1].residual, 0);
  EXPECT_EQ(t.run.solution.x, (V{-64, 128}));
  EXPECT_TRUE(t.run.solution.converged);
  // Same symmetric equations in reversed variable/equation order. Map back explicitly.
  const auto forward = gauss_seidel_solve({{2, -1}, {-1, 2}}, {1, 1}, 1, 0);
  const auto reverse = gauss_seidel_solve({{2, -1}, {-1, 2}}, {1, 1}, 1, 0);
  const V    reverse_mapped{reverse.solution.x[1], reverse.solution.x[0]};
  EXPECT_EQ(forward.solution.x, (V{.5, .75}));
  EXPECT_EQ(reverse_mapped, (V{.75, .5}));
}

TEST(GaussSeidelHistory, OriginalSystemStopsInclusivelyAndPreservesLegacyResult) {
  const M    a{{4, -1, 1}, {4, -8, 1}, {-2, 1, 5}};
  const V    b{7, -21, 15};
  const auto t = gauss_seidel_history(a, b);
  const auto s = gauss_seidel(a, b);
  ASSERT_TRUE(s.converged);
  EXPECT_EQ(s.iterations, 13);
  EXPECT_EQ(t.run.solution.x, s.x);
  EXPECT_EQ(t.run.solution.residual, s.residual);
  EXPECT_EQ(t.coordinates.size(), 39);
  EXPECT_EQ(t.run.work.off_diagonal_products, 78);
  EXPECT_EQ(t.run.work.residual_products, 117);
  EXPECT_EQ(t.run.work.residual_evaluations, 14);
  for (std::size_t i = 0; i < 3; ++i)
    EXPECT_NEAR(s.x[i], (V{2, 4, 3})[i], 3e-11);
  const auto boundary = gauss_seidel_history({{2, -1}, {-1, 2}}, {1, 1}, 20, .75, 0);
  EXPECT_EQ(boundary.run.solution.iterations, 1);
  EXPECT_TRUE(boundary.run.solution.converged);
  EXPECT_EQ(boundary.captured_entries, 16); // whole admitted bound exceeds actual storage.
  EXPECT_EQ(gauss_seidel_history_entries(3, 1000), 25004);
}

TEST(GaussSeidelHistory, ValidateCompleteInputBeforeInitialSuccessOrQuota) {
  const auto inf = std::numeric_limits<double>::infinity();
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  for (const M& a : {M{{0}}, M{{nan}}, M{{inf}}, M{{1, 2}}, M{}}) {
    EXPECT_THROW(gauss_seidel_history(a, {0}, 1000, 0, 1), std::invalid_argument);
    EXPECT_THROW(gauss_seidel_solve(a, {0}), std::invalid_argument);
  }
  EXPECT_THROW(gauss_seidel_history({{1}}, {nan}, 0, 0, 1), std::invalid_argument);
  EXPECT_THROW(gauss_seidel_history({{1}}, {inf}, 0, 0, 1), std::invalid_argument);
  for (double tol : {-1., inf, nan})
    EXPECT_THROW(gauss_seidel_history({{1}}, {0}, 0, tol, 1), std::invalid_argument);
  EXPECT_THROW(gauss_seidel_solve({{1}}, {0}, -1), std::invalid_argument);
  EXPECT_THROW(
      gauss_seidel_history({{1}}, {0}, std::numeric_limits<int>::min(), 0, 1),
      std::invalid_argument);
  EXPECT_THROW(gauss_seidel_history_entries(0, -1), std::invalid_argument);
  EXPECT_THROW(gauss_seidel_history({{1, 0}, {0, 0}}, {0, 0}, 0, 0, 1), std::invalid_argument);
}

TEST(GaussSeidelHistory, CheckedArithmeticRejectsUnrepresentableStatesAndNorms) {
  const auto huge = std::numeric_limits<double>::max();
  const auto tiny = std::numeric_limits<double>::denorm_min();
  EXPECT_THROW(gauss_seidel_history({{tiny}}, {huge}, 1, 0, 0), std::overflow_error);
  EXPECT_THROW(gauss_seidel_solve({{1, 0}, {0, 1}}, {huge, huge}), std::overflow_error);
  EXPECT_THROW(gauss_seidel_history({{1, huge}, {0, 1}}, {0, huge}, 1, 0, 0), std::overflow_error);
  EXPECT_THROW(
      gauss_seidel_history_entries(std::numeric_limits<std::size_t>::max(), 0), std::length_error);
  EXPECT_THROW(
      gauss_seidel_history_entries(std::numeric_limits<std::size_t>::max() / 2, 1),
      std::length_error);
  EXPECT_EQ(gauss_seidel_history_entries(0, std::numeric_limits<int>::max()), 1);
}

TEST(GaussSeidelHistory, WholeAdmissionIsDistinctFromLegacyOnlineResidualCap) {
  EXPECT_THROW(gauss_seidel_history({{2, -1}, {-1, 2}}, {1, 1}, 1, 0, 15), std::length_error);
  EXPECT_NO_THROW(gauss_seidel_history({{2, -1}, {-1, 2}}, {1, 1}, 1, 0, 16));
  EXPECT_THROW(gauss_seidel_history({{2}}, {0}, 5, 0, 2), std::length_error);
  EXPECT_NO_THROW(gauss_seidel_trace({{2}}, {0}, 5, 0, 1));
  EXPECT_THROW(gauss_seidel_trace({{2}}, {4}, 5, 0, 1), std::length_error);
  const auto zero = gauss_seidel_history({{2}}, {0}, 5, 0, 0);
  EXPECT_EQ(zero.captured_entries, 2);
  EXPECT_TRUE(zero.run.solution.converged);
  EXPECT_TRUE(zero.coordinates.empty());
  EXPECT_TRUE(zero.sweeps.empty());
  EXPECT_EQ(gauss_seidel_history_entries(1, 5), 27);
  const auto empty = gauss_seidel_history({}, {}, std::numeric_limits<int>::max(), 0, 1);
  EXPECT_EQ(empty.captured_entries, 1);
  EXPECT_TRUE(empty.run.solution.converged);
}

TEST(GaussSeidelHistory, SnapshotsAndResultAreIndependentOwnedCopies) {
  M          a{{2, -1}, {-1, 2}};
  V          b{1, 1};
  const auto olda = a;
  const auto oldb = b;
  auto       t    = gauss_seidel_history(a, b, 2, 0, 0);
  EXPECT_EQ(a, olda);
  EXPECT_EQ(b, oldb);
  const auto result         = t.run.solution.x;
  const auto next           = t.coordinates[1].before;
  a[0][0]                   = 999;
  b[0]                      = 999;
  t.coordinates[0].after[0] = 999;
  EXPECT_EQ(t.coordinates[1].before, next);
  EXPECT_EQ(t.run.solution.x, result);
  t.sweeps.back().x[0] = 999;
  t.initial[0]         = 999;
  EXPECT_EQ(t.run.solution.x, result);
  EXPECT_EQ(t.coordinates[0].before, (V{0, 0}));
}

TEST(GaussSeidelHistory, DenseWorkAndWholeHistoryUnitsIncludeZeroCoefficients) {
  for (std::size_t n = 0; n <= 64; ++n) {
    M a(n, V(n));
    for (std::size_t i = 0; i < n; ++i)
      a[i][i] = 2;
    const auto  t = gauss_seidel_history(a, V(n, 1), 1, 0, 0);
    const auto& w = t.run.work;
    EXPECT_EQ(w.coordinate_updates, n);
    EXPECT_EQ(w.divisions, n);
    EXPECT_EQ(w.off_diagonal_products, n * (n ? n - 1 : 0));
    EXPECT_EQ(w.residual_products, n * n);
    EXPECT_EQ(w.residual_evaluations, n ? 2 : 1);
    EXPECT_EQ(t.captured_entries, gauss_seidel_history_entries(n, 1));
    EXPECT_EQ(t.captured_entries, n ? 2 * n * n + 3 * n + 2 : 1);
    const auto no = gauss_seidel_solve(a, V(n, 1), 0, 0);
    EXPECT_EQ(no.solution.iterations, 0);
    EXPECT_EQ(no.work.coordinate_updates, 0);
    EXPECT_EQ(no.work.residual_evaluations, 1);
    EXPECT_EQ(no.solution.converged, n == 0);
    const auto z = gauss_seidel_solve(a, V(n, 0));
    EXPECT_TRUE(z.solution.converged);
    EXPECT_EQ(z.solution.iterations, 0);
    EXPECT_EQ(z.work.residual_products, 0);
  }
}

long double det(const std::array<std::array<long double, 3>, 3>& a) {
  return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
         a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
         a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
}
TEST(GaussSeidelHistory, IndependentCramerRuleOnThousandDominantIntegerSystems) {
  std::mt19937                       rng(129);
  std::uniform_int_distribution<int> u(-3, 3);
  for (int trial = 0; trial < 1000; ++trial) {
    M                                         a(3, V(3));
    V                                         b(3);
    std::array<std::array<long double, 3>, 3> exact{};
    for (std::size_t i = 0; i < 3; ++i) {
      double d = 1;
      for (std::size_t j = 0; j < 3; ++j)
        if (i != j) {
          a[i][j] = u(rng);
          d += std::abs(a[i][j]);
        }
      a[i][i] = d;
      b[i]    = u(rng);
      for (std::size_t j = 0; j < 3; ++j)
        exact[i][j] = a[i][j];
    }
    const auto denominator = det(exact);
    ASSERT_NE(denominator, 0);
    const auto t = gauss_seidel_history(a, b, 500, 1e-10, 0);
    ASSERT_TRUE(t.run.solution.converged);
    for (std::size_t j = 0; j < 3; ++j) {
      auto replaced = exact;
      for (std::size_t i = 0; i < 3; ++i)
        replaced[i][j] = b[i];
      EXPECT_NEAR(t.run.solution.x[j], static_cast<double>(det(replaced) / denominator), 2e-10);
    }
    EXPECT_EQ(t.run.solution.x, gauss_seidel_solve(a, b, 500, 1e-10).solution.x);
    EXPECT_LE(t.captured_entries, gauss_seidel_history_entries(3, 500));
    for (const auto& e : t.coordinates) {
      ASSERT_EQ(e.before.size(), 3);
      ASSERT_EQ(e.after.size(), 3);
      for (std::size_t j = 0; j < 3; ++j)
        if (j != e.row)
          EXPECT_EQ(e.before[j], e.after[j]);
      EXPECT_EQ(e.after[e.row], e.value);
      EXPECT_EQ(e.before[e.row], e.previous);
      EXPECT_EQ(e.value, static_cast<double>(e.numerator / a[e.row][e.row]));
    }
  }
}
} // namespace
