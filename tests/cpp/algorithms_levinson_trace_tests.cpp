#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/algorithms/linear_solvers.hpp>
#include <limits>
using namespace datamunge::algorithms;
TEST(LevinsonTrace, OriginalEightByEightAndActualPrefixes) {
  std::vector<double> row{4, 2, 1, .5, .25, .1, .05, .02}, rhs(8, 1);
  auto                t = levinson_trace(row, rhs);
  EXPECT_TRUE(t.analysis.solved);
  EXPECT_EQ(t.states.size(), 8);
  EXPECT_EQ(t.attempts.size(), 7);
  EXPECT_EQ(t.analysis.solution, levinson_solve(row, rhs));
  for (auto& s : t.states) {
    std::vector<double> r(row.begin(), row.begin() + s.order),
        b(rhs.begin(), rhs.begin() + s.order);
    auto independent = gaussian_elimination(symmetric_toeplitz_matrix(r), b);
    ASSERT_TRUE(independent.solved);
    for (std::size_t i = 0; i < s.order; ++i)
      EXPECT_NEAR(s.solution[i], independent.x[i], 1e-13);
    std::vector<double> e(s.order);
    e[0] = 1;
    EXPECT_LT(levinson_residual(r, e, s.forward).infinity_norm, 1e-13);
    e[0]     = 0;
    e.back() = 1;
    EXPECT_LT(levinson_residual(r, e, s.backward).infinity_norm, 1e-13);
  }
  EXPECT_EQ(t.analysis.work.dot_products, 84);
  EXPECT_EQ(t.analysis.work.vector_updates, 105);
  EXPECT_LT(levinson_residual(row, rhs, t.analysis.solution).backward_error, 1e-15);
}
TEST(LevinsonTrace, IndependentARInverseAndDenseReference) {
  for (int n : {2, 4, 8, 16, 32, 64})
    for (double rho : {-.8, -.2, .2, .8, .99}) {
      std::vector<double> row(n), rhs(n, 1);
      for (int i = 0; i < n; ++i)
        row[i] = std::pow(rho, i);
      auto t = levinson_trace(row, rhs);
      ASSERT_TRUE(t.analysis.solved);
      auto last = t.states.back();
      EXPECT_NEAR(last.forward[0], 1 / (1 - rho * rho), 1e-10);
      EXPECT_NEAR(last.forward[1], -rho / (1 - rho * rho), 1e-10);
      for (int i = 2; i < n; ++i)
        EXPECT_NEAR(last.forward[i], 0, 1e-10);
      auto g = gaussian_elimination(symmetric_toeplitz_matrix(row), rhs);
      ASSERT_TRUE(g.solved);
      for (int i = 0; i < n; ++i)
        EXPECT_NEAR(t.analysis.solution[i], g.x[i], 1e-10);
    }
}
TEST(LevinsonTrace, InvertibleMatrixCanBreakAtLeadingOrder) {
  std::vector<double> r{1, 1, 0}, b{1, 2, 3};
  auto                g = gaussian_elimination(symmetric_toeplitz_matrix(r), b);
  EXPECT_TRUE(g.solved);
  auto t = levinson_trace(r, b);
  EXPECT_FALSE(t.analysis.solved);
  EXPECT_EQ(t.analysis.breakdown_order, 2);
  EXPECT_EQ(t.analysis.solution.size(), 1);
  EXPECT_EQ(t.states.size(), 1);
  ASSERT_EQ(t.attempts.size(), 1);
  EXPECT_EQ(t.attempts[0].denominator, 0);
  EXPECT_THROW(levinson_solve(r, b), std::domain_error);
  EXPECT_FALSE(levinson_analyze({0, 1}, {1, 2}).solved);
  EXPECT_TRUE(levinson_analyze({-2, 0}, {2, 4}).solved);
}
TEST(LevinsonTrace, CompleteGuardsQuotaOwnershipAndEmpty) {
  double q = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(levinson_solve({1}, {}), std::invalid_argument);
  EXPECT_THROW(levinson_solve({1, q}, {1, 2}), std::invalid_argument);
  EXPECT_THROW(levinson_solve({1, 0}, {1, q}), std::invalid_argument);
  EXPECT_THROW(
      levinson_solve({std::numeric_limits<double>::denorm_min()}, {1}), std::overflow_error);
  auto t = levinson_trace({2, 1}, {1, 1});
  EXPECT_EQ(t.captured_entries, 16);
  EXPECT_THROW(levinson_trace({2, 1}, {1, 1}, 15), std::length_error);
  EXPECT_EQ(levinson_trace({2, 1}, {1, 1}, 16).captured_entries, 16);
  EXPECT_TRUE(levinson_trace({}, {}).analysis.solved);
  EXPECT_EQ(levinson_trace({}, {}).captured_entries, 0);
  std::vector<double> r{2, 1};
  auto                a = levinson_trace(r, {1, 1}, 0);
  r[0]                  = 4;
  EXPECT_EQ(a.first_row[0], 2);
  EXPECT_THROW(levinson_residual({1}, {1}, {}), std::invalid_argument);
  EXPECT_EQ(levinson_residual({}, {}, {}).backward_error, 0);
}
