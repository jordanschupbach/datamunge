#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <datamunge/algorithms/gauss_jordan.hpp>
#include <limits>
#include <numeric>
#include <random>
using namespace datamunge::algorithms;
namespace {
using Vector = std::vector<double>;
long long det(const GaussianMatrix& a) {
  if (a.empty())
    return 1;
  long long value = 0;
  for (std::size_t col = 0; col < a.size(); ++col) {
    GaussianMatrix minor(a.size() - 1, Vector(a.size() - 1));
    for (std::size_t i = 1; i < a.size(); ++i) {
      std::size_t j2 = 0;
      for (std::size_t j = 0; j < a.size(); ++j)
        if (j != col)
          minor[i - 1][j2++] = a[i][j];
    }
    value += (col % 2 ? -1 : 1) * static_cast<long long>(a[0][col]) * det(minor);
  }
  return value;
}
void exact(const GaussianMatrix& a, const Vector& b) {
  const auto d = det(a);
  const auto s = gauss_jordan_solve(a, b);
  ASSERT_EQ(s.result.solved, d != 0);
  if (!d) {
    EXPECT_TRUE(s.result.x.empty());
    EXPECT_TRUE(s.result.inverse.empty());
    return;
  }
  const auto n = a.size();
  ASSERT_EQ(s.result.x.size(), n);
  ASSERT_EQ(s.result.inverse.size(), n);
  for (std::size_t i = 0; i < n; ++i) {
    auto c = a;
    for (std::size_t k = 0; k < n; ++k)
      c[k][i] = b[k];
    EXPECT_NEAR(s.result.x[i], double(det(c)) / d, 3e-12);
    for (std::size_t j = 0; j < n; ++j) {
      GaussianMatrix minor(n - 1, Vector(n - 1));
      std::size_t    r2 = 0;
      for (std::size_t r = 0; r < n; ++r)
        if (r != j) {
          std::size_t c2 = 0;
          for (std::size_t col = 0; col < n; ++col)
            if (col != i)
              minor[r2][c2++] = a[r][col];
          ++r2;
        }
      const double inverse = double(((i + j) % 2 ? -1 : 1) * det(minor)) / d;
      EXPECT_NEAR(s.result.inverse[i][j], inverse, 3e-12);
    }
  }
}
void near(double actual, long double expected) {
  EXPECT_NEAR(actual, expected, 3e-13L * (1 + std::fabs(expected)));
}
void certificate(const GaussJordanTrace& t) {
  const auto               n = t.input.size();
  auto                     a = t.input;
  auto                     b = t.rhs;
  GaussianMatrix           inv(n, Vector(n));
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  for (std::size_t i = 0; i < n; ++i)
    inv[i][i] = 1;
  std::size_t col = 0, row = 0, pivots = 0, swaps = 0, comparisons = 0, normalizations = 0,
              eliminations = 0;
  int    phase             = 0;
  bool   rejected          = false;
  double maxfactor         = 0;
  for (const auto& e : t.events) {
    ASSERT_FALSE(rejected);
    ASSERT_LT(col, n);
    EXPECT_EQ(e.column, col);
    if (e.kind == "pivot" || e.kind == "reject") {
      ASSERT_EQ(phase, 0);
      std::size_t p = col;
      for (std::size_t i = col + 1; i < n; ++i)
        if (std::fabs(a[i][col]) > std::fabs(a[p][col]))
          p = i;
      comparisons += n - col - 1;
      EXPECT_EQ(e.row, p);
      EXPECT_EQ(e.value, a[p][col]);
      if (e.kind == "reject") {
        EXPECT_LT(std::fabs(e.value), t.run.cutoff);
        rejected = true;
      } else {
        EXPECT_GE(std::fabs(e.value), t.run.cutoff);
        swaps += p != col;
        std::swap(a[col], a[p]);
        std::swap(inv[col], inv[p]);
        std::swap(b[col], b[p]);
        std::swap(order[col], order[p]);
        ++pivots;
        phase = 1;
      }
    } else if (e.kind == "normalize") {
      ASSERT_EQ(phase, 1);
      EXPECT_EQ(e.row, col);
      EXPECT_EQ(e.value, a[col][col]);
      for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
          if (i == col) {
            near(e.matrix[i][j], static_cast<long double>(a[i][j]) / e.value);
            near(e.transform[i][j], static_cast<long double>(inv[i][j]) / e.value);
          } else {
            EXPECT_EQ(e.matrix[i][j], a[i][j]);
            EXPECT_EQ(e.transform[i][j], inv[i][j]);
          }
        }
        if (i == col)
          near(e.rhs[i], static_cast<long double>(b[i]) / e.value);
        else
          EXPECT_EQ(e.rhs[i], b[i]);
      }
      a   = e.matrix;
      inv = e.transform;
      b   = e.rhs;
      ++normalizations;
      row = 0;
      if (row == col)
        ++row;
      phase = 2;
      if (n == 1) {
        ++col;
        phase = 0;
      }
    } else if (e.kind == "eliminate") {
      ASSERT_EQ(phase, 2);
      ASSERT_LT(row, n);
      EXPECT_EQ(e.row, row);
      EXPECT_EQ(e.value, a[row][col]);
      maxfactor = std::max(maxfactor, std::fabs(e.value));
      for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
          if (i == row) {
            near(
                e.matrix[i][j],
                static_cast<long double>(a[i][j]) - static_cast<long double>(e.value) * a[col][j]);
            near(
                e.transform[i][j],
                static_cast<long double>(inv[i][j]) -
                    static_cast<long double>(e.value) * inv[col][j]);
          } else {
            EXPECT_EQ(e.matrix[i][j], a[i][j]);
            EXPECT_EQ(e.transform[i][j], inv[i][j]);
          }
        }
        if (i == row)
          near(
              e.rhs[i],
              static_cast<long double>(b[i]) - static_cast<long double>(e.value) * b[col]);
        else
          EXPECT_EQ(e.rhs[i], b[i]);
      }
      a   = e.matrix;
      inv = e.transform;
      b   = e.rhs;
      ++eliminations;
      ++row;
      if (row == col)
        ++row;
      if (row == n) {
        ++col;
        phase = 0;
      }
    } else
      FAIL() << e.kind;
    EXPECT_EQ(e.matrix, a);
    EXPECT_EQ(e.transform, inv);
    EXPECT_EQ(e.rhs, b);
    EXPECT_EQ(e.row_order, order);
    // Independently certify the accumulated row-operation transformation T.
    for (std::size_t i = 0; i < n; ++i) {
      long double rb = 0, scale = 1;
      for (std::size_t k = 0; k < n; ++k) {
        const auto product = static_cast<long double>(inv[i][k]) * t.rhs[k];
        rb += product;
        scale += std::fabs(product);
      }
      EXPECT_NEAR(rb, b[i], 1e-11L * scale);
      for (std::size_t j = 0; j < n; ++j) {
        long double product = 0, magnitude = 1;
        for (std::size_t k = 0; k < n; ++k) {
          const auto term = static_cast<long double>(inv[i][k]) * t.input[k][j];
          product += term;
          magnitude += std::fabs(term);
        }
        EXPECT_NEAR(product, a[i][j], 1e-11L * magnitude);
      }
    }
  }
  EXPECT_EQ(t.run.result.solved, !rejected);
  EXPECT_EQ(t.run.reduced, a);
  EXPECT_EQ(t.run.rhs, b);
  EXPECT_EQ(t.run.row_order, order);
  if (!rejected) {
    EXPECT_EQ(t.run.result.x, b);
    EXPECT_EQ(t.run.result.inverse, inv);
    EXPECT_EQ(col, n);
  } else {
    EXPECT_TRUE(t.run.result.x.empty());
    EXPECT_TRUE(t.run.result.inverse.empty());
  }
  EXPECT_EQ(t.run.accepted_pivots, pivots);
  const auto& w = t.run.work;
  EXPECT_EQ(w.identity_diagonal_writes, n);
  EXPECT_EQ(w.pivot_comparisons, comparisons);
  EXPECT_EQ(w.row_swaps, swaps);
  EXPECT_EQ(w.normalizations, normalizations);
  EXPECT_EQ(w.eliminations, eliminations);
  EXPECT_EQ(w.a_divisions, normalizations * n);
  EXPECT_EQ(w.transform_divisions, normalizations * n);
  EXPECT_EQ(w.rhs_divisions, normalizations);
  EXPECT_EQ(w.a_updates, eliminations * n);
  EXPECT_EQ(w.transform_updates, eliminations * n);
  EXPECT_EQ(w.rhs_updates, eliminations);
  EXPECT_EQ(t.run.max_clear_factor, maxfactor);
  EXPECT_EQ(t.captured_entries, n * n + 2 * n + t.events.size() * (1 + 2 * n * n + 4 * n));
  EXPECT_LE(t.captured_entries, gauss_jordan_trace_entries(n));
}
} // namespace
TEST(JordanChecked, IndependentDeterminantAdjugateAndCramerAllSmallSystems) {
  exact({}, {});
  for (int code = 0; code < 625; ++code) {
    int            v = code;
    GaussianMatrix a(2, Vector(2));
    for (auto& row : a)
      for (double& x : row) {
        x = v % 5 - 2;
        v /= 5;
      }
    for (int rhs = 0; rhs < 25; ++rhs)
      exact(a, {double(rhs % 5 - 2), double(rhs / 5 - 2)});
  }
  for (int code = 0; code < 19683; ++code) {
    int            v = code;
    GaussianMatrix a(3, Vector(3));
    for (auto& row : a)
      for (double& x : row) {
        x = v % 3 - 1;
        v /= 3;
      }
    Vector b(3);
    for (std::size_t i = 0; i < 3; ++i)
      b[i] = a[i][0] - 2 * a[i][1] + 3 * a[i][2];
    exact(a, b);
  }
}
TEST(JordanChecked, ActualEventAndTransformationCertificates) {
  const auto f = gaussian_elimination_example();
  certificate(gauss_jordan_trace(f.a, f.b));
  certificate(gauss_jordan_trace({{1, 1}, {2, 3}}, {3, 8}));
  std::mt19937_64                    rng(127);
  std::uniform_int_distribution<int> u(-5, 5);
  for (int t = 0; t < 300; ++t) {
    const auto     n = std::size_t(t % 9);
    GaussianMatrix a(n, Vector(n));
    Vector         b(n);
    for (auto& row : a)
      for (double& v : row)
        v = u(rng);
    for (double& v : b)
      v = u(rng);
    certificate(gauss_jordan_trace(a, b));
  }
}
TEST(JordanChecked, RawAboveClearingFactorAndEarliestTies) {
  const auto t = gauss_jordan_trace({{1, 100}, {0, 1}}, {101, 1});
  certificate(t);
  EXPECT_EQ(t.run.max_clear_factor, 100);
  EXPECT_EQ(t.events.back().kind, "eliminate");
  EXPECT_EQ(t.events.back().row, 0u);
  EXPECT_EQ(t.events.back().value, 100);
  EXPECT_EQ(t.run.work.row_swaps, 0u);
  EXPECT_EQ(t.run.result.inverse, (GaussianMatrix{{1, -100}, {0, 1}}));
  EXPECT_EQ(t.run.result.x, (Vector{1, 1}));
  const auto tie = gauss_jordan_trace({{1, 1}, {-1, 2}}, {2, 1});
  certificate(tie);
  EXPECT_EQ(tie.events[0].row, 0u);
}
TEST(JordanChecked, SuccessfulFullBlockOperationIdentities) {
  for (std::size_t n = 0; n <= 64; ++n) {
    const auto a = gaussian_elimination_growth_matrix(n);
    const auto s = gauss_jordan_solve(a, Vector(n, 1));
    ASSERT_TRUE(s.result.solved);
    const auto& w = s.work;
    EXPECT_EQ(w.identity_diagonal_writes, n);
    EXPECT_EQ(w.pivot_comparisons, n * (n ? n - 1 : 0) / 2);
    EXPECT_EQ(w.row_swaps, 0u);
    EXPECT_EQ(w.normalizations, n);
    EXPECT_EQ(w.eliminations, n * (n ? n - 1 : 0));
    EXPECT_EQ(w.a_divisions, n * n);
    EXPECT_EQ(w.transform_divisions, n * n);
    EXPECT_EQ(w.rhs_divisions, n);
    EXPECT_EQ(w.a_updates, n * n * (n ? n - 1 : 0));
    EXPECT_EQ(w.transform_updates, n * n * (n ? n - 1 : 0));
    EXPECT_EQ(w.rhs_updates, n * (n ? n - 1 : 0));
  }
}
TEST(JordanChecked, AbsoluteCutoffAndPartialTransformationNotInverse) {
  const auto below = std::nextafter(1e-12, 0.);
  EXPECT_FALSE(gauss_jordan({{below}}, {below}).solved);
  EXPECT_TRUE(gauss_jordan({{1e-12}}, {1e-12}).solved);
  const GaussianMatrix a{{1e-13, 0}, {0, 1e-13}};
  EXPECT_FALSE(gauss_jordan(a, {1e-13, 2e-13}).solved);
  EXPECT_EQ(gauss_jordan_solve(a, {1e-13, 2e-13}, 1e-15).result.x, (Vector{1, 2}));
  for (const Vector& b : std::array<Vector, 2>{{{1, 2}, {1, 3}}}) {
    const auto t = gauss_jordan_trace({{1, 2}, {2, 4}}, b);
    certificate(t);
    EXPECT_FALSE(t.run.result.solved);
    ASSERT_EQ(t.events.back().kind, "reject");
    EXPECT_EQ(t.run.accepted_pivots, 1u);
    const auto d = inverse_residual_summary(t.input, t.events.back().transform);
    EXPECT_GT(d.left_error_inf, .1L);
  }
}
TEST(JordanChecked, CompleteDomainsBeforeQuotaAndZeroPivot) {
  const auto inf = std::numeric_limits<double>::infinity(),
             nan = std::numeric_limits<double>::quiet_NaN();
  for (const auto& f : std::vector<GaussianSystem>{
           {{}, {1}},
           {{{1}}, {1, 2}},
           {{{1, 99}}, {1}},
           {{{1, 2}, {3}}, {1, 2}},
           {{{0, 0}, {0, nan}}, {0, 0}},
           {{{0, 0}, {0, 1}}, {0, inf}},
           {{{inf}}, {1}}}) {
    EXPECT_THROW((void)gauss_jordan(f.a, f.b), std::invalid_argument);
    EXPECT_THROW((void)gauss_jordan_solve(f.a, f.b), std::invalid_argument);
    EXPECT_THROW((void)gauss_jordan_trace(f.a, f.b, 1), std::invalid_argument);
  }
  for (double cutoff : {0., -1., inf, nan}) {
    EXPECT_THROW((void)gauss_jordan_solve({}, {}, cutoff), std::invalid_argument);
    EXPECT_THROW((void)gauss_jordan_trace({}, {}, 1, cutoff), std::invalid_argument);
  }
}
TEST(JordanChecked, StoredCoefficientAndInverseOverflowWithoutMutation) {
  const auto           m = std::numeric_limits<double>::max();
  const GaussianMatrix a{{m, m}, {-m, m}};
  const Vector         b{m, m};
  const auto           saved = a;
  EXPECT_THROW((void)gauss_jordan(a, b), std::overflow_error);
  EXPECT_THROW((void)gauss_jordan_solve(a, b), std::overflow_error);
  EXPECT_THROW((void)gauss_jordan_trace(a, b), std::overflow_error);
  EXPECT_EQ(a, saved);
  EXPECT_EQ(b, (Vector{m, m}));
  EXPECT_THROW(
      (void)gauss_jordan_trace({{1e-309}}, {0}, 0, std::numeric_limits<double>::denorm_min()),
      std::overflow_error);
}
TEST(JordanChecked, WholeAdmissionSuccessfulExactAndEarlyRejectConservativeOwned) {
  EXPECT_EQ(gauss_jordan_trace_entries(2), 110u);
  EXPECT_THROW((void)gauss_jordan_trace({{1, 1}, {2, 3}}, {3, 8}, 109), std::length_error);
  auto t = gauss_jordan_trace({{1, 1}, {2, 3}}, {3, 8}, 110);
  EXPECT_EQ(t.captured_entries, 110u);
  EXPECT_EQ(gauss_jordan_trace({{1, 1}, {2, 3}}, {3, 8}, 0).captured_entries, 110u);
  GaussianMatrix a{{0, 0}, {0, 0}};
  Vector         b{0, 0};
  EXPECT_THROW((void)gauss_jordan_trace(a, b, 25), std::length_error);
  auto r = gauss_jordan_trace(a, b, 110);
  certificate(r);
  EXPECT_EQ(r.captured_entries, 25u);
  a.clear();
  b.clear();
  r.events[0].matrix[0][0] = 7;
  EXPECT_EQ(r.input[0][0], 0);
  EXPECT_EQ(r.run.reduced[0][0], 0);
  t.events[0].transform[0][0] = 999;
  EXPECT_NE(t.events[1].transform[0][0], 999);
  EXPECT_NE(t.run.result.inverse[0][0], 999);
  const auto f        = gaussian_elimination_example();
  const auto original = gauss_jordan_trace(f.a, f.b);
  EXPECT_EQ(original.captured_entries, 387u);
  EXPECT_EQ(gauss_jordan_trace_entries(3), 387u);
  EXPECT_THROW(
      (void)gauss_jordan_trace_entries(std::numeric_limits<std::size_t>::max()), std::length_error);
  EXPECT_THROW(
      (void)gauss_jordan_trace_entries(std::numeric_limits<std::size_t>::max() / 8),
      std::length_error);
}
TEST(JordanChecked, IndependentUnsnappedTwoSidedInverseDiagnosticsAndDomains) {
  const GaussianMatrix a{{1, 2}, {0, 1}}, v{{1, 0}, {3, 1}};
  const auto           d = inverse_residual_summary(a, v, 12);
  EXPECT_EQ(d.left_product, (std::vector<std::vector<long double>>{{7, 2}, {3, 1}}));
  EXPECT_EQ(d.right_product, (std::vector<std::vector<long double>>{{1, 2}, {3, 7}}));
  EXPECT_EQ(d.left_error_inf, 8);
  EXPECT_EQ(d.right_error_inf, 9);
  EXPECT_EQ(d.a_inf, 3);
  EXPECT_EQ(d.candidate_inf, 4);
  EXPECT_EQ(d.norm_product, 12);
  EXPECT_THROW((void)inverse_residual_summary(a, v, 11), std::length_error);
  EXPECT_EQ(inverse_residual_summary(a, v, 0).left_error_inf, 8);
  const auto eps   = std::ldexp(1., -40);
  const auto small = inverse_residual_summary({{1}}, {{1 + eps}});
  EXPECT_EQ(small.left_product[0][0], 1.L + eps);
  EXPECT_EQ(small.left_error_inf, eps);
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(
      (void)inverse_residual_summary({{0, 0}, {0, 1}}, {{0, 0}, {0, nan}}, 1),
      std::invalid_argument);
  EXPECT_THROW((void)inverse_residual_summary({{1}}, {}, 1), std::invalid_argument);
  EXPECT_THROW((void)inverse_residual_summary({{1, 2}}, {{1}}, 1), std::invalid_argument);
  const auto m = std::numeric_limits<double>::max();
  if (std::numeric_limits<long double>::max_exponent > std::numeric_limits<double>::max_exponent) {
    const auto wide = inverse_residual_summary({{m}}, {{m}});
    EXPECT_TRUE(std::isfinite(wide.left_product[0][0]));
    EXPECT_GT(wide.left_product[0][0], static_cast<long double>(m));
  }
}
TEST(JordanChecked, EmptyAndRandomKnownSolutionsWithBothInverseSides) {
  const auto t = gauss_jordan_trace({}, {}, 1);
  certificate(t);
  EXPECT_TRUE(t.run.result.solved);
  EXPECT_TRUE(t.run.result.inverse.empty());
  EXPECT_EQ(t.captured_entries, 0u);
  EXPECT_EQ(gauss_jordan_trace_entries(0), 0u);
  const auto z = inverse_residual_summary({}, {}, 1);
  EXPECT_EQ(z.left_error_inf, 0);
  EXPECT_EQ(z.right_error_inf, 0);
  EXPECT_EQ(z.norm_product, 0);
  std::mt19937                       rng(127);
  std::uniform_int_distribution<int> u(-3, 3);
  for (std::size_t n = 1; n <= 32; ++n) {
    GaussianMatrix a(n, Vector(n));
    Vector         b(n), truth(n);
    for (auto& v : truth)
      v = u(rng);
    for (std::size_t i = 0; i < n; ++i) {
      double sum = 0;
      for (std::size_t j = 0; j < n; ++j)
        if (j != i) {
          a[i][j] = u(rng);
          sum += std::fabs(a[i][j]);
        }
      a[i][i] = sum + 1;
    }
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        b[i] += a[i][j] * truth[j];
    const auto s = gauss_jordan_solve(a, b);
    ASSERT_TRUE(s.result.solved);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_NEAR(s.result.x[i], truth[i], 1e-11);
    const auto d = inverse_residual_summary(a, s.result.inverse);
    EXPECT_LT(d.left_error_inf, 1e-11L);
    EXPECT_LT(d.right_error_inf, 1e-11L);
  }
}
