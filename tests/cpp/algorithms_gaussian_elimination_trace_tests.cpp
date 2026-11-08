#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <datamunge/algorithms/gaussian_elimination.hpp>
#include <limits>
#include <numeric>
#include <random>
using namespace datamunge::algorithms;
namespace {
using Vector = std::vector<double>;
long long det(const GaussianMatrix& a) {
  if (a.empty())
    return 1;
  if (a.size() == 1)
    return static_cast<long long>(a[0][0]);
  if (a.size() == 2)
    return static_cast<long long>(a[0][0] * a[1][1] - a[0][1] * a[1][0]);
  return static_cast<long long>(
      a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
      a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
      a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]));
}
void cramer(const GaussianMatrix& a, const Vector& b) {
  const auto d = det(a);
  const auto s = gaussian_elimination_solve(a, b);
  ASSERT_EQ(s.solution.solved, d != 0);
  if (!d) {
    EXPECT_TRUE(s.solution.x.empty());
    return;
  }
  ASSERT_EQ(s.solution.x.size(), b.size());
  for (std::size_t j = 0; j < b.size(); ++j) {
    auto c = a;
    for (std::size_t i = 0; i < b.size(); ++i)
      c[i][j] = b[i];
    EXPECT_NEAR(s.solution.x[j], double(det(c)) / double(d), 2e-12);
  }
  const auto r = gaussian_residual_summary(a, b, s.solution.x);
  EXPECT_LT(r.residual_inf, 1e-10L);
}
void near(double actual, long double expected) {
  EXPECT_NEAR(actual, expected, 1e-13L * (1 + std::fabs(expected)));
}
void certificate(const GaussianEliminationTrace& t) {
  const auto               n = t.input.size();
  auto                     a = t.input;
  auto                     b = t.rhs;
  Vector                   x;
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::size_t col = 0, eliminate = 0, pivots = 0, swaps = 0, comparisons = 0, updates = 0,
              products = 0, backs = 0, entries = n * n + 2 * n;
  double maxfactor = 0, maxentry = 0;
  for (const auto& row : a)
    for (double v : row)
      maxentry = std::max(maxentry, std::fabs(v));
  std::size_t next_row = 0;
  bool        rejected = false, back_phase = false;
  for (const auto& e : t.events) {
    ASSERT_FALSE(rejected);
    ASSERT_EQ(e.matrix.size(), n);
    ASSERT_EQ(e.rhs.size(), n);
    ASSERT_EQ(e.row_order.size(), n);
    if (e.kind == "pivot" || e.kind == "reject") {
      ASSERT_FALSE(back_phase);
      EXPECT_EQ(e.column, col);
      ASSERT_LT(col, n);
      std::size_t pivot = col;
      for (std::size_t row = col + 1; row < n; ++row)
        if (std::fabs(a[row][col]) > std::fabs(a[pivot][col]))
          pivot = row;
      comparisons += n - col - 1;
      EXPECT_EQ(e.row, pivot);
      EXPECT_EQ(e.value, a[pivot][col]);
      if (e.kind == "reject") {
        EXPECT_LT(std::fabs(e.value), t.run.cutoff);
        rejected = true;
      } else {
        EXPECT_GE(std::fabs(e.value), t.run.cutoff);
        swaps += pivot != col;
        std::swap(a[col], a[pivot]);
        std::swap(b[col], b[pivot]);
        std::swap(order[col], order[pivot]);
        ++pivots;
        next_row = col + 1;
        if (col + 1 == n)
          ++col;
      }
    } else if (e.kind == "eliminate") {
      ASSERT_FALSE(back_phase);
      ASSERT_LT(col, n);
      EXPECT_EQ(e.column, col);
      EXPECT_EQ(e.row, next_row++);
      ASSERT_GT(e.row, col);
      ASSERT_LT(e.row, n);
      const long double factor = static_cast<long double>(a[e.row][col]) / a[col][col];
      near(e.value, factor);
      EXPECT_LE(std::fabs(e.value), 1.0);
      maxfactor = std::max(maxfactor, std::fabs(e.value));
      for (std::size_t r = 0; r < n; ++r)
        for (std::size_t c = 0; c < n; ++c) {
          if (r == e.row && c >= col)
            near(
                e.matrix[r][c],
                static_cast<long double>(a[r][c]) - static_cast<long double>(e.value) * a[col][c]);
          else
            EXPECT_EQ(e.matrix[r][c], a[r][c]);
        }
      for (std::size_t r = 0; r < n; ++r) {
        if (r == e.row)
          near(
              e.rhs[r],
              static_cast<long double>(b[r]) - static_cast<long double>(e.value) * b[col]);
        else
          EXPECT_EQ(e.rhs[r], b[r]);
      }
      updates += n - col;
      ++eliminate;
      // Current row is the final row of this column, then the next pivot begins.
      if (e.row == n - 1)
        ++col;
      a = e.matrix;
      b = e.rhs;
    } else if (e.kind == "back") {
      back_phase = true;
      ASSERT_EQ(col, n);
      ASSERT_LT(backs, n);
      const auto i = n - 1 - backs;
      EXPECT_EQ(e.column, i);
      EXPECT_EQ(e.row, i);
      if (x.empty())
        x.assign(n, 0);
      long double s = b[i];
      for (std::size_t j = i + 1; j < n; ++j)
        s -= static_cast<long double>(a[i][j]) * x[j];
      near(e.value, s / a[i][i]);
      products += n - i - 1;
      x[i] = e.value;
      ++backs;
    } else
      FAIL() << e.kind;
    EXPECT_EQ(e.matrix, a);
    EXPECT_EQ(e.rhs, b);
    EXPECT_EQ(e.x, x);
    EXPECT_EQ(e.row_order, order);
    for (const auto& row : a)
      for (double v : row)
        maxentry = std::max(maxentry, std::fabs(v));
    entries += 1 + n + n * n + n + x.size() + n;
  }
  EXPECT_EQ(t.run.solution.solved, !rejected);
  EXPECT_EQ(t.run.upper, a);
  EXPECT_EQ(t.run.rhs, b);
  EXPECT_EQ(t.run.row_order, order);
  EXPECT_EQ(t.run.solution.x, x);
  EXPECT_EQ(t.run.accepted_pivots, pivots);
  EXPECT_EQ(t.run.work.pivot_comparisons, comparisons);
  EXPECT_EQ(t.run.work.row_swaps, swaps);
  EXPECT_EQ(t.run.work.eliminations, eliminate);
  EXPECT_EQ(t.run.work.rhs_updates, eliminate);
  EXPECT_EQ(t.run.work.coefficient_updates, updates);
  EXPECT_EQ(t.run.work.back_products, products);
  EXPECT_EQ(t.run.work.back_divisions, backs);
  EXPECT_EQ(t.run.max_multiplier, maxfactor);
  EXPECT_EQ(t.run.max_coefficient, maxentry);
  EXPECT_EQ(t.captured_entries, entries);
  EXPECT_LE(entries, gaussian_elimination_trace_entries(n));
}
} // namespace
TEST(GaussianChecked, IndependentIntegerDeterminantsAndCramerAllSmallSystems) {
  cramer({}, {});
  for (int code = 0; code < 625; ++code) {
    int            v = code;
    GaussianMatrix a(2, Vector(2));
    for (auto& row : a)
      for (auto& x : row) {
        x = v % 5 - 2;
        v /= 5;
      }
    for (int rhs = 0; rhs < 25; ++rhs)
      cramer(a, {double(rhs % 5 - 2), double(rhs / 5 - 2)});
  }
  for (int code = 0; code < 19683; ++code) {
    int            v = code;
    GaussianMatrix a(3, Vector(3));
    for (auto& row : a)
      for (auto& x : row) {
        x = v % 3 - 1;
        v /= 3;
      }
    Vector b(3);
    for (int i = 0; i < 3; ++i)
      b[i] = a[i][0] - 2 * a[i][1] + 3 * a[i][2];
    cramer(a, b);
  }
}
TEST(GaussianChecked, ActualEventCertificatesRandomAndOriginal) {
  const auto f = gaussian_elimination_example();
  certificate(gaussian_elimination_trace(f.a, f.b));
  std::mt19937_64                    rng(126);
  std::uniform_int_distribution<int> u(-9, 9);
  for (int trial = 0; trial < 300; ++trial) {
    const auto     n = std::size_t(trial % 10);
    GaussianMatrix a(n, Vector(n));
    Vector         b(n);
    for (auto& row : a)
      for (auto& v : row)
        v = u(rng);
    for (auto& v : b)
      v = u(rng);
    certificate(gaussian_elimination_trace(a, b));
  }
}
TEST(GaussianChecked, PivotSwapTieActualUpperAndNoPretendedExactness) {
  const auto t = gaussian_elimination_trace({{0, 1}, {2, 1}}, {1, 3});
  certificate(t);
  ASSERT_EQ(t.events.size(), 5u);
  EXPECT_EQ(t.events[0].row, 1u);
  EXPECT_EQ(t.events[1].kind, "eliminate");
  EXPECT_EQ(t.events[1].value, 0);
  EXPECT_EQ(t.run.row_order, (std::vector<std::size_t>{1, 0}));
  EXPECT_EQ(t.run.solution.x, (Vector{1, 1}));
  const auto tie = gaussian_elimination_trace({{1, 1}, {-1, 2}}, {2, 1});
  certificate(tie);
  EXPECT_EQ(tie.events[0].row, 0u);
  EXPECT_EQ(tie.run.work.row_swaps, 0u);
  const auto f = gaussian_elimination_example();
  const auto s = gaussian_elimination_solve(f.a, f.b);
  EXPECT_EQ(s.row_order, (std::vector<std::size_t>{1, 2, 0}));
  EXPECT_EQ(s.work.row_swaps, 2u);
  EXPECT_EQ(s.work.coefficient_updates, 8u);
  EXPECT_NEAR(s.upper[2][2], .2, 1e-15);
  EXPECT_NEAR(s.solution.x[1], 3, 1e-14);
}
TEST(GaussianChecked, WilkinsonGrowthDespiteBoundedMultipliersAndExactWork) {
  for (std::size_t n = 1; n <= 128; ++n) {
    const auto a = gaussian_elimination_growth_matrix(n);
    const auto s = gaussian_elimination_solve(a, Vector(n, 1));
    ASSERT_TRUE(s.solution.solved);
    EXPECT_EQ(s.work.row_swaps, 0u);
    EXPECT_EQ(s.work.pivot_comparisons, n * (n - 1) / 2);
    EXPECT_EQ(s.work.eliminations, n * (n - 1) / 2);
    EXPECT_EQ(s.work.coefficient_updates, n * (n - 1) * (n + 1) / 3);
    EXPECT_EQ(s.work.rhs_updates, n * (n - 1) / 2);
    EXPECT_EQ(s.work.back_products, n * (n - 1) / 2);
    EXPECT_EQ(s.work.back_divisions, n);
    EXPECT_EQ(s.max_multiplier, n > 1 ? 1 : 0);
    EXPECT_EQ(s.growth, std::ldexp(1.L, int(n - 1)));
  }
}
TEST(GaussianChecked, AbsoluteThresholdScaleEqualityAndSingularFailure) {
  const double below = std::nextafter(1e-12, 0.);
  EXPECT_FALSE(gaussian_elimination({{below}}, {below}).solved);
  EXPECT_TRUE(gaussian_elimination({{1e-12}}, {1e-12}).solved);
  const GaussianMatrix tiny{{1e-13, 0}, {0, 1e-13}};
  EXPECT_FALSE(gaussian_elimination(tiny, {1e-13, 2e-13}).solved);
  EXPECT_EQ(gaussian_elimination_solve(tiny, {1e-13, 2e-13}, 1e-15).solution.x, (Vector{1, 2}));
  for (Vector b : std::array<Vector, 2>{{{1, 2}, {1, 3}}}) {
    const auto t = gaussian_elimination_trace({{1, 2}, {2, 4}}, b);
    certificate(t);
    EXPECT_FALSE(t.run.solution.solved);
    EXPECT_EQ(t.events.back().kind, "reject");
    EXPECT_EQ(t.run.accepted_pivots, 1u);
    EXPECT_TRUE(t.run.solution.x.empty());
  }
}
TEST(GaussianChecked, FullValidationPrecedesQuotaAndEarlyCutoff) {
  const auto inf = std::numeric_limits<double>::infinity(),
             nan = std::numeric_limits<double>::quiet_NaN();
  for (const GaussianSystem f : std::vector<GaussianSystem>{
           {{}, {1}},
           {{{1, 2}, {3}}, {1, 2}},
           {{{1, 2, 3}, {3, 4}}, {1, 2}},
           {{{1}}, {1, 2}},
           {{{1}}, {}},
           {{{0, 0}, {0, nan}}, {0, 0}},
           {{{0, 0}, {0, 1}}, {0, inf}},
           {{{inf}}, {1}}}) {
    EXPECT_THROW((void)gaussian_elimination(f.a, f.b), std::invalid_argument);
    EXPECT_THROW((void)gaussian_elimination_solve(f.a, f.b), std::invalid_argument);
    EXPECT_THROW((void)gaussian_elimination_trace(f.a, f.b, 1), std::invalid_argument);
  }
  for (double c : {0., -1., inf, nan}) {
    EXPECT_THROW((void)gaussian_elimination_solve({}, {}, c), std::invalid_argument);
    EXPECT_THROW((void)gaussian_elimination_trace({}, {}, 1, c), std::invalid_argument);
  }
}
TEST(GaussianChecked, NonfiniteAssignmentsThrowWithoutCallerMutation) {
  const auto           m = std::numeric_limits<double>::max();
  const GaussianMatrix a{{m, m}, {-m, m}};
  const Vector         b{m, m};
  const auto           saved = a;
  EXPECT_THROW((void)gaussian_elimination(a, b), std::overflow_error);
  EXPECT_THROW((void)gaussian_elimination_solve(a, b), std::overflow_error);
  EXPECT_THROW((void)gaussian_elimination_trace(a, b), std::overflow_error);
  EXPECT_EQ(a, saved);
  EXPECT_EQ(b, (Vector{m, m}));
  EXPECT_THROW((void)gaussian_elimination({{1e-12}}, {m}), std::overflow_error);
}
TEST(GaussianChecked, ConservativeWholeQuotaOwnedSnapshotsAndVirtualOverflow) {
  GaussianMatrix a{{0, 1}, {2, 1}};
  Vector         b{1, 3};
  EXPECT_EQ(gaussian_elimination_trace_entries(2), 73u);
  EXPECT_THROW((void)gaussian_elimination_trace(a, b, 67), std::length_error);
  EXPECT_THROW((void)gaussian_elimination_trace(a, b, 72), std::length_error);
  auto t = gaussian_elimination_trace(a, b, 73);
  EXPECT_EQ(t.captured_entries, 67u);
  a[0].clear();
  b.clear();
  t.events[0].matrix[0][0] = 999;
  EXPECT_EQ(t.input[0], (Vector{0, 1}));
  EXPECT_EQ(t.events[1].matrix[0][0], 2);
  EXPECT_EQ(t.run.upper[0][0], 2);
  EXPECT_EQ(t.rhs, (Vector{1, 3}));
  EXPECT_EQ(gaussian_elimination_trace({{0, 1}, {2, 1}}, {1, 3}, 0).captured_entries, 67u);
  EXPECT_EQ(gaussian_elimination_growth_matrix(2, 6).size(), 2u);
  EXPECT_THROW((void)gaussian_elimination_growth_matrix(2, 5), std::length_error);
  for (auto n :
       {std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::size_t>::max() / 4}) {
    EXPECT_THROW((void)gaussian_elimination_trace_entries(n), std::length_error);
    EXPECT_THROW((void)gaussian_elimination_growth_matrix(n, 0), std::length_error);
  }
  // Early failure is not discounted by conservative admission.
  EXPECT_THROW((void)gaussian_elimination_trace({{0, 0}, {0, 0}}, {0, 0}, 67), std::length_error);
}
TEST(GaussianChecked, WidenedResidualBackwardVersusForwardAndDomains) {
  const auto s = gaussian_residual_summary({{2}}, {4}, {1});
  EXPECT_EQ(s.residual, (std::vector<long double>{2}));
  EXPECT_NEAR(s.backward_error, 1.L / 3, 1e-18L);
  const GaussianMatrix a{{1, 0}, {0, 1e-10}};
  const auto           r = gaussian_residual_summary(a, {1, 1e-10}, {1, 2});
  EXPECT_LT(r.backward_error, 4e-11L);
  EXPECT_GT(std::fabs(2. - 1), .9);
  EXPECT_NEAR(r.residual_inf, 1e-10L, 1e-25L);
  const auto z = gaussian_residual_summary({{0}}, {0}, {0});
  EXPECT_EQ(z.backward_error, 0);
  const double m = std::numeric_limits<double>::max();
  if (std::numeric_limits<long double>::max_exponent > std::numeric_limits<double>::max_exponent) {
    const auto wide = gaussian_residual_summary({{m}}, {0}, {m});
    EXPECT_TRUE(std::isfinite(wide.residual_inf));
    EXPECT_GT(wide.residual_inf, static_cast<long double>(m));
    EXPECT_EQ(wide.backward_error, 1);
  }
  EXPECT_THROW((void)gaussian_residual_summary({{1}}, {1}, {}), std::invalid_argument);
  EXPECT_THROW(
      (void)gaussian_residual_summary({{1}}, {1}, {std::numeric_limits<double>::infinity()}),
      std::invalid_argument);
  EXPECT_THROW(
      (void)gaussian_residual_summary(
          {{0, 0}, {0, std::numeric_limits<double>::quiet_NaN()}}, {0, 0}, {0, 0}),
      std::invalid_argument);
}
TEST(GaussianChecked, EmptyResultsAndRandomKnownSolutions) {
  const auto t = gaussian_elimination_trace({}, {}, 1);
  certificate(t);
  EXPECT_TRUE(t.run.solution.solved);
  EXPECT_EQ(t.captured_entries, 0u);
  EXPECT_TRUE(t.events.empty());
  EXPECT_EQ(t.run.growth, 1);
  EXPECT_EQ(gaussian_elimination_trace_entries(0), 0u);
  EXPECT_TRUE(gaussian_elimination_growth_matrix(0, 1).empty());
  EXPECT_EQ(gaussian_residual_summary({}, {}, {}).backward_error, 0);
  std::mt19937                       rng(126);
  std::uniform_int_distribution<int> u(-3, 3);
  for (std::size_t n = 1; n <= 64; ++n) {
    GaussianMatrix a(n, Vector(n));
    Vector         truth(n), b(n);
    for (auto& v : truth)
      v = u(rng);
    for (std::size_t i = 0; i < n; ++i) {
      double sum = 0;
      for (std::size_t j = 0; j < n; ++j)
        if (i != j) {
          a[i][j] = u(rng);
          sum += std::fabs(a[i][j]);
        }
      a[i][i] = sum + 1;
    }
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        b[i] += a[i][j] * truth[j];
    const auto s = gaussian_elimination_solve(a, b);
    ASSERT_TRUE(s.solution.solved);
    for (std::size_t i = 0; i < n; ++i)
      EXPECT_NEAR(s.solution.x[i], truth[i], 1e-11);
  }
}
