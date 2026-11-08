/// @file gauss_jordan.cpp
/// @brief Checked shared full-block reduction and actual-time evidence, with independent products.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/gauss_jordan.hpp>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace datamunge::algorithms {
/// @brief Private full-reduction admission, validation and actual execution helpers.
namespace gauss_jordan_detail {
/// @brief Checked count sum.
/// @param a First count.
/// @param b Second count.
/// @return Exact sum.
/// @throws std::length_error Sum exceeds size_t.
std::size_t add(std::size_t a, std::size_t b) {
  if (a > std::numeric_limits<std::size_t>::max() - b)
    throw std::length_error("Gauss-Jordan count sum");
  return a + b;
}
/// @brief Checked count product.
/// @param a First count.
/// @param b Second count.
/// @return Exact product.
/// @throws std::length_error Product exceeds size_t.
std::size_t mul(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::length_error("Gauss-Jordan count product");
  return a * b;
}
/// @brief Check matrix/vector dimensions and return cell count before allocation.
/// @param n Matrix order.
/// @return n² cell count.
/// @throws std::length_error Count or container capacity exceeded.
std::size_t cells(std::size_t n) {
  if (n > GaussianMatrix{}.max_size() || n > std::vector<double>{}.max_size() ||
      n > std::vector<std::size_t>{}.max_size())
    throw std::length_error("Gauss-Jordan dimensions");
  return mul(n, n);
}
/// @brief Check complete-work counters and return maximum actual event count.
/// @param n System order.
/// @return n²+n; zero for empty.
/// @throws std::length_error Count, work counter or container capacity exceeded.
std::size_t events(std::size_t n) {
  const auto square = cells(n);
  if (n)
    (void)mul(square, n - 1);
  return add(square, n);
}
/// @brief Validate every row length before all finite values, without allocation.
/// @param a Matrix to check.
/// @throws std::invalid_argument Matrix is not square or contains a nonfinite entry.
void square(const GaussianMatrix& a) {
  for (const auto& row : a)
    if (row.size() != a.size())
      throw std::invalid_argument("Gauss-Jordan square shape");
  for (const auto& row : a)
    for (double v : row)
      if (!std::isfinite(v))
        throw std::invalid_argument("Gauss-Jordan finite coefficients");
}
/// @brief Validate complete matrix, RHS and positive finite control without allocation.
/// @param a Square finite matrix.
/// @param b Matching finite RHS.
/// @param cutoff Absolute threshold.
/// @throws std::invalid_argument Invalid complete input or cutoff.
void validate(const GaussianMatrix& a, const std::vector<double>& b, double cutoff) {
  if (a.size() != b.size())
    throw std::invalid_argument("Gauss-Jordan RHS length");
  square(a);
  for (double v : b)
    if (!std::isfinite(v))
      throw std::invalid_argument("Gauss-Jordan finite RHS");
  if (!(cutoff > 0) || !std::isfinite(cutoff))
    throw std::invalid_argument("Gauss-Jordan positive finite cutoff");
}
/// @brief Check an actual stored double without changing the recurrence or its order.
/// @param v Stored result.
/// @throws std::overflow_error Result is nonfinite.
void finite(double v) {
  if (!std::isfinite(v))
    throw std::overflow_error("Gauss-Jordan nonfinite stored arithmetic");
}
/// @brief Check widened independent product/norm arithmetic.
/// @param v Computed long-double result.
/// @return The same finite value.
/// @throws std::overflow_error Widened result is nonfinite.
long double wide(long double v) {
  if (!std::isfinite(v))
    throw std::overflow_error("Inverse diagnostic arithmetic");
  return v;
}
/// @brief Append one actual-time snapshot only when a whole-admitted trace is requested.
/// @param trace Nullable admitted output.
/// @param kind Actual event name.
/// @param col Pivot column.
/// @param row Selected, normalized or cleared row.
/// @param value Actual pivot, divisor or raw clearing factor.
/// @param a Current coefficient block.
/// @param transform Current identity/transformation block.
/// @param b Current RHS.
/// @param order Cumulative row-interchange IDs.
/// @throws std::bad_alloc Snapshot allocation fails.
void capture(
    GaussJordanTrace*               trace,
    const char*                     kind,
    std::size_t                     col,
    std::size_t                     row,
    double                          value,
    const GaussianMatrix&           a,
    const GaussianMatrix&           transform,
    const std::vector<double>&      b,
    const std::vector<std::size_t>& order) {
  if (!trace)
    return;
  trace->events.push_back({kind, col, row, value, a, transform, b, order});
  const auto n = a.size();
  trace->captured_entries += 1 + 2 * n * n + 4 * n;
}
/// @brief Reduce the prevalidated owned augmented blocks with exactly the original double order.
/// Every full row division/update is retained, including zero factors and earlier
/// columns. A failed pivot returns no legacy inverse; trace retains its partial
/// transformation. Final result.inverse moves the successful working block.
/// @param a Owned coefficients, copied after admission or moved from legacy arguments.
/// @param b Owned matching RHS.
/// @param cutoff Positive finite absolute threshold.
/// @param trace Nullable whole-admitted trace output.
/// @return Actual final coefficient/RHS state, counters and legacy-layout result.
/// @throws std::overflow_error A stored assignment becomes nonfinite.
/// @throws std::bad_alloc Working, output or snapshot allocation fails.
GaussJordanRun
run(GaussianMatrix a, std::vector<double> b, double cutoff, GaussJordanTrace* trace) {
  const auto     n = a.size();
  GaussJordanRun out;
  out.cutoff = cutoff;
  GaussianMatrix inv(n, std::vector<double>(n, 0.0));
  out.row_order.resize(n);
  std::iota(out.row_order.begin(), out.row_order.end(), 0);
  for (std::size_t i = 0; i < n; ++i) {
    inv[i][i] = 1.0;
    ++out.work.identity_diagonal_writes;
  }
  for (std::size_t col = 0; col < n; ++col) {
    std::size_t pivot = col;
    for (std::size_t r = col + 1; r < n; ++r) {
      ++out.work.pivot_comparisons;
      if (std::fabs(a[r][col]) > std::fabs(a[pivot][col]))
        pivot = r;
    }
    if (std::fabs(a[pivot][col]) < cutoff) {
      capture(trace, "reject", col, pivot, a[pivot][col], a, inv, b, out.row_order);
      out.reduced = std::move(a);
      out.rhs     = std::move(b);
      return out;
    }
    if (pivot != col)
      ++out.work.row_swaps;
    std::swap(a[col], a[pivot]);
    std::swap(inv[col], inv[pivot]);
    std::swap(b[col], b[pivot]);
    std::swap(out.row_order[col], out.row_order[pivot]);
    ++out.accepted_pivots;
    capture(trace, "pivot", col, pivot, a[col][col], a, inv, b, out.row_order);
    const double d = a[col][col];
    for (std::size_t k = 0; k < n; ++k) {
      a[col][k] /= d;
      finite(a[col][k]);
      ++out.work.a_divisions;
      inv[col][k] /= d;
      finite(inv[col][k]);
      ++out.work.transform_divisions;
    }
    b[col] /= d;
    finite(b[col]);
    ++out.work.rhs_divisions;
    ++out.work.normalizations;
    capture(trace, "normalize", col, col, d, a, inv, b, out.row_order);
    for (std::size_t r = 0; r < n; ++r) {
      if (r == col)
        continue;
      const double f       = a[r][col];
      out.max_clear_factor = std::max(out.max_clear_factor, std::fabs(f));
      for (std::size_t k = 0; k < n; ++k) {
        a[r][k] -= f * a[col][k];
        finite(a[r][k]);
        ++out.work.a_updates;
        inv[r][k] -= f * inv[col][k];
        finite(inv[r][k]);
        ++out.work.transform_updates;
      }
      b[r] -= f * b[col];
      finite(b[r]);
      ++out.work.rhs_updates;
      ++out.work.eliminations;
      capture(trace, "eliminate", col, r, f, a, inv, b, out.row_order);
    }
  }
  out.result.solved  = true;
  out.rhs            = b;
  out.result.x       = std::move(b);
  out.result.inverse = std::move(inv);
  out.reduced        = std::move(a);
  return out;
}
} // namespace gauss_jordan_detail

/// @copydoc gauss_jordan_trace_entries()
std::size_t gauss_jordan_trace_entries(std::size_t n) {
  using namespace gauss_jordan_detail;
  const auto count = events(n), square = cells(n);
  if (count > std::vector<GaussJordanEvent>{}.max_size())
    throw std::length_error("Gauss-Jordan event capacity");
  return add(add(square, mul(2, n)), mul(count, add(1, add(mul(2, square), mul(4, n)))));
}
/// @copydoc gauss_jordan_solve()
GaussJordanRun
gauss_jordan_solve(const GaussianMatrix& a, const std::vector<double>& b, double cutoff) {
  gauss_jordan_detail::validate(a, b, cutoff);
  (void)gauss_jordan_detail::events(a.size());
  return gauss_jordan_detail::run(a, b, cutoff, nullptr);
}
/// @copydoc gauss_jordan_trace()
GaussJordanTrace gauss_jordan_trace(
    const GaussianMatrix&      a,
    const std::vector<double>& b,
    std::size_t                max_entries,
    double                     cutoff) {
  using namespace gauss_jordan_detail;
  validate(a, b, cutoff);
  const auto bound = gauss_jordan_trace_entries(a.size());
  if (max_entries && bound > max_entries)
    throw std::length_error("Gauss-Jordan whole history quota");
  GaussJordanTrace out;
  out.input            = a;
  out.rhs              = b;
  out.captured_entries = add(cells(a.size()), mul(2, a.size()));
  out.run              = run(a, b, cutoff, &out);
  return out;
}
/// @copydoc gauss_jordan()
GaussJordanResult gauss_jordan(std::vector<std::vector<double>> a, std::vector<double> b) {
  gauss_jordan_detail::validate(a, b, 1e-12);
  (void)gauss_jordan_detail::events(a.size());
  return gauss_jordan_detail::run(std::move(a), std::move(b), 1e-12, nullptr).result;
}
/// @copydoc inverse_residual_summary()
InverseResidualSummary inverse_residual_summary(
    const GaussianMatrix& a,
    const GaussianMatrix& candidate,
    std::size_t           max_entries) {
  using namespace gauss_jordan_detail;
  if (a.size() != candidate.size())
    throw std::invalid_argument("Inverse candidate order");
  square(a);
  square(candidate);
  const auto n = a.size();
  if (n > std::vector<long double>{}.max_size() ||
      n > std::vector<std::vector<long double>>{}.max_size())
    throw std::length_error("Inverse product dimensions");
  const auto entries = mul(2, add(n, cells(n)));
  if (max_entries && entries > max_entries)
    throw std::length_error("Inverse product quota");
  InverseResidualSummary out;
  out.left_product.assign(n, std::vector<long double>(n, 0));
  out.right_product.assign(n, std::vector<long double>(n, 0));
  for (std::size_t i = 0; i < n; ++i) {
    long double anorm = 0, vnorm = 0, lerr = 0, rerr = 0;
    for (std::size_t j = 0; j < n; ++j) {
      anorm            = wide(anorm + std::fabs(static_cast<long double>(a[i][j])));
      vnorm            = wide(vnorm + std::fabs(static_cast<long double>(candidate[i][j])));
      long double left = 0, right = 0;
      for (std::size_t k = 0; k < n; ++k) {
        left  = wide(left + wide(static_cast<long double>(a[i][k]) * candidate[k][j]));
        right = wide(right + wide(static_cast<long double>(candidate[i][k]) * a[k][j]));
      }
      out.left_product[i][j]  = left;
      out.right_product[i][j] = right;
      lerr                    = wide(lerr + std::fabs(wide(left - (i == j ? 1.L : 0.L))));
      rerr                    = wide(rerr + std::fabs(wide(right - (i == j ? 1.L : 0.L))));
    }
    out.a_inf           = std::max(out.a_inf, anorm);
    out.candidate_inf   = std::max(out.candidate_inf, vnorm);
    out.left_error_inf  = std::max(out.left_error_inf, lerr);
    out.right_error_inf = std::max(out.right_error_inf, rerr);
  }
  out.norm_product = wide(out.a_inf * out.candidate_inf);
  return out;
}
} // namespace datamunge::algorithms
