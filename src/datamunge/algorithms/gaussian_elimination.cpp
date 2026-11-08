/// @file gaussian_elimination.cpp
/// @brief Shared checked scalar elimination and actual-time histories, without numerical replay.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/gaussian_elimination.hpp>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace datamunge::algorithms {
/// @brief Internal arithmetic, validation, admission and actual shared execution.
namespace gaussian_elimination_detail {
/// @brief Add two virtual counts without wraparound.
/// @param a First count.
/// @param b Second count.
/// @return Exact sum.
/// @throws std::length_error Sum exceeds size_t.
std::size_t add(std::size_t a, std::size_t b) {
  if (a > std::numeric_limits<std::size_t>::max() - b)
    throw std::length_error("Gaussian count addition");
  return a + b;
}
/// @brief Multiply two virtual counts without wraparound.
/// @param a First count.
/// @param b Second count.
/// @return Exact product.
/// @throws std::length_error Product exceeds size_t.
std::size_t mul(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::length_error("Gaussian count multiplication");
  return a * b;
}
/// @brief Validate row/cell capacities before matrix construction.
/// @param n Matrix order.
/// @return Number of cells, excluding row records.
/// @throws std::length_error Arithmetic or vector capacities exceeded.
std::size_t cells(std::size_t n) {
  if (n > GaussianMatrix{}.max_size() || n > std::vector<double>{}.max_size() ||
      n > std::vector<std::size_t>{}.max_size())
    throw std::length_error("Gaussian matrix dimension");
  return mul(n, n);
}
/// @brief Check ordinary work counters and return maximum actual event count.
/// @param n Number of unknowns.
/// @return n(n-1)/2+2n; zero for n0.
/// @throws std::length_error Matrix dimensions or work counters exceed capacities.
std::size_t events(std::size_t n) {
  (void)cells(n);
  if (!n)
    return 0;
  const auto pairs = n % 2 ? mul(n, (n - 1) / 2) : mul(n / 2, n - 1);
  // Actual coefficient updates sum k*(k-1), k=1..n, equals n*(n-1)*(n+1)/3.
  auto a = n, b = n - 1, c = add(n, 1);
  if (a % 3 == 0)
    a /= 3;
  else if (b % 3 == 0)
    b /= 3;
  else
    c /= 3;
  (void)mul(mul(a, b), c);
  return add(pairs, mul(2, n));
}
/// @brief Validate all input shapes before scanning all finite entries; no allocation.
/// @param a Coefficient matrix.
/// @param b Matching RHS.
/// @return Largest absolute coefficient; zero for empty/allzero.
/// @throws std::invalid_argument Invalid complete shape or nonfinite entry.
double validate(const GaussianMatrix& a, const std::vector<double>& b) {
  const auto n = a.size();
  if (b.size() != n)
    throw std::invalid_argument("Gaussian RHS length");
  for (const auto& row : a)
    if (row.size() != n)
      throw std::invalid_argument("Gaussian square shape");
  double maximum = 0;
  for (const auto& row : a)
    for (double v : row) {
      if (!std::isfinite(v))
        throw std::invalid_argument("Gaussian finite coefficients");
      maximum = std::max(maximum, std::fabs(v));
    }
  for (double v : b)
    if (!std::isfinite(v))
      throw std::invalid_argument("Gaussian finite RHS");
  return maximum;
}
/// @brief Require a positive finite absolute pivot cutoff.
/// @param cutoff Absolute threshold.
/// @throws std::invalid_argument Cutoff is nonpositive or nonfinite.
void control(double cutoff) {
  if (!(cutoff > 0) || !std::isfinite(cutoff))
    throw std::invalid_argument("Gaussian positive finite cutoff");
}
/// @brief Verify an actual stored double assignment without changing its arithmetic.
/// @param value Computed assignment.
/// @throws std::overflow_error Assignment is not finite.
void finite(double value) {
  if (!std::isfinite(value))
    throw std::overflow_error("Gaussian nonfinite stored arithmetic");
}
/// @brief Verify widened diagnostic arithmetic in the platform's long-double format.
/// @param value Computed widened result.
/// @return Same finite value.
/// @throws std::overflow_error Widened arithmetic is not finite.
long double wide(long double value) {
  if (!std::isfinite(value))
    throw std::overflow_error("Gaussian residual arithmetic");
  return value;
}
/// @brief Append one actual-time event only when history is requested.
/// @param trace Nullable admitted trace output.
/// @param kind Actual event name.
/// @param col Pivot/back column.
/// @param row Selected/updated row.
/// @param value Actual pivot/factor/coordinate value.
/// @param a Current coefficient state.
/// @param b Current RHS.
/// @param x Empty or full current solution state.
/// @param order Current original-row IDs.
/// @throws std::bad_alloc Snapshot allocation fails.
void capture(
    GaussianEliminationTrace*       trace,
    const char*                     kind,
    std::size_t                     col,
    std::size_t                     row,
    double                          value,
    const GaussianMatrix&           a,
    const std::vector<double>&      b,
    const std::vector<double>&      x,
    const std::vector<std::size_t>& order) {
  if (!trace)
    return;
  trace->events.push_back({kind, col, row, value, a, b, x, order});
  const auto n = a.size();
  trace->captured_entries += 1 + n + n * n + b.size() + x.size() + order.size();
}
/// @brief Move final working storage into the ordinary run and finish observed growth.
/// @param out Run to complete.
/// @param a Current coefficients, moved from.
/// @param b Current RHS, moved from.
/// @throws std::overflow_error Observed growth cannot fit long double.
void finish(GaussianEliminationRun& out, GaussianMatrix& a, std::vector<double>& b) {
  out.upper = std::move(a);
  out.rhs   = std::move(b);
  out.growth =
      out.initial_max ? wide(static_cast<long double>(out.max_coefficient) / out.initial_max) : 1;
}
/// @brief Execute the original ordered double recurrence once, with optional actual captures.
/// Inputs/controls/count capacities have already been fully checked. Input values
/// arrive by value after admission, or moved from the legacy by-value arguments.
/// @param a Owned coefficient state.
/// @param b Owned RHS state.
/// @param cutoff Positive finite absolute pivot threshold.
/// @param initial_max Prevalidated coefficient maximum.
/// @param trace Optional whole-admitted history, null for ordinary solve.
/// @return Owned result and final actual state, including cutoff failures.
/// @throws std::overflow_error A stored arithmetic result or widened growth becomes nonfinite.
/// @throws std::bad_alloc State or snapshot allocation fails.
GaussianEliminationRun run(
    GaussianMatrix            a,
    std::vector<double>       b,
    double                    cutoff,
    double                    initial_max,
    GaussianEliminationTrace* trace) {
  const auto             n = a.size();
  GaussianEliminationRun out;
  out.cutoff      = cutoff;
  out.initial_max = out.max_coefficient = initial_max;
  out.row_order.resize(n);
  std::iota(out.row_order.begin(), out.row_order.end(), 0);
  for (std::size_t col = 0; col < n; ++col) {
    std::size_t pivot = col;
    for (std::size_t r = col + 1; r < n; ++r) {
      ++out.work.pivot_comparisons;
      if (std::fabs(a[r][col]) > std::fabs(a[pivot][col]))
        pivot = r;
    }
    if (std::fabs(a[pivot][col]) < cutoff) {
      capture(trace, "reject", col, pivot, a[pivot][col], a, b, {}, out.row_order);
      finish(out, a, b);
      return out;
    }
    if (pivot != col)
      ++out.work.row_swaps;
    std::swap(a[col], a[pivot]);
    std::swap(b[col], b[pivot]);
    std::swap(out.row_order[col], out.row_order[pivot]);
    ++out.accepted_pivots;
    capture(trace, "pivot", col, pivot, a[col][col], a, b, {}, out.row_order);
    for (std::size_t r = col + 1; r < n; ++r) {
      const double factor = a[r][col] / a[col][col];
      finite(factor);
      out.max_multiplier = std::max(out.max_multiplier, std::fabs(factor));
      for (std::size_t c = col; c < n; ++c) {
        a[r][c] -= factor * a[col][c];
        finite(a[r][c]);
        out.max_coefficient = std::max(out.max_coefficient, std::fabs(a[r][c]));
        ++out.work.coefficient_updates;
      }
      b[r] -= factor * b[col];
      finite(b[r]);
      ++out.work.rhs_updates;
      ++out.work.eliminations;
      capture(trace, "eliminate", col, r, factor, a, b, {}, out.row_order);
    }
  }
  std::vector<double> x(n, 0.0);
  for (std::size_t i = n; i-- > 0;) {
    double s = b[i];
    for (std::size_t j = i + 1; j < n; ++j) {
      s -= a[i][j] * x[j];
      finite(s);
      ++out.work.back_products;
    }
    x[i] = s / a[i][i];
    finite(x[i]);
    ++out.work.back_divisions;
    capture(trace, "back", i, i, x[i], a, b, x, out.row_order);
  }
  out.solution.solved = true;
  out.solution.x      = std::move(x);
  finish(out, a, b);
  return out;
}
} // namespace gaussian_elimination_detail

/// @copydoc gaussian_elimination_trace_entries()
std::size_t gaussian_elimination_trace_entries(std::size_t n) {
  using namespace gaussian_elimination_detail;
  const auto count = events(n), square = cells(n);
  if (count > std::vector<GaussianEliminationEvent>{}.max_size())
    throw std::length_error("Gaussian event capacity");
  return add(add(square, mul(2, n)), mul(count, add(1, add(square, mul(4, n)))));
}
/// @copydoc gaussian_elimination_solve()
GaussianEliminationRun
gaussian_elimination_solve(const GaussianMatrix& a, const std::vector<double>& b, double cutoff) {
  const auto maximum = gaussian_elimination_detail::validate(a, b);
  gaussian_elimination_detail::control(cutoff);
  (void)gaussian_elimination_detail::events(a.size());
  return gaussian_elimination_detail::run(a, b, cutoff, maximum, nullptr);
}
/// @copydoc gaussian_elimination_trace()
GaussianEliminationTrace gaussian_elimination_trace(
    const GaussianMatrix&      a,
    const std::vector<double>& b,
    std::size_t                max_entries,
    double                     cutoff) {
  const auto maximum = gaussian_elimination_detail::validate(a, b);
  gaussian_elimination_detail::control(cutoff);
  const auto bound = gaussian_elimination_trace_entries(a.size());
  if (max_entries && bound > max_entries)
    throw std::length_error("Gaussian whole history quota");
  GaussianEliminationTrace out;
  out.input            = a;
  out.rhs              = b;
  out.captured_entries = gaussian_elimination_detail::add(
      gaussian_elimination_detail::cells(a.size()), gaussian_elimination_detail::mul(2, a.size()));
  out.run = gaussian_elimination_detail::run(a, b, cutoff, maximum, &out);
  return out;
}
/// @copydoc gaussian_elimination()
LinearSolution gaussian_elimination(std::vector<std::vector<double>> a, std::vector<double> b) {
  const auto maximum = gaussian_elimination_detail::validate(a, b);
  (void)gaussian_elimination_detail::events(a.size());
  return gaussian_elimination_detail::run(std::move(a), std::move(b), 1e-12, maximum, nullptr)
      .solution;
}
/// @copydoc gaussian_elimination_example()
GaussianSystem gaussian_elimination_example() {
  return {{{2, 1, -1}, {-3, -1, 2}, {-2, 1, 2}}, {8, -11, -3}};
}
/// @copydoc gaussian_elimination_growth_matrix()
GaussianMatrix gaussian_elimination_growth_matrix(std::size_t n, std::size_t max_entries) {
  using namespace gaussian_elimination_detail;
  const auto entries = add(n, cells(n));
  if (max_entries && entries > max_entries)
    throw std::length_error("Gaussian growth matrix quota");
  GaussianMatrix a(n, std::vector<double>(n, 0));
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      a[i][j] = (i == j || j == n - 1) ? 1 : (i > j ? -1 : 0);
  return a;
}
/// @copydoc gaussian_residual_summary()
GaussianResidualSummary gaussian_residual_summary(
    const GaussianMatrix&      a,
    const std::vector<double>& b,
    const std::vector<double>& x) {
  using namespace gaussian_elimination_detail;
  (void)validate(a, b);
  if (x.size() != b.size())
    throw std::invalid_argument("Gaussian candidate length");
  for (double v : x)
    if (!std::isfinite(v))
      throw std::invalid_argument("Gaussian finite candidate");
  (void)cells(a.size());
  GaussianResidualSummary out;
  out.residual.resize(b.size());
  for (double v : x)
    out.x_inf = std::max(out.x_inf, std::fabs(static_cast<long double>(v)));
  for (std::size_t i = 0; i < b.size(); ++i) {
    long double product = 0, rowsum = 0;
    for (std::size_t j = 0; j < x.size(); ++j) {
      product = wide(product + wide(static_cast<long double>(a[i][j]) * x[j]));
      rowsum  = wide(rowsum + std::fabs(static_cast<long double>(a[i][j])));
    }
    out.residual[i]  = wide(static_cast<long double>(b[i]) - product);
    out.residual_inf = std::max(out.residual_inf, std::fabs(out.residual[i]));
    out.matrix_inf   = std::max(out.matrix_inf, rowsum);
    out.b_inf        = std::max(out.b_inf, std::fabs(static_cast<long double>(b[i])));
  }
  const auto denominator = wide(wide(out.matrix_inf * out.x_inf) + out.b_inf);
  out.backward_error     = denominator ? wide(out.residual_inf / denominator) : 0;
  return out;
}
} // namespace datamunge::algorithms
