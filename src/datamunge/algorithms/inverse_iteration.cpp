/// @file inverse_iteration.cpp
/// @brief One fixed-shift partial-pivot LU and actual normalized inverse iterates.
#include <algorithm>
#include <datamunge/algorithms/inverse_iteration.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private extended-precision numerical operations; no shared mutable state.
namespace {
/// @brief Owned extended-precision dense matrix for one fixed-shift factorization.
using WideMatrix = std::vector<std::vector<long double>>;
/// @brief Reject nonfinite intermediate arithmetic.
/// @param x Extended value.
/// @return Unchanged finite value.
/// @throws std::overflow_error On nonfinite arithmetic.
long double finite(long double x) {
  if (!std::isfinite(x))
    throw std::overflow_error("inverse iteration arithmetic");
  return x;
}
/// @brief Convert a diagnostic to finite double without silently overflowing.
/// @param x Extended diagnostic.
/// @return Finite double, allowing underflow to zero.
/// @throws std::overflow_error On nonfinite or out-of-double-range values.
double narrow(long double x) {
  finite(x);
  if (std::fabs(x) > std::numeric_limits<double>::max())
    throw std::overflow_error("inverse iteration diagnostic range");
  return static_cast<double>(x);
}
/// @brief Robustly normalize a finite extended vector into stored double entries.
/// @param y Borrowed solve/start direction.
/// @return Owned normalized double direction.
/// @throws std::domain_error On a zero direction.
/// @throws std::overflow_error On nonfinite direction/norm.
std::vector<double> normalize(const std::vector<long double>& y) {
  long double norm = 0;
  for (auto v : y)
    norm = finite(std::hypot(norm, finite(v)));
  if (norm == 0)
    throw std::domain_error("inverse iteration zero direction");
  std::vector<double> x;
  x.reserve(y.size());
  for (auto v : y)
    x.push_back(narrow(v / norm));
  return x;
}
/// @brief Diagnose one actual stored iterate against the original matrix.
/// @param A Borrowed original square matrix.
/// @param x Owned normalized stored iterate.
/// @param step Actual completed-solve count.
/// @param anorm Original Frobenius norm.
/// @return Owned iterate, Rayleigh quotient and both residuals.
/// @throws std::overflow_error On nonfinite/unrepresentable numerical results.
InverseIterationStep diagnose(
    const std::vector<std::vector<double>>& A,
    std::vector<double>                     x,
    std::size_t                             step,
    long double                             anorm) {
  std::vector<long double> ax(x.size());
  long double              dot = 0, xx = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    for (std::size_t j = 0; j < x.size(); ++j)
      ax[i] = finite(ax[i] + static_cast<long double>(A[i][j]) * x[j]);
    dot = finite(dot + static_cast<long double>(x[i]) * ax[i]);
    xx += static_cast<long double>(x[i]) * x[i];
  }
  const double value    = narrow(dot / xx);
  long double  residual = 0;
  for (std::size_t i = 0; i < x.size(); ++i)
    residual = finite(std::hypot(residual, ax[i] - static_cast<long double>(value) * x[i]));
  const auto denominator =
      finite((anorm + std::fabs(static_cast<long double>(value))) * std::sqrt(xx));
  return {
      step,
      {value, std::move(x)},
      narrow(residual),
      denominator == 0 ? 0 : narrow(residual / denominator)};
}
/// @brief Factor the actual fixed shift once using partial pivoting.
/// @param lu Mutable shifted matrix, overwritten with packed L/U factors.
/// @return Chronological pivot swaps, one per column.
/// @throws std::domain_error On an exact zero pivot; no regularization.
/// @throws std::overflow_error On nonfinite elimination arithmetic.
std::vector<std::size_t> factor(WideMatrix& lu) {
  const auto               n = lu.size();
  std::vector<std::size_t> pivots(n);
  for (std::size_t k = 0; k < n; ++k) {
    auto pivot = k;
    for (std::size_t i = k + 1; i < n; ++i)
      if (std::fabs(lu[i][k]) > std::fabs(lu[pivot][k]))
        pivot = i;
    if (lu[pivot][k] == 0)
      throw std::domain_error("inverse iteration singular shift");
    pivots[k] = pivot;
    std::swap(lu[k], lu[pivot]);
    for (std::size_t i = k + 1; i < n; ++i) {
      lu[i][k] = finite(lu[i][k] / lu[k][k]);
      for (std::size_t j = k + 1; j < n; ++j)
        lu[i][j] = finite(lu[i][j] - lu[i][k] * lu[k][j]);
    }
  }
  return pivots;
}
/// @brief Apply saved swaps and triangular solves to one actual iterate.
/// @param lu Borrowed packed partial-pivot LU.
/// @param pivots Borrowed chronological swaps from factor().
/// @param x Borrowed current stored direction.
/// @return Owned extended solution, before normalization.
/// @throws std::overflow_error On nonfinite solve arithmetic.
std::vector<long double>
solve(const WideMatrix& lu, const std::vector<std::size_t>& pivots, const std::vector<double>& x) {
  std::vector<long double> y(x.begin(), x.end());
  const auto               n = x.size();
  for (std::size_t k = 0; k < n; ++k)
    std::swap(y[k], y[pivots[k]]);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < i; ++j)
      y[i] = finite(y[i] - lu[i][j] * y[j]);
  for (std::size_t i = n; i-- > 0;) {
    for (std::size_t j = i + 1; j < n; ++j)
      y[i] = finite(y[i] - lu[i][j] * y[j]);
    y[i] = finite(y[i] / lu[i][i]);
  }
  return y;
}
/// @brief Validate complete sizes/input then execute the shared producing loop.
/// @param A Borrowed original matrix.
/// @param mu Fixed finite shift.
/// @param options Complete policies and exact solve count.
/// @param start Borrowed optional custom starting direction.
/// @param capture Whether to retain actual boundaries.
/// @return Final diagnostics/work and optional owned chronology.
/// @throws std::invalid_argument On input shape/finiteness/start violations.
/// @throws std::length_error On enabled complete caps or size overflow.
/// @throws std::domain_error On singular shift/zero direction.
/// @throws std::overflow_error On numerical range failure.
/// @throws std::bad_alloc On allocation failure.
InverseIterationTrace execute(
    const std::vector<std::vector<double>>& A,
    double                                  mu,
    InverseIterationOptions                 options,
    const std::vector<double>&              start,
    bool                                    capture) {
  const auto n = A.size();
  if (options.max_order && n > options.max_order)
    throw std::length_error("inverse iteration order cap");
  if (n == 0 || !std::isfinite(mu) || (!start.empty() && start.size() != n))
    throw std::invalid_argument("inverse iteration matrix/shift/start");
  if (capture) {
    if (options.iterations == std::numeric_limits<std::size_t>::max() ||
        options.iterations + 1 > std::vector<InverseIterationStep>().max_size() ||
        options.iterations + 1 > std::numeric_limits<std::size_t>::max() / n)
      throw std::length_error("inverse iteration history size");
    if (options.max_history_values && (options.iterations + 1) * n > options.max_history_values)
      throw std::length_error("inverse iteration complete history cap");
  }
  long double anorm = 0;
  for (const auto& row : A) {
    if (row.size() != n)
      throw std::invalid_argument("inverse iteration square matrix");
    for (double v : row) {
      if (!std::isfinite(v))
        throw std::invalid_argument("inverse iteration finite matrix");
      anorm = finite(std::hypot(anorm, static_cast<long double>(v)));
    }
  }
  std::vector<long double> initial(n, 1);
  if (!start.empty())
    for (std::size_t i = 0; i < n; ++i) {
      if (!std::isfinite(start[i]))
        throw std::invalid_argument("inverse iteration finite start");
      initial[i] = start[i];
    }
  auto                  current = diagnose(A, normalize(initial), 0, anorm);
  InverseIterationTrace out;
  if (capture)
    out.steps.push_back(current);
  WideMatrix               lu;
  std::vector<std::size_t> pivots;
  if (options.iterations) {
    lu.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      lu[i].assign(A[i].begin(), A[i].end());
      lu[i][i] = finite(lu[i][i] - mu);
    }
    pivots                 = factor(lu);
    out.run.factorizations = 1;
  }
  // k counts completed solves and may reach SIZE_MAX without incrementing past it.
  for (std::size_t k = 0; k < options.iterations;) {
    auto x = normalize(solve(lu, pivots, current.pair.vector));
    ++k;
    current        = diagnose(A, std::move(x), k, anorm);
    out.run.solves = k;
    if (capture)
      out.steps.push_back(current);
  }
  out.run.pair              = std::move(current.pair);
  out.run.residual          = current.residual;
  out.run.relative_residual = current.relative_residual;
  return out;
}
} // namespace
EigenPair inverse_iteration(const std::vector<std::vector<double>>& A, double mu, int iters) {
  if (iters < 0)
    throw std::invalid_argument("inverse iteration nonnegative count");
  return execute(A, mu, {static_cast<std::size_t>(iters), 0, 0}, {}, false).run.pair;
}
InverseIterationRun inverse_iteration_run(
    const std::vector<std::vector<double>>& A,
    double                                  mu,
    InverseIterationOptions                 options,
    const std::vector<double>&              start) {
  return execute(A, mu, options, start, false).run;
}
InverseIterationTrace inverse_iteration_trace(
    const std::vector<std::vector<double>>& A,
    double                                  mu,
    InverseIterationOptions                 options,
    const std::vector<double>&              start) {
  return execute(A, mu, options, start, true);
}
} // namespace datamunge::algorithms
