/// @file lanczos_iteration.cpp
/// @brief Shared checked original-order dense Lanczos producer and optional actual stage capture.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/lanczos_iteration.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Local complete validation, finite arithmetic, actual work and observation helpers.
namespace lanczos_detail {
/// @brief Add selected operation/capture counts without wrap.
/// @param count Actual mutable logical count.
/// @param amount Additional actual count.
/// @throws std::length_error On size_t overflow.
void add(std::size_t& count, std::size_t amount) {
  if (amount > std::numeric_limits<std::size_t>::max() - count)
    throw std::length_error("Lanczos count overflow");
  count += amount;
}
/// @brief Require a produced double to be finite.
/// @param x Actual component/coefficient/norm.
/// @return Unchanged finite value.
/// @throws std::overflow_error On nonfinite numeric value.
double finite(double x) {
  if (!std::isfinite(x))
    throw std::overflow_error("Lanczos nonfinite arithmetic");
  return x;
}
/// @brief Validate complete square symmetry, matched finite start and nonnegative requested steps.
/// @param A Complete borrowed original matrix.
/// @param b Complete borrowed original start.
/// @param m Actual requested step ceiling.
/// @throws std::invalid_argument On step/shape/finite/symmetry violations.
void validate(const std::vector<std::vector<double>>& A, const std::vector<double>& b, int m) {
  if (m < 0 || b.size() != A.size())
    throw std::invalid_argument("Lanczos steps/shape");
  const auto n = A.size();
  for (const auto& row : A) {
    if (row.size() != n)
      throw std::invalid_argument("Lanczos ragged matrix");
    for (double x : row)
      if (!std::isfinite(x))
        throw std::invalid_argument("Lanczos finite matrix");
  }
  for (double x : b)
    if (!std::isfinite(x))
      throw std::invalid_argument("Lanczos finite start");
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < i; ++j)
      if (A[i][j] != A[j][i])
        throw std::invalid_argument("Lanczos exact symmetry required");
}
/// @brief Local optional producing capture plus actual counters.
struct Observer {
  LanczosTrace* trace; ///< Borrowed complete output, null for ordinary mode.
  std::size_t   cap;   ///< Optional retained logical-entry ceiling, zero off.
  LanczosWork   work;  ///< Actual same-execution selected operations.

  /// @brief Admit original/stage/projection records before their operation.
  /// @param amount Additional logical entries.
  /// @throws std::length_error On count overflow or enabled quota.
  void admit(std::size_t amount) {
    if (!trace)
      return;
    auto next = trace->captured_entries;
    add(next, amount);
    if (cap && next > cap)
      throw std::length_error("Lanczos trace quota");
    trace->captured_entries = next;
  }
};
/// @brief Evaluate one original-order finite double inner product and count its actual call.
/// @param a First complete compatible vector.
/// @param b Second compatible vector.
/// @param o Actual local work/capture state.
/// @return Actual finite sequential double dot product.
/// @throws std::length_error On work count overflow.
/// @throws std::overflow_error On nonfinite intermediate sum.
double dot(const std::vector<double>& a, const std::vector<double>& b, Observer& o) {
  add(o.work.dot_products, 1);
  double sum = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    sum = finite(sum + finite(a[i] * b[i]));
  return sum;
}
/// @brief Evaluate an original-order norm with widened hypot only on square overflow/underflow.
/// @param x Complete finite vector.
/// @param o Actual local observation state.
/// @return Finite double norm, zero only for exact zero input.
/// @throws std::length_error On norm-call counter overflow.
/// @throws std::overflow_error On unrepresentable finite double norm.
double norm(const std::vector<double>& x, Observer& o) {
  add(o.work.norm_evaluations, 1);
  double squares = 0;
  bool   nonzero = false;
  for (double value : x) {
    squares += value * value;
    nonzero |= value != 0;
  }
  if (std::isfinite(squares) && (squares > 0 || !nonzero))
    return std::sqrt(squares);
  long double wide = 0;
  for (double value : x)
    wide = std::hypot(wide, static_cast<long double>(value));
  if (!std::isfinite(wide) || wide > std::numeric_limits<double>::max())
    throw std::overflow_error("Lanczos norm range");
  return finite(static_cast<double>(wide));
}
/// @brief Produce actual dense Lanczos recurrence, one full Gram pass and optional observations.
/// @param A Validated immutable complete symmetric matrix.
/// @param b Validated matched finite original start.
/// @param requested Nonnegative original requested step ceiling.
/// @param o Actual local work/capture state.
/// @return Owned actual original-layout coefficients/basis.
/// @throws std::length_error On capture/work/geometry/container limits.
/// @throws std::overflow_error On nonfinite produced arithmetic.
/// @throws std::bad_alloc On native/capture allocation failure.
LanczosResult run(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>&              b,
    int                                     requested,
    Observer&                               o) {
  const auto n = A.size();
  if (o.trace) {
    if (n && n > std::numeric_limits<std::size_t>::max() / n)
      throw std::length_error("Lanczos original matrix geometry");
    o.admit(n * n);
    o.admit(n);
    o.trace->matrix = A;
    o.trace->start  = b;
  }
  LanczosResult res;
  if (n == 0) {
    if (o.trace)
      o.trace->stop = LanczosStop::Empty;
    return res;
  }
  if (requested == 0)
    return res;
  std::vector<double> v  = b;
  const double        b0 = norm(v, o);
  if (o.trace)
    o.trace->initial_norm = b0;
  if (b0 == 0) {
    if (o.trace)
      o.trace->stop = LanczosStop::ZeroStart;
    return res;
  }
  for (double& x : v)
    x = finite(x / b0);
  std::vector<double> previous(n, 0);
  double              beta_previous = 0;
  const auto          limit         = std::min(n, static_cast<std::size_t>(requested));
  for (std::size_t j = 0; j < limit; ++j) {
    if (o.trace) {
      if (n > (std::numeric_limits<std::size_t>::max() - 1) / 3)
        throw std::length_error("Lanczos stage geometry");
      o.admit(1 + 3 * n);
    }
    res.V.push_back(v);
    add(o.work.matvecs, 1);
    std::vector<double> w(n, 0);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t k = 0; k < n; ++k) {
        w[i] = finite(w[i] + finite(A[i][k] * v[k]));
        add(o.work.matrix_products, 1);
      }
    LanczosStep step;
    if (o.trace) {
      step.iteration      = j;
      step.matrix_product = w;
    }
    const double alpha = dot(v, w, o);
    res.alpha.push_back(alpha);
    for (std::size_t i = 0; i < n; ++i)
      w[i] = finite(w[i] - finite(finite(alpha * v[i]) + finite(beta_previous * previous[i])));
    if (o.trace)
      step.three_term = w;
    for (std::size_t qi = 0; qi < res.V.size(); ++qi) {
      o.admit(1);
      const auto&  q = res.V[qi];
      const double h = dot(q, w, o);
      for (std::size_t i = 0; i < n; ++i)
        w[i] = finite(w[i] - finite(h * q[i]));
      add(o.work.reorth_projections, 1);
      if (o.trace)
        o.trace->projections.push_back({j, qi, h});
    }
    const double beta = norm(w, o);
    if (j + 1 < static_cast<std::size_t>(requested))
      res.beta.push_back(beta);
    if (o.trace) {
      step.alpha         = alpha;
      step.beta_previous = beta_previous;
      step.residual_norm = beta;
      step.residual      = w;
      o.trace->steps.push_back(std::move(step));
    }
    if (beta < 1e-12) {
      if (o.trace)
        o.trace->stop = LanczosStop::InvariantSubspace;
      break;
    }
    if (j + 1 == limit) {
      if (o.trace)
        o.trace->stop = limit == n ? LanczosStop::DimensionLimit : LanczosStop::StepLimit;
      break;
    }
    previous      = v;
    beta_previous = beta;
    for (double& x : w)
      x = finite(x / beta);
    v = std::move(w);
  }
  return res;
}
} // namespace lanczos_detail
LanczosResult
lanczos_iteration(const std::vector<std::vector<double>>& A, const std::vector<double>& b, int m) {
  lanczos_detail::validate(A, b, m);
  lanczos_detail::Observer o{nullptr, 0, {}};
  return lanczos_detail::run(A, b, m, o);
}
LanczosTrace lanczos_trace(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>&              b,
    int                                     m,
    std::size_t                             max_entries) {
  lanczos_detail::validate(A, b, m);
  LanczosTrace             out;
  lanczos_detail::Observer o{&out, max_entries, {}};
  out.result = lanczos_detail::run(A, b, m, o);
  out.work   = o.work;
  return out;
}
} // namespace datamunge::algorithms
