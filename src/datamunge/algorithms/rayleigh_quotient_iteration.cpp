/// @file rayleigh_quotient_iteration.cpp
/// @brief Shared residual-stopped changing-shift producer over canonical robust inverse iteration.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/rayleigh_quotient_iteration.hpp>
#include <limits>
#include <stdexcept>
#include <utility>
/// @brief Canonical symmetric eigenpair refinement implementation.
namespace datamunge::algorithms {
/// @brief Private changing-shift producer; all state belongs to one call.
namespace {
/// @brief Validate and execute one changing-shift iteration stream.
/// @param matrix Borrowed finite symmetric matrix.
/// @param start Borrowed matching nonzero direction.
/// @param o Execution policies.
/// @param capture Retain actual initial/post-solve states.
/// @return Owned final diagnostics and optional actual history.
RayleighRun solve(
    const std::vector<std::vector<double>>& matrix,
    const std::vector<double>&              start,
    const RayleighOptions&                  o,
    bool                                    capture) {
  const auto n = matrix.size();
  if (!n || start.size() != n || !std::isfinite(o.tolerance) || o.tolerance < 0)
    throw std::invalid_argument("Rayleigh: shape/start/tolerance");
  if (o.max_order && n > o.max_order)
    throw std::length_error("Rayleigh: order policy");
  long double scale = 0;
  for (const auto& row : matrix) {
    if (row.size() != n)
      throw std::invalid_argument("Rayleigh: nonsquare matrix");
    for (double x : row) {
      if (!std::isfinite(x))
        throw std::invalid_argument("Rayleigh: nonfinite matrix");
      scale = std::max(scale, std::fabs(static_cast<long double>(x)));
    }
  }
  auto A = matrix;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < i; ++j) {
      const long double a = A[i][j], b = A[j][i];
      if (std::fabs(a - b) > 64 * std::numeric_limits<double>::epsilon() * scale)
        throw std::invalid_argument("Rayleigh: nonsymmetric matrix");
      A[i][j] = A[j][i] = static_cast<double>(a / 2 + b / 2);
    }
  auto        current = inverse_iteration_run(A, 0, InverseIterationOptions{0, 0, 0}, start);
  RayleighRun out;
  out.matrix_vector_products = 1;
  std::size_t history        = 0;
  /// @brief Admit a real state record before the corresponding solve or copy.
  const auto admit = [&]() {
    if (!capture)
      return;
    if (n == std::numeric_limits<std::size_t>::max() ||
        n + 1 > std::numeric_limits<std::size_t>::max() - history ||
        (o.max_trace_entries &&
         (history > o.max_trace_entries || n + 1 > o.max_trace_entries - history)))
      throw std::length_error("Rayleigh: history policy");
    history += n + 1;
  };
  /// @brief Copy the actual matching pair and residuals into the result and optional record.
  /// @param shift Actual shift used for the most recent solve, zero for the initial state.
  const auto record = [&](double shift) {
    out.pair              = current.pair;
    out.residual          = current.residual;
    out.relative_residual = current.relative_residual;
    out.converged         = current.relative_residual <= o.tolerance;
    if (capture)
      out.steps.push_back(
          {out.solves,
           shift,
           current.pair,
           current.residual,
           current.relative_residual,
           out.converged});
  };
  admit();
  record(0);
  while (!out.converged && out.solves < o.max_iterations) {
    admit();
    const double shift = current.pair.value;
    if (out.matrix_vector_products > std::numeric_limits<std::size_t>::max() - 2)
      throw std::length_error("Rayleigh: work counter capacity");
    current =
        inverse_iteration_run(A, shift, InverseIterationOptions{1, 0, 0}, current.pair.vector);
    ++out.solves;
    ++out.factorizations;
    out.matrix_vector_products += 2;
    record(shift);
  }
  return out;
}
} // namespace
RayleighRun rayleigh_iteration_run(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>&              x,
    const RayleighOptions&                  o) {
  return solve(A, x, o, false);
}
RayleighRun rayleigh_iteration_trace(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>&              x,
    const RayleighOptions&                  o) {
  return solve(A, x, o, true);
}
EigenPair rayleigh_quotient_iteration(
    const std::vector<std::vector<double>>& A,
    std::vector<double>                     x,
    int                                     iters,
    std::vector<double>*                    residuals) {
  if (iters < 0)
    throw std::invalid_argument("Rayleigh: negative step budget");
  RayleighOptions o;
  o.max_iterations = static_cast<std::size_t>(iters);
  auto run = residuals ? rayleigh_iteration_trace(A, x, o) : rayleigh_iteration_run(A, x, o);
  if (residuals)
    for (const auto& s : run.steps)
      residuals->push_back(s.residual);
  return std::move(run.pair);
}
std::vector<std::vector<double>> rayleigh_example_matrix() {
  return {
      {2.5555777899999996, -1.7776222099999999, 0.44422221000000023},
      {-1.7776222099999999, 3.8891777899999997, -2.22257779},
      {0.44422221000000023, -2.22257779, 4.555977789999999}};
}
} // namespace datamunge::algorithms
