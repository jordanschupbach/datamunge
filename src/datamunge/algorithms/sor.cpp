/// @file sor.cpp
/// @brief One checked forward relaxed producer with independently recomputed residuals.
#include <cmath>
#include <datamunge/algorithms/sor.hpp>
#include <limits>
#include <stdexcept>
#include <utility>
/// @brief Canonical checked native producers and owned values.
namespace datamunge::algorithms {
/// @brief Private invocation-local checked producing helpers.
namespace {
/// @brief Convert actual finite widened arithmetic to a representable double.
/// @param x Actual widened value.
/// @return Representable double.
double narrow(long double x) {
  if (!std::isfinite(x) || std::abs(x) > std::numeric_limits<double>::max())
    throw std::overflow_error("SOR: arithmetic range");
  return static_cast<double>(x);
}
/// @brief Execute actual coordinate updates and post-sweep residuals.
/// @param A Complete matrix.
/// @param b Complete RHS.
/// @param o Controls/policies.
/// @param capture Retain sweeps.
/// @return Actual result.
SORRun run(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>&              b,
    const SOROptions&                       o,
    bool                                    capture) {
  const auto n = A.size(), limit = std::numeric_limits<std::size_t>::max();
  if (b.size() != n || !std::isfinite(o.omega) || o.omega <= 0 || o.omega >= 2 ||
      o.max_iterations < 0 || !std::isfinite(o.tolerance) || o.tolerance < 0)
    throw std::invalid_argument("SOR: shape/controls");
  if (n && n > limit / n)
    throw std::length_error("SOR: cells capacity");
  if (o.max_cells && n * n > o.max_cells)
    throw std::length_error("SOR: matrix policy");
  for (std::size_t i = 0; i < n; ++i) {
    if (A[i].size() != n || !std::isfinite(b[i]))
      throw std::invalid_argument("SOR: shape/RHS");
    for (double v : A[i])
      if (!std::isfinite(v))
        throw std::invalid_argument("SOR: finite matrix");
    if (A[i][i] == 0)
      throw std::invalid_argument("SOR: diagonal");
  }
  SORRun r;
  r.solution.x.assign(n, 0);
  std::size_t visits = 0, retained = 0;
  /// @brief Count one actual coefficient visit before overflow.
  /// @param counter Actual selected visit accumulator.
  auto visit = [&](std::size_t& counter) {
    if (visits == limit || counter == limit ||
        (o.max_coefficient_visits && visits >= o.max_coefficient_visits))
      throw std::length_error("SOR: coefficient visits");
    ++visits;
    ++counter;
  };
  /// @brief Recompute the actual widened Euclidean residual from the current iterate.
  /// @return Finite representable actual residual.
  auto residual = [&]() {
    long double norm = 0;
    for (std::size_t i = 0; i < n; ++i) {
      long double value = -static_cast<long double>(b[i]);
      for (std::size_t j = 0; j < n; ++j) {
        visit(r.residual_products);
        value += static_cast<long double>(A[i][j]) * r.solution.x[j];
      }
      norm = std::hypot(norm, value);
    }
    return narrow(norm);
  };
  r.solution.residual  = residual();
  r.initial_residual   = r.solution.residual;
  r.solution.converged = r.solution.residual <= o.tolerance;
  for (int it = 0; it < o.max_iterations && !r.solution.converged; ++it) {
    for (std::size_t i = 0; i < n; ++i) {
      long double value = b[i];
      for (std::size_t j = 0; j < n; ++j)
        if (j != i) {
          visit(r.off_diagonal_visits);
          value -= static_cast<long double>(A[i][j]) * r.solution.x[j];
        }
      const long double candidate = value / A[i][i];
      r.solution.x[i]             = narrow(
          (1 - static_cast<long double>(o.omega)) * r.solution.x[i] +
          static_cast<long double>(o.omega) * candidate);
      if (r.coordinate_updates == limit)
        throw std::length_error("SOR: update capacity");
      ++r.coordinate_updates;
    }
    r.solution.iterations = it + 1;
    r.solution.residual   = residual();
    r.solution.converged  = r.solution.residual <= o.tolerance;
    if (capture) {
      if (n == limit || retained > limit - n - 1)
        throw std::length_error("SOR: history capacity");
      retained += n + 1;
      if (o.max_history_values && retained > o.max_history_values)
        throw std::length_error("SOR: history");
      r.sweeps.push_back({r.solution.iterations, r.solution.x, r.solution.residual});
    }
  }
  return r;
}
} // namespace
SORRun sor_run(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>&              b,
    const SOROptions&                       o) {
  return run(A, b, o, false);
}
SORRun sor_trace(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>&              b,
    const SOROptions&                       o) {
  return run(A, b, o, true);
}
IterativeSolution sor(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>&              b,
    double                                  omega,
    int                                     cap,
    double                                  tol) {
  SOROptions o;
  o.omega          = omega;
  o.max_iterations = cap;
  o.tolerance      = tol;
  return sor_run(A, b, o).solution;
}
} // namespace datamunge::algorithms
