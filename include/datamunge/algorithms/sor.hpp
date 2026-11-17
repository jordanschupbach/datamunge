#pragma once
/// @file sor.hpp
/// @brief Checked actual successive over-relaxation sweeps and residuals.
#include <cstddef>
#include <datamunge/algorithms/iterative_solution.hpp>
#include <vector>
/// @brief Canonical checked native producers and owned values.
namespace datamunge::algorithms {
/// @brief SOROptions owns actual producer values or explicit invocation policies; fields state units and defaults.
/// @par Example
/// @code
/// const dm::algorithms::SOROptions value{};
/// std::print("{}\n",value.omega); // 1
/// @endcode
struct SOROptions {
  double omega{1}; ///< Finite relaxation factor strictly between0 and2.
  int    max_iterations{
      1000}; ///< Maximum complete sweeps; nonnegative, zero checks only initial residual.
  double tolerance{
      1e-10}; ///< Finite nonnegative absolute recomputed Euclidean residual threshold, inclusive.
  std::size_t max_cells{1000000}; ///< Complete square matrix cell count; zero disables policy.
  std::size_t max_coefficient_visits{
      200000000}; ///< Actual update off-diagonal visits plus residual products; zero disables.
  std::size_t max_history_values{
      1000000}; ///< Captured complete-sweep records plus copied iterate entries; zero disables.
};
/// @brief SORSweep owns actual producer values or explicit invocation policies; fields state units and defaults.
/// @par Example
/// @code
/// const dm::algorithms::SORSweep value{};
/// std::print("{}\n",value.iteration); // 0
/// @endcode
struct SORSweep {
  int                 iteration{}; ///< Actual completed sweep count, one-based.
  std::vector<double> x;           ///< Actual iterate immediately after this complete sweep.
  double              residual{};  ///< Actual independently recomputed ||A*x-b|| after this sweep.
};
/// @brief SORRun owns actual producer values or explicit invocation policies; fields state units and defaults.
/// @par Example
/// @code
/// const dm::algorithms::SORRun value{};
/// std::print("{}\n",value.coordinate_updates); // 0
/// @endcode
struct SORRun {
  IterativeSolution solution; ///< Actual final iterate, residual, sweep count and stopping flag.
  double            initial_residual{};    ///< Actual residual at the zero initial iterate.
  std::size_t       coordinate_updates{};  ///< Actual relaxed coordinate assignments.
  std::size_t       off_diagonal_visits{}; ///< Actual Aij*xj terms in coordinate updates.
  std::size_t
      residual_products{}; ///< Actual Aij*xj terms in initial/post-sweep residual recomputation.
  std::vector<SORSweep> sweeps; ///< Optional actual completed-sweep histories; no replay.
};
/// @brief Solve from a zero initial iterate by actual forward SOR sweeps.
/// Requires finite square A, matching finite b and nonzero diagonal. Every coordinate uses newest available values, forms a Gauss-Seidel candidate, then blends (1-omega)*old+omega*candidate. Uses widened sums and checked double assignments. Independently recomputes absolute residual, stopping at residual<=tolerance; empty and zero RHS may converge at iteration0. For symmetric positive definite A,0<omega<2 ensures convergence; this interval is not sufficient for arbitrary matrices and SPD is not verified. Complete sweep cap returns converged=false, never a convergence certificate from iteration count alone. Time O(k*n^2), working storage O(n) beyond input, optional full-sweep history O(k*n). Borrowed inputs, owned output and no shared state.
/// @param A Complete finite square coefficient matrix with nonzero diagonal.
/// @param b Complete finite matching right-hand side.
/// @param options Relaxation, cap, tolerance and actual work/capture policies; zero disables size/work policies only.
/// @return Owned actual iterate/residual/stopping condition, executed work and optional sweep states.
/// @throws std::invalid_argument Shape, nonfinite input, zero diagonal or invalid controls.
/// @throws std::length_error Complete geometry, actual work/history or counter capacity exceeded.
/// @throws std::overflow_error Actual relaxed assignment or residual not representable.
/// @par Example
/// @code
/// auto r=dm::algorithms::sor_run({{2}},{4});
/// std::print("{} {}\n",r.solution.x[0],r.solution.converged); // 2 true
/// @endcode
/// @par References
/// - D. M. Young, Iterative methods for solving partial difference equations of elliptic type, Trans. AMS76,1954, DOI10.1090/S0002-9947-1954-0059635-7.
/// @see sor()
SORRun sor_run(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>&              b,
    const SOROptions&                       options = {});
/// @brief Solve from a zero initial iterate by actual forward SOR sweeps with captured iterates.
/// Requires finite square A, matching finite b and nonzero diagonal. Every coordinate uses newest available values, forms a Gauss-Seidel candidate, then blends (1-omega)*old+omega*candidate. Uses widened sums and checked double assignments. Independently recomputes absolute residual, stopping at residual<=tolerance; empty and zero RHS may converge at iteration0. For symmetric positive definite A,0<omega<2 ensures convergence; this interval is not sufficient for arbitrary matrices and SPD is not verified. Complete sweep cap returns converged=false, never a convergence certificate from iteration count alone. Time O(k*n^2), working storage O(n) beyond input, optional full-sweep history O(k*n). Borrowed inputs, owned output and no shared state.
/// @param A Complete finite square coefficient matrix with nonzero diagonal.
/// @param b Complete finite matching right-hand side.
/// @param options Relaxation, cap, tolerance and actual work/capture policies; zero disables size/work policies only.
/// @return Owned actual iterate/residual/stopping condition, executed work and optional sweep states.
/// @throws std::invalid_argument Shape, nonfinite input, zero diagonal or invalid controls.
/// @throws std::length_error Complete geometry, actual work/history or counter capacity exceeded.
/// @throws std::overflow_error Actual relaxed assignment or residual not representable.
/// @par Example
/// @code
/// auto r=dm::algorithms::sor_trace({{2}},{4});
/// std::print("{} {}\n",r.solution.x[0],r.solution.converged); // 2 true
/// @endcode
/// @par References
/// - D. M. Young, Iterative methods for solving partial difference equations of elliptic type, Trans. AMS76,1954, DOI10.1090/S0002-9947-1954-0059635-7.
/// @see sor()
SORRun sor_trace(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>&              b,
    const SOROptions&                       options = {});
/// @brief Return the preserved checked SOR solution signature.
/// Same forward producer, inclusive actual absolute residual stopping and default size/work policies. Unlike the historical shortcut, zero-sweep results report the real initial residual.
/// @param a Finite square matrix with nonzero diagonal.
/// @param b Matching finite RHS.
/// @param omega Finite relaxation in(0,2).
/// @param max_iter Nonnegative complete sweep cap, default1000.
/// @param tol Finite nonnegative absolute residual tolerance, default1e-10.
/// @return Owned actual iterative stopping result.
/// @throws std::invalid_argument Malformed input/controls.
/// @throws std::length_error Default geometry/work/capacity exceeded.
/// @throws std::overflow_error Actual arithmetic not representable.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::sor({{2}},{4},1).x[0]); // 2
/// @endcode
/// @see sor_run()
IterativeSolution sor(
    const std::vector<std::vector<double>>& a,
    const std::vector<double>&              b,
    double                                  omega,
    int                                     max_iter = 1000,
    double                                  tol      = 1e-10);
} // namespace datamunge::algorithms
