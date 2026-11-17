#pragma once
/// @file rayleigh_quotient_iteration.hpp
/// @brief Residual-stopped symmetric Rayleigh iteration over the canonical robust inverse solve.
#include <cstddef>
#include <datamunge/algorithms/inverse_iteration.hpp>
#include <vector>
/// @brief Canonical eigenvalue methods and owned execution records.
namespace datamunge::algorithms {
/// @brief Input, convergence, step and actual-history policies.
/// @par Example
/// @code
/// const dm::algorithms::RayleighOptions o; std::print("{}\n",o.max_iterations); // 20
/// @endcode
/// @see rayleigh_iteration_run()
struct RayleighOptions {
  std::size_t max_iterations{20}; ///< Maximum actual changing-shift solves, including zero.
  double      tolerance{1e-14};   ///< Nonnegative finite relative residual stopping threshold.
  std::size_t max_order{256};     ///< Complete matrix order bound; zero disables.
  std::size_t max_trace_entries{
      1000000}; ///< Actual retained vector scalars plus record headers; zero disables.
};
/// @brief One actual initial or post-solve normalized direction and its original-matrix diagnostics.
/// @par Example
/// @code
/// const dm::algorithms::RayleighStep s; std::print("{}\n",s.iteration); // 0
/// @endcode
/// @see rayleigh_iteration_trace()
struct RayleighStep {
  std::size_t iteration{}; ///< Completed changing-shift solves; zero is the initial state.
  double      shift{};     ///< Actual shift used by this solve; zero for the initial diagnostic.
  EigenPair   pair;        ///< Owned normalized direction and its matching actual Rayleigh value.
  double      residual{};  ///< Actual Euclidean norm of A*x-value*x.
  double      relative_residual{}; ///< Residual divided by (Frobenius(A)+abs(value))*norm(x).
  bool        converged{};         ///< Whether this actual relative residual is at most tolerance.
};
/// @brief Owned actual result and selected work from changing-shift inverse iteration.
/// @par Example
/// @code
/// const dm::algorithms::RayleighRun r; std::print("{}\n",r.solves); // 0
/// @endcode
/// @see rayleigh_iteration_run()
struct RayleighRun {
  EigenPair   pair;       ///< Actual final normalized direction and its matching Rayleigh value.
  double      residual{}; ///< Actual final absolute residual.
  double      relative_residual{}; ///< Actual final Frobenius-scaled backward-error diagnostic.
  bool        converged{}; ///< Threshold met; a false result is an explicitly exhausted finite run.
  std::size_t solves{};    ///< Actual canonical changing-shift solves.
  std::size_t factorizations{}; ///< Actual one factorization per solve.
  std::size_t
      matrix_vector_products{}; ///< Actual diagnostic matvecs: one initial plus two per nested inverse run.
  std::vector<RayleighStep>
      steps; ///< Actual initial/post-solve history, only populated by the trace.
};
/// @brief Refine a symmetric eigenpair with actual Rayleigh shifts and robust canonical inverse solves.
/// Validates finite nonempty square symmetric input, accepting asymmetry up to64 machine eps times the largest input magnitude and averaging it. Requires a matching finite nonzero starting direction. The canonical inverse-iteration kernel supplies hypot-based normalization and extended-precision partial-pivot LU; no tiny-pivot regularization is invented. Check the original-matrix relative residual before each solve, stopping before an already converged shift becomes singular. Returns a flagged finite result on step exhaustion; an exact singular shift with a nonconverged direction throws rather than returning a zero vector. Each solve is O(n^3), O(n^2) working storage. Cubic convergence is local under the usual simple-eigenvalue/nondegeneracy conditions; the reached eigenpair need not be the one nearest the initial quotient. Borrowed inputs unchanged, outputs owned.
/// @param matrix Finite symmetric nonempty square matrix.
/// @param start Finite matching nonzero starting vector, normalized internally.
/// @param options Step, residual, input and history policies.
/// @return Actual final normalized pair, residual, convergence flag and selected work.
/// @throws std::invalid_argument Malformed matrix/start or invalid tolerance.
/// @throws std::domain_error Zero start or exact singular nonconverged shifted system.
/// @throws std::length_error Input, counter or retained capacity exceeded.
/// @throws std::overflow_error Nonfinite arithmetic or unrepresentable diagnostic.
/// @par Example
/// @code
/// auto r=dm::algorithms::rayleigh_iteration_run({{1,0},{0,3}},{1,0});
/// std::print("{} {}\n",r.converged,r.solves); // true 0
/// @endcode
/// @par References
/// - J. H. Wilkinson, The Algebraic Eigenvalue Problem (1965), ISBN 9780198534181
/// @see rayleigh_iteration_trace(), inverse_iteration_run()
RayleighRun rayleigh_iteration_run(
    const std::vector<std::vector<double>>& matrix,
    const std::vector<double>&              start,
    const RayleighOptions&                  options = {});
/// @brief Capture actual initial and post-solve changing-shift states from the same producer.
/// Every actual record retains n vector values plus one header, admitted before its solve/copy. Converged runs retain only their real completed states, never padded or repeated-prefix simulations. Final pair/diagnostics equal the last record. Other semantics and complexity match rayleigh_iteration_run(), with O(actual_steps*n) retained history.
/// @param matrix Finite symmetric square matrix.
/// @param start Finite nonzero matching start.
/// @param options Step, tolerance, order and actual-history policies.
/// @return Actual normalized history, shifts, residuals, final pair and work.
/// @throws std::invalid_argument Input/tolerance invalid.
/// @throws std::domain_error Zero direction or singular nonconverged shift.
/// @throws std::length_error Input/history/work capacity exceeded.
/// @throws std::overflow_error Nonfinite intermediate or diagnostic.
/// @par Example
/// @code
/// auto t=dm::algorithms::rayleigh_iteration_trace({{2,0},{0,4}},{1,0});
/// std::print("{}\n",t.steps.size()); // 1
/// @endcode
/// @see rayleigh_iteration_run()
RayleighRun rayleigh_iteration_trace(
    const std::vector<std::vector<double>>& matrix,
    const std::vector<double>&              start,
    const RayleighOptions&                  options = {});
/// @brief Return the actual residual-stopped Rayleigh pair through the preserved signature.
/// iters is now an explicit nonnegative maximum solve budget; convergence may stop earlier, and zero steps returns the actual normalized start quotient. If residuals is non-null, append the real initial and post-solve absolute residuals, without clearing existing values or padding converged states. Corrects the legacy overflow-normalized zero-vector result and stale quotient/vector mismatch. Inputs unchanged; residuals appends only after successful execution. Default relative tolerance1e-14 and order256; use the run API to configure them.
/// @param matrix Finite symmetric nonempty square input.
/// @param start Finite matching nonzero start.
/// @param iters Maximum solves, default20.
/// @param residuals Optional borrowed output vector, default null.
/// @return Owned actual final normalized eigenpair estimate.
/// @throws std::invalid_argument Negative budget or malformed input.
/// @throws std::domain_error Zero start or exact singular nonconverged shift.
/// @throws std::length_error Default input/history policies exceeded.
/// @throws std::overflow_error Arithmetic/diagnostic outside finite range.
/// @par Example
/// @code
/// auto p=dm::algorithms::rayleigh_quotient_iteration({{1,0},{0,3}},{0,2},0);
/// std::print("{}\n",p.value); // 3
/// @endcode
/// @see rayleigh_iteration_run()
EigenPair rayleigh_quotient_iteration(
    const std::vector<std::vector<double>>& matrix,
    std::vector<double>                     start,
    int                                     iters     = 20,
    std::vector<double>*                    residuals = nullptr);
/// @brief Return the original rounded-basis three-by-three Rayleigh-iteration fixture.
/// Preserves the original Q entries0.6667,-0.6667,0.3333 and nominal diagonal1,3,7, evaluating the old double product order once into an explicit native matrix fixture. Because the decimal Q is only approximately orthogonal, its actual eigenvalues are not exactly1,3,7. Owned constant-size output; no global mutable state.
/// @return Owned original three-by-three matrix.
/// @par Example
/// @code
/// auto a=dm::algorithms::rayleigh_example_matrix();
/// std::print("{}\n",a.size()); // 3
/// @endcode
/// @see rayleigh_iteration_trace()
std::vector<std::vector<double>> rayleigh_example_matrix();
} // namespace datamunge::algorithms
