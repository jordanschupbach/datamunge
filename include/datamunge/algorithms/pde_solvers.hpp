/// @file
/// @brief Uniform Poisson, periodic advection and scalar trapezoidal ODE solvers.
#pragma once
#include <datamunge/algorithms/trapezoidal_ode.hpp>
#include <datamunge/algorithms/lax_wendroff.hpp>

#include <datamunge/algorithms/crank_nicolson.hpp>
#include <functional>
#include <utility>
#include <vector>

/// @brief Canonical numerical algorithms and their owned result types.
namespace datamunge::algorithms {

/// @brief Owned grid, assembled Poisson system and numerical residual from one native solve.
/// Arrays own their storage; no source callback or caller references are retained.
/// source[i], rhs[i] and residual[i] refer to interior node x[i+1]; x and u
/// include both boundary nodes. The residual is A*u_interior-rhs for the
/// stored binary64 matrix and RHS, evaluated with long-double intermediates,
/// then rounded to double. It measures the discrete solve, not PDE accuracy.
/// @par Example
/// @code
/// const auto r=dm::algorithms::finite_difference_poisson_solve([](double){return 1.;},0,0,1);
/// std::cout << r.u[1] << ' ' << r.rhs[0]; // 0.125 1
/// @endcode
/// @see finite_difference_poisson_solve(), finite_difference_poisson()
struct FiniteDifferencePoissonResult {
    double h=0; ///< Uniform spacing 1/(n+1).
    double diagonal=0; ///< Repeated main diagonal 2/h² of the interior matrix.
    double off_diagonal=0; ///< Repeated sub/super diagonal -1/h².
    std::vector<double> x; ///< Owned n+2 coordinates; endpoints exactly zero and one.
    std::vector<double> source; ///< Owned n actual callback samples before boundary elimination.
    std::vector<double> rhs; ///< Owned n samples with left then right boundary contributions added.
    std::vector<double> u; ///< Owned n+2 finite solved values including the supplied endpoints.
    std::vector<double> residual; ///< Owned n rounded discrete residuals, with eliminated boundaries.
    double max_abs_residual=0; ///< Largest absolute entry of residual, in source units.
};

/// @brief Solve -u''=f on [0,1] with Dirichlet values by central differences and Thomas elimination.
/// Uses n interior nodes and returns their values with both boundaries. Calls f
/// exactly once at each (i+1)*h in increasing i order; exceptions propagate.
/// Source samples and boundary values must be finite. The signed grid count
/// n+2 must fit int; this is checked before allocation or callback evaluation.
/// O(n) arithmetic/storage plus callback costs, no shared state or input mutation.
/// Nodal error is O(h²) for a sufficiently smooth exact solution (for example C4),
/// apart from rounding; finite output alone does not certify this accuracy.
/// Reuses thomas_solve() on the internally assembled positive-definite system.
/// @param f Nonempty source callback, evaluated only at the n interior coordinates.
/// @param alpha Finite value at x=0, copied exactly to the first output entry.
/// @param beta Finite value at x=1, copied exactly to the last output entry.
/// @param n Positive number of interior nodes, at most INT_MAX-2.
/// @return Owned n+2 solved nodal values; no interpolation between nodes is implied.
/// @throws std::invalid_argument For nonpositive n, empty f or nonfinite boundaries/samples.
/// @throws std::length_error If n+2 cannot fit int or a vector capacity is exceeded.
/// @throws std::overflow_error If boundary assembly or the computed solution becomes nonfinite.
/// @throws std::bad_alloc If owned storage cannot be allocated.
/// @par Example
/// @code
/// const auto u=dm::algorithms::finite_difference_poisson([](double){return 0.;},2,4,1);
/// std::cout << u[0] << ' ' << u[1] << ' ' << u[2]; // 2 3 4
/// @endcode
/// @par References
/// - P.-O. Persson, Finite Difference Methods for PDEs, Boundary Value Problems
///   (the implementation uses the opposite sign, -u''=f),
///   https://persson.berkeley.edu/math228b/html/finite_difference_methods_for_PDEs.html .
/// @see finite_difference_poisson_solve(), FiniteDifferencePoissonResult, thomas_solve()
std::vector<double> finite_difference_poisson(const std::function<double(double)>& f, double alpha,
                                              double beta, int n);

/// @brief Solve the same Poisson problem and retain its actual grid, samples, matrix and residual.
/// Shares the ordinary solver's assembly and canonical thomas_solve(), without
/// replaying the solve or reevaluating f. Captures O(n) extra owned numeric output;
/// these arrays describe the solved system, not an iteration history. Recomputed
/// residuals use long-double intermediates and are not a PDE error certificate.
/// No input mutation/shared state; source callback exceptions propagate unchanged.
/// @param f Nonempty finite-valued source, sampled once per interior node in increasing order.
/// @param alpha Finite left Dirichlet value at zero.
/// @param beta Finite right Dirichlet value at one.
/// @param n Positive interior-node count, at most INT_MAX-2.
/// @return Owned actual assembly, all nodal values and rounded discrete residuals.
/// @throws std::invalid_argument For nonpositive n, empty f or nonfinite boundaries/samples.
/// @throws std::length_error If n+2 cannot fit int or a vector capacity is exceeded.
/// @throws std::overflow_error If assembly, solution or a stored residual becomes nonfinite.
/// @throws std::bad_alloc If owned storage cannot be allocated.
/// @par Example
/// @code
/// const auto r=dm::algorithms::finite_difference_poisson_solve([](double){return 1.;},0,0,1);
/// std::cout << r.diagonal << ' ' << r.off_diagonal << ' ' << r.max_abs_residual; // 8 -4 0
/// @endcode
/// @par References
/// - P.-O. Persson, Finite Difference Methods for PDEs, Boundary Value Problems,
///   https://persson.berkeley.edu/math228b/html/finite_difference_methods_for_PDEs.html .
/// @see finite_difference_poisson(), FiniteDifferencePoissonResult
FiniteDifferencePoissonResult finite_difference_poisson_solve(
    const std::function<double(double)>& f, double alpha, double beta, int n);



} // namespace datamunge::algorithms
