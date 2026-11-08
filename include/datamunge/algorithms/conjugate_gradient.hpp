#pragma once

/// @file conjugate_gradient.hpp
/// @brief Checked CG execution histories, a stationary baseline and Dirichlet fixtures.

#include <datamunge/algorithms/linear_solvers.hpp>
#include <cstddef>

namespace datamunge::algorithms {

/// @brief Why a checked conjugate-gradient run stopped.
/// Values describe computed arithmetic, not a positive-definiteness certificate.
/// @par Example
/// @code
/// const auto reason = dm::algorithms::CGStop::iteration_limit;
/// std::cout << (reason == dm::algorithms::CGStop::iteration_limit); // 1
/// @endcode
/// @see ConjugateGradientRun
enum class CGStop {
    converged,              ///< Recomputed residual norm is at most the absolute tolerance.
    iteration_limit,        ///< The requested number of completed steps was exhausted.
    nonpositive_curvature,  ///< A visited direction has computed p^T A p <= 0.
    residual_gap            ///< Recursive residual vanishes/reaches tolerance, true residual does not.
};

/// @brief Absolute stopping and optional actual-snapshot storage controls for CG.
/// Zero iterations is valid; zero max_trace_values means unlimited. Positive trace
/// caps count four vector entries per unknown per completed step, checked online
/// before completing the next step. A failing call can already have requested A p.
/// Solve-only calls ignore this storage cap.
/// @par Example
/// @code
/// dm::algorithms::ConjugateGradientOptions o;
/// std::cout << o.max_iterations << ' ' << o.max_trace_values; // 1000 1000000
/// @endcode
/// @see conjugate_gradient_trace(), conjugate_gradient_solve()
struct ConjugateGradientOptions {
    int max_iterations{1000};                ///< Nonnegative complete-step budget.
    double tolerance{1e-10};                 ///< Finite nonnegative absolute true-residual threshold.
    std::size_t max_trace_values{1000000};    ///< Stored scalar entries; zero is unlimited.
};

/// @brief Owned actual CG step, including its used direction and resulting iterate.
/// x and r are the resulting iterate and recursively updated residual; p and ap
/// are the direction and A p actually used. beta is the next-direction coefficient
/// only when has_next_direction is true. Norms/coefficients use long double;
/// vector storage and each recurrence assignment use checked double rounding.
/// @par Example
/// @code
/// const auto t = dm::algorithms::conjugate_gradient_trace(
///     std::vector<std::vector<double>>{{2}}, {4});
/// std::cout << t.steps[0].x[0] << ' ' << t.steps[0].alpha; // 2 0.5
/// @endcode
/// @see ConjugateGradientRun
struct ConjugateGradientStep {
    int iteration{};                         ///< One-based completed step number.
    std::vector<double> x;                    ///< Resulting actual iterate.
    std::vector<double> r;                    ///< Resulting actual recursive residual.
    std::vector<double> p;                    ///< Direction used for this step.
    std::vector<double> ap;                   ///< Actual rounded A p.
    long double alpha{};                     ///< Actual step coefficient.
    long double beta{};                      ///< Next-direction coefficient when used, otherwise zero.
    long double recursive_residual{};         ///< Norm of the stored recursively updated r.
    long double true_residual{};              ///< Recomputed norm of b - A x.
    bool has_next_direction{};                ///< Whether beta was used to form another direction.
};

/// @brief CG output with honest stop reason, counted products and optional owned steps.
/// solution.residual is the recomputed absolute norm; solution.converged is true
/// exactly for stop == CGStop::converged. Starting point is zero. There is one
/// A p product and one diagnostic A x product per completed step, with no replay.
/// A curvature failure also counts its attempted A p. Initial zero needs no call.
/// No initial snapshot is stored; initial_residual is ||b||. Returns own storage.
/// @par Example
/// @code
/// const auto t = dm::algorithms::conjugate_gradient_trace(
///     std::vector<std::vector<double>>{{2}}, {4});
/// std::cout << t.solution.converged << ' ' << t.products; // 1 2
/// @endcode
/// @see conjugate_gradient_trace(), CGStop
struct ConjugateGradientRun {
    IterativeSolution solution;               ///< Legacy-layout result with a checked true residual.
    CGStop stop{CGStop::iteration_limit};       ///< Explicit computed stopping reason.
    std::size_t products{};                   ///< All actual operator requests, including checks.
    long double initial_residual{};           ///< ||b||, also the zero-iteration true residual.
    long double recursive_residual{};         ///< Final recursively maintained norm.
    std::vector<ConjugateGradientStep> steps;  ///< Empty for solve-only calls.
};

/// @brief Solve dense SPD Ax=b from zero with checked true-residual stopping.
/// Validates every entry, square shape, b length, exact stored symmetry and
/// controls before work. Positive definiteness is a precondition: no Cholesky
/// certification is performed; only encountered nonpositive curvature is caught.
/// Empty/zero RHS can converge initially. Inclusive absolute tolerance; floating
/// CG need not finish in n steps. Dot products/norms use long double, vector
/// updates round to double. Nonfinite arithmetic cannot silently converge.
/// Time O(n² + k n²), scratch O(n); two dense products per completed step.
/// No input mutation or global state; concurrent calls are independent.
/// @param a Finite exactly symmetric square matrix; positive definite by contract.
/// @param b Finite RHS of matching size.
/// @param options Nonnegative iteration/tolerance controls; trace cap ignored.
/// @return Owned solution, stopping reason and work; steps is empty.
/// @throws std::invalid_argument Invalid shapes, entries, symmetry or controls.
/// @throws std::overflow_error A computed quantity cannot fit its working/storage format.
/// @par Example
/// @code
/// const auto r = dm::algorithms::conjugate_gradient_solve(
///     std::vector<std::vector<double>>{{2}}, {4});
/// std::cout << r.solution.x[0] << ' ' << r.steps.size(); // 2 0
/// @endcode
/// @par References
/// - M. R. Hestenes and E. Stiefel (1952), Methods of Conjugate Gradients for
///   Solving Linear Systems, https://doi.org/10.6028/jres.049.044 .
/// - True-residual diagnostics are an explicit addition to the recurrence.
/// @see conjugate_gradient_trace(), conjugate_gradient()
ConjugateGradientRun conjugate_gradient_solve(const std::vector<std::vector<double>>& a,
    const std::vector<double>& b,
    ConjugateGradientOptions options = {});

/// @brief Execute dense checked CG and retain actual steps without replay.
/// Same arithmetic, validation and stopping as conjugate_gradient_solve().
/// Stores four n-vectors per completed step, O(kn) extra storage. The actual
/// storage cap is checked after positive curvature, before updating; failure returns no partial
/// trace. It does not precharge the requested maximum number of iterations.
/// @param a Finite exactly symmetric square SPD matrix by contract.
/// @param b Finite matching RHS.
/// @param options Solve controls and online scalar-entry cap, zero unlimited.
/// @return Owned actual states, result and product count.
/// @throws std::invalid_argument Invalid inputs or controls.
/// @throws std::length_error The next actual snapshot would exceed the cap or size_t.
/// @throws std::overflow_error Nonfinite/unrepresentable computed arithmetic.
/// @par Example
/// @code
/// const auto t = dm::algorithms::conjugate_gradient_trace(
///     std::vector<std::vector<double>>{{2}}, {4});
/// std::cout << t.steps.size() << ' ' << t.steps[0].r[0]; // 1 0
/// @endcode
/// @par References
/// - Hestenes and Stiefel (1952), https://doi.org/10.6028/jres.049.044 .
/// @see conjugate_gradient_solve(), ConjugateGradientStep
ConjugateGradientRun conjugate_gradient_trace(const std::vector<std::vector<double>>& a,
    const std::vector<double>& b,
    ConjugateGradientOptions options = {});

/// @brief Solve matrix-free SPD Ax=b with the same checked recurrence as dense CG.
/// apply must be a deterministic linear SPD map on b.size() finite doubles.
/// Linearity/symmetry/definiteness cannot be certified through this interface.
/// Every returned vector is checked for size/finiteness. Callback exceptions
/// propagate. Two calls per completed step include a true-residual check; zero
/// RHS needs none. O(n) vector scratch plus callback storage, no input mutation.
/// @param apply Nonempty v -> A v callable satisfying the stated SPD contract.
/// @param b Finite RHS.
/// @param options Same absolute controls as the dense solve; trace cap ignored.
/// @return Owned result and actual call count, with no snapshots.
/// @throws std::invalid_argument Empty callback, bad vector size/finiteness or controls.
/// @throws std::overflow_error Unrepresentable computed arithmetic.
/// @par Example
/// @code
/// const dm::algorithms::LinearOperator a = [](const auto& v) {
///     return std::vector<double>{2*v[0]};
/// };
/// std::cout << dm::algorithms::conjugate_gradient_solve(a, {4}).solution.x[0]; // 2
/// @endcode
/// @par References
/// - Hestenes and Stiefel (1952), https://doi.org/10.6028/jres.049.044 .
/// @see conjugate_gradient_trace(), LinearOperator
ConjugateGradientRun conjugate_gradient_solve(
    const LinearOperator& apply, const std::vector<double>& b,
    ConjugateGradientOptions options = {});

/// @brief Retain matrix-free checked CG's actual states without replay.
/// Same operator contract, arithmetic and two-products-per-step diagnostics as
/// the matrix-free solve. Snapshot storage and online caps match the dense trace.
/// @param apply Nonempty deterministic linear SPD map with finite matching output.
/// @param b Finite RHS.
/// @param options Absolute controls and optional actual scalar-entry cap.
/// @return Owned result, actual calls and actual four-vector steps.
/// @throws std::invalid_argument Bad callback/results/RHS/controls.
/// @throws std::length_error Actual snapshot storage exceeds its positive cap or size_t.
/// @throws std::overflow_error Computed arithmetic cannot fit.
/// @par Example
/// @code
/// const dm::algorithms::LinearOperator a = [](const auto& v) {
///     return std::vector<double>{2*v[0]};
/// };
/// std::cout << dm::algorithms::conjugate_gradient_trace(a, {4}).steps.size(); // 1
/// @endcode
/// @par References
/// - Hestenes and Stiefel (1952), https://doi.org/10.6028/jres.049.044 .
/// @see conjugate_gradient_solve(), ConjugateGradientStep
ConjugateGradientRun conjugate_gradient_trace(
    const LinearOperator& apply, const std::vector<double>& b,
    ConjugateGradientOptions options = {});

/// @brief Owned actual residual history for the forward in-place Gauss-Seidel baseline.
/// residuals[0] is ||b|| and each subsequent entry follows a completed sweep.
/// Empty and zero RHS converge initially. The legacy result layout is preserved.
/// @par Example
/// @code
/// const auto t = dm::algorithms::gauss_seidel_trace({{2}}, {4});
/// std::cout << t.residuals.size() << ' ' << t.solution.x[0]; // 2 2
/// @endcode
/// @see gauss_seidel_trace()
struct GaussSeidelTrace {
    IterativeSolution solution;          ///< Final iterate and checked absolute residual.
    std::vector<double> residuals;       ///< Initial and actual post-sweep norms.
};

/// @brief Execute forward Gauss-Seidel while recording actual recomputed residuals.
/// Validates finite square shape, matching RHS, nonzero diagonal, nonnegative
/// controls before any sweep. Symmetry is not required; convergence is not
/// guaranteed for a general matrix. Inclusive absolute stopping. Uses widened
/// sums with checked double assignments. O(kn²) work, O(n+k) storage.
/// @param a Finite square matrix with nonzero diagonal.
/// @param b Finite matching RHS.
/// @param max_iter Nonnegative complete-sweep cap, default 1000.
/// @param tol Finite nonnegative absolute tolerance, default 1e-10.
/// @param max_history Positive actual residual-entry cap including the initial entry; zero unlimited.
/// @return Owned actual history and final solution.
/// @throws std::invalid_argument Invalid shape, finite values, diagonal or controls.
/// @throws std::length_error Actual history would exceed the positive cap.
/// @throws std::overflow_error Computed arithmetic cannot fit.
/// @par Example
/// @code
/// const auto t = dm::algorithms::gauss_seidel_trace({{2}}, {4});
/// std::cout << t.solution.converged << ' ' << t.residuals.back(); // 1 0
/// @endcode
/// @par References
/// - Barrett et al., Templates for the Solution of Linear Systems, 2nd ed.,
///   https://netlib.org/templates/templates.html , forward Gauss-Seidel.
/// @see gauss_seidel(), GaussSeidelTrace
GaussSeidelTrace gauss_seidel_trace(const std::vector<std::vector<double>>& a,
    const std::vector<double>& b, int max_iter = 1000, double tol = 1e-10,
    std::size_t max_history = 1000000);

/// @brief Build the dense n-by-n unscaled Dirichlet tridiagonal (-1,2,-1) matrix.
/// n=0 returns empty. Complete n² storage is checked before allocation. No dataset
/// or global state is used. O(n²) time/storage; prefer laplacian_1d_apply for sparse work.
/// @param n Number of interior unknowns.
/// @param max_entries Positive complete matrix-entry cap; zero unlimited.
/// @return Owned exactly represented matrix.
/// @throws std::length_error n² overflows size_t or exceeds the cap.
/// @par Example
/// @code
/// std::cout << dm::algorithms::laplacian_1d_matrix(2)[0][1]; // -1
/// @endcode
/// @par References
/// - Standard three-point second-difference discretisation; endpoints are homogeneous Dirichlet.
/// @see laplacian_1d_apply(), laplacian_1d_unit_solution()
std::vector<std::vector<double>> laplacian_1d_matrix(std::size_t n,
    std::size_t max_entries = 1000000);

/// @brief Apply the unscaled Dirichlet tridiagonal in O(n) time and storage.
/// Full finite-input validation precedes arithmetic; zero boundary values are
/// implicit. Widened sums narrow with range checks. Empty input returns empty.
/// @param x Finite interior vector.
/// @return Owned vector with y_i = 2x_i - x_(i-1) - x_(i+1).
/// @throws std::invalid_argument Nonfinite input.
/// @throws std::overflow_error A computed entry cannot fit double.
/// @par Example
/// @code
/// std::cout << dm::algorithms::laplacian_1d_apply({1,1})[0]; // 1
/// @endcode
/// @par References
/// - Standard three-point second difference with homogeneous Dirichlet endpoints.
/// @see laplacian_1d_matrix()
std::vector<double> laplacian_1d_apply(const std::vector<double>& x);

/// @brief Closed-form unit-RHS solution x_i = (i+1)(n-i)/2, zero-based i.
/// This is a fixture/oracle, not a CG solve. Complete dimension caps precede
/// allocation; widened arithmetic avoids integer products. n=0 returns empty.
/// O(n) time/storage; returned values round to double.
/// @param n Number of interior unknowns.
/// @param max_size Positive complete dimension cap; zero unlimited.
/// @return Owned analytical solution of laplacian_1d_matrix(n) x = ones.
/// @throws std::length_error n exceeds the positive cap.
/// @throws std::overflow_error A computed entry cannot fit double.
/// @par Example
/// @code
/// std::cout << dm::algorithms::laplacian_1d_unit_solution(3)[1]; // 2
/// @endcode
/// @par References
/// - Derived by substituting the discrete quadratic into 2x_i-x_(i-1)-x_(i+1)=1.
/// @see laplacian_1d_apply()
std::vector<double> laplacian_1d_unit_solution(std::size_t n,
    std::size_t max_size = 1000000);

} // namespace datamunge::algorithms
