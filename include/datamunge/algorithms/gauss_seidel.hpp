#pragma once

/// @file gauss_seidel.hpp
/// @brief Actual forward Gauss-Seidel coordinates, sweep states and counted dense work.
#include <datamunge/algorithms/conjugate_gradient.hpp>
#include <cstddef>

namespace datamunge::algorithms {

/// @brief Actual dense arithmetic counts from a forward Gauss-Seidel execution.
/// Counts include zero coefficients. One product is a multiply-subtract or
/// multiply-add, not one floating operation. Validation, copies, allocation and
/// norm arithmetic are excluded. The initial residual evaluation uses b alone.
/// @par Example
/// @code
/// const auto r = dm::algorithms::gauss_seidel_solve({{2}}, {4});
/// std::cout << r.work.divisions << ' ' << r.work.residual_products; // 1 1
/// @endcode
/// @see GaussSeidelRun
struct GaussSeidelWork {
    std::size_t coordinate_updates{}; ///< Completed in-place double assignments.
    std::size_t off_diagonal_products{}; ///< Executed off-diagonal multiply-subtracts.
    std::size_t divisions{}; ///< Completed coordinate divisions, one per assignment.
    std::size_t residual_products{}; ///< Dense row products for post-sweep A x, including zeros.
    std::size_t residual_evaluations{}; ///< Initial norm of b plus one norm per completed sweep.
};

/// @brief Owned final Gauss-Seidel result and work from the same execution.
/// Zero start; inclusive absolute tolerance. No history or hidden replay.
/// solution retains IterativeSolution's legacy layout. Inputs are never changed;
/// no global state is used. O(kn²) work and O(n) extra storage for k sweeps.
/// @par Example
/// @code
/// const auto r = dm::algorithms::gauss_seidel_solve({{2}}, {4});
/// std::cout << r.solution.x[0] << ' ' << r.work.coordinate_updates; // 2 1
/// @endcode
/// @see gauss_seidel_solve(), GaussSeidelHistory
struct GaussSeidelRun {
    IterativeSolution solution; ///< Final owned iterate, actual sweep count and checked residual.
    GaussSeidelWork work; ///< Actual arithmetic counts, including checks after the final sweep.
};

/// @brief One actual in-place coordinate assignment with its mixed read state.
/// row is zero-based, sweep is one-based. before contains new lower-index values
/// and old higher-index values; after changes only row. numerator is the widened
/// b[row] minus ordered off-diagonal products immediately before division.
/// Snapshots are independent owned copies, not reconstructions of the solver.
/// @par Example
/// @code
/// const auto t = dm::algorithms::gauss_seidel_history({{2,-1},{-1,2}}, {1,1}, 1, 0);
/// std::cout << t.coordinates[1].before[0] << ' ' << t.coordinates[1].value; // 0.5 0.75
/// @endcode
/// @see GaussSeidelSweep
struct GaussSeidelCoordinate {
    int sweep{}; ///< One-based current complete-sweep number.
    std::size_t row{}; ///< Zero-based coordinate being assigned.
    long double numerator{}; ///< Actual ordered widened numerator used by division.
    double previous{}; ///< Stored double at row before assignment.
    double value{}; ///< Checked rounded double after assignment.
    std::vector<double> before; ///< Actual mixed vector read during this coordinate update.
    std::vector<double> after; ///< Actual vector immediately after this coordinate update.
};

/// @brief One actual completed sweep and its recomputed residual.
/// residual uses widened A x row sums rounded to double, then widened b-A x
/// differences and hypot, finally checked to double. It is not an exact residual
/// or forward-error certificate. work is cumulative, including this residual.
/// @par Example
/// @code
/// const auto t = dm::algorithms::gauss_seidel_history({{2,-1},{-1,2}}, {1,1}, 1, 0);
/// std::cout << t.sweeps[0].residual << ' ' << t.sweeps[0].work.residual_products; // 0.75 4
/// @endcode
/// @see GaussSeidelCoordinate, GaussSeidelHistory
struct GaussSeidelSweep {
    int number{}; ///< One-based completed-sweep count.
    std::vector<double> x; ///< Owned actual end-of-sweep iterate.
    double residual{}; ///< Actual recomputed Euclidean stopping norm.
    GaussSeidelWork work; ///< Cumulative completed work from this execution.
};

/// @brief Owned initial vector, actual coordinate events, sweep states and final run.
/// captured_entries counts initial.size()+1 for the initial residual, then one
/// metadata record plus both n-vectors per coordinate and one metadata record
/// plus its n-vector per completed sweep. Record metadata includes scalar fields
/// and work. Units are entries, not bytes or peak RAM. Original input A/b, root
/// metadata, final ordinary run, working vectors and allocator storage are excluded.
/// All vectors are independent of caller and other records. O(kn²) history.
/// @par Example
/// @code
/// const auto t = dm::algorithms::gauss_seidel_history({{2}}, {4});
/// std::cout << t.coordinates.size() << ' ' << t.captured_entries; // 1 7
/// @endcode
/// @see gauss_seidel_history_entries(), gauss_seidel_history()
struct GaussSeidelHistory {
    GaussSeidelRun run; ///< Ordinary final output, excluded from history entry accounting.
    std::vector<double> initial; ///< Owned all-zero initial vector.
    double initial_residual{}; ///< Actual norm of b before any sweep.
    std::vector<GaussSeidelCoordinate> coordinates; ///< Actual ordered coordinate updates.
    std::vector<GaussSeidelSweep> sweeps; ///< Actual completed sweep states, no initial pseudo-sweep.
    std::size_t captured_entries{}; ///< Actual retained entries, no larger than whole admission bound.
};

/// @brief Count a complete worst-case Gauss-Seidel history before execution.
/// For n>0 the bound is n+1+K*[n*(1+2n)+(1+n)]; for n=0 it is 1 because the
/// empty system converges initially. Includes initial vector/norm. No early
/// success discount for a nonempty system, no allocation or work is performed.
/// @param n Nonnegative system dimension.
/// @param max_iter Nonnegative complete-sweep budget, default 1000.
/// @return Representable worst-case history entries; units follow GaussSeidelHistory.
/// @throws std::invalid_argument Negative budget.
/// @throws std::length_error Entry-count arithmetic overflows size_t.
/// @par Example
/// @code
/// std::cout << dm::algorithms::gauss_seidel_history_entries(2, 1); // 16
/// @endcode
/// @see gauss_seidel_history()
std::size_t gauss_seidel_history_entries(std::size_t n, int max_iter = 1000);

/// @brief Execute checked forward in-place Gauss-Seidel with actual work counts.
/// Validates all inputs before initial convergence. Uses the legacy zero start,
/// ordered widened sums, checked double assignments and recomputed residual from
/// rounded dense A x. Stops when residual<=tol or max_iter sweeps complete.
/// Nonzero diagonal is required; no dominance, symmetry or convergence claim is
/// inferred. Empty and zero RHS converge initially. No input mutation or shared
/// state. Dense O(kn²) work/O(n) extra space; zeros are still visited.
/// @param a Finite square coefficient matrix with nonzero diagonal.
/// @param b Finite matching right-hand side.
/// @param max_iter Nonnegative sweep budget, default 1000.
/// @param tol Finite nonnegative absolute Euclidean residual tolerance, default 1e-10.
/// @return Owned final iterate and counts from the actual execution, including budget exhaustion.
/// @throws std::invalid_argument Invalid shape, values, diagonal or controls.
/// @throws std::length_error Actual work counters overflow size_t.
/// @throws std::overflow_error Computed arithmetic cannot fit finite double or working arithmetic.
/// @par Example
/// @code
/// const auto r = dm::algorithms::gauss_seidel_solve({{2,-1},{-1,2}}, {1,1}, 1, 0);
/// std::cout << r.solution.x[0] << ' ' << r.solution.x[1]; // 0.5 0.75
/// @endcode
/// @par References
/// - Barrett et al., Templates for the Solution of Linear Systems, 2nd ed.,
///   https://netlib.org/linalg/html_templates/node14.html , forward Gauss-Seidel;
///   this implementation uses dense widened sums and an absolute recomputed norm.
/// @see gauss_seidel(), gauss_seidel_history()
GaussSeidelRun gauss_seidel_solve(const std::vector<std::vector<double>>& a,
    const std::vector<double>& b, int max_iter = 1000, double tol = 1e-10);

/// @brief Execute the shared Gauss-Seidel kernel while capturing its actual states.
/// Same arithmetic, validation and stopping as gauss_seidel_solve; no replay.
/// Checks the complete worst-case entry bound and vector capacities before
/// initialization or history allocation. A positive cap admits the full requested
/// budget even when the RHS would converge initially; zero disables only the
/// optional quota. No truncation, partial returned trace, or caller mutation.
/// Existing gauss_seidel_trace keeps its separate online residual-entry policy.
/// @param a Finite square coefficient matrix with nonzero diagonal.
/// @param b Finite matching RHS.
/// @param max_iter Nonnegative sweep budget, default 1000.
/// @param tol Finite nonnegative absolute Euclidean residual cutoff, default 1e-10.
/// @param max_entries Positive whole-history entry cap, default 1000000; zero disables quota.
/// @return Owned actual history and ordinary result; captured count may be below admitted bound.
/// @throws std::invalid_argument Invalid shape, values, diagonal or controls, checked before quota.
/// @throws std::length_error Whole count/capacity/quota or actual work counter overflows.
/// @throws std::overflow_error Computed arithmetic cannot fit finite double or working arithmetic.
/// @par Example
/// @code
/// const auto t = dm::algorithms::gauss_seidel_history({{2,-1},{-1,2}}, {1,1}, 1, 0, 16);
/// std::cout << t.captured_entries << ' ' << t.sweeps[0].x[1]; // 16 0.75
/// @endcode
/// @par References
/// - Barrett et al., https://netlib.org/linalg/html_templates/node14.html .
/// @see gauss_seidel_history_entries(), gauss_seidel_trace()
GaussSeidelHistory gauss_seidel_history(const std::vector<std::vector<double>>& a,
    const std::vector<double>& b, int max_iter = 1000, double tol = 1e-10,
    std::size_t max_entries = 1000000);

} // namespace datamunge::algorithms
