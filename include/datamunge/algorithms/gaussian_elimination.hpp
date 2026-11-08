#pragma once

/// @file gaussian_elimination.hpp
/// @brief Actual checked partial-pivot elimination, owned histories and numerical controls.

#include <datamunge/algorithms/linear_solvers.hpp>
#include <cstddef>
#include <string>

namespace datamunge::algorithms {

/// @brief Owned square coefficient rows for the checked direct solver.
/// No implicit symmetry, scaling, sparsity or matrix-factorization metadata.
/// @par Example
/// @code
/// dm::algorithms::GaussianMatrix a{{2}};
/// std::cout << a[0][0]; // 2
/// @endcode
/// @see gaussian_elimination_solve()
using GaussianMatrix = std::vector<std::vector<double>>;

/// @brief Actual arithmetic work in one partial-pivot solve, excluding validation and copies.
/// Counts record executed comparisons, nontrivial swaps and stored updates, even
/// on a pivot rejection. A coefficient update is one multiply-subtract, not one
/// floating operation. Checks, allocation, diagnostics and history are excluded.
/// @par Example
/// @code
/// const auto s = dm::algorithms::gaussian_elimination_solve({{2}}, {4});
/// std::cout << s.work.back_divisions; // 1
/// @endcode
/// @see GaussianEliminationRun
struct GaussianEliminationWork {
    std::size_t pivot_comparisons{}; ///< Strict absolute-value candidate comparisons.
    std::size_t row_swaps{};         ///< Accepted pivot rows different from the current row.
    std::size_t eliminations{};      ///< Completed below-pivot row updates, including zero factors.
    std::size_t coefficient_updates{}; ///< Updated a[row][column..n), including cancelled entry.
    std::size_t rhs_updates{};       ///< Completed transformed RHS assignments.
    std::size_t back_products{};     ///< Executed upper-triangular multiply-subtracts.
    std::size_t back_divisions{};    ///< Completed solution-coordinate divisions.
};

/// @brief Owned ordinary solve output, actual upper state, row order and element-growth evidence.
/// solution preserves the legacy LinearSolution layout. A rejected pivot leaves
/// solution.x empty and accepted_pivots identifies its zero-based column. This
/// is a computed cutoff failure, not a rank/conditioning certificate. upper/rhs
/// are actual stored doubles, including cancellation residues; row_order[i] is
/// the original row now at i. They remain available on rejection. Successful
/// empty input has accepted_pivots0 and solved=true. max_coefficient includes
/// the input and every stored coefficient assignment; RHS is excluded. growth
/// is max_coefficient/initial_max, or the placeholder1 for all-zero/empty A.
/// The scalar double recurrence is not widened; finite stored assignments are
/// checked, while rounding and underflow remain possible. No shared state.
/// @par Example
/// @code
/// const auto s = dm::algorithms::gaussian_elimination_solve({{0,1},{2,1}}, {1,3});
/// std::cout << s.solution.x[0] << ' ' << s.row_order[0]; // 1 1
/// @endcode
/// @see gaussian_elimination_trace(), GaussianEliminationWork
struct GaussianEliminationRun {
    LinearSolution solution;        ///< Owned legacy result: empty x on cutoff failure.
    GaussianMatrix upper;           ///< Actual final coefficient state, not packed LU factors.
    std::vector<double> rhs;         ///< Actual final transformed right-hand side.
    std::vector<std::size_t> row_order; ///< Original row IDs in current order.
    GaussianEliminationWork work;    ///< Actual completed work, no replay.
    std::size_t accepted_pivots{};   ///< n on success; failing column on cutoff rejection.
    double cutoff{1e-12};            ///< Positive finite absolute pivot cutoff used, equality accepted.
    double initial_max{};            ///< Largest absolute input coefficient, zero for empty A.
    double max_coefficient{};        ///< Largest absolute input/updated stored coefficient.
    double max_multiplier{};         ///< Largest absolute completed elimination factor, zero if none.
    long double growth{1};           ///< Observed stored-coefficient growth, not an accuracy bound.
};

/// @brief One actual accepted/rejected pivot, completed elimination or back-substitution step.
/// kind is pivot, reject, eliminate or back. column is zero-based; row is selected
/// current pivot row (before swapping), eliminated row or solved coordinate.
/// value is selected pivot, actual factor or assigned solution coordinate.
/// matrix/rhs/row_order are post-event owned copies, except reject precedes any
/// rejected swap. x is empty before back substitution and a full n-vector after
/// each back step; unsolved earlier coordinates are zero placeholders.
/// @par Example
/// @code
/// const auto t = dm::algorithms::gaussian_elimination_trace({{2}}, {4});
/// std::cout << t.events[1].kind << ' ' << t.events[1].x[0]; // back 2
/// @endcode
/// @see GaussianEliminationTrace
struct GaussianEliminationEvent {
    std::string kind;                ///< Actual event name: pivot/reject/eliminate/back.
    std::size_t column{};            ///< Zero-based current pivot or back-substitution column.
    std::size_t row{};               ///< Selected pivot row, eliminated row or solution coordinate.
    double value{};                  ///< Used pivot, multiplier or resulting solution value.
    GaussianMatrix matrix;          ///< Actual coefficient snapshot, including small cancellation residues.
    std::vector<double> rhs;         ///< Actual transformed RHS snapshot.
    std::vector<double> x;           ///< Empty before back, otherwise full solution state.
    std::vector<std::size_t> row_order; ///< Actual current-to-original row permutation.
};

/// @brief Owned inputs, actual elimination history and ordinary run from one shared execution.
/// Inputs and all event vectors are independent of caller and other snapshots.
/// captured_entries counts input rows/cells/RHS, plus each event's one metadata
/// record and matrix rows/cells/RHS/x/row_order. Ordinary final outputs, root/work
/// metadata, working storage and allocator bytes are excluded. Entries are not
/// bytes or peak RAM. Worst-case history costs O(n^4), ordinary work O(n^3).
/// @par Example
/// @code
/// const auto t = dm::algorithms::gaussian_elimination_trace({{2}}, {4});
/// std::cout << t.events.size() << ' ' << t.captured_entries; // 2 14
/// @endcode
/// @see gaussian_elimination_trace_entries()
struct GaussianEliminationTrace {
    GaussianMatrix input;           ///< Owned original finite square coefficient matrix.
    std::vector<double> rhs;         ///< Owned original finite RHS.
    GaussianEliminationRun run;      ///< Actual final state and result from this execution.
    std::vector<GaussianEliminationEvent> events; ///< Owned actual-time snapshots, no replay.
    std::size_t captured_entries{};  ///< Observed retained history entries under the documented units.
};

/// @brief Conservative whole-history entry admission for a virtual n-by-n solve.
/// With E=n(n-1)/2+2n, returns n²+2n+E*(1+n²+4n). Zero gives0. Counts owned input
/// and worst-case event vectors including matrix row records; not bytes or peak
/// RAM. Excludes ordinary output, work/root metadata, working state and allocator
/// bytes. Rejects arithmetic/container/counter capacities even without allocation.
/// No early-stop discount: a singular input can need fewer events than admitted.
/// @param n Virtual number of unknowns.
/// @return Conservative retained-entry bound (n2 gives73).
/// @throws std::length_error Arithmetic, work-counter or container limits exceeded.
/// @par Example
/// @code
/// std::cout << dm::algorithms::gaussian_elimination_trace_entries(2); // 73
/// @endcode
/// @see gaussian_elimination_trace()
std::size_t gaussian_elimination_trace_entries(std::size_t n);

/// @brief Checked scalar Gaussian elimination with absolute-cutoff partial pivoting.
/// Validates ALL square row lengths, matching RHS, finite input and positive
/// finite cutoff before input copies or state allocation. Uses earliest maximal
/// absolute pivot ties and the legacy double recurrence/order. Rejects strictly
/// below cutoff; equality is accepted. Failure does not distinguish inconsistent
/// from underdetermined or ill-scaled nonsingular systems. No rank/conditioning
/// certificate, relative scaling, iterative refinement or stored L factor. Empty
/// square input succeeds. O(n³) work/O(n²) storage, no caller mutation.
/// @param a Finite square coefficient rows.
/// @param b Finite RHS matching the number of rows.
/// @param cutoff Positive finite absolute pivot threshold, default1e-12.
/// @return Owned result, actual upper/RHS/permutation, work and growth evidence.
/// @throws std::invalid_argument Bad shape, nonfinite input or invalid cutoff.
/// @throws std::length_error Container or work-counter limits exceeded.
/// @throws std::overflow_error A stored arithmetic result or widened growth is nonfinite.
/// @throws std::bad_alloc Allocation of owned output or working storage fails.
/// @par Example
/// @code
/// const auto s = dm::algorithms::gaussian_elimination_solve({{2}}, {4});
/// std::cout << s.solution.solved << ' ' << s.solution.x[0]; // 1 2
/// @endcode
/// @par References
/// - LAPACK DGETRF, partial pivoting with row interchanges (blocked variant):
///   https://netlib.org/lapack/explore-html/d3/d6a/dgetrf_8f_source.html .
/// - J. H. Wilkinson (1961), Error Analysis of Direct Methods of Matrix Inversion,
///   Journal of the ACM8(3),281–330, doi:10.1145/321075.321076 .
/// @see gaussian_elimination(), gaussian_elimination_trace(), gaussian_residual_summary()
GaussianEliminationRun gaussian_elimination_solve(const GaussianMatrix& a, const std::vector<double>& b,
                                                 double cutoff = 1e-12);

/// @brief Capture actual partial-pivot elimination without replay, truncation or partial results.
/// Full input/control validation precedes conservative quota admission, which
/// precedes input copies and state allocation. Zero max_entries disables only
/// the optional cap; count/container/counter checks remain. Uses the same kernel
/// and recurrence as solve-only/legacy calls. Rejecting a small pivot is a normal
/// run result and its actual rejection event is retained. A thrown quota/domain/
/// arithmetic/allocation exception returns no partial trace or caller mutation.
/// O(n^4) worst-case history/time in addition to O(n³) arithmetic.
/// @param a Finite square coefficient matrix.
/// @param b Finite matching RHS.
/// @param max_entries Whole conservative entry cap, default1000000; zero disables it.
/// @param cutoff Positive finite absolute pivot cutoff, default1e-12.
/// @return Owned inputs, actual events, final run and observed entry count.
/// @throws std::invalid_argument Invalid complete input or cutoff.
/// @throws std::length_error Whole quota, count or container capacities exceeded.
/// @throws std::overflow_error A stored arithmetic result or widened growth is nonfinite.
/// @throws std::bad_alloc Allocation of owned inputs, events or state fails.
/// @par Example
/// @code
/// const auto t = dm::algorithms::gaussian_elimination_trace({{0,1},{2,1}}, {1,3}, 73);
/// std::cout << t.events.size() << ' ' << t.captured_entries; // 5 67
/// @endcode
/// @par References
/// - LAPACK DGETRF: https://netlib.org/lapack/explore-html/d3/d6a/dgetrf_8f_source.html .
/// @see gaussian_elimination_solve(), gaussian_elimination_trace_entries()
GaussianEliminationTrace gaussian_elimination_trace(const GaussianMatrix& a, const std::vector<double>& b,
                                                   std::size_t max_entries = 1000000,
                                                   double cutoff = 1e-12);

/// @brief Original report's fixed finite coefficient matrix and RHS, returned by value.
/// Deterministic owned rows and RHS; exact mathematical solution(2,3,-1).
/// Floating arithmetic yields an approximation. No RNG or external dataset.
/// @par Example
/// @code
/// const auto f = dm::algorithms::gaussian_elimination_example();
/// std::cout << f.a[0][0] << ' ' << f.b[0]; // 2 8
/// @endcode
/// @see gaussian_elimination_example(), gaussian_elimination_growth_matrix()
struct GaussianSystem {
    GaussianMatrix a;               ///< Owned square coefficient rows.
    std::vector<double> b;           ///< Owned matching right-hand side.
};

/// @brief Return the original three-equation report fixture without solving it.
/// Owned copies; exact mathematical solution(2,3,-1), no random state.
/// @return Coefficients{{2,1,-1},{-3,-1,2},{-2,1,2}}, RHS{8,-11,-3}.
/// @throws std::bad_alloc Fixture allocation fails.
/// @par Example
/// @code
/// const auto f = dm::algorithms::gaussian_elimination_example();
/// std::cout << f.a.size() << ' ' << f.b.size(); // 3 3
/// @endcode
/// @see GaussianSystem, gaussian_elimination_solve()
GaussianSystem gaussian_elimination_example();

/// @brief Wilkinson partial-pivot growth family with last column one and strict earliest ties.
/// Diagonal and last column are1, strict lower triangle-1, other entries0.
/// For n>0 and exact arithmetic, this solver's earliest-tie pivoting has no swaps,
/// |multiplier|=1 and coefficient growth2^(n-1). This is not the symmetric
/// eigenvalue Wilkinson matrix, nor a claim that every RHS has large error.
/// O(n²) owned storage/time. Empty n0 returns empty. Checks n+n² row/cell entries
/// and capacities before allocation; zero max_entries disables only optional cap.
/// @param n Matrix order.
/// @param max_entries Row/cell cap, default1000000; zero disables optional cap.
/// @return Owned square growth-example coefficient matrix.
/// @throws std::length_error Entry quota, arithmetic or container capacities exceeded.
/// @throws std::bad_alloc Matrix allocation fails.
/// @par Example
/// @code
/// const auto a = dm::algorithms::gaussian_elimination_growth_matrix(3);
/// std::cout << a[2][0] << ' ' << a[0][2]; // -1 1
/// @endcode
/// @par References
/// - N. J. Higham and D. J. Higham (1989), Large Growth Factors in Gaussian
///   Elimination with Pivoting, SIAM J. Matrix Anal. Appl.10(2),155–164,
///   doi:10.1137/0610012 .
/// - Author's explicit matrix: https://nhigham.com/2020/07/14/what-is-the-growth-factor-for-gaussian-elimination/ .
/// @see gaussian_elimination_solve(), gaussian_elimination_example()
GaussianMatrix gaussian_elimination_growth_matrix(std::size_t n, std::size_t max_entries = 1000000);

/// @brief Widened infinity-norm residual and normwise backward error for a supplied candidate.
/// These describe the candidate, not solver success, conditioning or forward
/// accuracy. residual is b-Ax in long double. backward_error is its infinity norm
/// divided by matrix_inf*x_inf+b_inf, with0/0defined0; empty givesallzeros.
/// Units differ from linalg::backward_error(), which uses Frobenius/2-norms.
/// @par Example
/// @code
/// const auto s = dm::algorithms::gaussian_residual_summary({{2}}, {4}, {1});
/// std::cout << s.residual_inf << ' ' << s.backward_error; // 2 0.333333
/// @endcode
/// @see gaussian_residual_summary()
struct GaussianResidualSummary {
    std::vector<long double> residual; ///< Owned recomputed b-Ax, rounded in long double.
    long double residual_inf{};       ///< Max absolute residual; zero for empty.
    long double matrix_inf{};         ///< Max coefficient absolute row sum; zero for empty.
    long double x_inf{};              ///< Max absolute candidate coordinate.
    long double b_inf{};              ///< Max absolute RHS coordinate.
    long double backward_error{};     ///< Normwise infinity backward error,0/0defined0.
};

/// @brief Recompute widened residual and infinity-norm backward error independently of elimination.
/// Validates full finite square A, matching b/x before any allocation. No caller
/// mutation or solver invocation; O(n²) time/O(n) output. Long double arithmetic
/// is checked for nonfinite results (it may have the same range as double).
/// A small residual/backward error alone does not bound forward solution error.
/// @param a Finite square coefficient matrix.
/// @param b Finite matching original RHS.
/// @param x Finite matching candidate solution, no stability/rank assertion.
/// @return Owned residual and normwise infinity summary; empty allzero.
/// @throws std::invalid_argument Bad shape, size or nonfinite entry.
/// @throws std::length_error Container or count capacities exceeded.
/// @throws std::overflow_error Widened arithmetic is nonfinite.
/// @throws std::bad_alloc Residual output allocation fails.
/// @par Example
/// @code
/// const auto s = dm::algorithms::gaussian_residual_summary({{2}}, {4}, {2});
/// std::cout << s.residual_inf << ' ' << s.backward_error; // 0 0
/// @endcode
/// @par References
/// - LAPACK Users' Guide, Further Details: Error Bounds for Linear Equation Solving,
///   https://netlib.org/lapack/lug/node81.html , normwise infinity formula.
/// @see gaussian_elimination_solve(), GaussianResidualSummary
GaussianResidualSummary gaussian_residual_summary(const GaussianMatrix& a, const std::vector<double>& b,
                                                  const std::vector<double>& x);

} // namespace datamunge::algorithms
