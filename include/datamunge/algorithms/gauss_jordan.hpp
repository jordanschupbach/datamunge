#pragma once

/// @file gauss_jordan.hpp
/// @brief Checked full augmented reduction, actual histories and unsnapped inverse diagnostics.
#include <datamunge/algorithms/gaussian_elimination.hpp>
#include <string>

namespace datamunge::algorithms {

/// @brief Actual full-block Gauss–Jordan work, excluding validation, allocation and snapshots.
/// A block update counts one stored multiply-subtract, not one floating operation.
/// Zero factors and already-processed columns are still updated, preserving the
/// legacy recurrence. Identity zero-fill is allocation; diagonal writes count.
/// @par Example
/// @code
/// const auto s = dm::algorithms::gauss_jordan_solve({{2}}, {4});
/// std::cout << s.work.a_divisions << ' ' << s.work.transform_divisions; // 1 1
/// @endcode
/// @see GaussJordanRun
struct GaussJordanWork {
    std::size_t identity_diagonal_writes{}; ///< Initial identity diagonal assignments.
    std::size_t pivot_comparisons{}; ///< Strict absolute-value candidate comparisons.
    std::size_t row_swaps{};         ///< Nontrivial accepted pivot interchanges of all blocks.
    std::size_t normalizations{};    ///< Completed pivot-row normalizations.
    std::size_t eliminations{};      ///< Completed updates of other rows, including zero factors.
    std::size_t a_divisions{};       ///< Stored divisions in the coefficient block.
    std::size_t transform_divisions{}; ///< Stored divisions in the identity/transformation block.
    std::size_t rhs_divisions{};     ///< Stored pivot-row RHS divisions.
    std::size_t a_updates{};         ///< Coefficient-block multiply-subtract assignments.
    std::size_t transform_updates{}; ///< Transformation-block multiply-subtract assignments.
    std::size_t rhs_updates{};       ///< RHS multiply-subtract assignments.
};

/// @brief Owned final reduction state and legacy result from a single checked execution.
/// result.inverse and result.x exist only on success; no duplicate final inverse
/// is stored. reduced/rhs are the actual final coefficient/RHS state on success
/// or cutoff failure. Partial transformation matrices on failure are available
/// through trace events, not result.inverse. accepted_pivots is n on success or
/// the rejected zero-based column. row_order records cumulative row interchanges
/// (row combinations mean rows no longer equal the original equations).
/// max_clear_factor is the largest raw f used against a normalized pivot row;
/// it is not the Gaussian pre-normalization ratio and need not be at most1.
/// Empty valid input succeeds with empty outputs. No accuracy/rank certificate.
/// @par Example
/// @code
/// const auto s = dm::algorithms::gauss_jordan_solve({{2}}, {4});
/// std::cout << s.result.x[0] << ' ' << s.result.inverse[0][0]; // 2 0.5
/// @endcode
/// @see gauss_jordan_solve(), GaussJordanTrace
struct GaussJordanRun {
    GaussJordanResult result;        ///< Legacy-layout result; both arrays empty on cutoff failure.
    GaussianMatrix reduced;         ///< Actual final coefficient block, no manufactured identity.
    std::vector<double> rhs;         ///< Actual final transformed RHS, including on failure.
    std::vector<std::size_t> row_order; ///< Cumulative current-to-original swap IDs.
    GaussJordanWork work;            ///< Actual completed work, no numerical replay.
    std::size_t accepted_pivots{};   ///< Number accepted before rejection, n on success.
    double cutoff{1e-12};            ///< Positive finite absolute cutoff; equality accepted.
    double max_clear_factor{};       ///< Largest absolute raw f actually used to clear another row.
};

/// @brief Owned actual accepted/rejected pivot, normalization or completed other-row update.
/// kind is pivot/reject/normalize/eliminate. column is the current zero-based
/// pivot column; row is selected pivot row before swapping, normalized pivot row
/// or updated row. value is selected pivot, normalization divisor or raw f.
/// A pivot event follows all row swaps; reject precedes any rejected swap.
/// transform starts as I and undergoes exactly the same row operations as matrix
/// and rhs: it is a partial row transformation, not yet an inverse certificate.
/// Every full matrix/RHS/permutation snapshot is independently owned.
/// @par Example
/// @code
/// const auto t = dm::algorithms::gauss_jordan_trace({{2}}, {4});
/// std::cout << t.events[1].kind << ' ' << t.events[1].transform[0][0]; // normalize 0.5
/// @endcode
/// @see GaussJordanTrace
struct GaussJordanEvent {
    std::string kind;                ///< Actual event name.
    std::size_t column{};            ///< Current pivot column.
    std::size_t row{};               ///< Selected, normalized or updated row.
    double value{};                  ///< Actual pivot, divisor or raw clearing factor.
    GaussianMatrix matrix;          ///< Actual current coefficient state.
    GaussianMatrix transform;       ///< Actual evolving identity block, not assumed inverse.
    std::vector<double> rhs;         ///< Actual current transformed RHS.
    std::vector<std::size_t> row_order; ///< Cumulative row-interchange IDs.
};

/// @brief Owned original system, actual-time events and final run, without replay.
/// captured_entries counts input matrix row records/cells and RHS plus one
/// metadata record and all matrix row records/cells/RHS/permutation per event.
/// Ordinary final outputs, root/work metadata, working storage and allocator bytes
/// are excluded. Entries are not bytes or peak RAM. Successful full histories
/// attain their admission bound; early rejection may retain much less. O(n^4)
/// worst-case history/copy cost versus O(n³) ordinary arithmetic/O(n²) storage.
/// @par Example
/// @code
/// const auto t = dm::algorithms::gauss_jordan_trace({{2}}, {4});
/// std::cout << t.events.size() << ' ' << t.captured_entries; // 2 17
/// @endcode
/// @see gauss_jordan_trace_entries()
struct GaussJordanTrace {
    GaussianMatrix input;           ///< Owned original coefficient matrix.
    std::vector<double> rhs;         ///< Owned original RHS.
    GaussJordanRun run;              ///< Result and actual final state of the same execution.
    std::vector<GaussJordanEvent> events; ///< Owned actual-time snapshots, no replay.
    std::size_t captured_entries{};  ///< Observed retained entries in documented units.
};

/// @brief Conservative whole-trace entry bound for a virtual n-by-n reduction.
/// Returns n²+2n+(n²+n)*(1+2n²+4n), including matrix row records. Zero gives0.
/// Successful histories attain this count; no early-stop discount is given to
/// rejected pivots. Checks arithmetic/container/work-counter capacities without
/// allocation. Ordinary output/root/work metadata, working state and allocator
/// bytes are excluded; entries are not bytes or peak RAM.
/// @param n Virtual system order.
/// @return Whole-history admission bound; n2 gives110.
/// @throws std::length_error Count, work-counter or container capacities exceeded.
/// @par Example
/// @code
/// std::cout << dm::algorithms::gauss_jordan_trace_entries(2); // 110
/// @endcode
/// @see gauss_jordan_trace()
std::size_t gauss_jordan_trace_entries(std::size_t n);

/// @brief Checked full [A|I|b] reduction with earliest-tie partial pivoting.
/// Validates ALL finite square A, matching finite b and positive finite cutoff
/// before input copies or inverse/state allocation. Preserves legacy ordered
/// double divisions and full-column updates of both blocks, even for zero
/// factors. Rejects strictly below the absolute cutoff; equality is accepted.
/// Failure is not a rank certificate or a distinction between inconsistent,
/// underdetermined and ill-scaled nonsingular systems. Empty succeeds. Stored
/// assignments are checked for nonfinite arithmetic; rounding/underflow and
/// inaccurate inverses remain possible. O(n³) time/O(n²) owned storage, no caller
/// mutation/shared state. Computing the inverse has explicit arithmetic cost.
/// @param a Finite square coefficient rows.
/// @param b Finite RHS matching the number of rows.
/// @param cutoff Positive finite absolute cutoff, default1e-12.
/// @return Owned final state and legacy-layout result, inverse only on success.
/// @throws std::invalid_argument Bad shape, nonfinite input or invalid cutoff.
/// @throws std::length_error Count/container/work-counter capacities exceeded.
/// @throws std::overflow_error A stored double assignment is nonfinite.
/// @throws std::bad_alloc Owned output or working allocation fails.
/// @par Example
/// @code
/// const auto s = dm::algorithms::gauss_jordan_solve({{2}}, {4});
/// std::cout << s.result.solved << ' ' << s.reduced[0][0]; // 1 1
/// @endcode
/// @par References
/// - N.-k. Tsao (1989), On the Equivalence of Gaussian Elimination and Gauss-Jordan
///   Reduction in Solving Linear Equations, NASA TM101466, Section3 (related
///   diagonal-reduction variant without pivoting; this code normalizes rows and
///   explicitly carries the identity block), https://ntrs.nasa.gov/citations/19890011339 .
/// - G. Peters and J. H. Wilkinson (1975), On the Stability of Gauss-Jordan
///   Elimination with Pivoting, Communications of the ACM18(1),20–24,
///   doi:10.1145/360569.360653 .
/// @see gauss_jordan(), gauss_jordan_trace(), inverse_residual_summary()
GaussJordanRun gauss_jordan_solve(const GaussianMatrix& a, const std::vector<double>& b,
                                 double cutoff = 1e-12);

/// @brief Capture actual full reduction after whole admission and complete validation.
/// Full input/control validation precedes admission, which precedes input copies
/// and inverse/state allocation. Zero max_entries disables only the optional
/// cap; arithmetic/container/work-counter checks remain. Uses the same kernel as
/// ordinary/legacy calls, without replay, truncation or partial returned traces.
/// Failed pivots produce a normal reject event and an empty legacy inverse;
/// their event transform is a partial row transformation. Successful history
/// reaches its conservative bound, unlike early-rejected runs. O(n^4) history and
/// copying in the worst case. No caller mutation or shared state.
/// @param a Finite square coefficients.
/// @param b Finite matching RHS.
/// @param max_entries Whole-history entry cap, default1000000; zero disables it.
/// @param cutoff Positive finite absolute cutoff, default1e-12.
/// @return Owned inputs, actual events, final run and observed entry count.
/// @throws std::invalid_argument Invalid complete input or cutoff.
/// @throws std::length_error Quota, arithmetic/container/work-counter limit exceeded.
/// @throws std::overflow_error A stored double assignment is nonfinite.
/// @throws std::bad_alloc Input, snapshot or working allocation fails.
/// @par Example
/// @code
/// const auto t = dm::algorithms::gauss_jordan_trace({{0,0},{0,0}}, {0,0}, 110);
/// std::cout << t.run.result.solved << ' ' << t.captured_entries; // 0 25
/// @endcode
/// @par References
/// - Tsao (1989), NASA TM101466, Section3: https://ntrs.nasa.gov/citations/19890011339 .
/// @see gauss_jordan_solve(), gauss_jordan_trace_entries()
GaussJordanTrace gauss_jordan_trace(const GaussianMatrix& a, const std::vector<double>& b,
                                   std::size_t max_entries = 1000000, double cutoff = 1e-12);

/// @brief Owned unsnapped products and infinity-norm residuals of a supplied inverse candidate.
/// left_product=A*V and right_product=V*A use checked long-double arithmetic;
/// each error norm is max_i sum_j |product_ij-I_ij|. a_inf/candidate_inf are max
/// absolute row sums. norm_product is their product, not a certified condition
/// number unless V is the true inverse. Even small residuals are computed
/// evidence, not interval certificates of accuracy. Empty products/norms are0.
/// @par Example
/// @code
/// const auto s = dm::algorithms::inverse_residual_summary({{2}}, {{0.5}});
/// std::cout << s.left_error_inf << ' ' << s.norm_product; // 0 1
/// @endcode
/// @see inverse_residual_summary()
struct InverseResidualSummary {
    std::vector<std::vector<long double>> left_product; ///< Actual widened A*V, without snapping.
    std::vector<std::vector<long double>> right_product; ///< Actual widened V*A, without snapping.
    long double left_error_inf{};    ///< Infinity norm of A*V-I.
    long double right_error_inf{};   ///< Infinity norm of V*A-I.
    long double a_inf{};             ///< Infinity norm of A.
    long double candidate_inf{};     ///< Infinity norm of supplied V.
    long double norm_product{};      ///< Product of supplied norms, not a conditioning certificate.
};

/// @brief Independently multiply and assess both sides of a supplied inverse candidate.
/// Validates both complete finite square matrices and equal order before quota
/// or allocation. Computes products and residual norms without solving, snapping
/// small entries or assuming V is an inverse. O(n³) time/O(n²) owned output.
/// Optional cap counts 2n+2n² product row records/cells; zero disables only the
/// optional cap. Checked container/count arithmetic remains. No caller mutation.
/// This is a numerical diagnostic, not a rigorous rank/conditioning certificate.
/// @param a Finite square coefficient matrix.
/// @param candidate Finite square candidate V of the same order.
/// @param max_entries Product row/cell cap, default1000000; zero disables it.
/// @return Widened A*V, V*A and infinity-norm summary; empty gives allzero.
/// @throws std::invalid_argument Bad complete shape, size or nonfinite entry.
/// @throws std::length_error Product quota, count or container capacities exceeded.
/// @throws std::overflow_error Widened product, sum or norm is nonfinite.
/// @throws std::bad_alloc Product output allocation fails.
/// @par Example
/// @code
/// const auto s = dm::algorithms::inverse_residual_summary({{2}}, {{1}});
/// std::cout << s.left_product[0][0] << ' ' << s.left_error_inf; // 2 1
/// @endcode
/// @par References
/// - LAPACK Users' Guide, Further Details: Error Bounds for Linear Equation
///   Solving, https://netlib.org/lapack/lug/node81.html (conditioning versus
///   residuals; products here assess a supplied inverse, not its solve accuracy).
/// @see gauss_jordan_solve(), gaussian_residual_summary(), InverseResidualSummary
InverseResidualSummary inverse_residual_summary(const GaussianMatrix& a, const GaussianMatrix& candidate,
                                                std::size_t max_entries = 1000000);

} // namespace datamunge::algorithms
