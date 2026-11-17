#pragma once
/// @file hungarian.hpp
/// @brief Checked square assignment with actual primal-dual execution histories.
#include <cstddef>
#include <limits>
#include <vector>
namespace datamunge::algorithms {
/// @brief Sentinel for an unmatched column or virtual start in public histories.
inline constexpr std::size_t kAssignmentUnmatched = std::numeric_limits<std::size_t>::max();
/// @brief Minimum-cost row-to-column permutation and computed total.
/// Owns its values, with the legacy field order and types preserved. Default is
/// an empty, unsolved value; successful solvers require a nonempty square input.
/// @par Example
/// @code
/// const auto r=dm::algorithms::hungarian({{9,2},{6,4}});
/// std::print("{} {}\n",r.assignment[0],r.cost); // 1 8
/// @endcode
/// @see hungarian()
struct AssignmentResult {
  std::vector<std::size_t>
         assignment; ///< Zero-based task for each row, a permutation after success.
  double cost{0};    ///< Computed row-order sum of selected input costs, finite after success.
};
/// @brief Optional complete-input and online history admission policies.
/// Zero disables the respective optional cap. Each event costs 1+4*n logical
/// entries (one event plus n entries in each of four owned vectors). The cap
/// excludes input copy, final result/potentials/work, vector capacity, scalar
/// metadata and ordinary/transient solver arrays; it is not peak bytes/RAM.
/// Untraced solves ignore the history cap. No partial history is returned on error.
/// @par Example
/// @code
/// const dm::algorithms::AssignmentOptions o;
/// std::print("{}\n",o.max_input_cells); // 1000000
/// @endcode
/// @see hungarian_trace()
struct AssignmentOptions {
  std::size_t max_input_cells{
      1000000}; ///< Complete borrowed n*n matrix entries, checked before working arrays/copy.
  std::size_t max_history_entries{
      1000000}; ///< Complete event entries, checked before the next producing transition.
};
/// @brief Kind of an actual captured Hungarian boundary.
/// Reprice follows a frontier scan and potential write; Augment follows all path
/// flips for one introduced row. Initial zero state is implicit, not captured.
/// @par Example
/// @code
/// const auto k=dm::algorithms::AssignmentEventKind::Reprice;
/// std::print("{}\n",k==dm::algorithms::AssignmentEventKind::Reprice); // true
/// @endcode
/// @see AssignmentEvent
enum class AssignmentEventKind {
  Reprice, ///< State immediately after a selected delta was applied, before entering next_column.
  Augment ///< State immediately after the matching was augmented; tree flags remain from the search.
};
/// @brief Owned actual state at one reprice or completed augmentation boundary.
/// All vectors have n entries and omit the private virtual column. Reprice's
/// matching precedes path flips, while Augment's matching includes them. Reached
/// flags mark columns already expanded, so next_column is not marked at Reprice.
/// Unmatched columns use kAssignmentUnmatched; no reference to the input is kept.
/// @par Example
/// @code
/// const auto t=dm::algorithms::hungarian_trace({{2}});
/// std::print("{} {}\n",t.events.size(),t.events.back().column_to_row[0]); // 2 0
/// @endcode
/// @see hungarian_trace()
struct AssignmentEvent {
  AssignmentEventKind kind{AssignmentEventKind::Reprice}; ///< Captured transition kind.
  std::size_t         introduced_row{0};                  ///< Zero-based outer row being added.
  std::size_t         expanded_row{0};   ///< Last expanded row supplying this frontier scan.
  std::size_t         next_column{0};    ///< Zero-based selected column; free endpoint for Augment.
  double              delta{0};          ///< Actual reprice delta; zero for an Augment event.
  std::vector<double> row_potentials;    ///< Computed n row potentials at this boundary.
  std::vector<double> column_potentials; ///< Computed n column potentials at this boundary.
  std::vector<std::size_t>
      column_to_row; ///< Actual zero-based partial matching or unmatched sentinel.
  std::vector<unsigned char> reached_columns; ///< Actual expanded-column flags, values zero or one.
};
/// @brief Selected actual operation counts from the shared producing kernel.
/// Counts omit input validation, allocations, copies, comparisons, result sum and
/// trace observation. They describe operations, not elapsed time or proof of exact
/// floating-point optimality. Overflow in counters throws rather than wraps.
/// @par Example
/// @code
/// const auto r=dm::algorithms::hungarian_run({{2}});
/// std::print("{} {}\n",r.work.slack_evaluations,r.work.path_writes); // 1 1
/// @endcode
/// @see hungarian_run()
struct AssignmentWork {
  std::size_t slack_evaluations{0}; ///< Actual cost-u-v evaluations at unexpanded real columns.
  std::size_t reprices{0};          ///< Actual frontier/potential-update rounds.
  std::size_t potential_pairs{0};   ///< Actual paired u/v writes, including virtual column zero.
  std::size_t path_writes{0};       ///< Actual column-matching writes while flipping paths.
};
/// @brief Final permutation, computed potentials and selected producing work.
/// Potentials provide numerical diagnostics: exact arithmetic theory certifies
/// optimality, while floating feasibility/tightness must be checked at a suitable
/// scale. Negative inputs need not be feasible against unintroduced rows during
/// execution; completed rows are handled by the active-prefix invariant.
/// @par Example
/// @code
/// const auto r=dm::algorithms::hungarian_run({{-3}});
/// std::print("{} {}\n",r.result.cost,r.row_potentials[0]); // -3 -3
/// @endcode
/// @see hungarian_run()
struct AssignmentRun {
  AssignmentResult    result;         ///< Owned final legacy-compatible assignment.
  std::vector<double> row_potentials; ///< Final computed row duals, excludes private sentinel.
  std::vector<double>
                 column_potentials; ///< Final computed column duals, excludes private sentinel.
  AssignmentWork work;              ///< Actual selected work from the solve that produced result.
};
/// @brief Owned original costs and actual state history from one solve.
/// History is generated inside the ordinary kernel, never by replaying a solved
/// assignment. Capture costs O(events*n) time/storage in addition to the solver.
/// @par Example
/// @code
/// const auto t=dm::algorithms::hungarian_trace({{2}});
/// std::print("{} {}\n",t.costs[0][0],t.run.result.cost); // 2 2
/// @endcode
/// @see hungarian_trace()
struct AssignmentTrace {
  AssignmentRun run; ///< Final state from this same execution.
  std::vector<std::vector<double>>
      costs; ///< Owned complete original costs, not altered by the solve.
  std::vector<AssignmentEvent>
      events; ///< Actual reprice and augmentation boundaries in execution order.
};
/// @brief Solve a finite nonempty square minimum-cost assignment.
/// Legacy entry point disables both optional caps; tie choices and ordinary finite arithmetic order are preserved.
/// No mutation, retained references or shared state. Square finite negative costs
/// are allowed; NaN/infinity are rejected. The deterministic ascending-column
/// strict-less tie rule is preserved. Ordinary solve O(n^3) time/O(n) auxiliary
/// storage excluding borrowed n*n input and owned outputs; trace adds its stated
/// copy/history costs. Floating subtraction/addition can round: this is a
/// numerical implementation, not an exact real or integer certificate. Any
/// nonfinite produced slack, potential, finite frontier update or total throws.
/// Maximize values by negating finite entries and interpreting the total sign.
/// @param cost Borrowed nonempty n*n finite costs, row i to column j.
/// @return Owned permutation and total.
/// @throws std::invalid_argument On empty, nonsquare or nonfinite input.
/// @throws std::length_error On dimension/counter overflow or an enabled complete admission cap.
/// @throws std::overflow_error On nonfinite producing arithmetic or no finite next frontier.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto r=dm::algorithms::hungarian({{2}});
/// std::print("{}\n",r.cost); // 2
/// @endcode
/// @par References
/// - H. W. Kuhn (1955), The Hungarian method for the assignment problem,
///   Naval Research Logistics Quarterly 2,83–97. doi:10.1002/nav.3800020109.
/// - J. Munkres (1957), Algorithms for the Assignment and Transportation Problems,
///   J. SIAM 5,32–38. doi:10.1137/0105003. Compact augmenting-path potential variant.
/// @see AssignmentOptions, AssignmentEvent, hungarian_run(), hungarian_trace()
AssignmentResult hungarian(const std::vector<std::vector<double>>& cost);
/// @brief Solve once and return computed duals and actual operation counts.
/// Uses optional complete input admission; history cap is ignored.
/// No mutation, retained references or shared state. Square finite negative costs
/// are allowed; NaN/infinity are rejected. The deterministic ascending-column
/// strict-less tie rule is preserved. Ordinary solve O(n^3) time/O(n) auxiliary
/// storage excluding borrowed n*n input and owned outputs; trace adds its stated
/// copy/history costs. Floating subtraction/addition can round: this is a
/// numerical implementation, not an exact real or integer certificate. Any
/// nonfinite produced slack, potential, finite frontier update or total throws.
/// Maximize values by negating finite entries and interpreting the total sign.
/// @param cost Borrowed nonempty n*n finite costs, row i to column j.
/// @param options Input/history policies, default one million entries each; zero disables.
/// @return Owned computed final state.
/// @throws std::invalid_argument On empty, nonsquare or nonfinite input.
/// @throws std::length_error On dimension/counter overflow or an enabled complete admission cap.
/// @throws std::overflow_error On nonfinite producing arithmetic or no finite next frontier.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto r=dm::algorithms::hungarian_run({{2}});
/// std::print("{}\n",r.result.cost); // 2
/// @endcode
/// @par References
/// - H. W. Kuhn (1955), The Hungarian method for the assignment problem,
///   Naval Research Logistics Quarterly 2,83–97. doi:10.1002/nav.3800020109.
/// - J. Munkres (1957), Algorithms for the Assignment and Transportation Problems,
///   J. SIAM 5,32–38. doi:10.1137/0105003. Compact augmenting-path potential variant.
/// @see AssignmentOptions, AssignmentEvent, hungarian_run(), hungarian_trace()
AssignmentRun hungarian_run(
    const std::vector<std::vector<double>>& cost,
    AssignmentOptions                       options = {});
/// @brief Capture actual reprices and augmentations from the shared solve.
/// Each complete event is admitted before its frontier scan or path flip; failure returns no partial result. Input is copied only after complete input validation/admission.
/// No mutation, retained references or shared state. Square finite negative costs
/// are allowed; NaN/infinity are rejected. The deterministic ascending-column
/// strict-less tie rule is preserved. Ordinary solve O(n^3) time/O(n) auxiliary
/// storage excluding borrowed n*n input and owned outputs; trace adds its stated
/// copy/history costs. Floating subtraction/addition can round: this is a
/// numerical implementation, not an exact real or integer certificate. Any
/// nonfinite produced slack, potential, finite frontier update or total throws.
/// Maximize values by negating finite entries and interpreting the total sign.
/// @param cost Borrowed nonempty n*n finite costs, row i to column j.
/// @param options Input/history policies, default one million entries each; zero disables.
/// @return Owned input, actual events and final state.
/// @throws std::invalid_argument On empty, nonsquare or nonfinite input.
/// @throws std::length_error On dimension/counter overflow or an enabled complete admission cap.
/// @throws std::overflow_error On nonfinite producing arithmetic or no finite next frontier.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto r=dm::algorithms::hungarian_trace({{2}});
/// std::print("{}\n",r.run.result.cost); // 2
/// @endcode
/// @par References
/// - H. W. Kuhn (1955), The Hungarian method for the assignment problem,
///   Naval Research Logistics Quarterly 2,83–97. doi:10.1002/nav.3800020109.
/// - J. Munkres (1957), Algorithms for the Assignment and Transportation Problems,
///   J. SIAM 5,32–38. doi:10.1137/0105003. Compact augmenting-path potential variant.
/// @see AssignmentOptions, AssignmentEvent, hungarian_run(), hungarian_trace()
AssignmentTrace hungarian_trace(
    const std::vector<std::vector<double>>& cost,
    AssignmentOptions                       options = {});
} // namespace datamunge::algorithms
