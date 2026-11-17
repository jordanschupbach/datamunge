#pragma once
/// @file hungarian_analysis.hpp
/// @brief Native assignment fixtures, numerical primal-dual checks and matching diagrams.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/hungarian.hpp>
#include <datamunge/plot/diagram.hpp>
#include <format>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Return the original three-worker assignment cost matrix.
/// Owns its finite integer-valued entries, no retained references or shared state.
/// @return Matrix {{9,2,7},{6,4,3},{5,8,1}}.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::hungarian(dm::algorithms::assignment_example_costs()).cost); // 9
/// @endcode
/// @see hungarian_trace()
inline std::vector<std::vector<double>> assignment_example_costs() {
  return {{9, 2, 7}, {6, 4, 3}, {5, 8, 1}};
}
/// @brief Construct a deterministic diagonal or all-tied square work-control matrix.
/// Diagonal costs zero and off-diagonal one give a strict diagonal optimum;
/// all-tied costs are zero everywhere. These are controlled workloads, not random
/// data or runtime forecasts. Complete n*n admission precedes allocation.
/// @param n Positive row/column count.
/// @param all_tied False (default) for strict diagonal, true for all zero.
/// @param max_cells Complete output cell cap, default one million; zero disables.
/// @return Owned finite square matrix.
/// @throws std::invalid_argument On n zero.
/// @throws std::length_error On n*n overflow or complete cap violation.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto c=dm::algorithms::assignment_work_costs(2);
/// std::print("{} {}\n",c[0][0],c[0][1]); // 0 1
/// @endcode
/// @see hungarian_run()
inline std::vector<std::vector<double>>
assignment_work_costs(std::size_t n, bool all_tied = false, std::size_t max_cells = 1000000) {
  if (!n)
    throw std::invalid_argument("assignment work positive size");
  if (n > std::numeric_limits<std::size_t>::max() / n || (max_cells && n * n > max_cells))
    throw std::length_error("assignment work complete cell cap");
  std::vector<std::vector<double>> out(n, std::vector<double>(n, all_tied ? 0. : 1.));
  for (std::size_t i = 0; i < n; ++i)
    out[i][i] = 0;
  return out;
}
/// @brief Numerical final-state diagnostics, not an exact optimality certificate.
/// Long-double accumulation is converted to finite doubles. Negative minimum
/// slack or nonzero tightness/gap can reflect rounding or an invalid candidate.
/// No acceptance tolerance is imposed; the caller chooses a scale-aware policy.
/// @par Example
/// @code
/// const dm::algorithms::AssignmentDiagnostics d;
/// std::print("{}\n",d.gap); // 0
/// @endcode
/// @see assignment_diagnostics()
struct AssignmentDiagnostics {
  double primal{0};        ///< Recomputed selected-edge cost, independent of stored result.cost.
  double dual{0};          ///< Sum of supplied final row and column potentials.
  double gap{0};           ///< Computed primal minus dual, can have either sign numerically.
  double minimum_slack{0}; ///< Minimum over every c_ij-u_i-v_j, can be negative.
  double maximum_matched_residual{0}; ///< Largest absolute reduced cost on selected edges.
};
/// @brief Check complete shapes/domains and compute numerical assignment diagnostics.
/// Requires nonempty square finite costs, finite n-entry potentials and a true
/// n-entry assignment permutation; a finite stored total is required but is not
/// trusted as the recomputed primal. O(n*n) work/O(n) validation storage, no solve,
/// mutation, retained references or shared state. Does not prove exact optimality.
/// @param costs Borrowed complete original costs.
/// @param run Borrowed final candidate, including permutation and potentials.
/// @return Finite computed primal/dual/gap/minimum-slack/matched-residual diagnostics.
/// @throws std::invalid_argument On any dimension, domain or permutation error.
/// @throws std::overflow_error If a computed diagnostic cannot fit a finite double.
/// @throws std::bad_alloc On validation storage allocation failure.
/// @par Example
/// @code
/// const auto c=dm::algorithms::assignment_example_costs();
/// const auto d=dm::algorithms::assignment_diagnostics(c,dm::algorithms::hungarian_run(c));
/// std::print("{} {} {}\n",d.primal,d.dual,d.gap); // 9 9 0
/// @endcode
/// @par References
/// - H. W. Kuhn (1955), The Hungarian method for the assignment problem,
///   doi:10.1002/nav.3800020109. Primal-dual diagnostics specialize assignment duality.
/// @see AssignmentDiagnostics, hungarian_run()
inline AssignmentDiagnostics assignment_diagnostics(
    const std::vector<std::vector<double>>& costs,
    const AssignmentRun&                    run) {
  const auto n = costs.size();
  if (!n || run.row_potentials.size() != n || run.column_potentials.size() != n ||
      run.result.assignment.size() != n || !std::isfinite(run.result.cost))
    throw std::invalid_argument("assignment diagnostic shape/domain");
  std::vector<unsigned char> seen(n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    if (costs[i].size() != n || !std::isfinite(run.row_potentials[i]) ||
        !std::isfinite(run.column_potentials[i]))
      throw std::invalid_argument("assignment diagnostic shape/domain");
    const auto j = run.result.assignment[i];
    if (j >= n || seen[j])
      throw std::invalid_argument("assignment diagnostic permutation");
    seen[j] = 1;
    for (double c : costs[i])
      if (!std::isfinite(c))
        throw std::invalid_argument("assignment diagnostic finite costs");
  }
  long double primal = 0, dual = 0, minimum = std::numeric_limits<long double>::infinity(),
              residual = 0;
  for (std::size_t i = 0; i < n; ++i) {
    primal += static_cast<long double>(costs[i][run.result.assignment[i]]);
    dual += static_cast<long double>(run.row_potentials[i]) + run.column_potentials[i];
    for (std::size_t j = 0; j < n; ++j) {
      const long double slack =
          static_cast<long double>(costs[i][j]) - run.row_potentials[i] - run.column_potentials[j];
      minimum = std::min(minimum, slack);
      if (j == run.result.assignment[i])
        residual = std::max(residual, std::abs(slack));
    }
  }
  AssignmentDiagnostics out{
      static_cast<double>(primal),
      static_cast<double>(dual),
      static_cast<double>(primal - dual),
      static_cast<double>(minimum),
      static_cast<double>(residual)};
  for (double x : {out.primal, out.dual, out.gap, out.minimum_slack, out.maximum_matched_residual})
    if (!std::isfinite(x))
      throw std::overflow_error("assignment diagnostic range");
  return out;
}
/// @brief Draw the supplied final matching with selected costs and computed potentials.
/// Native boxes/edges use actual supplied costs and computed row/column potentials;
/// Task boxes are reordered to align matched pairs; no matching is recomputed. Complete validation precedes layout. Edges are
/// the supplied permutation, not a claim of exact optimality. O(n) graphic
/// objects; the complete row quota is checked before diagnostic/layout allocation.
/// @param costs Borrowed original nonempty square finite matrix.
/// @param run Borrowed complete finite candidate, validated by assignment_diagnostics().
/// @param max_rows Complete display quota, default 8; zero disables optional quota.
/// @return Owned fitted native Diagram, editable by the caller.
/// @throws std::length_error If a nonzero complete row quota is exceeded.
/// @throws std::invalid_argument On malformed costs/candidate.
/// @throws std::overflow_error On diagnostic range failure.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto d=dm::algorithms::assignment_diagram({{2}},dm::algorithms::hungarian_run({{2}}));
/// std::print("{}\n",d.to_svg().empty()); // false
/// @endcode
/// @see assignment_diagnostics(), hungarian_trace()
inline graphics::Diagram assignment_diagram(
    const std::vector<std::vector<double>>& costs,
    const AssignmentRun&                    run,
    std::size_t                             max_rows = 8) {
  if (max_rows && costs.size() > max_rows)
    throw std::length_error("assignment display row cap");
  (void)assignment_diagnostics(costs, run);
  graphics::Diagram d;
  const auto        n = costs.size();

  for (std::size_t i = 0; i < n; ++i) {
    const double y = -4. * static_cast<double>(i);
    d.node(std::format("w{}", i), {0, y}, std::format("W{}\nu={:g}", i, run.row_potentials[i]));
    const auto j = run.result.assignment[i];
    d.node(std::format("t{}", j), {12, y}, std::format("T{}\nv={:g}", j, run.column_potentials[j]));
  }
  for (std::size_t i = 0; i < n; ++i) {
    const auto j = run.result.assignment[i];
    d.edge(std::format("w{}", i), std::format("t{}", j), std::format("cost {:g}", costs[i][j]));
  }
  return d;
}
} // namespace datamunge::algorithms
