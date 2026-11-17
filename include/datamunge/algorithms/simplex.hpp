#pragma once
/// @file simplex.hpp
/// @brief Checked feasible-origin primal simplex and actual pivot states.
#include <cstddef>
#include <vector>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms {
/// @brief SimplexStatus stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const auto value=dm::algorithms::SimplexStatus::Optimal;
/// std::print("{}\n",static_cast<int>(value)); // 0
/// @endcode
enum class SimplexStatus {
  Optimal,       ///< No reduced cost below -tolerance; approximate floating-point termination.
  Unbounded,     ///< Improving entering column has no pivot coefficient above tolerance.
  IterationLimit ///< Improving column remains when requested pivot limit is reached.
};
/// @brief SimplexResult stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::algorithms::SimplexResult value{};
/// std::print("{}\n",value.value); // 0
/// @endcode
struct SimplexResult {
  SimplexStatus status{
      SimplexStatus::Optimal}; ///< Actual stopping condition; not a rigorous numerical certificate.
  double value{};        ///< Current feasible-basis objective; zero in legacy unbounded result.
  std::vector<double> x; ///< Current structural values; empty in legacy unbounded result.
};
/// @brief SimplexOptions stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::algorithms::SimplexOptions value{};
/// std::print("{}\n",value.max_pivots); // 100000
/// @endcode
struct SimplexOptions {
  double      tolerance{1e-9}; ///< Finite positive entering/positive-column threshold.
  std::size_t max_pivots{
      100000}; ///< Maximum actual pivots; zero allows only initial termination check.
  std::size_t max_cells{1000000}; ///< Complete tableau cell count; zero disables policy.
  std::size_t max_history_values{
      1000000}; ///< Captured records plus tableau cells/basis entries; zero disables.
};
/// @brief SimplexPivot stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::algorithms::SimplexPivot value{};
/// std::print("{}\n",value.pivot); // 0
/// @endcode
struct SimplexPivot {
  std::size_t              entering{}; ///< Actual chosen smallest improving variable index.
  std::size_t              leaving{};  ///< Actual leaving basic variable before pivot.
  std::size_t              row{};      ///< Actual pivot row.
  double                   pivot{};    ///< Actual pre-normalization pivot coefficient.
  std::vector<std::size_t> basis;      ///< Actual basis after pivot.
  std::vector<std::vector<double>>
      tableau; ///< Actual complete tableau after pivot; objective is final row.
};
/// @brief SimplexRun stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::algorithms::SimplexRun value{};
/// std::print("{}\n",value.pivots); // 0
/// @endcode
struct SimplexRun {
  SimplexResult result;         ///< Actual stopping result.
  std::size_t   pivots{};       ///< Actual executed pivots.
  std::size_t   cell_updates{}; ///< Actual pivot-row divisions plus elimination cell subtractions.
  std::vector<SimplexPivot> history; ///< Optional actual post-pivot basis/tableau states.
};
/// @brief Maximize a finite feasible-origin inequality LP.
/// Maximize c dot x subject to A x<=b, x>=0, b>=0. Rectangular finite A with b.size()==A.size() and row width c.size(); zero constraints/variables allowed. Slack variables supply initial feasible basis; no PhaseI or infeasibility solver. Smallest improving reduced-cost index enters, minimum positive-column ratio leaves, exact ratio ties use smallest basic index (Bland). Arithmetic uses tolerance for entering/positive-column tests; exact-arithmetic Bland termination is not a floating-point guarantee. IterationLimit explicitly reports unresolved improvement. Optimal/Unbounded are tolerance-based outcomes, not certified decisions on ill-scaled LPs. Complete tableau/capture geometry and finite arithmetic checked. One pivot O((m+1)(m+n+1)); worst-case pivot count exponential. Borrowed inputs, owned output, no shared state.
/// @param c Complete finite objective coefficients.
/// @param A Complete finite rectangular constraint rows.
/// @param b Complete finite nonnegative right-hand sides.
/// @param options Tolerance, pivot limit, complete tableau and optional retained-value policies.
/// @return Owned actual current result, executed work and optional real tableaux.
/// @throws std::invalid_argument Malformed shape, nonfinite input, negative b or invalid tolerance.
/// @throws std::length_error Tableau/history/counter policy or capacity exceeded.
/// @throws std::domain_error Nonfinite tableau arithmetic or output objective.
/// @par Example
/// @code
/// auto r=dm::algorithms::simplex_run({5,4},{{6,4},{1,2}},{24,6});
/// std::print("{} {}\n",r.result.value,r.pivots); // 21 2
/// @endcode
/// @par References
/// - G. B. Dantzig, Linear Programming and Extensions,1963; R. G. Bland, New finite pivoting rules for the simplex method,1977, DOI10.1287/moor.2.2.103.
/// @see simplex_maximize()
SimplexRun simplex_run(
    const std::vector<double>&              c,
    const std::vector<std::vector<double>>& A,
    const std::vector<double>&              b,
    const SimplexOptions&                   options = {});
/// @brief Maximize a finite feasible-origin inequality LP with actual pivot tableaux.
/// Maximize c dot x subject to A x<=b, x>=0, b>=0. Rectangular finite A with b.size()==A.size() and row width c.size(); zero constraints/variables allowed. Slack variables supply initial feasible basis; no PhaseI or infeasibility solver. Smallest improving reduced-cost index enters, minimum positive-column ratio leaves, exact ratio ties use smallest basic index (Bland). Arithmetic uses tolerance for entering/positive-column tests; exact-arithmetic Bland termination is not a floating-point guarantee. IterationLimit explicitly reports unresolved improvement. Optimal/Unbounded are tolerance-based outcomes, not certified decisions on ill-scaled LPs. Complete tableau/capture geometry and finite arithmetic checked. One pivot O((m+1)(m+n+1)); worst-case pivot count exponential. Borrowed inputs, owned output, no shared state.
/// @param c Complete finite objective coefficients.
/// @param A Complete finite rectangular constraint rows.
/// @param b Complete finite nonnegative right-hand sides.
/// @param options Tolerance, pivot limit, complete tableau and optional retained-value policies.
/// @return Owned actual current result, executed work and optional real tableaux.
/// @throws std::invalid_argument Malformed shape, nonfinite input, negative b or invalid tolerance.
/// @throws std::length_error Tableau/history/counter policy or capacity exceeded.
/// @throws std::domain_error Nonfinite tableau arithmetic or output objective.
/// @par Example
/// @code
/// auto r=dm::algorithms::simplex_trace({5,4},{{6,4},{1,2}},{24,6});
/// std::print("{} {}\n",r.result.value,r.pivots); // 21 2
/// @endcode
/// @par References
/// - G. B. Dantzig, Linear Programming and Extensions,1963; R. G. Bland, New finite pivoting rules for the simplex method,1977, DOI10.1287/moor.2.2.103.
/// @see simplex_maximize()
SimplexRun simplex_trace(
    const std::vector<double>&              c,
    const std::vector<std::vector<double>>& A,
    const std::vector<double>&              b,
    const SimplexOptions&                   options = {});
/// @brief Return the preserved feasible-origin simplex result.
/// Uses the same checked producer/default policies; IterationLimit is now distinct from Optimal. Requires b>=0, not arbitrary inequality LPs.
/// @param c Finite objective.
/// @param A Finite rectangular constraints.
/// @param b Finite nonnegative right-hand side.
/// @return Owned stopping result.
/// @throws std::invalid_argument Malformed input.
/// @throws std::length_error Default geometry/capacity exceeded.
/// @throws std::domain_error Nonfinite arithmetic.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::simplex_maximize({5,4},{{6,4},{1,2}},{24,6}).value); // 21
/// @endcode
/// @see simplex_run()
SimplexResult simplex_maximize(
    const std::vector<double>&              c,
    const std::vector<std::vector<double>>& A,
    const std::vector<double>&              b);
} // namespace datamunge::algorithms
