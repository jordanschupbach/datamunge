#pragma once
/// @file thomas.hpp
/// @brief Checked actual no-pivot tridiagonal elimination and returned-solution residuals.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Own ThomasOptions values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::ThomasOptions r{};
/// (void)r;
/// @endcode
struct ThomasOptions {
  std::size_t max_rows{1000000};    ///< Complete diagonal geometry; zero disables.
  std::size_t max_records{1000000}; ///< Actual forward/back records; zero disables.
};
/// @brief Own ThomasStep values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::ThomasStep r{};
/// (void)r;
/// @endcode
struct ThomasStep {
  bool        backward{};   ///< Actual backward substitution rather than forward elimination.
  std::size_t row{};        ///< Actual row updated.
  double      multiplier{}; ///< Actual forward sub/pivot multiplier; zero for backward records.
  double
      pivot{};  ///< Actual current diagonal after elimination, or divisor during back substitution.
  double rhs{}; ///< Actual current transformed RHS.
  double value{}; ///< Actual computed x coordinate for backward records, zero for forward records.
};
/// @brief Own ThomasRun values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::ThomasRun r{};
/// (void)r;
/// @endcode
struct ThomasRun {
  std::vector<double>     x;        ///< Actual final owned solution.
  std::vector<double>     diagonal; ///< Actual final eliminated main diagonal.
  std::vector<double>     rhs;      ///< Actual final transformed RHS.
  std::vector<double>     residual; ///< Actual b-A*x at returned x using original active diagonals.
  double                  residual_norm{}; ///< Actual Euclidean residual accumulated by hypot.
  std::size_t             eliminations{};  ///< Actual forward row updates, max(n-1,0).
  std::size_t             substitutions{}; ///< Actual backward x computations, n.
  std::size_t             divisions{};     ///< Actual multiplier and x divisions, max(2n-1,0).
  std::vector<ThomasStep> steps; ///< Optional actual forward then reverse-row substitution records.
};
/// @brief Own ThomasGrid values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::ThomasGrid r{};
/// (void)r;
/// @endcode
struct ThomasGrid {
  std::vector<double> sub;         ///< Original row-major below diagonal; slot0 ignored by solver.
  std::vector<double> diagonal;    ///< Original complete main diagonal.
  std::vector<double> super;       ///< Original above diagonal; final slot ignored by solver.
  std::vector<double> rhs;         ///< Original finite forcing values.
  std::vector<double> coordinates; ///< Original interior x coordinates.
  std::vector<double>
      exact; ///< Continuous sine reference at those coordinates, not the exact discrete solution.
};
/// @brief Solve through actual checked tridiagonal elimination.
/// All four vectors must have identical length n and finite entries, including ignored sub[0]/super[n-1]. Empty matching vectors return empty. Execute the historical double forward elimination and backward substitution order, checking each actual pivot before division and rejecting nonfinite intermediates. This no-pivot solver requires nonzero elimination pivots; strict diagonal dominance or SPD structure gives familiar sufficient conditions, while arbitrary nonsingular matrices can still need pivoting. Recompute b-A*x at returned x using original active diagonals. A small residual is not an unconditional forward-error bound. O(n) work/owned storage, optional O(n) producing records and no shared state/caller mutation.
/// @param sub Complete finite below diagonal; first entry unused.
/// @param diag Complete finite main diagonal.
/// @param super Complete finite above diagonal; last entry unused.
/// @param rhs Complete finite RHS.
/// @param options Complete row and actual capture policies; zero disables.
/// @return Owned actual factors/transformed RHS, solution/residual and optional row states.
/// @throws std::invalid_argument Mismatched vectors or nonfinite supplied entries.
/// @throws std::domain_error Actual elimination pivot is zero.
/// @throws std::overflow_error Actual elimination/substitution/residual arithmetic nonfinite.
/// @throws std::length_error Complete row or actual record/counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::thomas_run({0},{2},{0},{4});
/// std::print("{}\n",r.x); // [2]
/// @endcode
/// @see thomas_solve()
ThomasRun thomas_run(
    const std::vector<double>& sub,
    const std::vector<double>& diag,
    const std::vector<double>& super,
    const std::vector<double>& rhs,
    const ThomasOptions&       options = {});
/// @brief Solve through actual checked tridiagonal elimination.
/// All four vectors must have identical length n and finite entries, including ignored sub[0]/super[n-1]. Empty matching vectors return empty. Execute the historical double forward elimination and backward substitution order, checking each actual pivot before division and rejecting nonfinite intermediates. This no-pivot solver requires nonzero elimination pivots; strict diagonal dominance or SPD structure gives familiar sufficient conditions, while arbitrary nonsingular matrices can still need pivoting. Recompute b-A*x at returned x using original active diagonals. A small residual is not an unconditional forward-error bound. O(n) work/owned storage, optional O(n) producing records and no shared state/caller mutation.
/// @param sub Complete finite below diagonal; first entry unused.
/// @param diag Complete finite main diagonal.
/// @param super Complete finite above diagonal; last entry unused.
/// @param rhs Complete finite RHS.
/// @param options Complete row and actual capture policies; zero disables.
/// @return Owned actual factors/transformed RHS, solution/residual and optional row states.
/// @throws std::invalid_argument Mismatched vectors or nonfinite supplied entries.
/// @throws std::domain_error Actual elimination pivot is zero.
/// @throws std::overflow_error Actual elimination/substitution/residual arithmetic nonfinite.
/// @throws std::length_error Complete row or actual record/counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::thomas_trace({0},{2},{0},{4});
/// std::print("{}\n",r.x); // [2]
/// @endcode
/// @see thomas_solve()
ThomasRun thomas_trace(
    const std::vector<double>& sub,
    const std::vector<double>& diag,
    const std::vector<double>& super,
    const std::vector<double>& rhs,
    const ThomasOptions&       options = {});
/// @brief Preserve no-pivot Thomas solution-only value semantics.
/// All four vectors must have identical length n and finite entries, including ignored sub[0]/super[n-1]. Empty matching vectors return empty. Execute the historical double forward elimination and backward substitution order, checking each actual pivot before division and rejecting nonfinite intermediates. This no-pivot solver requires nonzero elimination pivots; strict diagonal dominance or SPD structure gives familiar sufficient conditions, while arbitrary nonsingular matrices can still need pivoting. Recompute b-A*x at returned x using original active diagonals. A small residual is not an unconditional forward-error bound. O(n) work/owned storage, optional O(n) producing records and no shared state/caller mutation.
/// @param sub Complete finite below diagonal; first entry unused.
/// @param diag Complete finite main diagonal.
/// @param super Complete finite above diagonal; last entry unused.
/// @param rhs Complete finite RHS.
/// @return Owned actual solution.
/// @throws std::invalid_argument Malformed complete diagonals/RHS.
/// @throws std::domain_error Zero actual pivot.
/// @throws std::overflow_error Actual nonfinite arithmetic.
/// @throws std::length_error Default geometry/capacity exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::thomas_solve({0},{2},{0},{4})); // [2]
/// @endcode
/// @see thomas_run()
std::vector<double> thomas_solve(
    std::vector<double> sub,
    std::vector<double> diag,
    std::vector<double> super,
    std::vector<double> rhs);
/// @brief Generate the original sine-forcing Dirichlet Poisson tridiagonal fixture.
/// n interior nodes on(0,1), zero boundary values, h=1/(n+1), active sub/super-1/h^2 and diagonal2/h^2, RHS=pi^2*sin(pi*x), with unused boundary coefficients zero. The continuous sine reference differs from the discrete solution by second-order discretization error. Complete n<=1million; n0 returns empty. Owned values, no shared state.
/// @param n Complete interior row count, default49.
/// @return Owned original coefficients, forcing, coordinates and continuous sine reference.
/// @throws std::length_error More than1million rows.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::thomas_poisson_example().diagonal.size()); // 49
/// @endcode
/// @see thomas_run()
ThomasGrid thomas_poisson_example(std::size_t n = 49);
} // namespace datamunge::algorithms
