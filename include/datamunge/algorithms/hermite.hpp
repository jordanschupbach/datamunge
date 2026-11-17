#pragma once
/// @file hermite.hpp
/// @brief Checked piecewise cubic Hermite values and actual selected-segment basis terms.
#include <array>
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned actual basis and weighted terms from the producing Hermite evaluation.
/// No second computation, references, allocations or symbolic substitution.
/// Terms are in h00*y0,h10*h*m0,h01*y1,h11*h*m1 order; value uses the
/// retained left-associated sum. Slopes have dy/dx units, so width factors matter.
/// Roundoff remains visible; this is not an interval bound or exact rational result.
/// @par Example
/// @code
/// const auto e=dm::algorithms::hermite_evaluate({2,4},{1,3},{2,-1},3);
/// std::print("{} {}\n",e.t,e.value); // 0.5 2.75
/// @endcode
/// @see hermite_evaluate()
struct HermiteEvaluation {
  std::size_t           interval{0}; ///< Selected zero-based left interval index.
  double                x{0};        ///< Actual finite query, not clamped.
  double                width{0};    ///< Actual positive finite endpoint difference.
  double                t{0}; ///< Actual normalized coordinate (x-x0)/width, possibly outside[0,1].
  double                t2{0};    ///< Actual t*t.
  double                t3{0};    ///< Actual t2*t.
  std::array<double, 4> basis{};  ///< Actual h00,h10,h01,h11 in retained arithmetic order.
  std::array<double, 4> terms{};  ///< Actual four width-scaled weighted contributions.
  double                value{0}; ///< Actual sequential sum, without clamping or normalization.
  bool extrapolated{false};       ///< Query lies strictly outside the complete node range.
};
/// @brief Evaluate the unique endpoint-value/derivative cubic on the selected interval.
/// Validates all inputs before interval lookup. At interior nodes uses the interval
/// to their right; at/outside the final interval uses the last interval. Outside
/// the complete range the query is not clamped: that end cubic extrapolates.
/// All values/slopes/queries must be finite and widths finite positive. Failures
/// in normalized coordinate/basis/weighted-term arithmetic throw instead of
/// returning nonfinite values. Preserves ordinary legacy operation order.
/// O(n) validation plus O(log n) lookup; constant auxiliary storage, no allocation,
/// input mutation, retained references or shared mutable state. Exact-arithmetic
/// C1 joins and cubic reproduction do not imply bit-exact floating behavior.
/// @param xs Borrowed finite strictly increasing nodes, at least2.
/// @param ys Borrowed finite nodal values, same size as xs.
/// @param slopes Borrowed finite physical derivatives dy/dx, same size as xs.
/// @param x Finite query, extrapolation permitted.
/// @return Actual finite piecewise cubic value.
/// @throws std::invalid_argument On shape/nonfinite inputs or unordered/repeated nodes.
/// @throws std::overflow_error On nonfinite width or evaluation arithmetic.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::hermite_interpolate({0,1},{0,1},{0,0},.5)); // 0.5
/// @endcode
/// @par References
/// - NIST DLMF3.3(vi), Hermite interpolation, https://dlmf.nist.gov/3.3#vi .
/// @see hermite_evaluate(), hermite_evaluate_many()
double hermite_interpolate(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    const std::vector<double>& slopes,
    double                     x);
/// @brief Capture the actual selected segment, normalized coordinate, basis and weighted sum.
/// Same validation, interval choice, arithmetic and extrapolation as
/// hermite_interpolate(). This fixed-size trace owns values and needs no history
/// cap or replay. Constant record storage; all-node validation remains O(n).
/// @param xs Borrowed finite strictly increasing nodes, at least2.
/// @param ys Borrowed finite nodal values, same size as xs.
/// @param slopes Borrowed finite physical derivatives, same size as xs.
/// @param x Finite query, not clamped.
/// @return Owned actual evaluation record from the producing scalar kernel.
/// @throws std::invalid_argument On invalid shapes, nodes or finite input domain.
/// @throws std::overflow_error On nonfinite width/evaluation arithmetic.
/// @par Example
/// @code
/// const auto e=dm::algorithms::hermite_evaluate({0,1,2},{0,1,0},{0,0,0},1);
/// std::print("{} {}\n",e.interval,e.t); // 1 0
/// @endcode
/// @see HermiteEvaluation, hermite_interpolate()
HermiteEvaluation hermite_evaluate(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    const std::vector<double>& slopes,
    double                     x);
/// @brief Validate nodes once then evaluate complete finite queries through the shared kernel.
/// Each output uses actual legacy segment arithmetic, in input query order.
/// Validates shapes/all nodes and all queries before optional query cap/allocation;
/// no partial result on arithmetic failure. Empty queries return empty only after
/// node validation. O(n+q+q log n) work/O(q) output; no references retained or
/// input mutation. Caps count outputs, not bytes/peak RAM, and do not bound n.
/// @param xs Borrowed finite strictly increasing nodes, at least2.
/// @param ys Borrowed finite values, same size as xs.
/// @param slopes Borrowed finite dy/dx values, same size as xs.
/// @param queries Borrowed finite queries, arbitrary order; extrapolation allowed.
/// @param max_queries Complete output-count cap, default1m; zero disables optional policy.
/// @return Owned actual finite values, one per query in input order.
/// @throws std::invalid_argument On shapes/nodes/nonfinite inputs before policy caps.
/// @throws std::length_error On complete query cap/container size.
/// @throws std::overflow_error On nonfinite width/evaluation arithmetic.
/// @throws std::bad_alloc On output allocation failure.
/// @par Example
/// @code
/// const auto v=dm::algorithms::hermite_evaluate_many({0,1},{0,1},{0,0},{0,.5,1});
/// std::print("{} {}\n",v.size(),v[1]); // 3 0.5
/// @endcode
/// @see hermite_evaluate(), hermite_interpolate()
std::vector<double> hermite_evaluate_many(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    const std::vector<double>& slopes,
    const std::vector<double>& queries,
    std::size_t                max_queries = 1000000);
} // namespace datamunge::algorithms
