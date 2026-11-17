#pragma once
/// @file linear_interpolation.hpp
/// @brief Checked piecewise linear interpolation with reusable owned knots and actual interval-search/blend evidence.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief One actual interval-selection comparison from the shared producing search.
/// The bool field reports the actual role-specific predicate: kind0 uses<=,
/// kind1 uses>= and kind2 uses<. It is not always a strict less-than relation.
/// @par Example
/// @code
/// const dm::algorithms::LinearInterpolator f({0,1,3},{0,2,3});
/// std::print("{}\n",f.trace(.5).probes.front().knot); // 0
/// @endcode
/// @par References
/// - NIST Digital Library of Mathematical Functions,3.3 Interpolation, https://dlmf.nist.gov/3.3; local degree1 interpolation and its remainder.
/// @see LinearInterpolator, linear_interpolate()
struct LinearInterpolationProbe {
  std::size_t knot{0};          ///< Actual original knot index tested.
  double      abscissa{0};      ///< Actual finite original knot value.
  bool        predicate{false}; ///< Actual role-specific comparison result; see kind.
  unsigned    kind{
      0}; ///< Actual comparison role:0 query<=first,1 query>=penultimate,2 query<mid in upper-bound search.
};
/// @brief Owned actual selected interval endpoints, blend and branch/work evidence.
/// @par Example
/// @code
/// const dm::algorithms::LinearInterpolator f({0,1,3},{0,2,3});
/// std::print("{}\n",f.evaluate(.5).value); // 1
/// @endcode
/// @par References
/// - NIST Digital Library of Mathematical Functions,3.3 Interpolation, https://dlmf.nist.gov/3.3; local degree1 interpolation and its remainder.
/// @see LinearInterpolator, linear_interpolate()
struct LinearInterpolationEvaluation {
  double      query{0};    ///< Actual finite caller query, not clamped.
  std::size_t interval{0}; ///< Actual left endpoint index, interval+1 is right endpoint.
  double      left_x{0};   ///< Actual original left abscissa.
  double      right_x{0};  ///< Actual original right abscissa.
  double      left_y{0};   ///< Actual original left ordinate.
  double      right_y{0};  ///< Actual original right ordinate.
  long double fraction{
      0};          ///< Actual ordinary-double t widened for storage, or actual widened fallback t.
  double value{0}; ///< Actual finite produced value, exactly original y on a numeric knot match.
  bool   extrapolated{false}; ///< True only outside complete original knot range.
  bool   exact_node{false};   ///< True only for actual x equality with a selected endpoint.
  bool   widened{
      false}; ///< Whether nonfinite ordinary intermediate arithmetic triggered widened fallback.
  std::size_t endpoint_tests{0}; ///< Actual interval-selection first/penultimate guard comparisons.
  std::size_t binary_tests{
      0}; ///< Actual upper-bound mid comparisons; excludes validation and blend-branch checks.
};
/// @brief Owned original knots and actual interval-search probes plus producing blend result.
/// Conservative complete admission uses2*n+2+bit_width(n) records before
/// copying/evaluation; actual count can be smaller due to endpoint guards.
/// Excludes fields/capacities/result/runtime copies and bytes/RAM, default
/// million records;0 optional cap off. No interval or blend replay.
/// @par Example
/// @code
/// const dm::algorithms::LinearInterpolator f({0,1},{1,3});
/// std::print("{}\n",f.trace(.5).evaluation.value); // 2
/// @endcode
/// @par References
/// - NIST Digital Library of Mathematical Functions,3.3 Interpolation, https://dlmf.nist.gov/3.3; local degree1 interpolation and its remainder.
/// @see LinearInterpolator, linear_interpolate()
struct LinearInterpolationTrace {
  std::vector<double> xs; ///< Owned complete original strictly increasing finite abscissae.
  std::vector<double> ys; ///< Owned complete original finite matched ordinates.
  LinearInterpolationEvaluation
      evaluation; ///< Actual selected endpoints, t/value and branches/work.
  std::vector<LinearInterpolationProbe>
              probes; ///< Actual role-specific interval-selection comparisons, chronological order.
  std::size_t captured_entries{
      0}; ///< Actual2*n original coordinates plus actual search-comparison records.
};
/// @brief Own once-validated piecewise-linear knots for repeated finite interpolation/extrapolation queries.
/// Require at least2 finite strictly increasing xs and equal-length finite ys;
/// duplicates/descending knots rejected. Complete constructor validation O(n)
/// precedes owned copies. Each subsequent interval search is O(log n), O(1)
/// ordinary query storage; no complete validation repeated. Exterior queries
/// extrapolate along the nearest end segment without clamping query or value.
/// Numeric knot matches return the original ordinate exactly. Ordinary finite
/// y0+t*(y1-y0) operation order retained; nonfinite intermediates trigger a
/// widened same-line fallback with checked finite double output. No mutation
/// during inference or shared state; concurrent const queries are independent.
/// @par Example
/// @code
/// const dm::algorithms::LinearInterpolator f({0,1,3},{0,2,3});
/// std::print("{} {}\n",f(.5),f(4)); // 1 3.5
/// @endcode
/// @par References
/// - NIST Digital Library of Mathematical Functions,3.3 Interpolation, https://dlmf.nist.gov/3.3; local degree1 interpolation and its remainder.
/// @see linear_interpolate(), LinearInterpolationTrace
class LinearInterpolator {
 public:
  /// @brief Validate the complete knot pair before owning copies for repeated queries.
  /// @param xs Borrowed at least2 finite strictly increasing abscissae.
  /// @param ys Borrowed equally sized complete finite ordinates.
  /// @throws std::invalid_argument On complete shape/order/nonfinite violations.
  /// @throws std::bad_alloc On owned knot copies.
  /// @par Example
  /// @code
  /// const dm::algorithms::LinearInterpolator f({0,1},{1,3});
  /// std::print("{}\n",f(.5)); // 2
  /// @endcode
  /// @par References
  /// - NIST Digital Library of Mathematical Functions,3.3 Interpolation, https://dlmf.nist.gov/3.3; local degree1 interpolation and its remainder.
  /// @see evaluate()
  LinearInterpolator(const std::vector<double>& xs, const std::vector<double>& ys);
  /// @brief Evaluate one finite query using the shared producing search/blend.
  /// @param x Finite query, including exterior extrapolation.
  /// @return Actual finite interpolated/extrapolated value.
  /// @throws std::invalid_argument On nonfinite query.
  /// @throws std::overflow_error On unrepresentable produced fraction/value arithmetic.
  /// @par Example
  /// @code
  /// const dm::algorithms::LinearInterpolator f({0,1},{1,3});
  /// std::print("{}\n",f(.5)); // 2
  /// @endcode
  /// @par References
  /// - NIST Digital Library of Mathematical Functions,3.3 Interpolation, https://dlmf.nist.gov/3.3; local degree1 interpolation and its remainder.
  /// @see evaluate()
  double operator()(double x) const;
  /// @brief Evaluate the actual selected interval/blend and return explicit branch/search evidence.
  /// @param x Finite actual query; not clamped.
  /// @return Owned actual endpoint/fraction/result/work observations, without optional capture.
  /// @throws std::invalid_argument On nonfinite query.
  /// @throws std::overflow_error On unrepresentable producing fraction/value.
  /// @par Example
  /// @code
  /// const dm::algorithms::LinearInterpolator f({0,1},{1,3});
  /// std::print("{}\n",f.evaluate(.5).fraction); // 0.5
  /// @endcode
  /// @par References
  /// - NIST Digital Library of Mathematical Functions,3.3 Interpolation, https://dlmf.nist.gov/3.3; local degree1 interpolation and its remainder.
  /// @see trace()
  LinearInterpolationEvaluation evaluate(double x) const;
  /// @brief Capture owned original knots and the real interval-selection comparisons of this query.
  /// @param x Finite actual query, exterior extrapolation allowed.
  /// @param max_entries Conservative complete logical cap, default1000000;0 optional cap off.
  /// @return Owned complete original/probe/result evidence; no repeated evaluation.
  /// @throws std::invalid_argument On nonfinite query.
  /// @throws std::length_error On complete conservative capture geometry/quota.
  /// @throws std::overflow_error On unrepresentable producing arithmetic.
  /// @throws std::bad_alloc On owned capture allocation failure.
  /// @par Example
  /// @code
  /// const dm::algorithms::LinearInterpolator f({0,1},{1,3});
  /// std::print("{}\n",f.trace(.5).evaluation.value); // 2
  /// @endcode
  /// @par References
  /// - NIST Digital Library of Mathematical Functions,3.3 Interpolation, https://dlmf.nist.gov/3.3; local degree1 interpolation and its remainder.
  /// @see LinearInterpolationTrace
  LinearInterpolationTrace trace(double x, std::size_t max_entries = 1000000) const;
  /// @brief Borrow the owned validated abscissae without mutation.
  /// @return Const reference valid for this object's lifetime.
  const std::vector<double>& xs() const noexcept { return xs_; }
  /// @brief Borrow the owned complete matched ordinates without mutation.
  /// @return Const reference valid for this object's lifetime.
  const std::vector<double>& ys() const noexcept { return ys_; }

 private:
  std::vector<double> xs_; ///< Owned once-validated abscissae.
  std::vector<double> ys_; ///< Owned once-validated ordinates.
};
/// @brief Checked legacy piecewise-linear value with complete borrowed-input validation on every call.
/// Same interval/extrapolation/numeric-knot/arithmetic policy as LinearInterpolator;
/// scalar input validation costs O(n), then O(log n) search/O(1) runtime storage.
/// Use an owned LinearInterpolator to validate once for many queries. No input
/// copies/mutation/history/shared state. At internal knots select the segment
/// to their right, except the final two knots use the final segment.
/// @param xs Complete borrowed strictly increasing finite knots, size>=2.
/// @param ys Complete borrowed equally sized finite ordinates.
/// @param x Actual finite query, exterior extrapolation allowed.
/// @return Actual finite interpolation/extrapolation value.
/// @throws std::invalid_argument On full input shape/order/nonfinite violations.
/// @throws std::overflow_error On unrepresentable produced arithmetic.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::linear_interpolate({0,1,3,6},{1,4,10,19},2.5)); // 8.5
/// @endcode
/// @par References
/// - NIST Digital Library of Mathematical Functions,3.3 Interpolation, https://dlmf.nist.gov/3.3; local degree1 interpolation and its remainder.
/// @see LinearInterpolator
double linear_interpolate(const std::vector<double>& xs, const std::vector<double>& ys, double x);
/// @brief Owned original sine sampling/grid experiment and actual interpolated/error observations.
/// @par Example
/// @code
/// const auto s=dm::algorithms::linear_interpolation_sine_study(7,300);
/// std::print("{} {}\n",s.xs.size(),s.queries.size()); // 7 301
/// @endcode
/// @par References
/// - NIST Digital Library of Mathematical Functions,3.3 Interpolation, https://dlmf.nist.gov/3.3; local degree1 interpolation and its remainder.
/// @see LinearInterpolator, linear_interpolate()
struct LinearInterpolationSineStudy {
  std::vector<double> xs;      ///< Actual nodes0..2*pi, preserving the original uniform formula.
  std::vector<double> ys;      ///< Actual sin(xs) node values.
  std::vector<double> queries; ///< Actual complete uniform query grid0..2*pi.
  std::vector<double> exact;   ///< Actual analytic sin at every grid query, a function reference.
  std::vector<double> interpolated; ///< Actual native repeated-query interpolation values.
  double              spacing{0};   ///< Prescribed2*pi/(nodes-1) mathematical grid-spacing formula.
  double              error_reference{
      0}; ///< Prescribed spacing squared/8 smooth-sine reference, not a floating-point enclosure.
  double max_grid_error{0}; ///< Observed maximum absolute difference on the finite query grid only.
};
/// @brief Run the original uniform sine interpolation experiment using one once-validated native interpolator.
/// No external dataset/preprocessing, no inferred continuous maximum. Complete
/// node/grid geometry checked before allocation. Samples/queries use the original
///2*pi*i/(count-1) operation order; query intervals300 reproduces the figure and
///1000 the original error table. The h^2/8 line is an exact-arithmetic theory reference.
/// @param nodes Original node count>=2.
/// @param grid_intervals Positive uniform query interval count, default1000.
/// @return Owned complete nodes, truth, actual interpolant and finite-grid error/reference.
/// @throws std::invalid_argument On node count<2 or zero query interval count.
/// @throws std::length_error On complete node/grid container geometry.
/// @throws std::bad_alloc On owned study allocations.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::linear_interpolation_sine_study(7,300).queries.size()); // 301
/// @endcode
/// @par References
/// - NIST Digital Library of Mathematical Functions,3.3 Interpolation, https://dlmf.nist.gov/3.3; local degree1 interpolation and its remainder.
/// @see LinearInterpolator
LinearInterpolationSineStudy linear_interpolation_sine_study(
    std::size_t nodes,
    std::size_t grid_intervals = 1000);
} // namespace datamunge::algorithms
