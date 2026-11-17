#pragma once
/// @file monotone_cubic.hpp
/// @brief Prepared C1 Hermite interpolation with actual sequential slope limiting.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual mutations of one interval's shared endpoint tangents.
/// Records precede subsequent intervals, which can further decrease a shared
/// tangent. Final model tangents, rather than an earlier record, define evaluation.
/// @par Example
/// @code
/// dm::algorithms::MonotoneCubic c({0,1,2},{0,0,1});
/// std::print("{}\n",c.decisions()[0].flat); // true
/// @endcode
/// @see MonotoneCubic
struct MonotoneCubicDecision {
  std::size_t interval{0};          ///< Actual processed interval, left to right.
  double      secant{0};            ///< Interval ordinate difference divided by width.
  double      left_before{0};       ///< Shared left tangent before this interval.
  double      right_before{0};      ///< Shared right tangent before this interval.
  bool        flat{false};          ///< Whether the interval's secant is exactly zero.
  bool        left_clamped{false};  ///< Whether a wrong-sign left tangent was zeroed.
  bool        right_clamped{false}; ///< Whether a wrong-sign right tangent was zeroed.
  bool   scaled{false};  ///< Whether the nonnegative normalized tangent pair was radially scaled.
  double left_after{0};  ///< Tangent after this interval's actual limiter.
  double right_after{0}; ///< Tangent after this interval's actual limiter.
};
/// @brief Owned prepared shape-preserving cubic Hermite model with endpoint clamping.
/// Initial interior tangents average adjacent secants, endpoints use one-sided
/// secants. In left-to-right order, zero secants zero both tangents; wrong-sign
/// tangents are clamped before radial limiting to alpha^2+beta^2<=9. Later
/// updates only decrease shared tangent magnitudes. This sufficient region
/// gives interval monotonicity in exact arithmetic, including local extrema.
/// C1, generally not C2. Floating-point evaluation has roundoff. No smoothing.
/// Preparation O(n), owned storage O(n); each query O(log n). Independent const
/// queries share no mutation. Finite geometry/secants and results are required.
/// @par Example
/// @code
/// dm::algorithms::MonotoneCubic c({0,1,2},{0,0,1});
/// std::print("{} {}\n",c.evaluate(-1),c.evaluate(1.5)); // 0 0.375
/// @endcode
/// @par References
/// - F. N. Fritsch and R. E. Carlson (1980), Monotone piecewise cubic interpolation, SIAM Journal on Numerical Analysis17,238–246, doi:10.1137/0717021.
/// @see monotone_cubic_interpolate()
class MonotoneCubic {
 public:
  /// @brief Validate, own and prepare nodes, recording actual slope-limit decisions.
  /// @param xs At least two finite strictly increasing abscissae.
  /// @param ys Same-size finite ordinates, not required globally monotone.
  /// @param max_nodes Complete input-node cap, default1000000; zero disables.
  /// @throws std::invalid_argument If node shape, finiteness or order is invalid.
  /// @throws std::overflow_error If finite double widths/secants cannot be represented.
  /// @throws std::length_error If the complete node count exceeds a nonzero cap.
  /// @par Example
  /// @code
  /// dm::algorithms::MonotoneCubic c({0,1},{0,2});
  /// std::print("{}\n",c.evaluate(.5)); // 1
  /// @endcode
  /// @see MonotoneCubic
  MonotoneCubic(std::vector<double> xs, std::vector<double> ys, std::size_t max_nodes = 1000000);
  /// @brief Evaluate the prepared interpolant, clamping finite out-of-range queries.
  /// @param x Finite query; outside the node range returns its nearest endpoint ordinate.
  /// @return Cubic value, exact stored ordinate at every exact node.
  /// @throws std::invalid_argument If query is nonfinite.
  /// @throws std::overflow_error If arithmetic yields a nonfinite result.
  /// @par Example
  /// @code
  /// dm::algorithms::MonotoneCubic c({0,1},{0,2});
  /// std::print("{} {}\n",c.evaluate(.5),c.evaluate(5)); // 1 2
  /// @endcode
  /// @see MonotoneCubic
  double evaluate(double x) const;
  /// @brief Return owned abscissae as a borrowed const view.
  /// @return Model-lifetime abscissa reference.
  const std::vector<double>& xs() const noexcept { return xs_; }
  /// @brief Return owned ordinates as a borrowed const view.
  /// @return Model-lifetime ordinate reference.
  const std::vector<double>& ys() const noexcept { return ys_; }
  /// @brief Return final shared tangents after all interval limiting.
  /// @return Model-lifetime final tangent reference, in node order.
  const std::vector<double>& tangents() const noexcept { return tangents_; }
  /// @brief Return actual sequential slope-limit history.
  /// @return Model-lifetime actual decision reference, one per interval.
  const std::vector<MonotoneCubicDecision>& decisions() const noexcept { return decisions_; }

 private:
  std::vector<double>                xs_;        ///< Owned validated abscissae.
  std::vector<double>                ys_;        ///< Owned validated ordinates.
  std::vector<double>                tangents_;  ///< Final shared node tangents.
  std::vector<MonotoneCubicDecision> decisions_; ///< Actual interval limiter records.
};
/// @brief Prepare and evaluate one shape-preserving cubic query.
/// Same model/limiter as MonotoneCubic, including finite-input guards and endpoint
/// clamping. Preparation O(n), query O(log n), storage O(n). Reuse a prepared
/// model for repeated queries. The ordinary function disables the optional node
/// cap. Wrong-sign limiting and actual endpoint clamping correct legacy defects.
/// @param xs At least two finite strictly increasing abscissae.
/// @param ys Same-size finite ordinates.
/// @param x Finite query, clamped to the node range.
/// @return Prepared cubic value.
/// @throws std::invalid_argument If shape, finiteness or ordering is invalid.
/// @throws std::overflow_error If finite arithmetic geometry or result is unrepresentable.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::monotone_cubic_interpolate({0,1},{0,2},.5)); // 1
/// @endcode
/// @par References
/// - Fritsch and Carlson (1980), doi:10.1137/0717021.
/// @see MonotoneCubic
double
monotone_cubic_interpolate(const std::vector<double>& xs, const std::vector<double>& ys, double x);
} // namespace datamunge::algorithms
