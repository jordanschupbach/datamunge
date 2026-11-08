#pragma once
/// @file cubic_spline.hpp
/// @brief Checked natural cubic splines, prepared derivative evaluation and sampled error diagnostics.
#include <cstddef>
#include <functional>
#include <vector>

namespace datamunge::algorithms {
/// @brief Owned local-power coefficients S_i(x)=a_i+t*(b_i+t*(c_i+t*d_i)), t=x-xs[i].
/// Preserves the original five-vector layout. For n>=2, xs contains n finite
/// increasing knots and each coefficient vector has n-1 finite entries. Empty
/// models have all vectors empty; singleton models have one xs/a entry and no
/// b/c/d entries. Public mutation is permitted; evaluators validate the shape.
/// A manually supplied model need not be natural or C2 merely because it is
/// structurally valid. Coefficient units are y, y/x, y/x² and y/x³.
/// @par Example
/// @code
/// const auto s=dm::algorithms::natural_cubic_spline({0,1,2},{0,1,0});
/// std::print("{} {}\n",s.b,s.d); // [1.5, 0] [-0.5, 0.5]
/// @endcode
/// @par References
/// - C. de Boor, A Practical Guide to Splines (2001), https://link.springer.com/book/9780387953663
/// @see natural_cubic_spline(), PreparedCubicSpline
struct CubicSpline {
  std::vector<double> xs; ///< Owned knot coordinates, length n.
  std::vector<double> a; ///< Local constant coefficients; n-1 entries, or one for a singleton.
  std::vector<double> b; ///< Local linear coefficients, n-1 entries for n>=2.
  std::vector<double> c; ///< Local quadratic coefficients, half the left-knot second derivative.
  std::vector<double> d; ///< Local cubic coefficients, one per interval.
};

/// @brief Complete owned moments and actual Thomas-system records from one natural fit.
/// For k=max(n-2,0) interior unknowns, system vectors have length k. lower[0]
/// and upper[k-1] are structural zeros; pivots/reduced_rhs are the actual
/// post-elimination arrays, and multipliers[0]=0. Moments include both zero
/// natural endpoints; degenerate empty/singleton fits use zero/one zero moments.
/// All arrays own storage, with no prefix reruns or shared mutable state.
/// Counts are logical loop iterations, not timings or all memory operations.
/// @par Example
/// @code
/// const auto t=dm::algorithms::natural_cubic_spline_trace({0,1,2},{0,1,0});
/// std::print("{} {} {}\n",t.diagonal,t.rhs,t.moments); // [4] [-12] [0, -3, 0]
/// @endcode
/// @par References
/// - de Boor, A Practical Guide to Splines (2001), https://link.springer.com/book/9780387953663
/// @see natural_cubic_spline_trace(), CubicSpline
struct NaturalCubicSplineTrace {
  CubicSpline spline; ///< Owned final local-power model from this executing fit.
  std::vector<double> widths; ///< Actual adjacent knot differences, length n-1 for n>=2.
  std::vector<double> secants; ///< Actual adjacent ordinate differences divided by widths.
  std::vector<double> moments; ///< Actual solved second derivatives, length n.
  std::vector<double> lower; ///< Original interior-system lower diagonal, first entry zero.
  std::vector<double> diagonal; ///< Original interior-system diagonal before elimination.
  std::vector<double> upper; ///< Original interior-system upper diagonal, last entry zero.
  std::vector<double> rhs; ///< Original interior-system right-hand side.
  std::vector<double> multipliers; ///< Actual forward-elimination multipliers, first entry zero.
  std::vector<double> pivots; ///< Actual eliminated diagonal used in back substitution.
  std::vector<double> reduced_rhs; ///< Actual eliminated right-hand side.
  std::size_t eliminations=0; ///< Actual k-1 forward updates when k>0, otherwise zero.
  std::size_t back_substitutions=0; ///< Actual k interior unknowns recovered.
  std::size_t coefficient_intervals=0; ///< Actual n-1 intervals assembled when n>=2.
};

/// @brief Fit a natural C2 cubic interpolant using a checked linear-time moment system.
/// Finite strictly increasing knots and finite equally sized ordinates are
/// validated before allocation. Empty input yields the empty zero evaluator;
/// a singleton yields a constant, and two knots yield a line. For n>=2 the
/// ideal interpolant has zero endpoint second derivatives, not zero endpoint
/// slopes or globally linear extrapolation. It can overshoot monotone data.
/// O(n) time/storage, owned result, no input mutation or shared state.
/// @param xs Finite strictly increasing coordinates; empty/singleton conventions above.
/// @param ys Finite ordinates, exactly the same length as xs.
/// @return Owned local-power coefficients with the original CubicSpline layout.
/// @throws std::invalid_argument For nonfinite, unordered/duplicate knots or mismatched/nonfinite ordinates.
/// @throws std::overflow_error For nonrepresentable widths, secants, system arithmetic or coefficients.
/// @throws std::length_error If container capacities are exceeded.
/// @throws std::bad_alloc If working/result storage cannot be allocated.
/// @par Example
/// @code
/// const auto s=dm::algorithms::natural_cubic_spline({0,1,2},{0,1,0});
/// std::print("{}\n",dm::algorithms::cubic_spline_eval(s,.5)); // 0.6875
/// @endcode
/// @par References
/// - de Boor, A Practical Guide to Splines (2001), https://link.springer.com/book/9780387953663
/// - SciPy CubicSpline(bc_type="natural"), https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html
/// @see natural_cubic_spline_trace(), PreparedCubicSpline
CubicSpline natural_cubic_spline(const std::vector<double>& xs,const std::vector<double>& ys);

/// @brief Capture complete actual widths, secants, moments and elimination arrays from the fit.
/// Same canonical kernel and coefficients as natural_cubic_spline(). Full
/// knot/ordinate validation precedes the optional complete knot cap and all
/// size-dependent allocation. O(n) time/storage; results never truncate.
/// @param xs Finite strictly increasing coordinates, including allowed empty/singleton input.
/// @param ys Finite equally sized ordinates.
/// @param max_knots Complete knot cap, default 100000; zero disables only this optional cap.
/// @return Owned complete model, actual moment system and actual final factors/work.
/// @throws std::invalid_argument For invalid knots/ordinates, even beyond an exceeded cap.
/// @throws std::length_error If the complete cap or container limits are exceeded.
/// @throws std::overflow_error If fit arithmetic cannot be represented finitely.
/// @throws std::bad_alloc If owned/working storage cannot be allocated.
/// @par Example
/// @code
/// const auto t=dm::algorithms::natural_cubic_spline_trace({0,1,2},{0,1,0},3);
/// std::print("{} {}\n",t.pivots,t.back_substitutions); // [4] 1
/// @endcode
/// @par References
/// - de Boor (2001), https://link.springer.com/book/9780387953663
/// @see NaturalCubicSplineTrace, natural_cubic_spline()
NaturalCubicSplineTrace natural_cubic_spline_trace(const std::vector<double>& xs,const std::vector<double>& ys,std::size_t max_knots=100'000);

/// @brief Evaluate a mutable public spline record after complete structural validation.
/// Retains the original signature, zero/constant conventions and endpoint-piece
/// extrapolation: the interval index is clamped, the query itself is not.
/// Interior knots choose the right interval; the final knot uses the last.
/// Validation is O(n), then lookup O(log n), with O(1) auxiliary storage.
/// For repeated O(log n) evaluations, validate/copy once into PreparedCubicSpline.
/// A structurally valid manually supplied record is not a C2/natural certificate.
/// @param spline Read-only finite record with the documented CubicSpline shape.
/// @param x Finite query, inside or outside the knot interval.
/// @return Rounded value, zero for an empty model or the singleton constant.
/// @throws std::invalid_argument For nonfinite queries, malformed shapes, knots or coefficients.
/// @throws std::overflow_error For nonfinite evaluation offsets/arithmetic.
/// @par Example
/// @code
/// const auto s=dm::algorithms::natural_cubic_spline({0,1},{1,3});
/// std::print("{}\n",dm::algorithms::cubic_spline_eval(s,2)); // 5
/// @endcode
/// @par References
/// - SciPy CubicSpline extrapolation, https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html
/// @see PreparedCubicSpline
double cubic_spline_eval(const CubicSpline& spline,double x);

/// @brief Owned validated local-power spline for repeated value/derivative evaluation.
/// Comparable to a scalar SciPy PPoly with endpoint-piece extrapolation.
/// Construction checks/copies a public record once; evaluations never re-scan
/// all coefficients. No references to the input survive. Empty/singleton models
/// use the same zero/constant conventions as cubic_spline_eval(). Const methods
/// are safe for independent concurrent reads. Finite arithmetic is checked,
/// without a numerical error bound or a shape-preservation guarantee.
/// @par Example
/// @code
/// const dm::algorithms::PreparedCubicSpline p(dm::algorithms::natural_cubic_spline({0,1,2},{0,1,0}));
/// std::print("{} {}\n",p.evaluate(.5),p.evaluate(1,2)); // 0.6875 -3
/// @endcode
/// @par References
/// - SciPy PPoly, https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PPoly.html
/// @see CubicSpline, natural_cubic_spline()
class PreparedCubicSpline {
 public:
  /// @brief Own and validate a local-power spline snapshot once in O(n) time/storage.
  /// Does not certify natural boundary conditions or continuity of arbitrary coefficients.
  /// @param spline Record copied from lvalues or moved from rvalues; no input views retained.
  /// @throws std::invalid_argument For invalid shapes, nonfinite values or knot order.
  /// @throws std::length_error If copied container capacity is exceeded.
  /// @throws std::bad_alloc If copied storage cannot be allocated.
  /// @par Example
  /// @code
  /// const dm::algorithms::PreparedCubicSpline p(dm::algorithms::natural_cubic_spline({0,1},{0,1}));
  /// std::print("{}\n",p.evaluate(.25)); // 0.25
  /// @endcode
  /// @see coefficients(), evaluate()
  explicit PreparedCubicSpline(CubicSpline spline);

  /// @brief Evaluate a value or derivative of order zero through three in O(log n).
  /// Selects the right piece at interior knots and extends endpoint cubics
  /// outside the domain. The third derivative can jump at a knot. Order zero
  /// is the value; singleton/empty derivatives of positive order are zero.
  /// @param x Finite query coordinate, including extrapolation.
  /// @param derivative Derivative order in [0,3], default zero.
  /// @return Rounded value/derivative in y/x^derivative units; no state is mutated.
  /// @throws std::invalid_argument For a nonfinite query or unsupported order.
  /// @throws std::overflow_error If offset or polynomial arithmetic becomes nonfinite.
  /// @par Example
  /// @code
  /// const dm::algorithms::PreparedCubicSpline p(dm::algorithms::natural_cubic_spline({0,1,2},{0,1,0}));
  /// std::print("{} {}\n",p.evaluate(.5,1),p.evaluate(1,3)); // 1.125 3
  /// @endcode
  /// @par References
  /// - SciPy PPoly derivative evaluation, https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PPoly.html
  /// @see evaluate(const std::vector<double>&, unsigned), coefficients()
  double evaluate(double x,unsigned derivative=0) const;

  /// @brief Evaluate q finite queries in input order, with one shared validated model.
  /// All queries/order validate before output allocation. O(q log n) time and
  /// O(q) owned output; an empty query vector yields empty, even for empty models.
  /// @param queries Finite coordinates, in any order, including repeats and extrapolation.
  /// @param derivative Derivative order in [0,3], default zero.
  /// @return Owned corresponding values/derivatives, without retaining input views.
  /// @throws std::invalid_argument For any nonfinite query or unsupported order, even when empty.
  /// @throws std::overflow_error If any evaluation arithmetic becomes nonfinite.
  /// @throws std::length_error If output capacity is exceeded.
  /// @throws std::bad_alloc If output storage cannot be allocated.
  /// @par Example
  /// @code
  /// const dm::algorithms::PreparedCubicSpline p(dm::algorithms::natural_cubic_spline({0,1},{0,1}));
  /// std::print("{}\n",p.evaluate(std::vector<double>{2,.5,0})); // [2, 0.5, 0]
  /// @endcode
  /// @see evaluate(double, unsigned)
  std::vector<double> evaluate(const std::vector<double>& queries,unsigned derivative=0) const;

  /// @brief Borrow the owned validated record for inspection, without mutable access.
  /// @return Const reference valid until this object is destroyed or assigned; do not mutate it.
  const CubicSpline& coefficients() const noexcept { return model_; }
 private:
  CubicSpline model_; ///< Owned validated model, inaccessible to ordinary caller mutation.
};

/// @brief Owned finite-query error census for two scalar functions, with sampled extrema only.
/// Predictions and references are evaluated once each per query, in that order.
/// Empty input has has_samples=false and zero max/RMS/index placeholders.
/// Worst index is the first maximum absolute error, including first-zero ties.
/// A grid maximum is not a certified supremum between queries. RMS uses scaled
/// differences to avoid squaring large errors directly. All arrays are owned.
/// @par Example
/// @code
/// const auto e=dm::algorithms::interpolation_error_scan([](double x){return x;},[](double){return 0.;},{-2,0,2});
/// std::print("{} {}\n",e.max_abs_error,e.worst_index); // 2 0
/// @endcode
/// @see interpolation_error_scan()
struct InterpolationErrorScan {
  std::vector<double> queries; ///< Owned finite coordinates in supplied order.
  std::vector<double> predictions; ///< Actual finite model evaluations.
  std::vector<double> references; ///< Actual finite reference evaluations.
  std::vector<double> absolute_errors; ///< Finite absolute differences, in reference/model units.
  bool has_samples=false; ///< Whether sampled extrema and RMS refer to any queries.
  double max_abs_error=0; ///< Largest sampled absolute error, zero placeholder when empty.
  double rms_error=0; ///< Rounded sampled RMS difference, zero placeholder when empty.
  std::size_t worst_index=0; ///< First maximum-error query index, or unused zero for empty input.
};

/// @brief Evaluate and summarize a complete finite-query interpolation error census.
/// Checks callback presence and all query finiteness before the optional cap,
/// allocation or callback invocation. Then calls model and reference exactly
/// once per query in order; callback exceptions propagate unchanged. O(q) work
/// plus callback costs and O(q) owned storage, no input mutation/shared state.
/// @param model Nonempty model callback yielding finite scalar predictions.
/// @param reference Nonempty reference callback yielding finite scalar values.
/// @param queries Finite coordinates in any order; repeats/empty input are allowed.
/// @param max_queries Complete query cap, default 1000000; zero disables only this optional cap.
/// @return Owned actual evaluations, errors, first sampled maximum and scaled RMS.
/// @throws std::invalid_argument For empty callbacks, nonfinite queries or nonfinite callback values.
/// @throws std::length_error If the complete query cap or container capacities are exceeded.
/// @throws std::overflow_error If a finite difference or RMS cannot fit double storage.
/// @throws std::bad_alloc If owned storage cannot be allocated.
/// @par Example
/// @code
/// const auto e=dm::algorithms::interpolation_error_scan([](double x){return x;},[](double){return 0.;},{1,1});
/// std::print("{} {}\n",e.max_abs_error,e.rms_error); // 1 1
/// @endcode
/// @see InterpolationErrorScan, PreparedCubicSpline
InterpolationErrorScan interpolation_error_scan(const std::function<double(double)>& model,const std::function<double(double)>& reference,const std::vector<double>& queries,std::size_t max_queries=1'000'000);
} // namespace datamunge::algorithms
