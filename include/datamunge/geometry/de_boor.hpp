#pragma once

/// @file de_boor.hpp
/// @brief Validated planar B-spline evaluation, exact endpoint sampling and interpolation traces.

#include <datamunge/geometry/point2d.hpp>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/plot/plot.hpp>

#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::geometry {

/// @brief One initial or interpolated point in a de Boor construction triangle.
///
/// Rows at round zero copy active control points; their knot bounds and alpha are NaN.
/// Later rows record actual descending-index update order within each round.
/// @par Example
/// See DeBoorSpline::trace() for a complete construction.
/// @see DeBoorTrace
struct DeBoorStep {
    std::size_t round{0}; ///< Interpolation round, zero through degree.
    std::size_t local_index{0}; ///< Work-vector index j, round <= j <= degree.
    std::size_t control_index{0}; ///< Original index i = span-degree+j used in the knot weights.
    double left_knot{0.0}; ///< Lower knot u_i, NaN for initial points.
    double right_knot{0.0}; ///< Upper knot u_(i+degree-round+1), NaN for initial points.
    double alpha{0.0}; ///< Weight of the right parent, in [0,1]; NaN in round zero.
    Point2D point; ///< Point after this update, or an initial active control point.
};

/// @brief Owned interpolation triangle and final point from one de Boor evaluation.
///
/// A degree-p trace has (p+1)(p+2)/2 rows, including p+1 initial controls.
/// Changing fields manually can invalidate its table/plot interpretation. No input is borrowed.
/// @par Example
/// @code
/// const dm::geometry::DeBoorSpline spline(2,{0,0,0,1,1,1},{{0,0},{2,4},{4,0}});
/// const auto trace=spline.trace(.25);
/// std::print("{} {:.1f} {:.1f}\n",trace.steps.size(),trace.point.x,trace.point.y); // 6 1.0 1.5
/// @endcode
/// @par References
/// - C. de Boor (1972). On calculating with B-splines. Journal of Approximation Theory 6, 50-62.
///   https://doi.org/10.1016/0021-9045(72)90080-9
/// @see DeBoorSpline::trace(), DeBoorStep
struct DeBoorTrace {
    int degree{0}; ///< Polynomial degree of the evaluated curve.
    std::size_t span{0}; ///< Active nonempty knot span k.
    double parameter{0.0}; ///< Evaluated parameter, including either domain endpoint.
    Point2D point; ///< Final curve point after degree rounds.
    std::vector<DeBoorStep> steps; ///< Initial active controls, then actual descending-index updates.

    /// @brief Tabulates every initial point and interpolation weight for DataFrame::to_md().
    /// @return Owned numeric columns round, local, control, left_knot, right_knot, alpha, x, y.
    /// Initial bounds and weights are NaN, not missing data masks. Work/storage are O(p^2).
    /// @par Example
    /// @code
    /// const dm::geometry::DeBoorSpline spline(2,{0,0,0,1,1,1},{{0,0},{2,4},{4,0}});
    /// std::print("{}\n",spline.trace(.25).table().nrows()); // 6
    /// @endcode
    /// @see DeBoorTrace, plot()
    dstruct::DataFrame table() const;

    /// @brief Plots successive construction polygons and their surviving points.
    /// @return Owned editable RPlot; colours distinguish rounds and the final point.
    /// No equal-coordinate aspect ratio is imposed. Cost is O(p^2); legends grow with p.
    /// @par Example
    /// @code
    /// const dm::geometry::DeBoorSpline spline(2,{0,0,0,1,1,1},{{0,0},{2,4},{4,0}});
    /// auto plot=spline.trace(.25).plot();
    /// plot.size(800,500).save_svg("de_boor_triangle.svg");
    /// @endcode
    /// @see table(), DeBoorSpline::plot_curve()
    graphics::RPlot plot() const;
};

/// @brief An owned, validated planar B-spline for repeated de Boor evaluation.
///
/// Degree p is nonnegative, control count N >= p+1, and there are N+p+1 finite
/// nondecreasing knots. Knot multiplicity is at most p+1; both domain-end spans
/// must have positive width. The domain is [knots[p],knots[N]]. Interior knots use
/// the right-hand polynomial, including at a discontinuity; the final endpoint uses
/// the last span's left limit. No extrapolation, fitting, periodicity or rational weights.
/// Construction copies/moves its inputs and costs O(N+p). Const reads may run concurrently.
/// The evaluated point uses O(log N+p^2) work and O(p) scratch after validation.
/// Convex blending controls absolute propagation of control-point errors, not relative
/// error near a zero coordinate, nor sensitivity to perturbing nearly coincident knots.
/// @par Example
/// @code
/// const dm::geometry::DeBoorSpline spline(2,{0,0,0,1,1,1},{{0,0},{2,4},{4,0}});
/// const auto point=spline.evaluate(.25);
/// std::print("{:.1f} {:.1f}\n",point.x,point.y); // 1.0 1.5
/// @endcode
/// @par References
/// - C. de Boor (1972). On calculating with B-splines. Journal of Approximation Theory 6, 50-62.
///   https://doi.org/10.1016/0021-9045(72)90080-9
/// @see de_boor(), DeBoorTrace, fda::BSplineBasis
class DeBoorSpline {
public:
    /// @brief Validates and owns one planar B-spline, retaining all knot multiplicities.
    /// @param degree Nonnegative polynomial degree p, less than control.size().
    /// @param knots Finite nondecreasing N+p+1 knots; multiplicity <= p+1 and nonempty end spans.
    /// @param control N finite planar points, in basis/coefficient order.
    /// @throws std::invalid_argument For invalid degree, count, order, values, multiplicity or domain.
    /// @par Example
    /// See the class example for a quadratic Bezier represented as a clamped B-spline.
    /// @see DeBoorSpline, domain()
    DeBoorSpline(int degree, std::vector<double> knots, std::vector<Point2D> control);

    /// @brief Polynomial degree of the curve segments.
    /// @return Nonnegative degree p.
    int degree() const noexcept { return degree_; }
    /// @brief Full knot sequence, preserving repeated knots.
    /// @return Borrowed immutable knots; valid until the spline is destroyed or assigned.
    const std::vector<double>& knots() const noexcept { return knots_; }
    /// @brief Control points in their original basis order.
    /// @return Borrowed immutable control net; valid until destruction or assignment.
    const std::vector<Point2D>& control() const noexcept { return control_; }
    /// @brief Closed parameter interval covered by the nonempty end spans.
    /// @return Endpoints (knots[p],knots[N]), with first < second.
    std::pair<double,double> domain() const noexcept;

    /// @brief Evaluates a curve point by descending-index knot-weighted interpolation.
    /// @param t Finite parameter inside domain(), inclusive; interior knots select the right span.
    /// @return Curve point; a clamped spline interpolates both endpoint controls exactly.
    /// @throws std::invalid_argument For non-finite or out-of-domain t.
    /// @par Example
    /// See the class example. For clamped knots, evaluate(domain().second) needs no epsilon offset.
    /// @see trace(), sample(), de_boor()
    Point2D evaluate(double t) const;

    /// @brief Evaluates while retaining every initial point, knot weight and interpolation result.
    /// @param t Finite in-domain parameter, inclusive, with the same side convention as evaluate().
    /// @return Owned trace with the same final point as evaluate(); O(log N+p^2) work, O(p^2) storage.
    /// @throws std::invalid_argument For non-finite or out-of-domain t.
    /// @par Example
    /// See DeBoorTrace's class example and table() example.
    /// @see DeBoorTrace, evaluate()
    DeBoorTrace trace(double t) const;

    /// @brief Samples uniformly in parameter, including both exact domain endpoints.
    /// @param samples Number of points, at least two; not an arc-length spacing request.
    /// @return Owned samples in increasing parameter order; O(samples*(log N+p^2)) work.
    /// At full-multiplicity knots this discrete vector may jump; plot_curve() separates pieces.
    /// @throws std::invalid_argument For samples < 2.
    /// @par Example
    /// @code
    /// const dm::geometry::DeBoorSpline spline(1,{0,0,1,1},{{0,0},{2,4}});
    /// std::print("{}\n",spline.sample(5).back().x); // 2
    /// @endcode
    /// @see sample_table(), plot_curve(), de_boor_curve()
    std::vector<Point2D> sample(int samples) const;

    /// @brief Evaluates chosen parameters and reports their active knot spans.
    /// @param parameters Finite in-domain values in any order; empty is accepted.
    /// @return Owned numeric columns t, span, first_control, last_control, x, y.
    /// All indices are zero-based, rows preserve input order, cost is O(M*(log N+p^2)).
    /// @throws std::invalid_argument If any parameter is invalid; no partial table is returned.
    /// @par Example
    /// @code
    /// const dm::geometry::DeBoorSpline spline(1,{0,0,1,1},{{0,0},{2,4}});
    /// const auto table=spline.sample_table({0,.5,1});
    /// std::print("{}\n",table.numeric_column("x").values()[1]); // 1
    /// @endcode
    /// @see sample(), trace()
    dstruct::DataFrame sample_table(const std::vector<double>& parameters) const;

    /// @brief Plots the control polygon and sampled curve, keeping knot pieces separate.
    /// @param samples_per_span Number of points per positive-width span, at least two, default 40.
    /// @return Editable owned RPlot. A span ends at its left-limit value; a new series starts
    /// at the next knot, so discontinuities are never connected by a spurious line. Finite
    /// sampling is a visual approximation, without a geometric tolerance or equal-aspect guarantee.
    /// @throws std::invalid_argument For samples_per_span < 2.
    /// @par Example
    /// @code
    /// const dm::geometry::DeBoorSpline spline(2,{0,0,0,1,1,1},{{0,0},{2,4},{4,0}});
    /// auto plot=spline.plot_curve();
    /// plot.size(800,500).save_svg("de_boor_curve.svg");
    /// @endcode
    /// @see sample(), DeBoorTrace::plot()
    graphics::RPlot plot_curve(int samples_per_span=40) const;

private:
    int degree_; ///< Validated nonnegative segment degree.
    std::vector<double> knots_; ///< Owned nondecreasing finite knots with nonempty end spans.
    std::vector<Point2D> control_; ///< Owned finite control net, at least degree+1 points.

    /// @brief Validates t and locates its right-hand nonempty span, except at the final endpoint.
    /// @param t Finite in-domain parameter.
    /// @return Span index k in [degree,N-1], with knots[k] < knots[k+1].
    /// @throws std::invalid_argument For non-finite or out-of-domain t.
    std::size_t find_span(double t) const;

    /// @brief Runs the shared interpolation kernel on a preselected valid span.
    /// @param t Finite parameter within span, inclusive on its right for plotting left limits.
    /// @param span Nonempty span index.
    /// @param trace Optional output receiving all steps; null avoids trace allocation.
    /// @return Final planar curve point; O(p^2) work, O(p) scratch excluding trace.
    Point2D evaluate_span(double t,std::size_t span,DeBoorTrace* trace) const;
};

/// @brief Evaluates a planar B-spline once with full input validation.
///
/// This compatibility entry point constructs a DeBoorSpline on every call, costing
/// O(N+p^2) work and O(N+p) storage. Retain a DeBoorSpline for repeated evaluation.
/// @param degree Nonnegative polynomial degree p, less than control.size().
/// @param knots Finite nondecreasing N+p+1 knots satisfying DeBoorSpline's domain/multiplicity rules.
/// @param control N finite control points in coefficient order.
/// @param t Finite in-domain parameter, including either exact endpoint.
/// @return Curve point; interior knots select the right polynomial and the last endpoint the left limit.
/// @throws std::invalid_argument For invalid spline data or parameter.
/// @par Example
/// @code
/// const auto point=dm::geometry::de_boor(1,{0,0,1,1},{{0,0},{2,4}},.25);
/// std::print("{} {}\n",point.x,point.y); // 0.5 1
/// @endcode
/// @par References
/// - C. de Boor (1972). On calculating with B-splines. https://doi.org/10.1016/0021-9045(72)90080-9
/// @see DeBoorSpline, de_boor_curve()
inline Point2D de_boor(int degree,const std::vector<double>& knots,
                       const std::vector<Point2D>& control,double t) {
    return DeBoorSpline(degree,knots,control).evaluate(t);
}

/// @brief Samples a validated B-spline at uniformly spaced parameters including exact endpoints.
/// @param degree Nonnegative polynomial degree p.
/// @param knots Finite nondecreasing N+p+1 knots satisfying DeBoorSpline's validation rules.
/// @param control N finite control points, N>=p+1.
/// @param samples Number of samples, at least two.
/// @return Owned point vector; O(N+samples*(log N+p^2)) work and O(N+p+samples) storage.
/// @throws std::invalid_argument For invalid spline data or samples < 2.
/// @par Example
/// @code
/// const auto points=dm::geometry::de_boor_curve(1,{0,0,1e-15,1e-15},{{0,0},{2,4}},3);
/// std::print("{} {}\n",points.front().x,points.back().x); // 0 2
/// @endcode
/// @see DeBoorSpline::sample(), de_boor()
inline std::vector<Point2D> de_boor_curve(int degree,const std::vector<double>& knots,
                                       const std::vector<Point2D>& control,int samples) {
    return DeBoorSpline(degree,knots,control).sample(samples);
}

} // namespace datamunge::geometry
