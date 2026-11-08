#pragma once
/// @file
/// @brief Planar Bezier evaluation, actual interpolation triangles, subdivision and bounded sampling.
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {
/// @brief Owned two-double planar point, retaining the original interpolation API layout.
/// Coordinates are unvalidated plain values; curve functions require finite x/y.
/// Default construction gives (0,0). No ownership outside these two scalar fields.
/// @par Example
/// @code
/// const dm::algorithms::PlanarPoint p{2,3};std::print("{}\n",p); // PlanarPoint(2, 3)
/// @endcode
/// @see de_casteljau()
struct PlanarPoint {
  double x{0.0}; ///< Horizontal coordinate, default zero.
  double y{0.0}; ///< Vertical coordinate, default zero.
};

/// @brief Checked complete interpolation-triangle geometry for m control points.
/// Empty input has zero counts. For m>0, there are m levels, m(m+1)/2
/// stored points including the controls, m(m-1)/2 point blends and twice
/// that many coordinate blends. These are potential exact loop counts, not
/// elapsed time or counts of individual machine arithmetic instructions.
/// @par Example
/// @code
/// const auto s=dm::algorithms::de_casteljau_shape(4);
/// std::print("{} {} {}\n",s.stored_points,s.point_blends,s.coordinate_blends); // 10 6 12
/// @endcode
/// @see de_casteljau_shape(), DeCasteljauTrace
struct DeCasteljauShape {
  std::size_t control_points{0}; ///< Number m of input points; nonempty degree is m-1.
  std::size_t stored_points{0}; ///< All complete rows, including initial controls.
  std::size_t point_blends{0}; ///< m(m-1)/2 two-coordinate interpolation updates.
  std::size_t coordinate_blends{0}; ///< Twice point_blends; no instruction/timing claim.
};

/// @brief Owned actual interpolation levels and final value from one kernel execution.
/// levels[0] copies the controls; subsequent rows contain actual increasing-index
/// updates, with sizes m-1 down to one. Empty input retains the legacy zero-point
/// sentinel and has no levels. point_blends increments inside the actual loop,
/// excluding copies, validation and allocation. No triangle is reconstructed by
/// replay. Finite t outside [0,1] extrapolates and has no convex-hull guarantee.
/// Fields own their values; public edits are unchecked and calls share no state.
/// @par Example
/// @code
/// const auto a=dm::algorithms::de_casteljau_trace({{0,0},{1,2},{3,3},{4,0}},.5);
/// std::print("{} {} {}\n",a.point,a.levels.size(),a.point_blends); // PlanarPoint(2, 1.875) 4 6
/// @endcode
/// @see de_casteljau_trace(), de_casteljau()
struct DeCasteljauTrace {
  double parameter{0}; ///< Original finite parameter, possibly outside the unit interval.
  PlanarPoint point; ///< Actual final value, or (0,0) for empty controls.
  DeCasteljauShape shape; ///< Checked full triangle geometry.
  std::vector<std::vector<PlanarPoint>> levels; ///< All actual owned rows, including initial controls.
  std::size_t point_blends{0}; ///< Actual loop updates, equal to shape.point_blends on success.
};

/// @brief Owned degree-preserving left/right Bezier control nets from one triangle.
/// At split s in [0,1], left(u)=original(s*u) and
/// right(u)=original(s+(1-s)*u) in exact arithmetic. Both nets have the
/// original control count and share a copied meeting point; no views remain.
/// At s=0 the left net is constant; at s=1 the right is constant. Degree-zero
/// input gives one identical point in each net. No tessellation or tolerance-
/// based adaptive stopping is performed. Floating identities hold approximately.
/// @par Example
/// @code
/// const auto s=dm::algorithms::de_casteljau_subdivide({{0,0},{2,4},{4,0}},.5);
/// std::print("{} {} {}\n",s.left.back(),s.right.front(),s.point_blends); // PlanarPoint(2, 2) PlanarPoint(2, 2) 3
/// @endcode
/// @see de_casteljau_subdivide(), de_casteljau_sample()
struct BezierSubdivision {
  double parameter{0}; ///< Finite original split parameter in [0,1].
  std::vector<PlanarPoint> left; ///< First point of every row, in increasing row order.
  std::vector<PlanarPoint> right; ///< Last point of every row, in decreasing row order.
  std::size_t point_blends{0}; ///< Actual updates needed to form the two diagonals.
};

/// @brief Count a complete de Casteljau triangle without allocating or evaluating it.
/// Products are cancelled before multiplication. O(1) time/storage, no mutation
/// or shared state. Zero controls has all zero counts; one has one stored point.
/// @param control_points Number m of controls, any representable size_t.
/// @return Exact geometry, or an exception before arithmetic wraparound.
/// @throws std::length_error Any stored-point or blend count exceeds size_t.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::de_casteljau_shape(3).point_blends); // 3
/// @endcode
/// @see de_casteljau_trace()
DeCasteljauShape de_casteljau_shape(std::size_t control_points);

/// @brief Evaluate a planar Bezier by the original repeated weighted-blend recurrence.
/// Uses (1-t)*left+t*right in increasing index order on a copied work vector.
/// Preserves the original name/signature, PlanarPoint layout, interpolation.hpp
/// include and empty-control result (0,0). A singleton is constant. All controls
/// and t must be finite, including the empty-input parameter. Finite parameters
/// outside [0,1] retain polynomial extrapolation, subject to finite intermediate
/// arithmetic. In exact arithmetic, t=0/1 interpolates endpoints and [0,1]
/// yields convex combinations; ordinary rounding is not an exact hull certificate
/// or a universal small relative-error guarantee near cancellation. Nonfinite
/// intermediates reject even when another algebraic arrangement could be finite.
/// O(m^2+1) time and O(m) owned temporary storage; input unchanged, no shared state.
/// @param control Borrowed finite planar controls; empty retains the zero sentinel.
/// @param t Finite interpolation or extrapolation parameter.
/// @return Actual final point, or (0,0) for empty controls.
/// @throws std::invalid_argument Any nonfinite coordinate or parameter.
/// @throws std::overflow_error A weighted coordinate blend is nonfinite.
/// @throws std::length_error Temporary vector capacity is exceeded.
/// @throws std::bad_alloc Allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::de_casteljau({{0,0},{1,2},{3,3},{4,0}},.5)); // PlanarPoint(2, 1.875)
/// @endcode
/// @par References
/// - P. de Casteljau (1985), Formes a poles, Mathematiques et CAO vol. 2, Hermes, Paris. Bibliographic record: https://bibliotheque.map.cnrs.fr/index.php?id=86&lvl=notice_display .
/// - G. Farin (2002), Curves and Surfaces for CAGD, 5th ed., ISBN 1-55860-737-4, de Casteljau/Bernstein/subdivision topics; https://www.farinhansford.com/books/cagd/toc.html . This implementation retains the original double weighted-sum ordering.
/// @see de_casteljau_trace(), de_casteljau_subdivide()
PlanarPoint de_casteljau(const std::vector<PlanarPoint>& control,double t);

/// @brief Capture complete actual rows from the same evaluator as de_casteljau().
/// No replay or independent reconstruction. Empty input has zero stored points;
/// its sentinel does not define a Bezier curve. The complete triangle cap includes
/// initial controls and is checked after full finite validation, before copying
/// or allocating rows. Zero disables only the optional cap; count/capacity checks
/// remain. O(m^2+1) time and O(m^2) owned history, with O(m) temporary work.
/// @param control Borrowed finite controls, unchanged; empty allowed as legacy sentinel.
/// @param t Finite parameter; outside [0,1] extrapolates.
/// @param max_points Maximum complete stored triangle points, default 1000000; zero disables.
/// @return Owned actual triangle, final point, checked geometry and measured blend count.
/// @throws std::invalid_argument Any nonfinite input coordinate or parameter.
/// @throws std::length_error Triangle arithmetic, complete cap or container capacity exceeded.
/// @throws std::overflow_error Any weighted blend is nonfinite.
/// @throws std::bad_alloc Allocation fails.
/// @par Example
/// @code
/// const auto a=dm::algorithms::de_casteljau_trace({{0,0},{2,4},{4,0}},.25,6);
/// std::print("{} {}\n",a.levels[1][0],a.point); // PlanarPoint(0.5, 1) PlanarPoint(1, 1.5)
/// @endcode
/// @par References
/// - P. de Casteljau (1985), Formes a poles; G. Farin (2002), Curves and Surfaces for CAGD, https://www.farinhansford.com/books/cagd/toc.html .
/// @see de_casteljau(), DeCasteljauTrace
DeCasteljauTrace de_casteljau_trace(const std::vector<PlanarPoint>& control,double t,
                                  std::size_t max_points=1000000);

/// @brief Extract both exact-subcurve control diagonals during a single evaluation.
/// Only O(m) work/output storage is used; no full triangle is retained. Left
/// diagonal runs from the original first control to B(t); reversed right diagonal
/// runs from that same computed B(t) to the original last control. The exact
/// reparameterization identities are documented on BezierSubdivision. Validation
/// precedes output-count caps, and the inclusive count is 2m including both copies
/// of the meeting point. At most m(m-1)/2 actual point blends; no curve replay.
/// @param control At least one finite planar control, unchanged.
/// @param t Finite split parameter in [0,1], including degenerate endpoint splits.
/// @param max_points Maximum combined left/right control points; default 1000000, zero disables.
/// @return Two owned control nets and actual point-blend count.
/// @throws std::invalid_argument Empty controls, nonfinite input or t outside [0,1].
/// @throws std::length_error Geometry arithmetic, output cap or vector capacity exceeded.
/// @throws std::overflow_error Any weighted blend is nonfinite.
/// @throws std::bad_alloc Allocation fails.
/// @par Example
/// @code
/// const auto s=dm::algorithms::de_casteljau_subdivide({{0,0},{2,4},{4,0}},.5,6);
/// std::print("{} {}\n",s.left.size(),s.right.size()); // 3 3
/// @endcode
/// @par References
/// - P. de Casteljau (1985), Formes a poles; subdivision as described in https://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node13.html .
/// @see BezierSubdivision, de_casteljau_trace()
BezierSubdivision de_casteljau_subdivide(const std::vector<PlanarPoint>& control,double t,
                                         std::size_t max_points=1000000);

/// @brief Sample uniform parameter intervals with both requested endpoints included.
/// Returns segments+1 points. Parameters use std::lerp(first,last,i/segments),
/// with first and last assigned directly; no repeatedly added parameter step.
/// Equal or descending endpoints are valid, as is finite extrapolation. Distinct
/// grid parameters may round to duplicates in tiny intervals. All controls and
/// parameter bounds are validated once, before complete output/work caps and
/// allocation. Empty controls return the legacy zero sentinel at every parameter;
/// this is not a curve. A singleton is constant. Output count and prospective
/// total point blends (segments+1)*m(m-1)/2 use checked arithmetic. Zero disables
/// either optional cap independently, not arithmetic/capacity checks. O(m+k(m^2+1))
/// work for k=segments+1, O(k+m) output/temporary storage, no shared/input mutation.
/// @param control Borrowed finite planar controls, unchanged.
/// @param segments Positive number of uniform intervals, default 100.
/// @param first Finite initial parameter, default zero.
/// @param last Finite final parameter, default one.
/// @param max_points Maximum complete output points, default 1000000; zero disables.
/// @param max_blends Maximum prospective total point blends, default 100000000; zero disables.
/// @return Owned complete point sequence with values at first and last endpoints.
/// @throws std::invalid_argument Zero segments or any nonfinite input.
/// @throws std::length_error Count/work arithmetic, complete cap or container capacity exceeded.
/// @throws std::overflow_error Any evaluated weighted blend is nonfinite.
/// @throws std::bad_alloc Allocation fails.
/// @par Example
/// @code
/// const auto p=dm::algorithms::de_casteljau_sample({{0,0},{2,4},{4,0}},4);
/// std::print("{} {} {}\n",p.size(),p.front(),p.back()); // 5 PlanarPoint(0, 0) PlanarPoint(4, 0)
/// @endcode
/// @see de_casteljau(), de_casteljau_subdivide()
std::vector<PlanarPoint> de_casteljau_sample(const std::vector<PlanarPoint>& control,
                                           std::size_t segments=100,double first=0,double last=1,
                                           std::size_t max_points=1000000,std::size_t max_blends=100000000);
} // namespace datamunge::algorithms
