#pragma once
/// @file cohen_sutherland.hpp
/// @brief Checked closed-rectangle line clipping with actual region codes and edge decisions.
#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

/// @brief Owned visible endpoint coordinates with the historical field order and types.
/// A rejected result has visible=false and four zero coordinates. A visible
/// zero-length segment is valid, including tangency to a corner. Coordinates
/// preserve input direction. No input references or shared state are retained.
/// @par Example
/// @code
/// const auto c=dm::algorithms::cohen_sutherland_clip(-1,1,3,1,0,0,2,2);
/// std::print("{} {} {}\n",c.visible,c.x0,c.x1); // true 0 2
/// @endcode
/// @see cohen_sutherland_clip(), CohenSutherlandTrace
struct ClippedSegment {
  bool visible=false; ///< Whether the closed segment intersects the closed rectangle.
  double x0=0; ///< First visible x coordinate, or zero on rejection.
  double y0=0; ///< First visible y coordinate, or zero on rejection.
  double x1=0; ///< Second visible x coordinate, or zero on rejection.
  double y1=0; ///< Second visible y coordinate, or zero on rejection.
};

/// @brief Terminal or edge-replacement decision of one Cohen-Sutherland iteration.
/// Edge priority is Top, Bottom, Right, Left, after the accept/reject tests.
/// The first outside endpoint is selected; otherwise the second is selected.
/// @par Example
/// @code
/// const auto t=dm::algorithms::cohen_sutherland_clip_trace(-1,1,3,1,0,0,2,2);
/// std::print("{}\n",t.steps[0].action==dm::algorithms::ClipAction::Left); // true
/// @endcode
/// @see CohenSutherlandStep
enum class ClipAction {
  Accept, ///< Both outcodes are zero; retain the current endpoints.
  Reject, ///< Outcodes share an outside bit; discard the complete current segment.
  Top, ///< Replace the selected outside endpoint at y=ymax.
  Bottom, ///< Replace the selected outside endpoint at y=ymin.
  Right, ///< Replace the selected outside endpoint at x=xmax.
  Left ///< Replace the selected outside endpoint at x=xmin.
};

/// @brief Owned pre-decision endpoints/codes and the actual selected edge intersection.
/// Endpoints/intersections are long double working values before final double
/// conversion. Bit values are left=1,right=2,bottom=4,top=8; boundary points
/// are inside their half-plane. Terminal records use replaced_endpoint=2,
/// new_code=0 and zero intersection coordinates, which are not geometric data.
/// @par Example
/// @code
/// const auto t=dm::algorithms::cohen_sutherland_clip_trace(-1,1,3,1,0,0,2,2);
/// std::print("{} {}\n",unsigned(t.steps[0].code0),t.steps[0].intersection_x); // 1 0
/// @endcode
/// @see CohenSutherlandTrace, ClipAction
struct CohenSutherlandStep {
  long double x0=0; ///< First current x before this decision.
  long double y0=0; ///< First current y before this decision.
  long double x1=0; ///< Second current x before this decision.
  long double y1=0; ///< Second current y before this decision.
  std::uint8_t code0=0; ///< First current four-bit outcode.
  std::uint8_t code1=0; ///< Second current four-bit outcode.
  ClipAction action=ClipAction::Accept; ///< Actual terminal test or selected edge.
  std::size_t replaced_endpoint=2; ///< Zero or one for an edge cut; two for a terminal decision.
  long double intersection_x=0; ///< Actual computed replacement x for an edge decision.
  long double intersection_y=0; ///< Actual computed replacement y for an edge decision.
  std::uint8_t new_code=0; ///< Replacement endpoint's recomputed outcode, for edge decisions only.
};

/// @brief Owned final result and at most five actual decisions from one executing clip.
/// There are at most four intersections, because each processed closed
/// half-plane contains both remaining endpoints and stays satisfied. Interpolation
/// keeps each free intersection coordinate between the current endpoint values;
/// the selected coordinate is pinned to its rectangle boundary. These are
/// floating computations, not exact predicates: near-corner decisions can differ
/// from real arithmetic. Counts exclude validation, coordinate comparisons and
/// presentation. No references, global state or replay are retained.
/// @par Example
/// @code
/// const auto t=dm::algorithms::cohen_sutherland_clip_trace(-1,1,3,1,0,0,2,2);
/// std::print("{} {} {}\n",t.intersections,t.steps.size(),t.outcode_evaluations); // 2 3 4
/// @endcode
/// @see cohen_sutherland_clip_trace()
struct CohenSutherlandTrace {
  ClippedSegment result; ///< Historical visible result, or canonical rejected zeros.
  std::vector<CohenSutherlandStep> steps; ///< Actual pre-decision records, ending in Accept or Reject.
  std::size_t intersections=0; ///< Executed edge cuts, from zero through four.
  std::size_t outcode_evaluations=0; ///< Two initial endpoint codes plus one per replacement.
  std::size_t or_tests=0; ///< Actual accept-mask tests, one per decision.
  std::size_t and_tests=0; ///< Actual reject-mask tests; skipped on acceptance.
};

/// @brief Return a validated point's four-bit code relative to a closed ordered rectangle.
/// Left=1, right=2, bottom=4, top=8; corner regions combine bits. Equality to an
/// edge is inside. Zero-width/height rectangles are valid. All six coordinates
/// are checked before classification, with no tolerance or allocation. O(1)
/// work/storage, no mutation or shared state.
/// @param x Finite point x coordinate.
/// @param y Finite point y coordinate.
/// @param xmin Finite inclusive minimum x, no greater than xmax.
/// @param ymin Finite inclusive minimum y, no greater than ymax.
/// @param xmax Finite inclusive maximum x.
/// @param ymax Finite inclusive maximum y.
/// @return Combined outside mask; zero means the point is in the closed rectangle.
/// @throws std::invalid_argument If any coordinate is nonfinite or bounds are reversed.
/// @par Example
/// @code
/// std::print("{}\n",unsigned(dm::algorithms::cohen_sutherland_outcode(-1,3,0,0,2,2))); // 9
/// @endcode
/// @par References
/// - A. Kriegl (2003), Cohen-Sutherland Line-Clipping Algorithm, University of Vienna,
///   https://www.mat.univie.ac.at/~kriegl/Skripten/CG/node31.html.
/// @see cohen_sutherland_clip_trace()
std::uint8_t cohen_sutherland_outcode(double x,double y,double xmin,double ymin,double xmax,double ymax);

/// @brief Clip a directed segment to a closed finite rectangle with checked widened arithmetic.
/// Fully validates all eight coordinates and ordered bounds before trivial
/// acceptance/rejection. Empty-area rectangles, corner tangency, axis-parallel
/// and zero-length segments are valid. Endpoint zero is replaced when outside,
/// otherwise endpoint one; edge priority is top,bottom,right,left. Free coordinates
/// use the original supporting segment and a widened line determinant before
/// division, preserving small integer corner cancellations and avoiding drift
/// from earlier rounded cuts; unrepresentable products fall back to std::lerp.
/// Current-endpoint-range projection retains already satisfied half-planes. The selected
/// edge coordinate is pinned exactly. Output converts to double only at acceptance.
/// Input direction and the historical result layout are retained. Intersections
/// can differ in last bits from the former double product-before-division;
/// widening is not an exact-real guarantee near corners or extreme scales.
/// At most four cuts/five decisions, O(1) work/storage, no allocation/shared state.
/// @param x0 Finite first endpoint x.
/// @param y0 Finite first endpoint y.
/// @param x1 Finite second endpoint x.
/// @param y1 Finite second endpoint y.
/// @param xmin Finite inclusive minimum x, no greater than xmax.
/// @param ymin Finite inclusive minimum y, no greater than ymax.
/// @param xmax Finite inclusive maximum x.
/// @param ymax Finite inclusive maximum y.
/// @return Owned visible endpoints in input direction, or false with zero coordinates.
/// @throws std::invalid_argument If any coordinate is nonfinite or bounds are reversed.
/// @throws std::overflow_error If a widened difference/ratio/conversion is nonfinite,
///   or a numerical failure violates the four-edge progress bound. Platforms
///   without a wider long double can reject extreme finite endpoint differences.
/// @par Example
/// @code
/// const auto c=dm::algorithms::cohen_sutherland_clip(2,-3,8,13,0,0,10,10);
/// std::print("{} {} {} {}\n",c.x0,c.y0,c.x1,c.y1); // 3.125 0 6.875 10
/// @endcode
/// @par References
/// - A. Kriegl (2003), Cohen-Sutherland Line-Clipping Algorithm,
///   https://www.mat.univie.ac.at/~kriegl/Skripten/CG/node31.html (outcodes and half-plane tests).
/// - C++ working draft, Linear interpolation, https://eel.is/c++draft/c.math.lerp
///   (bounded finite fallback when determinant products are unrepresentable).
/// @see cohen_sutherland_clip_trace(), cohen_sutherland_outcode()
ClippedSegment cohen_sutherland_clip(double x0,double y0,double x1,double y1,double xmin,double ymin,double xmax,double ymax);

/// @brief Record actual outcodes, selected boundaries and terminal decision without replay.
/// Uses exactly the same checked kernel, validation, priority, boundary inclusion
/// and numerical behavior as cohen_sutherland_clip(). Trace coordinates retain
/// the long-double working values, before the visible result's double conversion.
/// Vector storage owns at most five records, so work/storage are O(1). Concurrent
/// calls use independent state. See CohenSutherlandTrace for named counters.
/// @param x0 Finite first endpoint x.
/// @param y0 Finite first endpoint y.
/// @param x1 Finite second endpoint x.
/// @param y1 Finite second endpoint y.
/// @param xmin Finite inclusive minimum x, no greater than xmax.
/// @param ymin Finite inclusive minimum y, no greater than ymax.
/// @param xmax Finite inclusive maximum x.
/// @param ymax Finite inclusive maximum y.
/// @return Owned actual decisions and the same result as the output-only call.
/// @throws std::invalid_argument If any coordinate is nonfinite or bounds are reversed.
/// @throws std::overflow_error If checked working arithmetic, conversion or edge progress fails.
/// @throws std::bad_alloc If the constant-size record vector cannot be allocated.
/// @throws std::length_error If record storage exceeds the vector's capacity.
/// @par Example
/// @code
/// const auto t=dm::algorithms::cohen_sutherland_clip_trace(-1,0.5,0.5,2,0,0,1,1);
/// std::print("{} {} {}\n",t.result.visible,t.intersections,t.steps.size()); // false 1 2
/// @endcode
/// @par References
/// - A. Kriegl (2003), Cohen-Sutherland Line-Clipping Algorithm,
///   https://www.mat.univie.ac.at/~kriegl/Skripten/CG/node31.html.
/// @see cohen_sutherland_clip(), CohenSutherlandStep
CohenSutherlandTrace cohen_sutherland_clip_trace(double x0,double y0,double x1,double y1,double xmin,double ymin,double xmax,double ymax);

} // namespace datamunge::algorithms
