#pragma once
/// @file point_in_polygon.hpp
/// @brief Exact stored-coordinate even-odd ray decisions with explicit boundary classification.
#include <cstddef>
#include <datamunge/geometry/point2d.hpp>
#include <vector>
namespace datamunge::geometry {
/// @brief Boundary-aware classification of a finite closed vertex path.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::classify_point_in_polygon({0,1},{{0,0},{2,0},{2,2},{0,2}})==dm::geometry::PolygonLocation::Boundary); // true
/// @endcode
enum class PolygonLocation {
  Outside,  ///< Even parity and not on any closed edge; empty path is outside.
  Boundary, ///< Exact collinearity plus inclusive edge bounds, including a singleton vertex.
  Inside    ///< Odd parity and not on any closed edge.
};
/// @brief One actual predecessor-to-current edge test in the native ray traversal.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::point_in_polygon_trace({1,1},{{0,0},{2,0},{2,2},{0,2}}).edges[0].current); // 0
/// @endcode
struct PolygonRayEdge {
  std::size_t previous;    ///< Original previous vertex index; last vertex closes the path.
  std::size_t current;     ///< Original current vertex index.
  bool        straddles;   ///< Actual half-open y-straddle test using strict > comparisons.
  int         orientation; ///< Exact orientation(current,previous,query), in {-1,0,1}.
  bool        boundary;    ///< Query exactly on this inclusive edge of the represented coordinates.
  bool
      toggled; ///< Actual strict rightward crossing decision; horizontal/collinear edges never toggle.
  bool inside_after; ///< Actual accumulated parity immediately after this edge.
};
/// @brief Owned actual parity/boundary outcome and optional producing edge history.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::point_in_polygon_trace({1,1},{{0,0},{2,0},{2,2},{0,2}}).inside); // true
/// @endcode
struct PolygonRayTrace {
  bool inside{
      false}; ///< Historical semi-open parity result; boundary points can have either parity.
  PolygonLocation location{
      PolygonLocation::Outside};     ///< Explicit boundary classification overriding parity.
  std::vector<PolygonRayEdge> edges; ///< Actual original edge order, not replay.
};
/// @brief Evaluate the historical even-odd semi-open ray ownership rule robustly.
/// Uses exact orientation signs of stored finite binary64 coordinates instead
/// of overflowing/cancelling ray-intersection division. Strict y-straddling and
/// rightward crossing conventions are preserved; boundary ownership depends on
/// the edge, so use classify_point_in_polygon() when boundary matters. This
/// evaluates an arbitrary closed path's even-odd fill; it does not certify a
/// simple polygon. Empty is false. O(n) fixed-width predicate calls; unchanged
/// borrowed inputs, no retained references or shared mutable state.
/// @param point Complete finite query coordinates.
/// @param vertices Complete finite closed-path vertices; no repeated closing vertex is required.
/// @return Actual even-odd parity, including semi-open boundary ownership.
/// @throws std::invalid_argument If any query or vertex coordinate is nonfinite.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::point_in_polygon({1,1},{{0,0},{2,0},{2,2},{0,2}})); // true
/// @endcode
/// @par References
/// - W. R. Franklin, PNPOLY, https://wrfranklin.org/Research/Short_Notes/pnpoly.html .
///   This implementation uses exact orientation signs rather than floating intersections.
[[nodiscard]] bool point_in_polygon(const Point2D& point, const std::vector<Point2D>& vertices);
/// @brief Classify exact boundary membership separately from even-odd interior parity.
/// A singleton has only its point as boundary; a two-vertex path has only its
/// segment as boundary; an empty path is outside. No simplicity test is performed.
/// @param point Finite query point, unchanged.
/// @param vertices Complete finite closed path, unchanged.
/// @return Boundary if any inclusive represented edge contains the query; otherwise parity classification.
/// @throws std::invalid_argument If any coordinate is nonfinite.
/// @par Example
/// @code
/// const auto location=dm::geometry::classify_point_in_polygon({0,1},{{0,0},{2,0},{2,2},{0,2}});std::print("{}\n",location==dm::geometry::PolygonLocation::Boundary); // true
/// @endcode
PolygonLocation classify_point_in_polygon(
    const Point2D&              point,
    const std::vector<Point2D>& vertices);
/// @brief Capture every actual edge decision of the shared ray/boundary producer.
/// Retains n fixed edge records; no intersection reconstruction or second traversal.
/// @param point Complete finite query point.
/// @param vertices Complete finite closed-path vertices, unchanged.
/// @param max_edges Retained edge-record cap, default one million; zero disables policy.
/// @return Owned actual parity, explicit location and complete producing edge records.
/// @throws std::invalid_argument If any coordinate is nonfinite.
/// @throws std::length_error If retained history policy is exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::point_in_polygon_trace({1,1},{{0,0},{2,0},{2,2},{0,2}}).edges.size()); // 4
/// @endcode
PolygonRayTrace point_in_polygon_trace(
    const Point2D&              point,
    const std::vector<Point2D>& vertices,
    std::size_t                 max_edges = 1000000);
} // namespace datamunge::geometry
