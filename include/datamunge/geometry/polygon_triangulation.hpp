#pragma once
/// @file polygon_triangulation.hpp
/// @brief Checked simple-polygon ear clipping with exact stored-coordinate predicates.
#include <cstddef>
#include <datamunge/geometry/delaunay.hpp>
#include <datamunge/geometry/point2d.hpp>
#include <datamunge/geometry/polygon.hpp>
#include <vector>
namespace datamunge::geometry {
/// @brief Legacy floating helper retained for source compatibility; the producer uses exact predicates.
namespace detail {
/// @brief Legacy unvalidated double-sign triangle containment primitive; robust triangulation uses exact signs.
/// @param p Query point.
/// @param a First triangle vertex.
/// @param b Second triangle vertex.
/// @param c Third triangle vertex.
/// @return Original rounded-sign closed-triangle test, without finite or degeneracy guarantees.
inline bool
point_in_triangle(const Point2D& p, const Point2D& a, const Point2D& b, const Point2D& c) {
  const double d1      = cross(a, b, p);
  const double d2      = cross(b, c, p);
  const double d3      = cross(c, a, p);
  const bool   has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
  const bool   has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);
  return !(has_neg && has_pos);
}
} // namespace detail

/// @brief Actual candidate-ear decision at one stage of the remaining ring.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::triangulate_polygon_trace({{0,0},{2,0},{2,2},{0,2}}).candidates[0].clipped); // true
/// @endcode
struct PolygonEarCandidate {
  std::size_t ring_size;   ///< Remaining vertices before this candidate.
  Triangle    triangle;    ///< Original previous/current/next vertex ids.
  int         orientation; ///< Exact local orientation in normalized CCW ring.
  std::size_t
      containment_tests; ///< Actual other-vertex containment queries before the first blocker.
  bool
      has_blocker; ///< True when another remaining vertex lies on or inside the candidate triangle.
  std::size_t blocker; ///< First actual blocking original id, meaningful only when has_blocker.
  bool        clipped; ///< True when this actual candidate is removed and emitted.
};
/// @brief One actual emitted ear and its resulting remaining original ids.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::triangulate_polygon_trace({{0,0},{2,0},{2,2},{0,2}}).clips[0].remaining.size()); // 3
/// @endcode
struct PolygonEarClip {
  Triangle                 triangle;  ///< Actual emitted original-index triangle.
  std::vector<std::size_t> remaining; ///< Actual ring after erasing that ear's center vertex.
};
/// @brief Predicate-work and retained-history policies, independently disabled by zero.
/// @par Example
/// @code
/// const dm::geometry::PolygonTriangulationOptions o;std::print("{}\n",o.max_predicates); // 1000000
/// @endcode
struct PolygonTriangulationOptions {
  std::size_t max_predicates{
      1000000}; ///< Actual exact orientation-call cap, including simplicity validation.
  std::size_t max_trace_entries{
      1000000}; ///< Initial ids, candidate/clip records and post-clip remaining ids; excludes final triangles.
};
/// @brief Owned triangulation and selected actual predicate/candidate counts.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::triangulate_polygon_run({{0,0},{2,0},{2,2},{0,2}}).triangles.size()); // 2
/// @endcode
struct PolygonTriangulationRun {
  std::vector<Triangle>
              triangles;       ///< Original vertex ids; n-2 nondegenerate triangles for valid n>=3.
  bool        reversed{false}; ///< True if an n>3 clockwise ring was actually reversed.
  std::size_t orientations{0}; ///< Actual exact orientation calls, including input validation.
  std::size_t candidates{0};   ///< Actual candidate centers inspected during clipping.
  std::size_t containment_tests{0}; ///< Actual other-point closed-triangle queries.
};
/// @brief Owned actual ear decisions and post-erasure ring histories.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::triangulate_polygon_trace({{0,0},{2,0},{2,2},{0,2}}).initial_ring.size()); // 4
/// @endcode
struct PolygonTriangulationTrace {
  PolygonTriangulationRun run; ///< Output and counts from the producing run.
  std::vector<std::size_t>
      initial_ring; ///< Actual initial original-index ring after optional reversal.
  std::vector<PolygonEarCandidate>
      candidates; ///< Actual tests, including rejected reflex/blocked vertices.
  std::vector<PolygonEarClip>
      clips; ///< Actual emitted ears and post-erasure rings; final triangle is in run.triangles.
};
/// @brief Triangulate a finite simple polygon using exact orientation-based ear decisions.
/// Rejects duplicate vertices, zero area, overlapping adjacent edges and every
/// nonadjacent edge intersection/touch. A repeated closing vertex is a duplicate.
/// Collinear forward boundary vertices are allowed. Uses the exact turn at the
/// lexicographically least vertex for simple-polygon winding, avoiding area
/// cancellation. O(n^2) validation and O(n^3) worst-case ear search, O(n) working
/// storage. n<3 returns empty after finite validation. For n=3 the historical
/// vertex order is retained; larger clockwise rings are reversed internally.
/// Input is borrowed unchanged; no references or mutable global state survive.
/// @param polygon Finite simple boundary in either winding, without duplicate closing vertex.
/// @return Owned original-index triangles; valid n>=3 produces exactly n-2.
/// @throws std::invalid_argument If coordinates or polygon geometry are invalid.
/// @throws std::length_error If default predicate policy or representable storage is exceeded.
/// @throws std::runtime_error If exact ear search cannot finish a validated polygon.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::triangulate_polygon({{0,0},{2,0},{2,2},{0,2}}).size()); // 2
/// @endcode
/// @par References
/// - D. Eberly, Triangulation by Ear Clipping, https://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf .
///   This implementation uses exact finite-binary64 orientation predicates.
[[nodiscard]] std::vector<Triangle> triangulate_polygon(const std::vector<Point2D>& polygon);
/// @brief Execute checked ear clipping and return selected actual operation counts.
/// @param polygon Complete finite simple boundary, unchanged.
/// @param options Predicate/history policies; history cap is unused without capture.
/// @return Owned triangles and actual predicate/candidate counts.
/// @throws std::invalid_argument If coordinates or geometry are invalid.
/// @throws std::length_error If predicate policy or storage is exceeded.
/// @throws std::runtime_error If validated ear search cannot finish.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::triangulate_polygon_run({{0,0},{1,0},{0,1}}).triangles.size()); // 1
/// @endcode
PolygonTriangulationRun triangulate_polygon_run(
    const std::vector<Point2D>& polygon,
    PolygonTriangulationOptions options = {});
/// @brief Observe actual ear tests and post-erasure rings from the shared producer.
/// History can grow quadratically in candidate records plus ring copies;
/// limits count logical records/ids, not allocator overhead or peak bytes.
/// @param polygon Complete finite simple boundary, unchanged.
/// @param options Predicate and retained-entry policies, checked at actual call/copy sites.
/// @return Owned output, actual candidates and actual post-clip rings.
/// @throws std::invalid_argument If coordinates or geometry are invalid.
/// @throws std::length_error If predicate/history policy or storage is exceeded.
/// @throws std::runtime_error If validated ear search cannot finish.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::triangulate_polygon_trace({{0,0},{2,0},{2,2},{0,2}}).clips.size()); // 1
/// @endcode
PolygonTriangulationTrace triangulate_polygon_trace(
    const std::vector<Point2D>& polygon,
    PolygonTriangulationOptions options = {});
} // namespace datamunge::geometry
