#pragma once
/// @file shoelace.hpp
/// @brief Anchored compensated polygon signed area and actual edge terms.
#include <cstddef>
#include <datamunge/geometry/point2d.hpp>
#include <vector>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::geometry {
/// @brief ShoelaceOptions stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::geometry::ShoelaceOptions value{};
/// std::print("{}\n",value.max_vertices); // 1000000
/// @endcode
struct ShoelaceOptions {
  std::size_t max_vertices{1000000}; ///< Complete input vertex policy; zero disables.
  std::size_t max_records{1000000};  ///< Actual captured edge records; zero disables.
};
/// @brief ShoelaceEdge stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::geometry::ShoelaceEdge value{};
/// std::print("{}\n",value.from); // 0
/// @endcode
struct ShoelaceEdge {
  std::size_t from{};        ///< Actual preceding vertex index; first record closes last to first.
  std::size_t to{};          ///< Actual next vertex index.
  long double determinant{}; ///< Actual anchored cross product in long double.
  long double signed_partial_area{}; ///< Actual compensated signed sum divided by two.
};
/// @brief ShoelaceRun stores owned actual values or explicit invocation policies; field units and defaults are described below.
/// @par Example
/// @code
/// const dm::geometry::ShoelaceRun value{};
/// std::print("{}\n",value.area); // 0
/// @endcode
struct ShoelaceRun {
  double      signed_area{}; ///< CCW positive algebraic area, rounded to double.
  double      area{}; ///< Absolute algebraic area; not union area for self-crossing boundaries.
  std::size_t edge_terms{};        ///< Actual evaluated determinants, zero below three vertices.
  std::vector<ShoelaceEdge> edges; ///< Optional actual native term histories.
};
/// @brief Compute signed polygon area.
/// Implicitly closes the vertex sequence; repeated closing vertex is allowed. Coordinates must be finite. Fewer than three vertices return zero after validation. Subtracts the first vertex before long-double products and uses Neumaier compensation to reduce translation cancellation. Approximate arithmetic, not an exact predicate or guaranteed correctly rounded result. For a simple boundary Green's theorem gives enclosed area; self-crossings give signed algebraic area and may cancel. No simplicity/convexity test. Time O(n), working storage O(1) without optional O(n) histories; input is borrowed during call, result owns records; no shared state.
/// @param vertices Complete finite planar boundary in traversal order.
/// @param options Complete size and optional actual history bounds; zero disables policies.
/// @return Owned signed/absolute area, actual work and optional edge histories.
/// @throws std::invalid_argument Nonfinite coordinate.
/// @throws std::length_error Input or capture policy exceeded.
/// @throws std::overflow_error Intermediate or final area is not representable.
/// @par Example
/// @code
/// auto r=dm::geometry::shoelace_run({{0,0},{4,0},{0,3}});
/// std::print("{} {}\n",r.signed_area,r.edge_terms); // 6 3
/// @endcode
/// @par References
/// - Green's theorem applied to one half of the boundary integral x dy - y dx.
/// @see signed_polygon_area()
ShoelaceRun shoelace_run(const std::vector<Point2D>& vertices, const ShoelaceOptions& options = {});
/// @brief Compute signed polygon area and capture actual edge terms.
/// Implicitly closes the vertex sequence; repeated closing vertex is allowed. Coordinates must be finite. Fewer than three vertices return zero after validation. Subtracts the first vertex before long-double products and uses Neumaier compensation to reduce translation cancellation. Approximate arithmetic, not an exact predicate or guaranteed correctly rounded result. For a simple boundary Green's theorem gives enclosed area; self-crossings give signed algebraic area and may cancel. No simplicity/convexity test. Time O(n), working storage O(1) without optional O(n) histories; input is borrowed during call, result owns records; no shared state.
/// @param vertices Complete finite planar boundary in traversal order.
/// @param options Complete size and optional actual history bounds; zero disables policies.
/// @return Owned signed/absolute area, actual work and optional edge histories.
/// @throws std::invalid_argument Nonfinite coordinate.
/// @throws std::length_error Input or capture policy exceeded.
/// @throws std::overflow_error Intermediate or final area is not representable.
/// @par Example
/// @code
/// auto r=dm::geometry::shoelace_trace({{0,0},{4,0},{0,3}});
/// std::print("{} {}\n",r.signed_area,r.edge_terms); // 6 3
/// @endcode
/// @par References
/// - Green's theorem applied to one half of the boundary integral x dy - y dx.
/// @see signed_polygon_area()
ShoelaceRun shoelace_trace(
    const std::vector<Point2D>& vertices,
    const ShoelaceOptions&      options = {});
/// @brief Return CCW-positive signed polygon area.
/// Uses the same anchored compensated producer and default bounds. Simple boundaries give enclosed area; self-crossing boundaries can cancel. Not an exact sign predicate.
/// @param vertices Borrowed complete finite planar boundary.
/// @return Area in squared coordinate units; zero below three vertices.
/// @throws std::invalid_argument Nonfinite coordinate.
/// @throws std::length_error Default vertex bound exceeded.
/// @throws std::overflow_error Area not representable.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::signed_polygon_area({{0,0},{4,0},{0,3}})); // 6
/// @endcode
/// @see shoelace_run()
[[nodiscard]] double signed_polygon_area(const std::vector<Point2D>& vertices);
/// @brief Return absolute algebraic polygon area.
/// Uses the same anchored compensated producer and default bounds. Simple boundaries give enclosed area; self-crossing boundaries can cancel. Not an exact sign predicate.
/// @param vertices Borrowed complete finite planar boundary.
/// @return Area in squared coordinate units; zero below three vertices.
/// @throws std::invalid_argument Nonfinite coordinate.
/// @throws std::length_error Default vertex bound exceeded.
/// @throws std::overflow_error Area not representable.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::polygon_area({{0,0},{4,0},{0,3}})); // 6
/// @endcode
/// @see shoelace_run()
[[nodiscard]] double polygon_area(const std::vector<Point2D>& vertices);
/// @brief Generate a finite regular polygon for the unit-circle area limit.
/// Angles start at zero and increase counterclockwise; pure owned fixture, no shared state.
/// @param n Number of corners, at least three and at most one million.
/// @return Owned cos/sin unit-circle corners.
/// @throws std::invalid_argument n below three.
/// @throws std::length_error n exceeds one million.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::shoelace_regular_polygon(4).size()); // 4
/// @endcode
/// @see shoelace_run()
std::vector<Point2D> shoelace_regular_polygon(std::size_t n);
} // namespace datamunge::geometry
