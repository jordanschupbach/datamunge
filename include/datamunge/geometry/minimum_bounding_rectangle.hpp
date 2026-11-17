#pragma once
/// @file minimum_bounding_rectangle.hpp
/// @brief Hull-edge orientation scan with actual projected extents and candidate areas.
#include <cstddef>
#include <datamunge/geometry/point2d.hpp>
#include <vector>
namespace datamunge::geometry {
/// @brief Owned oriented rectangle with four adjacent ordered corners.
/// Width follows the chosen hull edge; height follows its left perpendicular.
/// @par Example
/// @code
/// auto r=dm::geometry::minimum_bounding_rectangle({{0,0},{3,0},{3,2},{0,2}});
/// std::print("{}\n",r.area); // 6
/// @endcode
/// @see minimum_bounding_rectangle()
struct MinimumBoundingRectangle {
  std::vector<Point2D> corners;   ///< Four owned adjacent corners.
  double               width{0};  ///< Extent along chosen unit hull edge.
  double               height{0}; ///< Extent along perpendicular.
  double               area{0};   ///< Width times height, in squared coordinate units.
};
/// @brief Actual projected extents and area of one tested hull orientation.
/// Projections are relative to hull[edge]; direction is its unit outgoing edge.
/// @par Example
/// @code
/// auto t=dm::geometry::minimum_bounding_rectangle_trace({{0,0},{3,0},{3,2},{0,2}});
/// std::print("{}\n",t.candidates.front().area); // 6
/// @endcode
/// @see MinimumBoundingRectangleTrace
struct BoundingRectangleCandidate {
  std::size_t edge{0};   ///< Actual tested hull-edge index.
  Point2D     direction; ///< Unit outgoing hull edge.
  double      min_u{0};  ///< Actual minimum parallel projection.
  double      max_u{0};  ///< Actual maximum parallel projection.
  double      min_v{0};  ///< Actual minimum perpendicular projection.
  double      max_v{0};  ///< Actual maximum perpendicular projection.
  double      area{0};   ///< Actual candidate area.
  bool selected{false};  ///< Strict improvement at this iteration, not necessarily final winner.
};
/// @brief Owned original points, canonical hull and actual orientation candidates.
/// Entry units are input points, hull points, candidate records and four final
/// corners. Projection count is observed work, not elapsed time.
/// @par Example
/// @code
/// auto t=dm::geometry::minimum_bounding_rectangle_trace({{0,0},{3,0},{3,2},{0,2}});
/// std::print("{} {}\n",t.candidates.size(),t.projections); // 4 16
/// @endcode
/// @see minimum_bounding_rectangle_trace()
struct MinimumBoundingRectangleTrace {
  std::vector<Point2D>                    input; ///< Owned original finite points.
  std::vector<Point2D>                    hull;  ///< Actual canonical counterclockwise convex hull.
  std::vector<BoundingRectangleCandidate> candidates; ///< Actual hull-edge scan order.
  MinimumBoundingRectangle result;              ///< Actual first strictly smallest-area rectangle.
  std::size_t              projections{0};      ///< Actual projected hull points, h squared.
  std::size_t              captured_entries{0}; ///< Admitted capture units.
};
/// @brief Scan all convex-hull-edge orientations for a minimum-area rectangle.
/// Every tested orientation projects all hull points: O(n log n+h^2) work,
/// O(n+h) driver storage. This is not the linear rotating-calipers walk.
/// Exact-sign canonical hull construction precedes floating projection geometry.
/// First strict area improvement wins ties. Duplicates/interior points allowed;
/// a hull with fewer than three vertices is rejected. Floating roundoff remains.
/// @param points Finite coordinates with a nondegenerate hull.
/// @return Owned final corners, side lengths and area.
/// @throws std::invalid_argument If points are nonfinite or hull is degenerate.
/// @throws std::overflow_error If projection geometry/area/corners are nonfinite.
/// @par Example
/// @code
/// auto r=dm::geometry::minimum_bounding_rectangle({{0,0},{3,0},{3,2},{0,2}});
/// std::print("{}\n",r.area); // 6
/// @endcode
/// @par References
/// - H. Freeman and R. Shapira (1975), Determining the minimum-area encasing rectangle for an arbitrary closed curve, Communications of the ACM18,409–413, doi:10.1145/360881.360919.
/// @see minimum_bounding_rectangle_trace()
MinimumBoundingRectangle minimum_bounding_rectangle(const std::vector<Point2D>& points);
/// @brief Capture actual extents and selection decisions from the ordinary producer.
/// No replay. Same geometry/complexity as ordinary execution, plus O(n+h)
/// history. Actual entry admission throws before exceeding the optional cap;
/// no partial result escapes. Zero disables the cap.
/// @param points Original finite nondegenerate point set, retained by value.
/// @param max_entries Capture quota, default1000000; zero disables.
/// @return Owned actual hull, extents, work and rectangle.
/// @throws std::invalid_argument If coordinates or hull are invalid.
/// @throws std::overflow_error If finite projection geometry cannot be represented.
/// @throws std::length_error If quota or work count is exceeded.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// auto t=dm::geometry::minimum_bounding_rectangle_trace({{0,0},{3,0},{3,2},{0,2}});
/// std::print("{}\n",t.result.area); // 6
/// @endcode
/// @see MinimumBoundingRectangleTrace
MinimumBoundingRectangleTrace minimum_bounding_rectangle_trace(
    const std::vector<Point2D>& points,
    std::size_t                 max_entries = 1000000);
/// @brief Generate four adjacent corners of a rotated origin-anchored rectangle.
/// Uses native sine/cosine; floating rounding may perturb exact collinearity.
/// @param width Finite strictly positive width, default6.
/// @param height Finite strictly positive height, default2.
/// @param radians Finite rotation angle, default pi/6 (original30degrees).
/// @return Owned rotated points(0,0),(width,0),(width,height),(0,height).
/// @throws std::invalid_argument If dimensions/angle are invalid.
/// @throws std::overflow_error If rotated coordinates are nonfinite.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::bounding_rectangle_demo().size()); // 4
/// @endcode
/// @see minimum_bounding_rectangle()
std::vector<Point2D>
bounding_rectangle_demo(double width = 6, double height = 2, double radians = 0.523598775598298873);
} // namespace datamunge::geometry
