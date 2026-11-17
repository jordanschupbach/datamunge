#pragma once
/// @file clipped_voronoi.hpp
/// @brief Checked native clipped voronoi and actual producer state.
#include <cstddef>
#include <datamunge/geometry/point2d.hpp>
#include <datamunge/plot/plot.hpp>
#include <vector>
namespace datamunge::geometry {
/// @brief Axis-aligned finite clipping viewport.
/// @par Example
/// @code
/// dm::geometry::VoronoiBox box{0,10,0,10};
/// @endcode
/// @par References
/// - The companion report derives the implemented recurrence.
/// @see Related operations in this header.
struct VoronoiBox {
  double xmin{0}; ///< Finite left boundary, strictly below xmax.
  double xmax{1}; ///< Finite right boundary.
  double ymin{0}; ///< Finite bottom boundary, strictly below ymax.
  double ymax{1}; ///< Finite top boundary.
};
/// @brief Actual cell polygon after one competitor half-plane.
/// @par Example
/// @code
/// dm::geometry::VoronoiClipStep r;
/// @endcode
/// @par References
/// - The companion report derives the implemented recurrence.
/// @see Related operations in this header.
struct VoronoiClipStep {
  std::size_t          site{0};       ///< Actual owning site ID.
  std::size_t          competitor{0}; ///< Actual competing site ID.
  std::vector<Point2D> polygon;       ///< Actual polygon after clipping; empty allowed.
};
/// @brief Actual bounded Voronoi cells and selected clipping work.
/// @par Example
/// @code
/// auto r=dm::geometry::clipped_voronoi({{.25,.5},{.75,.5}},{0,1,0,1});
/// @endcode
/// @par References
/// - The companion report derives the implemented recurrence.
/// @see Related operations in this header.
struct ClippedVoronoi {
  std::vector<Point2D> sites; ///< Owned original sites in cell order.
  VoronoiBox           box;   ///< Actual viewport.
  std::vector<std::vector<Point2D>>
                               cells; ///< Actual closed polygons with implicit last-to-first edge.
  std::vector<VoronoiClipStep> steps; ///< First requested actual clipping snapshots.
  std::size_t halfplanes{0}; ///< Actual competitor half-planes processed, including empty polygons.
  std::size_t edge_tests{0}; ///< Actual polygon-edge clipping decisions.
};
/// @brief Construct bounded nearest-site cells by actual half-plane clipping.
/// Each cell starts at the viewport and intersects 2*(p_j-p_i) dot x <= |p_j|^2-|p_i|^2. Coordinates have absolute magnitude<=1e100; long-double intermediates reduce cancellation but do not supply exact predicates. Boundaries are shared ties. At most20000000 actual edge tests. No Delaunay replay or shared state.
/// @param sites Distinct finite byte-free points, <=1000; empty accepted.
/// @param box Finite positive-area viewport; sites may lie outside.
/// @param max_records Actual snapshots retained, default0; at most1000000 captured vertices.
/// @return Owned cells in input order and actual clip state/counters.
/// @throws std::invalid_argument Invalid bounds, nonfinite/oversized coordinates or duplicate sites.
/// @throws std::length_error Site/work/history ceiling exceeded.
/// @throws std::overflow_error Nonfinite intersection.
/// @par Example
/// @code
/// auto r=dm::geometry::clipped_voronoi({{2,5},{8,5}},{0,10,0,10},2);
/// @endcode
/// @par References
/// - The companion report derives the implemented recurrence.
/// @see Related operations in this header.
ClippedVoronoi
clipped_voronoi(const std::vector<Point2D>& sites, VoronoiBox box, std::size_t max_records = 0);
/// @brief Plot actual clipped cells and their sites at equal Cartesian scales.
/// @param result Unmodified clipped_voronoi output.
/// @return Owned plot; caller saves the figure.
/// @par Example
/// @code
/// auto p=dm::geometry::clipped_voronoi_plot(dm::geometry::clipped_voronoi({{.25,.5},{.75,.5}},{0,1,0,1}));
/// @endcode
/// @par References
/// - The companion report derives the implemented recurrence.
/// @see Related operations in this header.
graphics::RPlot clipped_voronoi_plot(const ClippedVoronoi& result);
/// @brief Return the original24-site seeded report fixture.
/// Two draws per site, x then y; exact stream follows the C++ standard-library distribution.
/// @return Owned mt19937 seed1234567 points, coordinates uniform[1,9).
/// @par Example
/// @code
/// auto sites=dm::geometry::voronoi_example_sites();
/// @endcode
/// @par References
/// - The companion report derives the implemented recurrence.
/// @see Related operations in this header.
std::vector<Point2D> voronoi_example_sites();

} // namespace datamunge::geometry
