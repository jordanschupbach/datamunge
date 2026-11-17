#pragma once
/// @file sutherland_hodgman.hpp
/// @brief Checked rectangular polygon clipping with actual half-plane decisions.
#include <cstddef>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Own one Cartesian point as x,y double coordinates.
/// Finite values are required by clipping consumers; pair construction itself performs no validation.
/// @par Example
/// @code
/// dm::algorithms::Point2 p{1,2};
/// std::print("{}\n",p.first); // 1
/// @endcode
/// @see sutherland_hodgman_run()
using Point2 = std::pair<double, double>;
/// @brief Own the SutherlandOptions contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::SutherlandOptions r{};
/// (void)r;
/// @endcode
/// @see sutherland_hodgman_run()
struct SutherlandOptions {
  std::size_t max_vertices{
      1000000}; ///< Complete input and actual per-phase output vertex count; zero disables.
  std::size_t max_edge_visits{
      20000000}; ///< Actual closed-walk edge classifications across at most four phases; zero disables.
  std::size_t max_history_values{
      1000000}; ///< Actual decision/phase records plus copied point scalar entries; zero disables.
};
/// @brief Own the SutherlandStep contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::SutherlandStep r{};
/// (void)r;
/// @endcode
/// @see sutherland_hodgman_run()
struct SutherlandStep {
  int                 boundary{};        ///< Actual0left,1right,2bottom,3top half-plane.
  Point2              previous;          ///< Actual previous input vertex for this edge.
  Point2              current;           ///< Actual current input vertex.
  bool                previous_inside{}; ///< Actual inclusive half-plane predicate at previous.
  bool                current_inside{};  ///< Actual inclusive half-plane predicate at current.
  std::vector<Point2> emitted; ///< Actual0..2 emitted points for this edge, in producing order.
};
/// @brief Own the SutherlandPhase contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::SutherlandPhase r{};
/// (void)r;
/// @endcode
/// @see sutherland_hodgman_run()
struct SutherlandPhase {
  int boundary{}; ///< Actual processed rectangle boundary.
  std::vector<Point2>
      after; ///< Actual complete output walk after this phase, including duplicate boundary hits.
};
/// @brief Own the SutherlandRun contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::SutherlandRun r{};
/// (void)r;
/// @endcode
/// @see sutherland_hodgman_run()
struct SutherlandRun {
  std::vector<Point2> polygon;  ///< Actual final ordered clipped walk; empty if completely removed.
  std::size_t         phases{}; ///< Actual processed boundaries; early empty stops further phases.
  std::size_t         edge_visits{};   ///< Actual classified previous/current edge pairs.
  std::size_t         intersections{}; ///< Actual crossing intersections computed.
  std::size_t emissions{}; ///< Actual output pushes across all phases, not final vertex count.
  std::vector<SutherlandStep>  steps;   ///< Optional actual edge decisions/emissions.
  std::vector<SutherlandPhase> history; ///< Optional actual post-phase walks.
};
/// @brief Clip the actual subject walk through four rectangle half-planes.
/// Finite input points and ordered finite inclusive rectangle bounds required; zero-width/height rectangles allowed. Process left,right,bottom,top, carrying each actual output walk to the next phase. Both inside emits current; leaving emits intersection; entering emits intersection then current; both outside emits nothing. Stable intersections reuse the checked widened native Cohen-Sutherland half-plane segment primitive with other finite bounds spanning all double coordinates. No tolerance or topology repair: duplicate boundary points may remain. A simple convex subject yields its clipped convex polygon; a concave subject may encode disconnected pieces as one connected boundary walk with coincident bridge edges, not a general multipolygon boolean result. Input self-intersections are not classified or repaired. Empty input returns empty. Four phases give linear output-sensitive work/storage in input size; actual policies bound emitted/captured geometry. Owned results, no shared state.
/// @param polygon Borrowed complete finite ordered subject walk.
/// @param xmin Finite inclusive lower x boundary.
/// @param ymin Finite inclusive lower y boundary.
/// @param xmax Finite upper x boundary at least xmin.
/// @param ymax Finite upper y boundary at least ymin.
/// @param options Input/output, actual edge-work and capture policies; zero disables.
/// @return Owned actual clipped walk, executed counts and optional phase/edge states.
/// @throws std::invalid_argument Nonfinite points/bounds or reversed rectangle.
/// @throws std::length_error Complete input or actual emitted/work/history capacity exceeded.
/// @throws std::overflow_error Checked native intersection cannot be represented.
/// @par Example
/// @code
/// auto r=dm::algorithms::sutherland_hodgman_run({{0,0},{1,0},{0,1}},0,0,2,2);
/// std::print("{}\n",r.polygon.size()); // 3
/// @endcode
/// @par References
/// - Sutherland and Hodgman, Reentrant Polygon Clipping, Communications of the ACM17 (1974),32-42.
/// @see sutherland_hodgman_clip()
SutherlandRun sutherland_hodgman_run(
    const std::vector<Point2>& polygon,
    double                     xmin,
    double                     ymin,
    double                     xmax,
    double                     ymax,
    const SutherlandOptions&   options = {});
/// @brief Clip the actual subject walk through four rectangle half-planes.
/// Finite input points and ordered finite inclusive rectangle bounds required; zero-width/height rectangles allowed. Process left,right,bottom,top, carrying each actual output walk to the next phase. Both inside emits current; leaving emits intersection; entering emits intersection then current; both outside emits nothing. Stable intersections reuse the checked widened native Cohen-Sutherland half-plane segment primitive with other finite bounds spanning all double coordinates. No tolerance or topology repair: duplicate boundary points may remain. A simple convex subject yields its clipped convex polygon; a concave subject may encode disconnected pieces as one connected boundary walk with coincident bridge edges, not a general multipolygon boolean result. Input self-intersections are not classified or repaired. Empty input returns empty. Four phases give linear output-sensitive work/storage in input size; actual policies bound emitted/captured geometry. Owned results, no shared state.
/// @param polygon Borrowed complete finite ordered subject walk.
/// @param xmin Finite inclusive lower x boundary.
/// @param ymin Finite inclusive lower y boundary.
/// @param xmax Finite upper x boundary at least xmin.
/// @param ymax Finite upper y boundary at least ymin.
/// @param options Input/output, actual edge-work and capture policies; zero disables.
/// @return Owned actual clipped walk, executed counts and optional phase/edge states.
/// @throws std::invalid_argument Nonfinite points/bounds or reversed rectangle.
/// @throws std::length_error Complete input or actual emitted/work/history capacity exceeded.
/// @throws std::overflow_error Checked native intersection cannot be represented.
/// @par Example
/// @code
/// auto r=dm::algorithms::sutherland_hodgman_trace({{0,0},{1,0},{0,1}},0,0,2,2);
/// std::print("{}\n",r.polygon.size()); // 3
/// @endcode
/// @par References
/// - Sutherland and Hodgman, Reentrant Polygon Clipping, Communications of the ACM17 (1974),32-42.
/// @see sutherland_hodgman_clip()
SutherlandRun sutherland_hodgman_trace(
    const std::vector<Point2>& polygon,
    double                     xmin,
    double                     ymin,
    double                     xmax,
    double                     ymax,
    const SutherlandOptions&   options = {});
/// @brief Preserve checked rectangular Sutherland-Hodgman output order.
/// Finite input points and ordered finite inclusive rectangle bounds required; zero-width/height rectangles allowed. Process left,right,bottom,top, carrying each actual output walk to the next phase. Both inside emits current; leaving emits intersection; entering emits intersection then current; both outside emits nothing. Stable intersections reuse the checked widened native Cohen-Sutherland half-plane segment primitive with other finite bounds spanning all double coordinates. No tolerance or topology repair: duplicate boundary points may remain. A simple convex subject yields its clipped convex polygon; a concave subject may encode disconnected pieces as one connected boundary walk with coincident bridge edges, not a general multipolygon boolean result. Input self-intersections are not classified or repaired. Empty input returns empty. Four phases give linear output-sensitive work/storage in input size; actual policies bound emitted/captured geometry. Owned results, no shared state.
/// @param polygon Borrowed complete finite ordered subject walk.
/// @param xmin Finite inclusive lower x boundary.
/// @param ymin Finite inclusive lower y boundary.
/// @param xmax Finite upper x boundary at least xmin.
/// @param ymax Finite upper y boundary at least ymin.
/// @return Owned actual final clipped vertex walk.
/// @throws std::invalid_argument Malformed finite geometry.
/// @throws std::length_error Default complete/actual work/output budget exceeded.
/// @throws std::overflow_error Unrepresentable checked intersection.
/// @par Example
/// @code
/// auto p=dm::algorithms::sutherland_hodgman_clip({{0,0},{1,0},{0,1}},0,0,2,2);
/// std::print("{}\n",p.size()); // 3
/// @endcode
/// @see sutherland_hodgman_run()
std::vector<Point2> sutherland_hodgman_clip(
    const std::vector<Point2>& polygon,
    double                     xmin,
    double                     ymin,
    double                     xmax,
    double                     ymax);
} // namespace datamunge::algorithms
