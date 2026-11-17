#pragma once
/// @file rotating_calipers.hpp
/// @brief Exact-area rotating antipodes with actual pointer events and widened farthest-pair distances.
#include <cstddef>
#include <datamunge/geometry/convex_hull.hpp>
#include <datamunge/geometry/minimum_bounding_rectangle.hpp>
#include <datamunge/geometry/point2d.hpp>
#include <vector>
/// @brief Canonical planar support geometry.
namespace datamunge::geometry {
/// @brief Owned actual farthest endpoints and finite Euclidean separation.
/// @par Example
/// @code
/// const dm::geometry::DiameterResult value{};
/// std::print("{}\n",value.distance); // 0
/// @endcode
struct DiameterResult {
  Point2D a;          ///< First actual canonical hull endpoint.
  Point2D b;          ///< Second actual canonical hull endpoint.
  double  distance{}; ///< Finite Euclidean separation in coordinate units.
};
/// @brief Complete input, actual-event and retained-event policies; zero disables a cap.
/// @par Example
/// @code
/// const dm::geometry::DiameterOptions value{};
/// std::print("{}\n",value.max_points); // 1000000
/// @endcode
struct DiameterOptions {
  std::size_t max_points{1000000};        ///< Complete original point count.
  std::size_t max_events{10000000};       ///< Actual area comparisons plus farthest-pair tests.
  std::size_t max_trace_entries{1000000}; ///< Actual retained area/pair events.
};
/// @brief Kind of one actual antipodal support decision.
/// @par Example
/// @code
/// const auto kind=dm::geometry::CaliperEventKind::AreaAdvance;
/// std::print("{}\n",static_cast<int>(kind)); // 0
/// @endcode
enum class CaliperEventKind {
  AreaAdvance, ///< Exact area comparison advanced the antipode.
  AreaStop,    ///< Exact comparison stopped advancing, including equal-area ties.
  PairTest     ///< Actual widened Euclidean candidate-pair comparison.
};
/// @brief One actual support-pointer decision or farthest-pair test.
/// @par Example
/// @code
/// const dm::geometry::CaliperEvent value{};
/// std::print("{}\n",value.area_sign); // 0
/// @endcode
struct CaliperEvent {
  CaliperEventKind kind{};  ///< Actual producing branch.
  std::size_t      first{}; ///< Edge start for area tests, first candidate endpoint for pair tests.
  std::size_t      second{}; ///< Current antipode or second actual candidate endpoint.
  std::size_t      next{};   ///< Next proposed antipode; zero outside area tests.
  int    area_sign{}; ///< Exact sign of next minus current baseline area; zero for pair tests.
  double distance{};  ///< Actual tested pair distance; zero outside pair tests.
  bool   improved{};  ///< Whether this actual candidate became the strictly larger best distance.
};
/// @brief Owned actual hull/diameter and selected rotating-pointer work.
/// @par Example
/// @code
/// const dm::geometry::DiameterRun value{};
/// std::print("{}\n",value.pointer_advances); // 0
/// @endcode
struct DiameterRun {
  DiameterResult
      result; ///< Actual winning pair, first encountered wins equal represented distances.
  std::vector<Point2D>      hull;         ///< Owned canonical CCW hull used by this execution.
  std::size_t               area_tests{}; ///< Actual exact baseline-area comparisons.
  std::size_t               pair_tests{}; ///< Actual widened Euclidean candidate tests.
  std::size_t               pointer_advances{}; ///< Actual committed forward antipode changes.
  std::vector<CaliperEvent> events; ///< Actual optional chronological support/pair events.
};
/// @brief Find the farthest hull pair using actual rotating antipodes.
/// Computes the canonical exact-orientation convex hull first, then advances a CCW antipodal pointer while the next point has strictly greater baseline area, using cross_difference_sign() without rounded cross-product subtraction. Tests both edge endpoints against the current antipode; equal areas also test the next antipode. Two-point/collinear hulls need one pair test. Distance uses widened hypot and narrows only the final finite scalar. Ties retain the first encountered represented distance, so an equally far pair can differ from the historical exhaustive scan. Hull O(N log N), caliper walk O(H) area/pair tests plus the exact predicate constant; working/output O(N+H), optional O(H) events. Complete finite inputs required; at least two distinct points. Inputs unchanged and results owned. Counters exclude hull construction/validation and predicate internal arithmetic.
/// @param points Complete finite planar points; duplicate/interior points allowed.
/// @param options Input, actual-event and retained-event bounds.
/// @return Owned canonical hull, actual farthest pair and selected pointer work.
/// @throws std::invalid_argument Nonfinite or fewer than two distinct points.
/// @throws std::length_error Input, actual event, history or capacity policy exceeded.
/// @throws std::overflow_error Farthest distance cannot be represented as finite double.
/// @par Example
/// @code
/// auto r=dm::geometry::polygon_diameter_run({{0,0},{3,0},{3,4},{0,4}});
/// std::print("{}\n",r.result.distance); // 5
/// @endcode
/// @par References
/// - Toussaint (1983), Solving geometric problems with the rotating calipers, https://www-cgrl.cs.mcgill.ca/~godfried/publications/calipers.pdf.
/// @see polygon_diameter(), cross_difference_sign()
DiameterRun polygon_diameter_run(
    const std::vector<Point2D>& points,
    const DiameterOptions&      options = {});
/// @brief Find the farthest hull pair using actual rotating antipodes and retain their events.
/// Computes the canonical exact-orientation convex hull first, then advances a CCW antipodal pointer while the next point has strictly greater baseline area, using cross_difference_sign() without rounded cross-product subtraction. Tests both edge endpoints against the current antipode; equal areas also test the next antipode. Two-point/collinear hulls need one pair test. Distance uses widened hypot and narrows only the final finite scalar. Ties retain the first encountered represented distance, so an equally far pair can differ from the historical exhaustive scan. Hull O(N log N), caliper walk O(H) area/pair tests plus the exact predicate constant; working/output O(N+H), optional O(H) events. Complete finite inputs required; at least two distinct points. Inputs unchanged and results owned. Counters exclude hull construction/validation and predicate internal arithmetic.
/// @param points Complete finite planar points; duplicate/interior points allowed.
/// @param options Input, actual-event and retained-event bounds.
/// @return Owned canonical hull, actual farthest pair and selected pointer work.
/// @throws std::invalid_argument Nonfinite or fewer than two distinct points.
/// @throws std::length_error Input, actual event, history or capacity policy exceeded.
/// @throws std::overflow_error Farthest distance cannot be represented as finite double.
/// @par Example
/// @code
/// auto r=dm::geometry::polygon_diameter_trace({{0,0},{3,0},{3,4},{0,4}});
/// std::print("{}\n",r.result.distance); // 5
/// @endcode
/// @par References
/// - Toussaint (1983), Solving geometric problems with the rotating calipers, https://www-cgrl.cs.mcgill.ca/~godfried/publications/calipers.pdf.
/// @see polygon_diameter(), cross_difference_sign()
DiameterRun polygon_diameter_trace(
    const std::vector<Point2D>& points,
    const DiameterOptions&      options = {});
/// @brief Return the actual farthest pair through the preserved diameter signature.
/// Uses the shared checked O(H) rotating antipodes instead of the former O(H^2) exhaustive pair scan. Retains any valid farthest pair; default execution policies apply.
/// @param points Complete finite points with at least two distinct locations.
/// @return Owned endpoints and finite Euclidean diameter.
/// @throws std::invalid_argument Invalid/degenerate input.
/// @throws std::length_error Default policy exceeded.
/// @throws std::overflow_error Unrepresentable distance.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::polygon_diameter({{0,0},{3,4}}).distance); // 5
/// @endcode
/// @par References
/// - Toussaint (1983), Solving geometric problems with the rotating calipers, https://www-cgrl.cs.mcgill.ca/~godfried/publications/calipers.pdf.
/// @see polygon_diameter_run()
DiameterResult polygon_diameter(const std::vector<Point2D>& points);
/// @brief Generate the original fifty-point Gaussian diameter fixture in the canonical library.
/// Uses mt19937(seed31337), one standard-normal distribution, then x=3*next and y=2*next in original order. Distribution realization follows the C++ standard-library implementation; it is a controlled synthetic input, not a real-data sample.
/// @param seed MT seed, default31337.
/// @return Owned original50 planar observations.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::diameter_example_cloud().size()); // 50
/// @endcode
/// @see polygon_diameter_trace()
std::vector<Point2D> diameter_example_cloud(unsigned seed = 31337);
} // namespace datamunge::geometry
