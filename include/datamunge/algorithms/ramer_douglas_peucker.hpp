#pragma once
/// @file ramer_douglas_peucker.hpp
/// @brief Checked segment-distance Ramer-Douglas-Peucker simplification and actual split history.
#include <cstddef>
#include <utility>
#include <vector>
/// @brief Canonical polyline algorithms and owned records.
namespace datamunge::algorithms {
/// @brief One finite Cartesian polyline vertex (x,y).
/// @par Example
/// @code
/// const dm::algorithms::RDPPoint p{1,2};
/// std::print("{}\n",p.first); // 1
/// @endcode
/// @see rdp_segment_distance()
using RDPPoint = std::pair<double, double>;
/// @brief Input, selected-work and retained record policies; zero disables a cap.
/// @par Example
/// @code
/// const dm::algorithms::RDPOptions x{};
/// std::print("{}\n",x.max_points); // 1000000
/// @endcode
/// @see ramer_douglas_peucker_run()
struct RDPOptions {
  std::size_t max_points{1000000};          ///< Largest admitted vertex count.
  std::size_t max_distance_tests{10000000}; ///< Actual interior-point segment-distance tests.
  std::size_t max_trace_entries{1000000};   ///< Actual range records retained.
};
/// @brief One actual nontrivial range considered by the iterative split producer.
/// @par Example
/// @code
/// const dm::algorithms::RDPStep x{};
/// std::print("{}\n",x.split); // false
/// @endcode
/// @see ramer_douglas_peucker_run()
struct RDPStep {
  std::size_t lo{};       ///< First endpoint index, inclusive.
  std::size_t hi{};       ///< Last endpoint index, inclusive.
  std::size_t farthest{}; ///< First farthest interior vertex; lo if every distance is zero.
  double      distance{}; ///< Largest actual finite-segment distance on this range.
  bool        split{};    ///< Whether distance strictly exceeds epsilon.
};
/// @brief Owned simplification and actual selected work.
/// @par Example
/// @code
/// const dm::algorithms::RDPResult x{};
/// std::print("{}\n",x.points.size()); // 0
/// @endcode
/// @see ramer_douglas_peucker_run()
struct RDPResult {
  std::vector<RDPPoint>    points;  ///< Retained vertices in original order.
  std::vector<std::size_t> indices; ///< Corresponding original vertex indices.
  std::vector<RDPStep>     steps;   ///< Actual pre-order decisions, only populated by the trace.
  std::size_t distance_tests{};     ///< Actual interior-point segment-distance evaluations.
  double      max_segment_error{};  ///< Maximum vertex distance to its assigned retained chord.
};
/// @brief Measure Euclidean distance to a finite segment, with endpoint clamping.
/// Uses extended precision for differences, projection and hypot; coincident endpoints reduce to ordinary point distance. Does not use the distance to an infinite supporting line. O(1) time/storage and no mutation.
/// @param point Finite point.
/// @param a Finite first endpoint.
/// @param b Finite second endpoint.
/// @return Nonnegative finite Euclidean distance.
/// @throws std::invalid_argument Any coordinate is nonfinite.
/// @throws std::overflow_error Distance is outside the finite double range.
/// @par Example
/// @code
/// std::print("{:.0f}\n",dm::algorithms::rdp_segment_distance({2,0},{0,0},{1,0})); // 1
/// @endcode
/// @see ramer_douglas_peucker_run()
double rdp_segment_distance(const RDPPoint& point, const RDPPoint& a, const RDPPoint& b);
/// @brief Simplify a polyline using actual finite-segment distances.
/// Always keeps the endpoints of nonempty input and splits at the first farthest vertex only when its distance strictly exceeds epsilon. Iterative depth-first traversal avoids recursive stack overflow. For each accepted chord every original vertex, hence every connecting original segment by convexity of the segment-distance tube, lies within epsilon of that chord (up to floating roundoff). This is a directed geometric error guarantee, not a minimum-vertex or topology guarantee. Worst-case O(n^2) distance tests, O(n) working storage; caller input unchanged.
/// @param points Finite ordered vertices; duplicate/coincident vertices are permitted.
/// @param epsilon Finite nonnegative distance tolerance.
/// @param options Input and selected-work policy.
/// @return Retained original points/indices, actual tests and assigned-chord maximum error.
/// @throws std::invalid_argument Nonfinite coordinate or invalid epsilon, even for short input.
/// @throws std::length_error Input or distance-test budget exceeded.
/// @throws std::overflow_error Actual distance exceeds double range.
/// @par Example
/// @code
/// auto r=dm::algorithms::ramer_douglas_peucker_run({{0,0},{2,0},{1,0}},0.5);
/// std::print("{}\n",r.points.size()); // 3
/// @endcode
/// @par References
/// - D. H. Douglas and T. K. Peucker (1973), https://doi.org/10.3138/FM57-6770-U75U-7727
/// @see ramer_douglas_peucker_trace()
RDPResult ramer_douglas_peucker_run(
    const std::vector<RDPPoint>& points,
    double                       epsilon,
    const RDPOptions&            options = {});
/// @brief Capture every actual nontrivial split/accept decision from the same producer.
/// Records before selecting child ranges; left child is visited before right. One retained record per considered range, admitted before scanning. Zero cap disables history policy. Other semantics and complexity match ramer_douglas_peucker_run().
/// @param points Finite ordered vertices.
/// @param epsilon Finite nonnegative tolerance.
/// @param options Input, selected-distance and retained-record bounds.
/// @return Actual ranges, farthest vertices, decisions and final original-index provenance.
/// @throws std::invalid_argument Invalid coordinate or tolerance.
/// @throws std::length_error Any execution policy exceeded.
/// @throws std::overflow_error Actual distance exceeds double range.
/// @par Example
/// @code
/// auto r=dm::algorithms::ramer_douglas_peucker_trace({{0,0},{1,1},{2,0}},0.5);
/// std::print("{}\n",r.steps[0].farthest); // 1
/// @endcode
/// @see ramer_douglas_peucker_run()
RDPResult ramer_douglas_peucker_trace(
    const std::vector<RDPPoint>& points,
    double                       epsilon,
    const RDPOptions&            options = {});
/// @brief Return retained vertices using the checked finite-segment producer.
/// Signature preserved; formerly accepted invalid inputs now throw, and endpoint-overshooting vertices use finite-segment distance rather than infinite-line distance. Uses the default execution policies; use the run API to configure them.
/// @param points Finite ordered vertices.
/// @param epsilon Finite nonnegative distance tolerance.
/// @return Retained original vertices in order.
/// @throws std::invalid_argument Invalid coordinate or tolerance.
/// @throws std::length_error Default policy exceeded.
/// @throws std::overflow_error Distance exceeds double range.
/// @par Example
/// @code
/// auto p=dm::algorithms::ramer_douglas_peucker({{0,0},{1,0},{2,0}},0);
/// std::print("{}\n",p.size()); // 2
/// @endcode
/// @see ramer_douglas_peucker_run()
std::vector<RDPPoint> ramer_douglas_peucker(const std::vector<RDPPoint>& points, double epsilon);
/// @brief Generate the original smooth wiggly polyline as native reusable input.
/// Returns (x,sin(x)+0.3*sin(3.1*x)) for x=i*step. The default201 vertices with step0.05 reproduce the original report. Count0 returns empty; finite positive step is still validated. O(count) time and owned storage, no shared mutable state.
/// @param count Vertices to generate, default201.
/// @param step Positive finite x increment, default0.05.
/// @param max_points Allocation policy, default1000000; zero disables.
/// @return Owned finite ordered points.
/// @throws std::invalid_argument Nonpositive or nonfinite step.
/// @throws std::length_error Point allocation policy exceeded.
/// @throws std::overflow_error Generated x is not finite.
/// @par Example
/// @code
/// auto p=dm::algorithms::rdp_waveform();
/// std::print("{}\n",p.size()); // 201
/// @endcode
/// @see ramer_douglas_peucker_run()
std::vector<RDPPoint>
rdp_waveform(std::size_t count = 201, double step = 0.05, std::size_t max_points = 1000000);
} // namespace datamunge::algorithms
