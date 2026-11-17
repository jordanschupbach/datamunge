#pragma once
/// @file quickhull.hpp
/// @brief Exact-predicate iterative planar Quickhull with actual candidate-set histories.
#include <cstddef>
#include <datamunge/geometry/point2d.hpp>
#include <vector>
/// @brief Canonical planar geometry and owned execution records.
namespace datamunge::geometry {
/// @brief Input, predicate-work and retained-index policies; zero disables a bound.
/// @par Example
/// @code
/// const dm::geometry::QuickhullOptions x{};
/// std::print("{}\n",x.max_points); // 1000000
/// @endcode
/// @see quickhull_run()
struct QuickhullOptions {
  std::size_t max_points{1000000};        ///< Maximum original input vertices.
  std::size_t max_predicates{10000000};   ///< Actual orientation/area-comparison calls.
  std::size_t max_trace_entries{1000000}; ///< Actual candidate/child indices plus record headers.
};
/// @brief One actual nonempty directed-baseline Quickhull subproblem.
/// @par Example
/// @code
/// const dm::geometry::QuickhullStep x{};
/// std::print("{}\n",x.a); // 0
/// @endcode
/// @see quickhull_run()
struct QuickhullStep {
  std::size_t              a{};        ///< Baseline start index in the sorted unique point table.
  std::size_t              b{};        ///< Baseline end index in the sorted unique point table.
  std::size_t              farthest{}; ///< First point with greatest exact signed baseline area.
  std::vector<std::size_t> candidates; ///< Actual strict-left candidates before selecting farthest.
  std::vector<std::size_t> left;       ///< Actual strict-left child for a to farthest.
  std::vector<std::size_t> right;      ///< Actual strict-left child for farthest to b.
};
/// @brief Owned strict hull, exact decision work and optional actual histories.
/// @par Example
/// @code
/// const dm::geometry::QuickhullResult x{};
/// std::print("{}\n",x.predicates); // 0
/// @endcode
/// @see quickhull_run()
struct QuickhullResult {
  std::vector<Point2D>       hull;   ///< Counterclockwise strict hull, no repeated closing vertex.
  std::vector<Point2D>       points; ///< Lexicographically sorted unique input point table.
  std::vector<QuickhullStep> steps;  ///< Actual preorder candidate decisions, only for the trace.
  std::size_t
      orientation_tests{}; ///< Actual exact orientation predicates, including final winding check.
  std::size_t area_comparisons{}; ///< Actual exact farthest-point area comparisons.
  std::size_t predicates{};       ///< Sum of orientation tests and area comparisons.
};
/// @brief Compute the strict planar Quickhull using exact binary64 predicate signs.
/// Validates every coordinate before sorting; duplicates collapse. Strict-left filtering removes collinear edge points. Farthest decisions compare exact signed areas, including near-collinear/extreme finite coordinates. Iterative traversal avoids recursive stack overflow. Worst-case O(n^2) predicate work, typical balanced O(n log n), O(n) working storage plus owned outputs; no distribution-free average guarantee. Empty/singleton/two-point/collinear inputs return their strict endpoints. Input is unchanged.
/// @param points Owned finite Cartesian points.
/// @param options Input and selected-predicate policies.
/// @return Strict CCW hull, sorted unique table and actual exact predicate counts.
/// @throws std::invalid_argument Any nonfinite coordinate.
/// @throws std::length_error Input or selected-predicate policy exceeded.
/// @par Example
/// @code
/// auto r=dm::geometry::quickhull_run({{0,0},{1,0},{0,1},{0.2,0.2}});
/// std::print("{}\n",r.hull.size()); // 3
/// @endcode
/// @par References
/// - C. B. Barber, D. P. Dobkin and H. Huhdanpaa (1996), https://doi.org/10.1145/235815.235821; this is the simpler planar farthest-point variant.
/// @see quickhull_trace(), cross_difference_sign()
QuickhullResult quickhull_run(std::vector<Point2D> points, const QuickhullOptions& options = {});
/// @brief Capture actual candidate and child sets from the same exact Quickhull producer.
/// One nonempty subproblem record costs one header plus all candidate/left/right indices, admitted before copying. Histories use indices into the owned sorted point table. Other behavior and selected-work limits match quickhull_run(); history can be quadratic and is independently bounded.
/// @param points Owned finite Cartesian points.
/// @param options Input, predicate and retained-index policies.
/// @return Strict hull, actual decisions and selected predicate counts.
/// @throws std::invalid_argument Nonfinite coordinate.
/// @throws std::length_error Any execution/history policy exceeded.
/// @par Example
/// @code
/// auto r=dm::geometry::quickhull_trace({{0,0},{1,0},{0,1}});
/// std::print("{}\n",r.steps.size()); // 1
/// @endcode
/// @see quickhull_run()
QuickhullResult quickhull_trace(std::vector<Point2D> points, const QuickhullOptions& options = {});
/// @brief Return the strict CCW hull using the shared exact-predicate producer.
/// Signature preserved. Near-degenerate/extreme-range decisions now use exact signs rather than rounded cross products. Uses the default policies; choose quickhull_run() to configure them.
/// @param points Owned finite input points, duplicates permitted.
/// @return Strict hull; empty/tiny/collinear conventions match quickhull_run().
/// @throws std::invalid_argument Nonfinite coordinate.
/// @throws std::length_error Default input or selected-work policy exceeded.
/// @par Example
/// @code
/// auto h=dm::geometry::quickhull({{0,0},{1,0},{0.5,0}});
/// std::print("{}\n",h.size()); // 2
/// @endcode
/// @see quickhull_run()
std::vector<Point2D> quickhull(std::vector<Point2D> points);
} // namespace datamunge::geometry
