#pragma once
/// @file graham_scan.hpp
/// @brief Exact-sign polar-angle Graham scan with original identities and actual stack events.
#include <datamunge/geometry/convex_hull.hpp>

namespace datamunge::geometry {
/// @brief Selected native preparation and scan work, excluding validation and allocation.
/// Sort counts depend on the standard library and input order. Angular and sweep
/// tests call the exact binary64 orientation predicate; other comparisons do not.
/// The scan pushes each distinct point once when there are at least three.
/// @par Example
/// @code
/// const auto a=dm::geometry::graham_scan_analyze({{0,0},{1,0},{0,1}});
/// std::print("{} {}\n",a.work.pushes,a.work.pops); // 3 0
/// @endcode
/// @see GrahamAnalysis
struct GrahamWork {
  std::size_t input_points{0};              ///< Validated original records, duplicates included.
  std::size_t distinct_points{0};           ///< Exact coordinate classes after deduplication.
  std::size_t lexicographic_comparisons{0}; ///< Actual initial record-sort comparisons.
  std::size_t duplicate_comparisons{0};     ///< Actual adjacent coordinate-equality comparisons.
  std::size_t pivot_comparisons{0}; ///< Distinct records examined after the first for the pivot.
  std::size_t angular_tests{0};     ///< Actual exact orientation calls in polar sorting.
  std::size_t sweep_tests{0};       ///< Actual exact orientation calls in the stack sweep.
  std::size_t pushes{0};    ///< Actual stack pushes; zero for the fewer-than-three shortcut.
  std::size_t pops{0};      ///< Actual non-left-turn stack removals.
  std::size_t max_stack{0}; ///< Greatest actual scan stack size, zero on the shortcut.
};
/// @brief Owned strict hull, earliest original identities and actual selected work.
/// With at least three distinct points, output starts at the lowest-y then
/// lowest-x pivot and runs counterclockwise without repeating the first vertex.
/// Collinear input returns the pivot and farthest endpoint. Empty/singleton/two
/// distinct points retain the legacy lexicographic shortcut, with no scan work.
/// Exact coordinate equality merges signed zeros; the earliest matching original
/// record supplies the coordinate representation. No caller storage is retained.
/// @par Example
/// @code
/// const auto a=dm::geometry::graham_scan_analyze({{2,0},{0,0},{0,2},{1,1}});
/// std::print("{} {}\n",a.hull.size(),a.input_indices[0]); // 3 1
/// @endcode
/// @see graham_scan_analyze(), GrahamTrace
struct GrahamAnalysis {
  std::vector<Point2D>     hull; ///< Owned unclosed strict hull of represented coordinates.
  std::vector<std::size_t> input_indices; ///< Earliest original identities aligned with hull.
  GrahamWork               work;          ///< Selected actual preparation/scan counts.
};
/// @brief One actual test, pop or push, referring to trace.ordered indices.
/// Constant-size edits avoid quadratic stack snapshots. A test leaves the stack
/// unchanged; a,b and turn describe its actual operands/sign. On edits those
/// fields are zero placeholders. point is the edited point, or candidate on tests.
/// @par Example
/// @code
/// const auto t=dm::geometry::graham_scan_trace({{0,0},{1,0},{0,1}});
/// std::print("{}\n",t.events[0].stack_size); // 1
/// @endcode
/// @see HullAction, GrahamTrace
struct GrahamEvent {
  HullAction  action{};      ///< Actual operation, reusing the common hull action enum.
  std::size_t candidate{0};  ///< Pending point in actual scan order.
  std::size_t point{0};      ///< Edited point, or candidate for a test.
  std::size_t a{0};          ///< Next-to-top operand for a test, otherwise zero.
  std::size_t b{0};          ///< Top operand for a test, otherwise zero.
  int         turn{0};       ///< Exact orientation sign on a test, otherwise zero.
  std::size_t stack_size{0}; ///< Actual size immediately after the event.
};
/// @brief Owned actual angular order, strict hull and complete chronological stack edits.
/// ordered starts with the chosen pivot for at least three distinct points; the
/// legacy small-input shortcut instead retains lexicographic order and no events.
/// Each indexed point owns its earliest original coordinate representation.
/// Capture performs no extra sorting or orientation replay. Copying owns storage.
/// @par Example
/// @code
/// const auto t=dm::geometry::graham_scan_trace({{0,0},{1,0},{0,1}});
/// std::print("{} {}\n",t.ordered.size(),t.analysis.hull.size()); // 3 3
/// @endcode
/// @see graham_scan_trace()
struct GrahamTrace {
  GrahamAnalysis                analysis; ///< Actual result, identities and selected work.
  std::vector<HullIndexedPoint> ordered; ///< Actual unique scan order, pivot first except shortcut.
  std::vector<GrahamEvent>      events; ///< Actual whole test/pop/push records, without truncation.
};
/// @brief Compute the strict convex hull by exact-sign polar sorting and a left-turn sweep.
/// Validates finite coordinates before internal indexed copies/sorting. The legacy
/// by-value argument copy occurs at the call boundary. Exact equality merges
/// duplicates, retaining earliest representations. Ordering/degeneracies follow
/// GrahamAnalysis. O(n log n) comparisons and O(n) scan tests; each exact sign has
/// bounded fixed-integer cost for binary64. No tolerance or squared-distance
/// tie breaking: collinear pivot rays are sorted by increasing y then x.
/// Handles all finite binary64 coordinates, including subnormals and large values.
/// No global state; independent calls may run concurrently. No optional limits or
/// history allocation; near-degenerate results can differ from the old rounded
/// cross-product implementation, which did not ensure a valid angular ordering.
/// @param points Finite planar records, copied by value and not retained.
/// @return Owned unclosed strict hull of represented coordinates.
/// @throws std::invalid_argument If any coordinate is nonfinite.
/// @throws std::length_error On work/container exhaustion.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto h=dm::geometry::graham_scan({{0,0},{2,0},{2,2},{0,2},{1,1}});
/// std::print("{}\n",h.size()); // 4
/// @endcode
/// @par References
/// - R. L. Graham (1972), An Efficient Algorithm for Determining the Convex Hull
///   of a Finite Planar Set, Information Processing Letters 1, 132–133,
///   https://doi.org/10.1016/0020-0190(72)90045-2 .
/// @see orientation_sign(), graham_scan_analyze(), graham_scan_trace()
std::vector<Point2D> graham_scan(std::vector<Point2D> points);
/// @brief Compute the same exact hull and add selected orientation requests to a counter.
/// Adds angular_tests+sweep_tests only after successful completion. Initial
/// lexicographic sort, deduplication, pivot checks, copying and validation are
/// excluded. The signed initial counter may be negative; it is unchanged on
/// failure, and overflow throws. Other semantics match graham_scan(points).
/// @param points Finite planar records, copied by value.
/// @param orientation_tests Borrowed accumulating counter, updated only on success.
/// @return Owned strict hull, identical to the uncounted overload.
/// @throws std::invalid_argument On nonfinite coordinates.
/// @throws std::length_error On work/container exhaustion.
/// @throws std::overflow_error If the selected total or counter addition exceeds long long.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// long long n=7; const auto h=dm::geometry::graham_scan({{0,0},{1,0}},n);
/// std::print("{} {}\n",h.size(),n); // 2 7
/// @endcode
/// @see graham_scan_analyze()
std::vector<Point2D> graham_scan(std::vector<Point2D> points, long long& orientation_tests);
/// @brief Execute the same exact-sign Graham kernel with source identities and actual work.
/// Full finite validation precedes max_points and internal indexed allocation.
/// max_points counts original records including duplicates, defaults one million;
/// zero disables it. max_events is ignored without capture. O(n) working/output
/// storage; no event history or input references are retained.
/// @param points Borrowed finite planar records, unmodified.
/// @param options Shared hull limits; default one million original points.
/// @return Owned result, identities and selected native work.
/// @throws std::invalid_argument For any nonfinite coordinate before point-cap checks.
/// @throws std::length_error On point cap or work/container exhaustion.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto a=dm::geometry::graham_scan_analyze({{0,0},{1,0},{2,0}});
/// std::print("{} {}\n",a.hull.size(),a.work.pops); // 2 1
/// @endcode
/// @see GrahamAnalysis, ConvexHullOptions, graham_scan()
GrahamAnalysis graham_scan_analyze(
    const std::vector<Point2D>& points,
    ConvexHullOptions           options = {});
/// @brief Capture actual shared-kernel stack events with online whole-record admission.
/// Same exact sorting, output and work as analyze; O(n) ordered records and edits
/// add no orientation calls. max_events counts test/pop/push records only, with
/// default one million and zero disabling the policy cap. Admission precedes the
/// corresponding orientation call or stack mutation. Initial sorting/normalization
/// and ordinary work/output are excluded; limits are not bytes or peak RAM.
/// No replay, partial trace, truncation or retained input references.
/// @param points Borrowed finite input, unmodified.
/// @param options Complete point and online event caps, each zero unlimited.
/// @return Owned actual unique order, result and chronological edits.
/// @throws std::invalid_argument For any nonfinite coordinate before caps.
/// @throws std::length_error On point/event caps or work/container exhaustion.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::geometry::graham_scan_trace({{0,0},{1,0},{0,1}});
/// std::print("{}\n",t.analysis.work.pushes); // 3
/// @endcode
/// @see GrahamTrace, graham_scan_analyze()
GrahamTrace graham_scan_trace(const std::vector<Point2D>& points, ConvexHullOptions options = {});
} // namespace datamunge::geometry
