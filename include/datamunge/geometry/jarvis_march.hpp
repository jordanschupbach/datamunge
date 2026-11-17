#pragma once
/// @file jarvis_march.hpp
/// @brief Exact-sign sorted gift wrapping with original identities and actual candidate events.
#include <datamunge/geometry/convex_hull.hpp>
namespace datamunge::geometry {
/// @brief Selected actual preparation and wrapping work, not time or memory accesses.
/// For u>=3 distinct points and h returned vertices, orientation_tests=h*(u-2)
/// and wrap_steps=h. The fewer-than-three shortcut performs no wrapping tests.
/// Sorting counts depend on input and standard library. Collinear comparisons
/// use sorted index order along a ray, never squared floating distances.
/// @par Example
/// @code
/// const auto a=dm::geometry::jarvis_march_analyze({{0,0},{2,0},{0,2}});
/// std::print("{} {}\n",a.work.wrap_steps,a.work.orientation_tests); // 3 3
/// @endcode
/// @see jarvis_march_analyze()
struct JarvisWork {
  std::size_t input_points{0};          ///< Validated original records, duplicates included.
  std::size_t distinct_points{0};       ///< Exact coordinate classes after preparation.
  std::size_t sort_comparisons{0};      ///< Actual lexicographic record comparisons.
  std::size_t duplicate_comparisons{0}; ///< Actual adjacent coordinate-equality tests.
  std::size_t orientation_tests{0};     ///< Actual exact binary64 determinant-sign calls.
  std::size_t collinear_tests{
      0}; ///< Actual sorted-ray farthest comparisons after a zero determinant.
  std::size_t candidate_updates{0}; ///< Actual incumbent changes, excluding initial seeds.
  std::size_t wrap_steps{0};        ///< Actual complete hull-edge scans, zero for small shortcuts.
};
/// @brief Owned strict unclosed hull, earliest source identities and selected work.
/// Starts at lexicographically least coordinate and runs counterclockwise for
/// noncollinear inputs. Collinear input returns its two extremes; empty/singleton/
/// two-distinct inputs use the lexicographic shortcut. Signed zeros coalesce,
/// keeping the earliest input coordinate representation. No input references.
/// @par Example
/// @code
/// const auto a=dm::geometry::jarvis_march_analyze({{2,0},{0,0},{0,2},{1,1}});
/// std::print("{} {}\n",a.hull.size(),a.input_indices[0]); // 3 1
/// @endcode
/// @see JarvisTrace
struct JarvisAnalysis {
  std::vector<Point2D>     hull; ///< Owned actual strict hull, with no repeated first point.
  std::vector<std::size_t> input_indices; ///< Earliest original identities aligned with hull.
  JarvisWork               work;          ///< Actual selected preparation/wrapping operations.
};
/// @brief Phase of an actual sorted Jarvis wrap operation.
/// Start seeds the candidate and emits the current hull origin; Test inspects
/// one other point; Select commits the chosen next vertex, including closure.
/// @par Example
/// @code
/// const auto t=dm::geometry::jarvis_march_trace({{0,0},{2,0},{0,2}});
/// std::print("{}\n",t.events[0].action==dm::geometry::JarvisAction::Start); // true
/// @endcode
/// @see JarvisEvent
enum class JarvisAction {
  Start, ///< Actual origin emission and cyclic-successor candidate initialization.
  Test,  ///< Actual exact orientation and optional collinear farthest comparison.
  Select ///< Actual completed candidate selection for this edge.
};
/// @brief Scalar actual wrap state using indices into JarvisTrace::sorted.
/// Test contains its actual incumbent before/after inspection and exact sign;
/// farther is meaningful only for a zero turn. Start/Select have examined=0,
/// turn=0 and farther=false placeholders, with equal before/after candidate.
/// No full hull snapshots or search replay; all state is owned scalar metadata.
/// @par Example
/// @code
/// const auto t=dm::geometry::jarvis_march_trace({{0,0},{2,0},{0,2}});
/// std::print("{} {}\n",t.events[0].origin,t.events[0].before); // 0 1
/// @endcode
/// @see JarvisTrace
struct JarvisEvent {
  JarvisAction action{JarvisAction::Start}; ///< Actual operation phase.
  std::size_t  step{0};                     ///< Zero-based actual hull-edge scan.
  std::size_t  origin{0};                   ///< Current emitted hull vertex in sorted storage.
  std::size_t  before{0};                   ///< Incumbent before this operation.
  std::size_t  examined{0};    ///< Actually tested point; placeholder zero outside Test.
  std::size_t  after{0};       ///< Incumbent after this operation.
  int          turn{0};        ///< Exact orientation sign on Test, placeholder zero otherwise.
  bool         farther{false}; ///< Actual same-ray farthest outcome on zero-turn Test only.
};
/// @brief Owned actual lexicographic distinct input and chronological wrapping events.
/// No caller references, repeated-prefix runs or reconstruction of candidate
/// decisions. Copying owns storage. Empty/small shortcuts retain sorted points
/// with no events. With u>=3, exactly u events per actual hull edge are captured.
/// @par Example
/// @code
/// const auto t=dm::geometry::jarvis_march_trace({{0,0},{2,0},{0,2}});
/// std::print("{} {}\n",t.sorted.size(),t.events.size()); // 3 9
/// @endcode
/// @see jarvis_march_trace()
struct JarvisTrace {
  JarvisAnalysis analysis; ///< Actual same-execution hull, identities and selected work.
  std::vector<HullIndexedPoint>
      sorted; ///< Actual lexicographic distinct records with earliest identities.
  std::vector<JarvisEvent> events; ///< Actual Start/Test/Select chronology without truncation.
};
/// @brief Compute the strict hull by sorted exact-sign Jarvis gift wrapping.
/// Validates finite input before internal indexed copying/sorting; the legacy
/// by-value argument copy occurs at the call boundary. Exact equality deduplicates
/// coordinates/signed zeros and retains earliest representations. Output order/
/// degeneracies follow JarvisAnalysis. Each scan starts at the cyclic sorted
/// successor and skips that original seed and origin, preserving the original
/// scan rule. Exact signs replace rounded cross products; collinear same-ray
/// farthest ties use sorted indices, avoiding squared-distance overflow/underflow.
/// A true hull vertex cannot lie between opposite-ray input points. No rounded
/// area reversal is needed. Handles all finite represented binary64 coordinates;
/// near-degenerate results can differ from the former rounded implementation.
/// Total O(n log n+u*h) comparison/predicate work, O(n) workspace/output; sorting
/// is retained, so the complete wrapper is not merely O(n*h). Exact signs have
/// fixed bounded integer cost. No shared state, mutation or retained references.
/// @param points Finite planar records, copied by value.
/// @return Owned unclosed strict hull, counterclockwise when noncollinear.
/// @throws std::invalid_argument On any nonfinite coordinate.
/// @throws std::length_error On work/container exhaustion.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto h=dm::geometry::jarvis_march({{0,0},{2,0},{2,2},{0,2},{1,1}});
/// std::print("{} {} {}\n",h.size(),h[1].x,h[1].y); // 4 2 0
/// @endcode
/// @par References
/// - Jarvis (1973), On the identification of the convex hull of a finite set of points in the plane, Information Processing Letters2(1):18-21: https://doi.org/10.1016/0020-0190(73)90020-3.
/// @see jarvis_march_analyze(), jarvis_march_trace(), orientation_sign()
std::vector<Point2D> jarvis_march(std::vector<Point2D> points);
/// @brief Accumulate actual Jarvis orientation tests only after a successful complete run.
/// Same kernel/order/numerical/ownership/cost policies as jarvis_march(). Adds
/// h*(u-2) when u>=3, zero for the small shortcut; excludes sorting, duplicate,
/// tie comparisons and allocation. Rejects a selected count above LLONG_MAX or
/// accumulated signed overflow; the caller's counter remains unchanged on error.
/// Existing negative initial counter values are permitted.
/// @param points Finite points, legacy by-value copy not retained.
/// @param orientation_tests Mutable signed accumulator, updated only on complete success.
/// @return Owned strict hull with the same ordering as the value-only overload.
/// @throws std::invalid_argument On nonfinite input, counter unchanged.
/// @throws std::length_error On work/container exhaustion.
/// @throws std::overflow_error On selected/accumulated signed count overflow.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// long long count=7;
/// const auto h=dm::geometry::jarvis_march({{0,0},{2,0},{0,2}},count);
/// std::print("{} {}\n",h.size(),count); // 3 10
/// @endcode
/// @see jarvis_march()
std::vector<Point2D> jarvis_march(std::vector<Point2D> points, long long& orientation_tests);
/// @brief Analyze the same exact-sign sorted gift wrap without event history.
/// Borrowed finite input is unchanged and unretained. Full finite validation
/// precedes max_points admission/internal copies. Default point cap million;
/// zero disables it. max_events is ignored. Complete caps count original
/// records, not duplicates after removal, capacity, bytes or peak RAM. Same
/// O(n log n+u*h) work/O(n) storage and numerical policies as jarvis_march().
/// @param points Borrowed finite original records, duplicates allowed.
/// @param options Complete point count and ignored event cap, default ConvexHullOptions.
/// @return Owned actual hull/earliest identities and selected work.
/// @throws std::invalid_argument On nonfinite input before internal copy/sort.
/// @throws std::length_error On enabled complete point cap or work/container exhaustion.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto a=dm::geometry::jarvis_march_analyze({{0,0},{2,0},{2,2},{0,2},{1,1}});
/// std::print("{} {}\n",a.hull.size(),a.work.orientation_tests); // 4 12
/// @endcode
/// @see JarvisAnalysis
JarvisAnalysis jarvis_march_analyze(
    const std::vector<Point2D>& points,
    ConvexHullOptions           options = {});
/// @brief Capture actual candidate selection events from the same producing kernel.
/// Owns actual sorted distinct records and chronological scalar Start/Test/Select
/// events, without re-sorting/replaying. Point admission is complete before
/// copying; event admission is online before each operation. Default million
/// caps, zero disables either; no partial prefix returns on failure. Additional
/// O(u*h) event records, O(n) sorted/output/working storage. Counts exclude
/// scalar summaries, vector capacities and byte/peak RAM; input is not copied
/// into a second original-order history. Same exact-sign policy as jarvis_march().
/// @param points Borrowed complete finite original records.
/// @param options Complete original-point and actual event limits.
/// @return Owned same-execution sorted records, events and actual analysis.
/// @throws std::invalid_argument On nonfinite input before internal copying.
/// @throws std::length_error On enabled cap or work/container exhaustion.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::geometry::jarvis_march_trace({{0,0},{2,0},{0,2}});
/// std::print("{}\n",t.analysis.work.orientation_tests); // 3
/// @endcode
/// @see JarvisTrace
JarvisTrace jarvis_march_trace(const std::vector<Point2D>& points, ConvexHullOptions options = {});
} // namespace datamunge::geometry
