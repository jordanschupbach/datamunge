#pragma once
/// @file convex_hull.hpp
/// @brief Exact-sign Andrew monotone chain with source identities and actual stack edits.
#include <datamunge/geometry/orientation.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::geometry {
/// @brief Limits for complete input storage and actual optional hull trace events.
/// Full finite validation precedes max_points preflight; max_events is an online
/// count of actual test/pop/push records, ignored by untraced calls. Zero disables
/// either cap. No partial result escapes a failed cap.
/// @par Example
/// @code
/// dm::geometry::ConvexHullOptions o;
/// std::cout << o.max_points << ' ' << o.max_events; // 1000000 1000000
/// @endcode
/// @see convex_hull_trace()
struct ConvexHullOptions {
    std::size_t max_points{1000000}; ///< Complete original-record cap, including duplicates.
    std::size_t max_events{1000000}; ///< Actual trace-record cap.
};
/// @brief Owned distinct coordinate with the earliest matching input identity.
/// Exact equality coalesces duplicates and signed zeros; sorting ties by original
/// index retains the earliest coordinate representation. Output owns the point.
/// @par Example
/// @code
/// const auto t=dm::geometry::convex_hull_trace({{1,0},{0,0},{1,0}});
/// std::cout << t.sorted[1].input_index; // 0
/// @endcode
/// @see ConvexHullTrace
struct HullIndexedPoint {
    Point2D point;                 ///< Copied original coordinate.
    std::size_t input_index{};     ///< Earliest matching original record index.
};
/// @brief Actual sorting, deduplication and two-scan work, with checked counters.
/// Comparison counts depend on the standard-library sort and original order.
/// For at least two distinct points each pass pushes each point once. Tests
/// count exact orientation requests; every nonpositive test causes one pop.
/// max_stack is the greatest length of either separate chain, not their sum.
/// @par Example
/// @code
/// const auto a=dm::geometry::convex_hull_analyze({{0,0},{1,0},{0,1}});
/// std::cout << a.work.distinct_points << ' ' << a.work.pushes; // 3 6
/// @endcode
/// @see ConvexHullAnalysis
struct ConvexHullWork {
    std::size_t input_points{};          ///< Fully validated original point count.
    std::size_t distinct_points{};       ///< Unique coordinate count after sorting.
    std::size_t sort_comparisons{};      ///< Actual lexicographic record comparisons.
    std::size_t duplicate_comparisons{}; ///< Actual adjacent-equality comparisons.
    std::size_t orientation_tests{};     ///< Exact determinant-sign calls during scans.
    std::size_t pushes{};                ///< Actual separate-chain pushes.
    std::size_t pops{};                  ///< Actual non-left-turn removals.
    std::size_t max_stack{};             ///< Maximum single-chain length reached.
};
/// @brief Select the forward lower or reverse upper monotone-chain scan.
/// @par Example
/// @code
/// std::cout << (dm::geometry::HullPass::lower==dm::geometry::HullPass::lower); // 1
/// @endcode
/// @see HullEvent
enum class HullPass { lower, ///< Left-to-right lower scan.
                       upper  ///< Right-to-left upper scan.
};
/// @brief Actual constant-size stack event; no full stack snapshots are copied.
/// @par Example
/// @code
/// std::cout << (dm::geometry::HullAction::pop==dm::geometry::HullAction::pop); // 1
/// @endcode
/// @see HullEvent
enum class HullAction { test, ///< Exact orientation test; stack unchanged.
                         pop, ///< Remove a non-left-turn top.
                         push ///< Append the current candidate.
};
/// @brief Actual scan event, referring to indices in the trace's sorted unique points.
/// candidate is the pending point; point is the pushed/popped point, or candidate
/// for a test. a,b,turn are meaningful only for test events and otherwise zero.
/// stack_size is the actual size after the event (unchanged for tests). Together
/// these compact edits describe every stack state without quadratic snapshots.
/// @par Example
/// @code
/// const auto t=dm::geometry::convex_hull_trace({{0,0},{1,0}});
/// std::cout << t.events[0].stack_size; // 1
/// @endcode
/// @see ConvexHullTrace, HullPass, HullAction
struct HullEvent {
    HullPass pass{};           ///< Producing scan direction.
    HullAction action{};       ///< Actual test, pop or push.
    std::size_t candidate{};   ///< Pending sorted unique point.
    std::size_t point{};       ///< Actual edited point, or candidate for a test.
    std::size_t a{};           ///< Test's next-to-top point; otherwise zero placeholder.
    std::size_t b{};           ///< Test's top point; otherwise zero placeholder.
    int turn{};               ///< Exact test sign -1,0,+1; otherwise zero placeholder.
    std::size_t stack_size{};  ///< Size immediately after this event.
};
/// @brief Owned strict hull, earliest original identities and actual work without history.
/// Vertices start at the lexicographic minimum and run counterclockwise, without
/// repeating the first. Collinear edge points are omitted. Empty gives no points;
/// one distinct point gives one; a collinear set gives its sorted two endpoints.
/// Identity vector aligns one-for-one with hull vertices.
/// @par Example
/// @code
/// const auto a=dm::geometry::convex_hull_analyze({{1,0},{0,0},{0,1}});
/// std::cout << a.hull.size() << ' ' << a.input_indices[0]; // 3 1
/// @endcode
/// @see convex_hull_analyze()
struct ConvexHullAnalysis {
    std::vector<Point2D> hull;              ///< Owned unclosed strict hull.
    std::vector<std::size_t> input_indices; ///< Earliest original identities in hull order.
    ConvexHullWork work;                   ///< Actual producing operations.
};
/// @brief Owned actual sorted identities, compact stack edits and completed scan chains.
/// lower/upper contain indices into sorted and include their endpoints. Inputs
/// with fewer than two distinct points take no scans and have empty chains/events.
/// Trace capture performs no extra sorting or orientation replay. All storage is
/// independent of the caller, result and other snapshots; O(n) extra space.
/// @par Example
/// @code
/// const auto t=dm::geometry::convex_hull_trace({{0,0},{1,0}});
/// std::cout << t.lower.size() << ' ' << t.upper.size(); // 2 2
/// @endcode
/// @see convex_hull_trace()
struct ConvexHullTrace {
    ConvexHullAnalysis analysis;           ///< Actual hull, identities and work.
    std::vector<HullIndexedPoint> sorted;  ///< Actual sorted distinct input records.
    std::vector<HullEvent> events;         ///< Actual test/pop/push edits.
    std::vector<std::size_t> lower;        ///< Actual final lower stack.
    std::vector<std::size_t> upper;        ///< Actual final upper stack.
};
/// @brief Compute Andrew's strict monotone-chain hull using exact binary64 orientation signs.
/// Validates all coordinates before internal indexed copies or sorting; the
/// original by-value argument copy occurs at the call boundary. Exact coordinate
/// equality removes duplicates (including signed zero); the earliest matching
/// original record supplies its representation. Returns lexicographic-minimum
/// first, counterclockwise, no repeated endpoint and no collinear edge interiors.
/// Empty/singleton/two-point/collinear outputs follow ConvexHullAnalysis. O(n log n)
/// comparisons plus O(n) bounded exact-sign tests, O(n) scratch/output, no global
/// state. The legacy by-value signature is retained and has no optional caps.
/// Other geometry routines that use cross() retain their own floating contracts.
/// @param points Finite planar records, copied by value as in the original API.
/// @return Owned strict hull of the represented coordinates.
/// @throws std::invalid_argument Any nonfinite coordinate.
/// @throws std::length_error Work/container size exhaustion.
/// @par Example
/// @code
/// const auto h=dm::geometry::convex_hull({{0,0},{2,0},{2,2},{0,2},{1,1}});
/// std::cout << h.size(); // 4
/// @endcode
/// @par References
/// - A. M. Andrew (1979), Another efficient algorithm for convex hulls in two
///   dimensions, https://doi.org/10.1016/0020-0190(79)90072-3 .
/// @see convex_hull_analyze(), convex_hull_trace(), orientation_sign()
[[nodiscard]] std::vector<Point2D> convex_hull(std::vector<Point2D> points);
/// @brief Execute the same exact-sign hull kernel with identities and actual work.
/// Same ordering, degeneracy and complexity contract as convex_hull(). Full finite
/// validation precedes the complete max_points cap and scratch allocation. No
/// events or completed-chain snapshots are retained; event cap is ignored.
/// @param points Finite original records; not mutated.
/// @param options Complete original-record cap, zero unlimited; defaults one million.
/// @return Owned hull, earliest original indices and actual counters.
/// @throws std::invalid_argument Nonfinite coordinate.
/// @throws std::length_error Complete point cap or work/container exhaustion.
/// @par Example
/// @code
/// const auto a=dm::geometry::convex_hull_analyze({{0,0},{1,0},{2,0}});
/// std::cout << a.hull.size() << ' ' << a.work.distinct_points; // 2 3
/// @endcode
/// @par References
/// - Andrew (1979), https://doi.org/10.1016/0020-0190(79)90072-3 .
/// @see convex_hull(), ConvexHullOptions
ConvexHullAnalysis convex_hull_analyze(const std::vector<Point2D>& points, ConvexHullOptions options = {});
/// @brief Capture the hull kernel's actual stack events without replay or quadratic snapshots.
/// Same computation/work as analyze, plus O(n) owned sorted records, compact
/// test/pop/push edits and completed lower/upper chains. Positive max_events is
/// checked online on actual records, including pushes; zero disables the cap.
/// @param points Finite original records; not mutated.
/// @param options Complete point and actual event caps.
/// @return Owned actual sorted coordinates, identities, work and scan edits.
/// @throws std::invalid_argument Nonfinite coordinate.
/// @throws std::length_error Point/event cap or work/container exhaustion.
/// @par Example
/// @code
/// const auto t=dm::geometry::convex_hull_trace({{0,0},{1,0}});
/// std::cout << t.events.size(); // 4
/// @endcode
/// @par References
/// - Andrew (1979), https://doi.org/10.1016/0020-0190(79)90072-3 .
/// @see convex_hull_analyze(), HullEvent
ConvexHullTrace convex_hull_trace(const std::vector<Point2D>& points, ConvexHullOptions options = {});
/// @brief Draw unsnapped independent standard-normal x,y pairs with a local 32-bit MT stream.
/// Reproduces the original hull report's mt19937/normal_distribution call order,
/// including its cached normal variate. Distribution mappings can differ between
/// standard libraries; a seed alone does not promise portable coordinate bits.
/// Complete count cap precedes allocation. Empty count is valid. O(count) expected
/// work/storage; no global RNG or mutation. Finite draws are checked before return.
/// @param count Number of points, default forty.
/// @param seed Local mt19937 seed, default 20240728.
/// @param max_points Complete draw cap, default one million; zero unlimited.
/// @return Owned unsnapped normal point cloud.
/// @throws std::length_error Complete draw cap/container exhaustion.
/// @throws std::overflow_error A generated coordinate is nonfinite.
/// @par Example
/// @code
/// std::cout << dm::geometry::sample_gaussian_points(40,20240728).size(); // 40
/// @endcode
/// @par References
/// - C++ normal_distribution specification: https://eel.is/c++draft/rand.dist.norm.normal .
/// @see convex_hull(), random_point_cloud()
std::vector<Point2D> sample_gaussian_points(std::size_t count = 40, std::uint32_t seed = 20240728,
    std::size_t max_points = 1000000);
} // namespace datamunge::geometry
