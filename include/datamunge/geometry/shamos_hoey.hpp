#pragma once
/// @file shamos_hoey.hpp
/// @brief Exact-predicate endpoint sweep with a native AVL status and explicit vertical events.
#include <cstddef>
#include <datamunge/geometry/segment_intersection.hpp>
#include <limits>
#include <utility>
#include <vector>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::geometry {
/// @brief Actual endpoint-sweep event category in equal-x processing order.
/// @par Example
/// @code
/// const auto value=dm::geometry::ShamosEventKind::Start;
/// std::print("{}\n",static_cast<int>(value)); // 0
/// @endcode
enum class ShamosEventKind {
  Start,    ///< Insert nonvertical segment, before same-x verticals or removals.
  Vertical, ///< Query a closed vertical interval; never insert it into status.
  End       ///< Remove nonvertical segment after same-x insertions/vertical queries.
};
/// @brief Complete segment, actual predicate and optional event-history bounds; zero disables a cap.
/// @par Example
/// @code
/// const dm::geometry::ShamosOptions value{};
/// std::print("{}\n",value.max_segments); // 1000000
/// @endcode
struct ShamosOptions {
  std::size_t max_segments{
      1000000}; ///< Complete input segment count, also bounded by int node ids.
  std::size_t max_pair_tests{10000000}; ///< Actual complete closed-segment predicate calls.
  std::size_t max_height_tests{
      10000000}; ///< Actual exact orientation calls in AVL height searches.
  std::size_t max_trace_entries{1000000}; ///< Actual event, pair-test and height-test records.
};
/// @brief One actual completed/early-terminal endpoint event and its resulting status geometry.
/// @par Example
/// @code
/// const dm::geometry::ShamosEvent value{};
/// std::print("{}\n",value.active_after); // 0
/// @endcode
struct ShamosEvent {
  double          x{};       ///< Actual represented sweep coordinate.
  ShamosEventKind kind{};    ///< Actual insertion, vertical query or removal.
  std::size_t     segment{}; ///< Original caller segment identity.
  std::size_t
      active_after{}; ///< Actual nonvertical status size after this processing, including early detection.
  int tree_height_after{}; ///< Actual AVL root height, zero when empty.
};
/// @brief One actual exact query-point versus active-segment height comparison.
/// @par Example
/// @code
/// const dm::geometry::ShamosHeightTest value{};
/// std::print("{}\n",value.sign); // 0
/// @endcode
struct ShamosHeightTest {
  std::size_t segment{}; ///< Actual incoming/vertical query identity.
  std::size_t other{};   ///< Actual active segment identity tested.
  int         sign{}; ///< Exact orientation of the active left-to-right segment and query endpoint.
};
/// @brief One actual complete closed-segment test on original identities.
/// @par Example
/// @code
/// const dm::geometry::ShamosPairTest value{};
/// std::print("{}\n",value.intersects); // false
/// @endcode
struct ShamosPairTest {
  std::size_t first{};  ///< Actual first tested original id.
  std::size_t second{}; ///< Actual second tested original id.
  bool intersects{}; ///< Actual canonical exact decision, without approximate point construction.
};
/// @brief Owned actual sweep decision, first discovered witness and selected status/predicate work.
/// @par Example
/// @code
/// const dm::geometry::ShamosRun value{};
/// std::print("{}\n",value.found); // false
/// @endcode
struct ShamosRun {
  bool        found{}; ///< Actual discovered nonempty closed-segment intersection.
  std::size_t first{
      std::numeric_limits<std::size_t>::max()}; ///< Original first witness id, sentinel when none.
  std::size_t second{
      std::numeric_limits<std::size_t>::max()}; ///< Original second witness id, sentinel when none.
  std::size_t events_processed{};        ///< Actual visited endpoint/vertical events before return.
  std::size_t pair_tests{};              ///< Actual full canonical segment predicate calls.
  std::size_t height_tests{};            ///< Actual exact orientation comparisons in tree searches.
  std::size_t rotations{};               ///< Actual single AVL rotations; double balance uses two.
  std::vector<ShamosEvent>      events;  ///< Optional actual completed/terminal events.
  std::vector<ShamosHeightTest> heights; ///< Optional actual search comparisons.
  std::vector<ShamosPairTest>   pairs;   ///< Optional actual witness/neighbor tests.
};
/// @brief Detect any exact represented-coordinate closed-segment intersection.
/// Normalizes nonvertical endpoints left-to-right. Same-x starts precede vertical queries, which precede ends. Native AVL status stores the current nonvertical order; insertion compares the incoming left endpoint with an active line using exact orientation, then tests predecessor/successor pairs. Removal uses direct identity-to-node addresses and tests newly adjacent neighbors. No comparator references a changing sweep coordinate, so there is no mutable std::set ordering or epsilon comparison. A vertical interval queries the first active height at or above its lower endpoint; one complete test decides whether that candidate lies below the upper endpoint. Same-x vertical intervals are sorted by lower y and adjacent intervals tested, supporting point segments and closed endpoint contacts. Complete pair tests use canonical exact binary64 predicates without approximate point construction. Before the first intersection, active order cannot exchange without a tested neighboring pair; orbiting through endpoints therefore detects any contact. O(n log n) bounded-word geometric/search work and O(n) intrinsic storage, optional O(actual events/tests) histories. Returned witness is the first discovered original identity pair, not all pairs or an intersection-count certificate. Complete input/event geometry checked before allocation; actual work/capture admitted before corresponding work. Inputs unchanged, outputs owned; independent calls share no state.
/// @param input Borrowed complete finite closed segments as endpoint pairs; duplicates/point segments allowed.
/// @param options Complete input and actual pair/search/capture limits.
/// @return Owned actual decision, original witness ids, selected actual work and optional producing histories.
/// @throws std::invalid_argument Any nonfinite coordinate.
/// @throws std::length_error Complete segment/event/node or actual work/history capacity/policy exceeded.
/// @par Example
/// @code
/// auto r=dm::geometry::shamos_hoey_run({{{0,0},{2,2}},{{0,2},{2,0}}});
/// std::print("{}\n",r.found); // true
/// @endcode
/// @par References
/// - Shamos and Hoey (1976), Geometric intersection problems, 17th Annual Symposium on Foundations of Computer Science, pp.208-215; native AVL status and explicit closed vertical-event handling here.
/// @see segments_intersect(), any_segments_intersect()
ShamosRun shamos_hoey_run(
    const std::vector<std::pair<Point2D, Point2D>>& input,
    const ShamosOptions&                            options = {});
/// @brief Detect any exact represented-coordinate closed-segment intersection with native event/search/witness histories.
/// Normalizes nonvertical endpoints left-to-right. Same-x starts precede vertical queries, which precede ends. Native AVL status stores the current nonvertical order; insertion compares the incoming left endpoint with an active line using exact orientation, then tests predecessor/successor pairs. Removal uses direct identity-to-node addresses and tests newly adjacent neighbors. No comparator references a changing sweep coordinate, so there is no mutable std::set ordering or epsilon comparison. A vertical interval queries the first active height at or above its lower endpoint; one complete test decides whether that candidate lies below the upper endpoint. Same-x vertical intervals are sorted by lower y and adjacent intervals tested, supporting point segments and closed endpoint contacts. Complete pair tests use canonical exact binary64 predicates without approximate point construction. Before the first intersection, active order cannot exchange without a tested neighboring pair; orbiting through endpoints therefore detects any contact. O(n log n) bounded-word geometric/search work and O(n) intrinsic storage, optional O(actual events/tests) histories. Returned witness is the first discovered original identity pair, not all pairs or an intersection-count certificate. Complete input/event geometry checked before allocation; actual work/capture admitted before corresponding work. Inputs unchanged, outputs owned; independent calls share no state.
/// @param input Borrowed complete finite closed segments as endpoint pairs; duplicates/point segments allowed.
/// @param options Complete input and actual pair/search/capture limits.
/// @return Owned actual decision, original witness ids, selected actual work and optional producing histories.
/// @throws std::invalid_argument Any nonfinite coordinate.
/// @throws std::length_error Complete segment/event/node or actual work/history capacity/policy exceeded.
/// @par Example
/// @code
/// auto r=dm::geometry::shamos_hoey_trace({{{0,0},{2,2}},{{0,2},{2,0}}});
/// std::print("{}\n",r.found); // true
/// @endcode
/// @par References
/// - Shamos and Hoey (1976), Geometric intersection problems, 17th Annual Symposium on Foundations of Computer Science, pp.208-215; native AVL status and explicit closed vertical-event handling here.
/// @see segments_intersect(), any_segments_intersect()
ShamosRun shamos_hoey_trace(
    const std::vector<std::pair<Point2D, Point2D>>& input,
    const ShamosOptions&                            options = {});
/// @brief Return the exact endpoint-sweep decision through the preserved signature.
/// Uses the same native AVL producer/default bounds without optional retained events. Includes vertical, point, overlapping and endpoint contacts; fixes unsafe mutable-comparator ordering and epsilon ties in the historical status set.
/// @param input Complete borrowed finite closed segments.
/// @return True exactly when some pair of distinct original segments intersects.
/// @throws std::invalid_argument Nonfinite coordinate.
/// @throws std::length_error Default complete/actual work capacity or policy exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::any_segments_intersect({{{0,0},{2,0}},{{0,1},{2,1}}})); // false
/// @endcode
/// @par References
/// - Shamos and Hoey (1976), Geometric intersection problems, 17th Annual Symposium on Foundations of Computer Science, pp.208-215; native AVL status and explicit closed vertical-event handling here.
/// @see shamos_hoey_run()
bool any_segments_intersect(const std::vector<std::pair<Point2D, Point2D>>& input);
/// @brief Construct controlled disjoint horizontal sweep segments with common x support.
/// @param count Complete requested segments, default16.
/// @param gap Finite positive vertical spacing, default1.
/// @param length Finite positive common x length, default10.
/// @param max_segments Complete input cap, default1000000; zero disables.
/// @return Owned segments (0,i*gap) to (length,i*gap).
/// @throws std::invalid_argument Nonpositive/nonfinite gap or length.
/// @throws std::length_error Complete input/capacity exceeded.
/// @throws std::overflow_error A distinct finite y coordinate cannot be represented.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::shamos_horizontal_segments(3).size()); // 3
/// @endcode
/// @see shamos_hoey_run()
std::vector<std::pair<Point2D, Point2D>> shamos_horizontal_segments(
    std::size_t count        = 16,
    double      gap          = 1,
    double      length       = 10,
    std::size_t max_segments = 1000000);
} // namespace datamunge::geometry
