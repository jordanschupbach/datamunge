#pragma once
/// @file graph_coloring.hpp
/// @brief Degree-ordered first-fit coloring with loop-deletion semantics and actual decision
/// records.
#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::algorithms {
/// @brief Owned legacy greedy labels and contiguous palette size.
/// Empty graphs use no colors; each nonempty edgeless graph uses color zero.
/// The output is proper for the graph after deleting loops, not an optimality
/// certificate or a proper coloring of an original graph containing a loop.
/// @par Example
/// @code
/// const auto c=dm::algorithms::greedy_coloring(2,{{0,1}});
/// std::print("{} {}\n",c.color[1],c.num_colors); // 1 2
/// @endcode
/// @see greedy_coloring(), is_proper_coloring()
struct ColoringResult {
  std::vector<std::size_t>
              color;         ///< Actual labels by vertex, contiguous from zero in produced results.
  std::size_t num_colors{0}; ///< Actual distinct palette size; zero only for empty output.
};
/// @brief Complete input caps and online actual-event cap for analysis/capture.
/// All endpoints are validated before caps or adjacency allocation. Vertices
/// include isolates; edges count original records, including loops/duplicates.
/// Event units are whole neighbor/probe/assignment records, not bytes or peak RAM.
/// @par Example
/// @code
/// const dm::algorithms::ColoringOptions o;
/// std::print("{}\n",o.max_events); // 1000000
/// @endcode
/// @see greedy_coloring_trace()
struct ColoringOptions {
  std::size_t max_vertices{1000000}; ///< Complete vertex cap; zero disables policy limit.
  std::size_t max_edges{1000000};    ///< Original edge-record cap; zero disables policy limit.
  std::size_t max_events{
      1000000}; ///< Actual whole trace-event cap; zero disables it; ignored without capture.
};
/// @brief Selected actual preparation and first-fit work, excluding validation and CPU costs.
/// Degrees and adjacency entries include parallel edges, but not loops.
/// Marker initialization is measured in logical slots; writes count every actual
/// assignment of true, including repeated writes from duplicate edges.
/// @par Example
/// @code
/// const auto a=dm::algorithms::greedy_coloring_analyze(2,{{0,1}});
/// std::print("{} {}\n",a.work.neighbor_visits,a.work.assignments); // 2 2
/// @endcode
/// @see ColoringAnalysis
struct ColoringWork {
  std::size_t vertices{0};      ///< Original vertex count.
  std::size_t edge_records{0};  ///< Original edge records, including loops and duplicates.
  std::size_t ignored_loops{0}; ///< Actual skipped self-loop records.
  std::size_t adjacency_entries{
      0}; ///< Actual adjacency entries appended, two per nonloop edge record.
  std::size_t sort_comparisons{0}; ///< Actual degree/index comparator calls; library-dependent.
  std::size_t marker_slots{0};     ///< Actual false-initialized availability slots, sum(degree+1).
  std::size_t neighbor_visits{0};  ///< Actual adjacency entries inspected during the sweep.
  std::size_t marker_writes{0};    ///< Actual true writes from already-colored in-range neighbors.
  std::size_t color_probes{0};     ///< Actual availability reads, including the first free color.
  std::size_t assignments{0};      ///< Actual completed vertex assignments.
  std::size_t max_degree{0};       ///< Maximum adjacency-list length, counting parallel edges.
};
/// @brief Select an actual neighbor inspection, availability probe or completed assignment.
/// @par Example
/// @code
/// const auto a=dm::algorithms::ColoringAction::probe;
/// std::print("{}\n",a==dm::algorithms::ColoringAction::probe); // true
/// @endcode
/// @see ColoringEvent
enum class ColoringAction {
  neighbor, ///< Inspect one adjacency entry, possibly uncolored.
  probe,    ///< Read one availability marker, including the first free one.
  assign    ///< Store the chosen vertex label.
};
/// @brief One actual constant-size first-fit event, without marker-array snapshots.
/// vertex is always the vertex being processed. For neighbor events, neighbor
/// is the actual adjacent vertex; colored states whether its label exists, color
/// is that label or zero placeholder, and flag states whether a true marker write
/// occurred. For probes color is the tested label and flag its blocked value.
/// For assignments color is the committed label. Other fields are placeholders.
/// @par Example
/// @code
/// const auto t=dm::algorithms::greedy_coloring_trace(1,{});
/// std::print("{}\n",t.events.back().color); // 0
/// @endcode
/// @see ColoringTrace
struct ColoringEvent {
  ColoringAction action{};    ///< Actual producing operation.
  std::size_t    vertex{0};   ///< Current vertex ID.
  std::size_t    neighbor{0}; ///< Inspected adjacent ID for neighbor events; otherwise zero.
  std::size_t    color{0}; ///< Actual label/probe, or zero placeholder for an uncolored neighbor.
  bool           colored{false}; ///< Neighbor already colored; false placeholder for other actions.
  bool flag{false}; ///< Actual marker write on neighbor events, blocked marker on probes; otherwise
                    ///< false.
};
/// @brief Owned producing result, actual degree ordering and selected work, without event history.
/// order sorts multigraph degree descending, then vertex ID ascending. Degrees
/// exclude self-loops. Inputs are not retained. Copying owns all vector storage.
/// @par Example
/// @code
/// const auto a=dm::algorithms::greedy_coloring_analyze(3,{{0,1},{1,2}});
/// std::print("{}\n",a.order[0]); // 1
/// @endcode
/// @see greedy_coloring_analyze()
struct ColoringAnalysis {
  ColoringResult           result; ///< Actual produced colors and contiguous palette size.
  std::vector<std::size_t> degree; ///< Actual adjacency lengths, aligned with vertex IDs.
  std::vector<std::size_t> order;  ///< Actual processing order, each vertex exactly once.
  ColoringWork             work;   ///< Selected actual preparation and sweep operations.
};
/// @brief Owned actual preparation, output and chronological constant-size sweep events.
/// Capturing performs no replay or extra degree sorting. Online admission occurs
/// before each neighbor read/marker write, availability read or assignment.
/// Default cap is one million records; zero disables it. Preparation, ordinary
/// output/work, capacities and allocation overhead are excluded. No truncation or
/// partial trace escapes failure. History is O(n+m) records, not full marker arrays.
/// @par Example
/// @code
/// const auto t=dm::algorithms::greedy_coloring_trace(1,{});
/// std::print("{}\n",t.events.size()); // 2
/// @endcode
/// @see greedy_coloring_trace(), ColoringOptions
struct ColoringTrace {
  ColoringAnalysis analysis; ///< Actual producing result, order, degrees and work.
  std::vector<ColoringEvent>
      events; ///< Owned chronological whole neighbor/probe/assignment records.
};
/// @brief First-fit vertex coloring in descending-degree, ascending-ID order.
/// Undirected adjacency preserves input edge order and parallel entries. Endpoints
/// must be valid, including on ignored loops. Self-loops are deleted: the result
/// is proper for that projected loopless multigraph, not the original looped
/// graph. Duplicate edges do not add coloring constraints but CAN change the
/// degree order, chosen labels and palette size. The heuristic is not optimal.
/// Empty input uses zero colors; a nonempty isolate set uses one. Legacy signature
/// and ordinary labels are preserved. Full endpoint validation precedes internal
/// adjacency allocation. No optional caps, retained inputs or event history.
/// O(n log n+m) selected work, O(n+m) working/output space, where m counts nonloop
/// records; input validation also visits loops. Independent calls share no state.
/// @param n Vertex count, valid IDs 0..n-1.
/// @param edges Borrowed undirected records, not mutated or retained.
/// @return Owned actual first-fit labels, contiguous from zero.
/// @throws std::invalid_argument For any endpoint outside [0,n).
/// @throws std::length_error On checked work or container exhaustion.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto c=dm::algorithms::greedy_coloring(3,{{0,1},{1,2},{2,0}});
/// std::print("{}\n",c.num_colors); // 3
/// @endcode
/// @par References
/// - D. J. A. Welsh, M. B. Powell (1967), An upper bound for the chromatic number
///   of a graph and its application to timetabling problems,
///   https://doi.org/10.1093/comjnl/10.1.85 . Degree-ordered first-fit convention here.
/// @see greedy_coloring_analyze(), greedy_coloring_trace(), is_proper_coloring()
ColoringResult greedy_coloring(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges);
/// @brief Execute the same native first-fit kernel with actual preparation and work.
/// Same loop-deletion/parallel-edge/order/output contract as greedy_coloring().
/// Full endpoint validation precedes complete vertex/edge caps and adjacency
/// allocation. Zero disables either cap. No events; max_events is ignored.
/// @param n Complete vertex count, including isolates.
/// @param edges Original records, including loops/duplicates, borrowed only during call.
/// @param options Complete input caps, each default one million.
/// @return Owned producing result, degree/order vectors and selected actual work.
/// @throws std::invalid_argument On any invalid endpoint, before caps/allocation.
/// @throws std::length_error On input caps, checked counts or container limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto a=dm::algorithms::greedy_coloring_analyze(3,{{0,1},{1,2}});
/// std::print("{}\n",a.work.adjacency_entries); // 4
/// @endcode
/// @see ColoringAnalysis, ColoringOptions
ColoringAnalysis greedy_coloring_analyze(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    ColoringOptions                                         options = {});
/// @brief Capture shared-kernel actual neighbor, probe and assignment events online.
/// Same result/work/preparation as analyze, with O(n+m) optional owned history.
/// Whole-record admission and exclusions follow ColoringTrace. No replay, input
/// retention, marker snapshots, truncation or partial result on failure.
/// @param n Complete vertex count.
/// @param edges Borrowed undirected input records, unmodified.
/// @param options Complete input caps and online event cap; defaults one million each.
/// @return Owned actual producing preparation, result and complete sweep history.
/// @throws std::invalid_argument On any invalid endpoint before caps/allocation.
/// @throws std::length_error On input/event caps, counters or container exhaustion.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::greedy_coloring_trace(2,{{0,1}});
/// std::print("{}\n",t.analysis.work.color_probes); // 3
/// @endcode
/// @see ColoringTrace, greedy_coloring_analyze()
ColoringTrace greedy_coloring_trace(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    ColoringOptions                                         options = {});
/// @brief Check unequal labels across nonloop edges, with the same loop-deletion policy.
/// Returns false for wrong vector size, any invalid endpoint or a monochromatic
/// nonloop edge. Any size_t labels are allowed, including gaps and SIZE_MAX; this
/// does not certify contiguous labels, a palette count, optimality, or validity
/// for an original graph containing loops. No allocation/mutation or retained
/// references. O(numeric size check+edge records), constant auxiliary storage.
/// @param n Vertex count.
/// @param edges Borrowed undirected original records; loops ignored only after range checks.
/// @param color Borrowed labels indexed by vertex, length must equal n.
/// @return Whether all endpoints are valid and all nonloop constraints are satisfied.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::is_proper_coloring(1,{{0,0}},{7})); // true
/// @endcode
/// @see greedy_coloring()
[[nodiscard]] bool is_proper_coloring(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    const std::vector<std::size_t>&                         color);
} // namespace datamunge::algorithms
