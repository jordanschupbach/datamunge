#pragma once
/// @file prim.hpp
/// @brief Validated lazy-heap Prim with actual frontier events and selected work.
#include <datamunge/algorithms/mst_result.hpp>
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Selected-event and retained-event limits for native Prim.
/// Zero independently disables either policy; input validation and adjacency construction precede selected event counting.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::PrimOptions{}.max_operations); // 10000000
/// @endcode
struct PrimOptions {
  std::size_t max_operations{
      10000000}; ///< Actual adjacency inspections plus heap pushes/pops; excludes validation and comparisons inside the heap.
  std::size_t max_trace_records{
      1000000}; ///< Actual pushed/popped frontier records retained; not bytes.
};
/// @brief One actual lazy-frontier insertion or removal.
/// A removal is accepted only when its target had not joined; accepted removals are the actual emitted edges.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::prim_trace(2,{{0,1,2}}).events.back().accepted); // true
/// @endcode
struct PrimEvent {
  bool        push;           ///< True for a push; false for an actual heap pop.
  std::size_t from;           ///< Actual in-tree endpoint at insertion.
  std::size_t to;             ///< Actual candidate target, possibly already joined by pop time.
  double      weight;         ///< Actual candidate edge weight.
  std::size_t frontier_after; ///< Actual heap size immediately after the push/pop.
  bool        accepted;       ///< True only when this popped edge was emitted.
  std::size_t reached_after;  ///< Actual number of joined vertices after this event.
};
/// @brief Actual minimum tree of the start component and selected producer work.
/// Disconnected inputs return one component tree, not a whole-graph forest.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::prim_run(2,{{0,1,2}}).tree.total_weight); // 2
/// @endcode
struct PrimRun {
  MinimumSpanningTree tree; ///< Actual emitted tree, weight and complete-graph connectivity flag.
  std::size_t         reached{0}; ///< Actual vertices joined, including start.
  std::size_t         pushes{0};  ///< Actual heap insertions, including seed self-loop candidates.
  std::size_t         pops{0};    ///< Actual heap removals, including stale entries.
  std::size_t         stale{0};   ///< Actual removals whose target was already joined.
  std::size_t         adjacency_entries{
      0};            ///< Actual adjacency entries inspected at seed and accepted vertices.
  std::size_t operations{0}; ///< Sum of pushes,pops and adjacency_entries.
};
/// @brief Owned ordinary Prim result and its actual lazy-frontier events.
/// Capture observes the producing heap; it does not restart the search to reconstruct selected cuts.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::prim_trace(2,{{0,1,2}}).events.size()); // 2
/// @endcode
struct PrimTrace {
  PrimRun     run;      ///< Actual component tree and selected work.
  std::size_t start{0}; ///< Actual initially joined vertex.
  std::vector<PrimEvent>
      events; ///< Actual heap push/pop order with immediate post-event state sizes.
};
/// @brief Grow the native lazy-heap minimum tree from a chosen start.
/// Builds undirected adjacency in input order; permits finite negative weights, parallel edges and self-loops. The min-heap uses lexicographic(weight,from,to) ties and lazy deletion. Only start's component is spanned; complete connectivity is reported. Complete geometry/finite validation precedes heap operations. With parallel edges the ordinary bound is O(E log(E+1)+V), O(V+E) storage, rather than assuming a simple graph. Inputs are borrowed unchanged and state is local. Default selected-operation policy is ten million events.
/// @param n Positive vertex count; zero has no valid start.
/// @param edges Borrowed undirected endpoint/finite-weight tuples.
/// @param start Initial vertex<n, default0.
/// @return Owned actual component tree.
/// @throws std::invalid_argument If start/endpoints are out of range or any weight is nonfinite.
/// @throws std::length_error If a selected-operation/history policy or representable storage is exceeded.
/// @throws std::overflow_error If an actual total-weight addition is nonfinite.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::prim(2,{{0,1,2}}).total_weight); // 2
/// @endcode
/// @par References
/// - R. C. Prim (1957), Shortest connection networks and some generalizations, Bell System Technical Journal36,1389–1401, doi:10.1002/j.1538-7305.1957.tb01515.x.
MinimumSpanningTree
prim(std::size_t n, const std::vector<WeightedEdge>& edges, std::size_t start = 0);
/// @brief Grow the native lazy-heap minimum tree from a chosen start.
/// Builds undirected adjacency in input order; permits finite negative weights, parallel edges and self-loops. The min-heap uses lexicographic(weight,from,to) ties and lazy deletion. Only start's component is spanned; complete connectivity is reported. Complete geometry/finite validation precedes heap operations. With parallel edges the ordinary bound is O(E log(E+1)+V), O(V+E) storage, rather than assuming a simple graph. Inputs are borrowed unchanged and state is local. Default selected-operation policy is ten million events.
/// @param n Positive vertex count; zero has no valid start.
/// @param edges Borrowed undirected endpoint/finite-weight tuples.
/// @param start Initial vertex<n, default0.
/// @param options Selected-event and retained-record policies; no history is kept by prim_run.
/// @return Owned actual component tree and selected work.
/// @throws std::invalid_argument If start/endpoints are out of range or any weight is nonfinite.
/// @throws std::length_error If a selected-operation/history policy or representable storage is exceeded.
/// @throws std::overflow_error If an actual total-weight addition is nonfinite.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::prim_run(2,{{0,1,2}}).reached); // 2
/// @endcode
/// @par References
/// - R. C. Prim (1957), Shortest connection networks and some generalizations, Bell System Technical Journal36,1389–1401, doi:10.1002/j.1538-7305.1957.tb01515.x.
PrimRun prim_run(
    std::size_t                      n,
    const std::vector<WeightedEdge>& edges,
    std::size_t                      start   = 0,
    PrimOptions                      options = {});
/// @brief Grow the native lazy-heap minimum tree from a chosen start.
/// Builds undirected adjacency in input order; permits finite negative weights, parallel edges and self-loops. The min-heap uses lexicographic(weight,from,to) ties and lazy deletion. Only start's component is spanned; complete connectivity is reported. Complete geometry/finite validation precedes heap operations. With parallel edges the ordinary bound is O(E log(E+1)+V), O(V+E) storage, rather than assuming a simple graph. Inputs are borrowed unchanged and state is local. Default selected-operation policy is ten million events.
/// @param n Positive vertex count; zero has no valid start.
/// @param edges Borrowed undirected endpoint/finite-weight tuples.
/// @param start Initial vertex<n, default0.
/// @param options Selected-event and retained-record policies; no history is kept by prim_run.
/// @return Owned actual component tree, selected work and producing event history.
/// @throws std::invalid_argument If start/endpoints are out of range or any weight is nonfinite.
/// @throws std::length_error If a selected-operation/history policy or representable storage is exceeded.
/// @throws std::overflow_error If an actual total-weight addition is nonfinite.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::prim_trace(2,{{0,1,2}}).run.reached); // 2
/// @endcode
/// @par References
/// - R. C. Prim (1957), Shortest connection networks and some generalizations, Bell System Technical Journal36,1389–1401, doi:10.1002/j.1538-7305.1957.tb01515.x.
PrimTrace prim_trace(
    std::size_t                      n,
    const std::vector<WeightedEdge>& edges,
    std::size_t                      start   = 0,
    PrimOptions                      options = {});
} // namespace datamunge::algorithms
