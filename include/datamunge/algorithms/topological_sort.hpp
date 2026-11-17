#pragma once
/// @file topological_sort.hpp
/// @brief Actual checked Kahn queue emissions, indegrees and partial cycle diagnostics.
#include <cstddef>
#include <string>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Own TopologicalSortResult values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TopologicalSortResult r{};
/// (void)r;
/// @endcode
struct TopologicalSortResult {
  std::vector<std::size_t>
       order; ///< Actual complete legal order on DAGs; empty on cyclic input for compatibility.
  bool is_dag{true}; ///< Actual all vertices emitted; empty graph is a DAG.
};
/// @brief Own TopologicalOptions values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TopologicalOptions r{};
/// (void)r;
/// @endcode
struct TopologicalOptions {
  std::size_t max_vertices{1000000}; ///< Complete vertex geometry; zero disables.
  std::size_t max_edges{
      20000000}; ///< Complete directed edge entries, duplicates included; zero disables.
  std::size_t max_records{1000000}; ///< Actual captured queue/edge events; zero disables.
};
/// @brief Own TopologicalStep values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TopologicalStep r{};
/// (void)r;
/// @endcode
struct TopologicalStep {
  std::string action;   ///< Actual source, emit, edge or cycle event.
  std::size_t vertex{}; ///< Actual source/emitted/edge-tail vertex; n sentinel for cycle.
  std::size_t other{};  ///< Actual edge successor, n when unused.
  std::size_t
      degree_before{}; ///< Actual successor indegree before an edge decrement; zero for other events.
  std::size_t degree_after{}; ///< Actual successor indegree after decrement; zero for other events.
  std::size_t ready{};        ///< Actual queue size after this event.
  std::size_t emitted{};      ///< Actual cumulative emitted vertices after this event.
};
/// @brief Own TopologicalRun values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TopologicalRun r{};
/// (void)r;
/// @endcode
struct TopologicalRun {
  TopologicalSortResult result; ///< Actual compatibility order/status.
  std::vector<std::size_t>
      emitted; ///< Actual partial or complete FIFO emission order, retained on cycles.
  std::vector<std::size_t>
      positions; ///< Actual emission position per vertex, n sentinel if never emitted.
  std::vector<std::size_t>
      levels; ///< Actual longest-path lower bound propagated from emitted sources; exact DAG levels on acyclic input.
  std::vector<std::size_t> initial_indegree; ///< Actual complete seeded incoming-edge counts.
  std::vector<std::size_t>
      remaining_indegree; ///< Actual remaining indegrees after queue exhaustion.
  std::size_t
      edge_visits{}; ///< Actual successor decrements executed, not necessarily all edges on cycles.
  std::size_t                  enqueues{}; ///< Actual source/new-ready queue pushes.
  std::vector<TopologicalStep> steps;      ///< Optional actual emitting and decrementing events.
};
/// @brief Sort a DAG through the actual checked Kahn FIFO producer.
/// Validate all endpoints and complete vertex/edge policies before allocating adjacency. Count duplicate edges separately. Seed FIFO sources in increasing vertex order, then visit outgoing edges in input order and enqueue vertices when indegree becomes0. Deterministic for a fixed edge list, but not globally lexicographically minimal and may change after reordering same edges. Each vertex/edge processed at most once, O(V+E) work/storage. If fewer than n vertices emit, a directed cycle exists; blocked leftovers can also be descendants of a cycle, not solely cycle members. Compatibility order is cleared on failure while run.emitted retains actual partial progress. Empty graph is a DAG; self loops are cycles. Owned output, no shared state.
/// @param n Complete vertex count.
/// @param edges Borrowed complete directed edge entries.
/// @param options Complete graph and actual capture budgets; zero disables.
/// @return Owned actual legal/partial emissions, positions/levels/indegrees, work and optional states.
/// @throws std::invalid_argument Complete edge endpoint outside0..n-1.
/// @throws std::length_error Complete geometry or actual history/counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::topological_sort_run(3,{{0,1},{1,2}});
/// std::print("{}\n",r.result.order); // [0, 1, 2]
/// @endcode
/// @par References
/// - Kahn, Topological Sorting of Large Networks, CACM5 (1962),558-562.
/// @see topological_sort()
TopologicalRun topological_sort_run(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    const TopologicalOptions&                               options = {});
/// @brief Sort a DAG through the actual checked Kahn FIFO producer.
/// Validate all endpoints and complete vertex/edge policies before allocating adjacency. Count duplicate edges separately. Seed FIFO sources in increasing vertex order, then visit outgoing edges in input order and enqueue vertices when indegree becomes0. Deterministic for a fixed edge list, but not globally lexicographically minimal and may change after reordering same edges. Each vertex/edge processed at most once, O(V+E) work/storage. If fewer than n vertices emit, a directed cycle exists; blocked leftovers can also be descendants of a cycle, not solely cycle members. Compatibility order is cleared on failure while run.emitted retains actual partial progress. Empty graph is a DAG; self loops are cycles. Owned output, no shared state.
/// @param n Complete vertex count.
/// @param edges Borrowed complete directed edge entries.
/// @param options Complete graph and actual capture budgets; zero disables.
/// @return Owned actual legal/partial emissions, positions/levels/indegrees, work and optional states.
/// @throws std::invalid_argument Complete edge endpoint outside0..n-1.
/// @throws std::length_error Complete geometry or actual history/counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::topological_sort_trace(3,{{0,1},{1,2}});
/// std::print("{}\n",r.result.order); // [0, 1, 2]
/// @endcode
/// @par References
/// - Kahn, Topological Sorting of Large Networks, CACM5 (1962),558-562.
/// @see topological_sort()
TopologicalRun topological_sort_trace(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    const TopologicalOptions&                               options = {});
/// @brief Preserve complete Kahn order or empty cyclic result with checked geometry.
/// Validate all endpoints and complete vertex/edge policies before allocating adjacency. Count duplicate edges separately. Seed FIFO sources in increasing vertex order, then visit outgoing edges in input order and enqueue vertices when indegree becomes0. Deterministic for a fixed edge list, but not globally lexicographically minimal and may change after reordering same edges. Each vertex/edge processed at most once, O(V+E) work/storage. If fewer than n vertices emit, a directed cycle exists; blocked leftovers can also be descendants of a cycle, not solely cycle members. Compatibility order is cleared on failure while run.emitted retains actual partial progress. Empty graph is a DAG; self loops are cycles. Owned output, no shared state.
/// @param n Complete vertex count.
/// @param edges Borrowed complete directed edge entries.
/// @return Owned actual complete order/status.
/// @throws std::invalid_argument Invalid endpoint.
/// @throws std::length_error Default geometry or capacity exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::topological_sort(2,{{0,1}}).order); // [0, 1]
/// @endcode
/// @see topological_sort_run()
TopologicalSortResult topological_sort(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges);
/// @brief Check a proposed order as a complete permutation satisfying every directed edge.
/// Wrong length, duplicate/out-of-range vertex, invalid edge endpoint or backward/self edge yieldsfalse. Valid empty graph/order yieldstrue. O(n+edges) validation time and O(n) positions; no mutation/shared state. This checks a supplied witness and does not run Kahn or impose the producer default budgets.
/// @param n Complete vertex count.
/// @param edges Borrowed complete directed edge entries.
/// @param order Borrowed proposed complete vertex permutation.
/// @return True exactly for a complete legal topological ordering.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::is_topological_order(2,{{0,1}},{0,1})); // true
/// @endcode
/// @see topological_sort()
bool is_topological_order(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    const std::vector<std::size_t>&                         order);
} // namespace datamunge::algorithms
