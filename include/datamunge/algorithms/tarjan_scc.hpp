#pragma once
/// @file tarjan_scc.hpp
/// @brief Checked iterative Tarjan discovery, low-link and actual component-pop histories.
#include <cstddef>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Own the TarjanSCCOptions contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TarjanSCCOptions r{};
/// (void)r;
/// @endcode
/// @see tarjan_scc_run()
struct TarjanSCCOptions {
  std::size_t max_vertices{1000000}; ///< Complete vertex count; zero disables.
  std::size_t max_edges{
      20000000}; ///< Complete directed adjacency entries, including duplicates/loops; zero disables.
  std::size_t max_records{1000000}; ///< Actual captured state events; zero disables.
};
/// @brief Own the TarjanSCCStep contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TarjanSCCStep r{};
/// (void)r;
/// @endcode
/// @see tarjan_scc_run()
struct TarjanSCCStep {
  std::string action;        ///< Actual discover, edge, back-edge, pop, finish or propagate event.
  int         vertex{};      ///< Actual current event vertex.
  int         other{-1};     ///< Actual edge endpoint/finished child, or-1 without partner.
  int         discovery{};   ///< Actual discovery number of vertex at this event.
  int         low{};         ///< Actual current low-link of vertex.
  int         component{-1}; ///< Actual assigned component, or-1 while unassigned.
  std::size_t active{};      ///< Actual SCC-stack size after this event.
};
/// @brief Own the TarjanSCCRun contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TarjanSCCRun r{};
/// (void)r;
/// @endcode
/// @see tarjan_scc_run()
struct TarjanSCCRun {
  std::vector<int>
      components; ///< Actual pop-order component ID per vertex, preserving legacy traversal IDs.
  std::vector<int> discovery;         ///< Actual increasing discovery number per vertex.
  std::vector<int> low;               ///< Actual final propagated low-link per vertex.
  std::size_t      component_count{}; ///< Actual completed SCC pops.
  std::size_t      discoveries{};     ///< Actual first-visit events, exactly n.
  std::size_t      edge_visits{};     ///< Actual processed directed adjacency entries.
  std::size_t
      active_edge_updates{}; ///< Actual already-discovered on-stack edge low-link updates, including unchanged values.
  std::size_t child_propagations{}; ///< Actual completed DFS child low-link propagations.
  std::size_t peak_frames{};        ///< Actual maximum explicit DFS stack size.
  std::vector<TarjanSCCStep>
      steps; ///< Optional actual state events with no borrowed graph storage.
};
/// @brief Find directed strongly connected components through the actual iterative Tarjan producer.
/// Validate n>=0, exact adjacency row count and every neighbor0..n-1 before traversal. Empty graph returns empty. Directed loops and duplicate entries are retained and counted. Visit roots in increasing order and neighbors in caller order. Explicit-stack DFS avoids recursion depth limits. For a discovered active neighbor use its discovery number, while a finished child propagates its low-link; a node with low==discovery pops one SCC. Component IDs are reverse finishing/pop order, not canonical graph labels; compare partitions when adjacency order changes. O(V+E) work/storage, optional O(actual events) history, independent owned results and no shared state.
/// @param n Complete nonnegative int vertex count.
/// @param adj Borrowed complete directed adjacency rows.
/// @param options Complete graph geometry and actual capture policies; zero disables budgets.
/// @return Owned actual partition, discovery/low arrays, executed counts and optional events.
/// @throws std::invalid_argument Negative count, row-count mismatch or invalid neighbor.
/// @throws std::length_error Complete geometry or actual capture/counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::tarjan_scc_run(2,{{1},{0}});
/// std::print("{}\n",r.component_count); // 1
/// @endcode
/// @par References
/// - Tarjan, Depth-First Search and Linear Graph Algorithms, SIAM Journal on Computing1 (1972),146-160.
/// @see tarjan_scc()
TarjanSCCRun tarjan_scc_run(
    int                                  n,
    const std::vector<std::vector<int>>& adj,
    const TarjanSCCOptions&              options = {});
/// @brief Find directed strongly connected components through the actual iterative Tarjan producer.
/// Validate n>=0, exact adjacency row count and every neighbor0..n-1 before traversal. Empty graph returns empty. Directed loops and duplicate entries are retained and counted. Visit roots in increasing order and neighbors in caller order. Explicit-stack DFS avoids recursion depth limits. For a discovered active neighbor use its discovery number, while a finished child propagates its low-link; a node with low==discovery pops one SCC. Component IDs are reverse finishing/pop order, not canonical graph labels; compare partitions when adjacency order changes. O(V+E) work/storage, optional O(actual events) history, independent owned results and no shared state.
/// @param n Complete nonnegative int vertex count.
/// @param adj Borrowed complete directed adjacency rows.
/// @param options Complete graph geometry and actual capture policies; zero disables budgets.
/// @return Owned actual partition, discovery/low arrays, executed counts and optional events.
/// @throws std::invalid_argument Negative count, row-count mismatch or invalid neighbor.
/// @throws std::length_error Complete geometry or actual capture/counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::tarjan_scc_trace(2,{{1},{0}});
/// std::print("{}\n",r.component_count); // 1
/// @endcode
/// @par References
/// - Tarjan, Depth-First Search and Linear Graph Algorithms, SIAM Journal on Computing1 (1972),146-160.
/// @see tarjan_scc()
TarjanSCCRun tarjan_scc_trace(
    int                                  n,
    const std::vector<std::vector<int>>& adj,
    const TarjanSCCOptions&              options = {});
/// @brief Preserve actual pop-order Tarjan SCC IDs with checked graph inputs.
/// Validate n>=0, exact adjacency row count and every neighbor0..n-1 before traversal. Empty graph returns empty. Directed loops and duplicate entries are retained and counted. Visit roots in increasing order and neighbors in caller order. Explicit-stack DFS avoids recursion depth limits. For a discovered active neighbor use its discovery number, while a finished child propagates its low-link; a node with low==discovery pops one SCC. Component IDs are reverse finishing/pop order, not canonical graph labels; compare partitions when adjacency order changes. O(V+E) work/storage, optional O(actual events) history, independent owned results and no shared state.
/// @param n Complete nonnegative vertex count.
/// @param adj Borrowed directed adjacency rows.
/// @return Owned actual component ID per vertex.
/// @throws std::invalid_argument Malformed count/rows/neighbor.
/// @throws std::length_error Default complete geometry or capacity exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::tarjan_scc(2,{{1},{0}})); // [0, 0]
/// @endcode
/// @see tarjan_scc_run()
std::vector<int> tarjan_scc(int n, const std::vector<std::vector<int>>& adj);
} // namespace datamunge::algorithms
