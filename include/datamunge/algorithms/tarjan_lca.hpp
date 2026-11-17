#pragma once
/// @file tarjan_lca.hpp
/// @brief Checked offline LCA with actual union-find, finish and query-answer events.
#include <cstddef>
#include <string>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Own the TarjanLCAOptions contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TarjanLCAOptions r{};
/// (void)r;
/// @endcode
/// @see tarjan_lca_run()
struct TarjanLCAOptions {
  std::size_t max_vertices{1000000}; ///< Complete vertices; zero disables policy.
  std::size_t max_queries{1000000};  ///< Complete query count; zero disables.
  std::size_t max_records{1000000};  ///< Actual captured fixed-size events; zero disables.
};
/// @brief Own the TarjanLCAStep contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TarjanLCAStep r{};
/// (void)r;
/// @endcode
/// @see tarjan_lca_run()
struct TarjanLCAStep {
  std::string action;           ///< Actual enter, black, answer or union event.
  std::size_t vertex{};         ///< Actual event node, or parent receiving a union.
  std::size_t other{};          ///< Actual query partner/merged child, n when unused.
  std::size_t query{};          ///< Actual query index when answering, q when unused.
  std::size_t representative{}; ///< Actual union-find representative for this event.
  std::size_t ancestor{};       ///< Actual representative ancestor at this event.
};
/// @brief Own the TarjanLCARun contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TarjanLCARun r{};
/// (void)r;
/// @endcode
/// @see tarjan_lca_run()
struct TarjanLCARun {
  std::vector<std::size_t> lca;      ///< Actual answer per original query order.
  std::vector<std::size_t> entered;  ///< Actual DFS entry order.
  std::vector<std::size_t> finished; ///< Actual subtree-finished/black order.
  std::size_t
      preflight_nodes{}; ///< Actual validation traversal nodes, separate from producing DFS.
  std::size_t
      preflight_edges{};     ///< Actual validation adjacency entries, including reverse directions.
  std::size_t edge_visits{}; ///< Actual producing DFS adjacency entries.
  std::size_t query_visits{}; ///< Actual endpoint-list entries inspected; self queries occur twice.
  std::size_t
      answer_writes{}; ///< Actual completed answer assignments, including repeated self-query writes.
  std::size_t unions{}; ///< Actual completed child-to-parent merges.
  std::size_t
      find_calls{}; ///< Actual union-find calls, including union rank lookup and event-free producer queries.
  std::size_t
      parent_hops{}; ///< Actual representative-search parent-edge traversals; excludes compression assignment pass.
  std::vector<TarjanLCAStep> steps; ///< Optional actual producer events, retaining IDs only.
};
/// @brief Answer offline lowest-common-ancestor queries through actual finish/union states.
/// Requires nonempty n-node undirected tree, root<n, exactly n-1 edges, valid complete edge/query IDs. A separate explicit validation traversal rejects loops, duplicates, cycles and disconnected inputs before query processing; its work is disclosed. Producing explicit-stack DFS marks a node black after its children. For a query whose partner is already black, answer ancestor[find(partner)]. After finishing a child, union its set with the parent by rank and reset the merged representative ancestor to parent. Self queries are installed at both endpoints and may write the same answer twice. Path compression/rank give O((n+q)*alpha(n)) producing work, plus linear validation and owned storage; actual event capture does not perform extra finds. No recursion, caller mutation, shared state or retained pointers.
/// @param n Complete positive node count.
/// @param root Complete valid root index.
/// @param tree_edges Borrowed complete n-1 undirected edges.
/// @param queries Borrowed complete offline query pairs.
/// @param options Complete node/query and actual capture budgets; zero disables policies.
/// @return Owned actual answers, traversal orders, disclosed work and optional events.
/// @throws std::invalid_argument Malformed non-tree input, root or query IDs.
/// @throws std::length_error Complete node/query or actual capture/counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::tarjan_lca_run(3,0,{{0,1},{0,2}},{{1,2}});
/// std::print("{}\n",r.lca); // [0]
/// @endcode
/// @par References
/// - Tarjan, Applications of Path Compression on Balanced Trees, JACM26 (1979),690-715.
/// @see tarjan_offline_lca()
TarjanLCARun tarjan_lca_run(
    std::size_t                                             n,
    std::size_t                                             root,
    const std::vector<std::pair<std::size_t, std::size_t>>& tree_edges,
    const std::vector<std::pair<std::size_t, std::size_t>>& queries,
    const TarjanLCAOptions&                                 options = {});
/// @brief Answer offline lowest-common-ancestor queries through actual finish/union states.
/// Requires nonempty n-node undirected tree, root<n, exactly n-1 edges, valid complete edge/query IDs. A separate explicit validation traversal rejects loops, duplicates, cycles and disconnected inputs before query processing; its work is disclosed. Producing explicit-stack DFS marks a node black after its children. For a query whose partner is already black, answer ancestor[find(partner)]. After finishing a child, union its set with the parent by rank and reset the merged representative ancestor to parent. Self queries are installed at both endpoints and may write the same answer twice. Path compression/rank give O((n+q)*alpha(n)) producing work, plus linear validation and owned storage; actual event capture does not perform extra finds. No recursion, caller mutation, shared state or retained pointers.
/// @param n Complete positive node count.
/// @param root Complete valid root index.
/// @param tree_edges Borrowed complete n-1 undirected edges.
/// @param queries Borrowed complete offline query pairs.
/// @param options Complete node/query and actual capture budgets; zero disables policies.
/// @return Owned actual answers, traversal orders, disclosed work and optional events.
/// @throws std::invalid_argument Malformed non-tree input, root or query IDs.
/// @throws std::length_error Complete node/query or actual capture/counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::tarjan_lca_trace(3,0,{{0,1},{0,2}},{{1,2}});
/// std::print("{}\n",r.lca); // [0]
/// @endcode
/// @par References
/// - Tarjan, Applications of Path Compression on Balanced Trees, JACM26 (1979),690-715.
/// @see tarjan_offline_lca()
TarjanLCARun tarjan_lca_trace(
    std::size_t                                             n,
    std::size_t                                             root,
    const std::vector<std::pair<std::size_t, std::size_t>>& tree_edges,
    const std::vector<std::pair<std::size_t, std::size_t>>& queries,
    const TarjanLCAOptions&                                 options = {});
/// @brief Preserve checked offline LCA answer order with default policies.
/// Requires nonempty n-node undirected tree, root<n, exactly n-1 edges, valid complete edge/query IDs. A separate explicit validation traversal rejects loops, duplicates, cycles and disconnected inputs before query processing; its work is disclosed. Producing explicit-stack DFS marks a node black after its children. For a query whose partner is already black, answer ancestor[find(partner)]. After finishing a child, union its set with the parent by rank and reset the merged representative ancestor to parent. Self queries are installed at both endpoints and may write the same answer twice. Path compression/rank give O((n+q)*alpha(n)) producing work, plus linear validation and owned storage; actual event capture does not perform extra finds. No recursion, caller mutation, shared state or retained pointers.
/// @param n Complete positive node count.
/// @param root Complete valid root index.
/// @param tree_edges Borrowed complete n-1 undirected edges.
/// @param queries Borrowed complete offline query pairs.
/// @return Owned answer IDs in original query order.
/// @throws std::invalid_argument Malformed tree/root/query.
/// @throws std::length_error Default geometry or representational capacity exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::tarjan_offline_lca(3,0,{{0,1},{0,2}},{{1,2}})); // [0]
/// @endcode
/// @see tarjan_lca_run()
std::vector<std::size_t> tarjan_offline_lca(
    std::size_t                                             n,
    std::size_t                                             root,
    const std::vector<std::pair<std::size_t, std::size_t>>& tree_edges,
    const std::vector<std::pair<std::size_t, std::size_t>>& queries);
} // namespace datamunge::algorithms
