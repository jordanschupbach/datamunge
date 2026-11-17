#pragma once
/// @file reverse_delete.hpp
/// @brief Checked reverse-delete forests and actual whole-graph connectivity decisions.
#include <cstddef>
#include <datamunge/algorithms/mst_result.hpp>
#include <vector>
/// @brief Canonical graph producers and owned histories.
namespace datamunge::algorithms {
/// @brief Input, actual scanned-edge and retained-mask policies; zero disables a cap.
/// @par Example
/// @code
/// const dm::algorithms::ReverseDeleteOptions value{};
/// std::print("{}\n",value.max_vertices); // 1000000
/// @endcode
struct ReverseDeleteOptions {
  std::size_t max_vertices{1000000};      ///< Complete vertex count.
  std::size_t max_edges{1000000};         ///< Complete input edge count.
  std::size_t max_edge_visits{10000000};  ///< Actual edge-slot inspections in connectivity passes.
  std::size_t max_trace_entries{1000000}; ///< Mask scalars plus one header per deletion decision.
};
/// @brief One actual temporary edge omission and its committed decision.
/// @par Example
/// @code
/// const dm::algorithms::ReverseDeleteStep value{};
/// std::print("{}\n",value.edge_index); // 0
/// @endcode
struct ReverseDeleteStep {
  std::size_t edge_index{}; ///< Original caller edge id, including parallel/self-loop identity.
  std::size_t components_without{}; ///< Actual whole-graph components with the tested edge omitted.
  bool        removed{};            ///< Actual safe deletion committed by this decision.
  std::vector<char> kept;           ///< Actual post-decision mask in caller input order.
};
/// @brief Owned actual minimum forest and selected connectivity work.
/// @par Example
/// @code
/// const dm::algorithms::ReverseDeleteRun value{};
/// std::print("{}\n",value.edge_visits); // 0
/// @endcode
struct ReverseDeleteRun {
  MinimumSpanningTree      forest; ///< Actual forest, retaining original edge orientation.
  std::vector<std::size_t> order;  ///< Actual descending-weight original ids; ties by input id.
  std::vector<char>        kept;   ///< Actual final original-id mask.
  std::size_t              initial_components{};  ///< Actual original whole-graph component count.
  std::size_t              connectivity_passes{}; ///< Actual initial plus one pass per input edge.
  std::size_t edge_visits{}; ///< Actual inspected edge slots, including omitted/deleted slots.
  std::size_t
      union_attempts{}; ///< Actual retained, nonomitted union calls including cycles/self-loops.
  std::vector<ReverseDeleteStep> steps; ///< Actual optional deletion decision histories.
};
/// @brief Execute the checked reverse-delete forest producer.
/// Validates all finite weights/endpoints before work. Sorts edge ids descending by weight and then ascending input id. Temporarily omits each edge, recomputes whole-graph components with canonical disjoint sets, and deletes exactly when the original component count is preserved. Supports disconnected graphs, negative weights, self-loops and parallel edges. Selected edge visits include every slot inspected by every connectivity pass; validation/sorting/vertex scans are excluded. Time O(E log E+(E+1)*(V+E alpha(V))); O(V+E) working/result storage plus optional O(E^2) masks. Inputs borrowed unchanged; outputs owned, no shared state. Finite total-weight overflow throws. Trace admission costs E+1 per actual decision, before its connectivity pass; zero disables only a policy.
/// @param n Number of vertices, including zero.
/// @param edges Complete undirected edge list, preserving caller ids.
/// @param options Input/work/history policies, default bounded.
/// @return Owned forest and actual connectivity work, with optional same-pass masks.
/// @throws std::invalid_argument Invalid endpoint or nonfinite weight.
/// @throws std::length_error Any requested input, work, history or count capacity exceeded.
/// @throws std::overflow_error Forest weight cannot be represented as finite double.
/// @par Example
/// @code
/// auto t=dm::algorithms::reverse_delete_run(3,{{0,1,1},{1,2,2},{0,2,8}});
/// std::print("{}\n",t.forest.total_weight); // 3
/// @endcode
/// @par References
/// - Kruskal (1956), On the Shortest Spanning Subtree of a Graph and the Traveling Salesman Problem, doi:10.1090/S0002-9939-1956-0078686-7.
/// @see reverse_delete()
ReverseDeleteRun reverse_delete_run(
    std::size_t                      n,
    const std::vector<WeightedEdge>& edges,
    const ReverseDeleteOptions&      options = {});
/// @brief Execute the checked reverse-delete forest producer with actual decision masks.
/// Validates all finite weights/endpoints before work. Sorts edge ids descending by weight and then ascending input id. Temporarily omits each edge, recomputes whole-graph components with canonical disjoint sets, and deletes exactly when the original component count is preserved. Supports disconnected graphs, negative weights, self-loops and parallel edges. Selected edge visits include every slot inspected by every connectivity pass; validation/sorting/vertex scans are excluded. Time O(E log E+(E+1)*(V+E alpha(V))); O(V+E) working/result storage plus optional O(E^2) masks. Inputs borrowed unchanged; outputs owned, no shared state. Finite total-weight overflow throws. Trace admission costs E+1 per actual decision, before its connectivity pass; zero disables only a policy.
/// @param n Number of vertices, including zero.
/// @param edges Complete undirected edge list, preserving caller ids.
/// @param options Input/work/history policies, default bounded.
/// @return Owned forest and actual connectivity work, with optional same-pass masks.
/// @throws std::invalid_argument Invalid endpoint or nonfinite weight.
/// @throws std::length_error Any requested input, work, history or count capacity exceeded.
/// @throws std::overflow_error Forest weight cannot be represented as finite double.
/// @par Example
/// @code
/// auto t=dm::algorithms::reverse_delete_trace(3,{{0,1,1},{1,2,2},{0,2,8}});
/// std::print("{}\n",t.forest.total_weight); // 3
/// @endcode
/// @par References
/// - Kruskal (1956), On the Shortest Spanning Subtree of a Graph and the Traveling Salesman Problem, doi:10.1090/S0002-9939-1956-0078686-7.
/// @see reverse_delete()
ReverseDeleteRun reverse_delete_trace(
    std::size_t                      n,
    const std::vector<WeightedEdge>& edges,
    const ReverseDeleteOptions&      options = {});
/// @brief Return the canonical checked reverse-delete forest through the preserved signature.
/// Uses the same producer/default policies. Equal-weight outputs now have explicit input-id ties; any minimum forest is valid. Complete zero/singleton graphs are connected by convention.
/// @param n Number of vertices.
/// @param edges Complete undirected finite weighted edges.
/// @return Owned minimum forest of every input component.
/// @throws std::invalid_argument Invalid edge.
/// @throws std::length_error Default policy exceeded.
/// @throws std::overflow_error Unrepresentable forest weight.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::reverse_delete(3,{{0,1,1},{1,2,2},{0,2,8}}).total_weight); // 3
/// @endcode
/// @par References
/// - Kruskal (1956), On the Shortest Spanning Subtree of a Graph and the Traveling Salesman Problem, doi:10.1090/S0002-9939-1956-0078686-7.
/// @see reverse_delete_run()
MinimumSpanningTree reverse_delete(std::size_t n, const std::vector<WeightedEdge>& edges);
/// @brief Construct a controlled connected weighted cycle for reverse-delete work studies.
/// For n>=3, consecutive edges have weights1,...,n-1 and the closing edge weight n. The heaviest cycle edge is the first safe deletion, leaving a known spanning path; no random input or MST solver is hidden.
/// @param n Vertex count, valid3..100000.
/// @return Owned n weighted undirected edges.
/// @throws std::invalid_argument Fewer than three cycle vertices.
/// @throws std::length_error More than100000 vertices.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::reverse_delete_cycle(4).size()); // 4
/// @endcode
/// @see reverse_delete_run()
std::vector<WeightedEdge> reverse_delete_cycle(std::size_t n);
} // namespace datamunge::algorithms
