#pragma once
/// @file kruskal.hpp
/// @brief Stable undirected minimum spanning forests with actual union-find work and owned scan history.
#include <datamunge/algorithms/mst_result.hpp>
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {

/// @brief Logical counts from one actual stable Kruskal execution.
/// Validation, sorting, path-halving steps, allocation, copying and CPU time are excluded.
/// find_calls is exactly twice scanned_edges, including cycles and self-loops.
///
///
/// @par Example
/// @code
/// const auto r=dm::algorithms::kruskal_work(3,{{0,1,1},{1,2,2},{0,2,3}});
/// std::print("{} {} {}\n",r.work.input_edges,r.work.scanned_edges,r.work.find_calls); // 3 2 4
/// @endcode
///
/// @see kruskal_work(), KruskalTrace
struct KruskalWork {
    std::size_t input_edges=0; ///< Number of validated supplied edge records.
    std::size_t scanned_edges=0; ///< Actually visited sorted edges before early tree completion.
    std::size_t accepted_edges=0; ///< Actual successful component merges, equal to result.edges.size().
    std::size_t cycle_rejections=0; ///< Scanned edges whose endpoints already shared a root.
    std::size_t find_calls=0; ///< Two actual root searches per scanned edge.
};

/// @brief Owned minimum spanning forest and actual work counts.
/// No graph/history is borrowed or retained. Edges keep caller endpoint orientation.
/// Equal finite weights preserve input order; output total sums accepted weights in that order.
///
///
/// @par Example
/// @code
/// const auto r=dm::algorithms::kruskal_work(3,{{0,1,2}});
/// std::print("{} {}\n",r.result.is_connected,r.result.total_weight); // false 2
/// @endcode
///
/// @see kruskal_work(), MinimumSpanningTree
struct KruskalOutcome {
    MinimumSpanningTree result; ///< Selected forest, checked finite total and whole-graph connectivity.
    KruskalWork work; ///< Actual logical counts from this execution.
};

/// @brief One actual sorted edge scan with pre-merge roots and post-decision components.
/// input_index identifies the owned input edge in KruskalTrace. Roots follow actual
/// path halving and rank ties, not canonical component labels. No DSU replay is used.
///
///
/// @par Example
/// @code
/// const auto t=dm::algorithms::kruskal_trace(2,{{0,1,3}});
/// std::print("{} {} {}\n",t.steps[0].input_index,t.steps[0].accepted,t.steps[0].components_after); // 0 true 1
/// @endcode
///
/// @see KruskalTrace
struct KruskalStep {
    std::size_t input_index=0; ///< Original zero-based edge record index.
    std::size_t root_u=0; ///< Actual first endpoint root before this merge decision.
    std::size_t root_v=0; ///< Actual second endpoint root before this merge decision.
    bool accepted=false; ///< True exactly when the roots differed and a merge succeeded.
    std::size_t components_after=0; ///< Number of forest components after the decision.
};

/// @brief Owned input edges and actual scanned-edge decisions from the shared forest kernel.
/// Capture C=M+S counts M input edge records and S actual step records, not their
/// individual scalar fields. The complete M is checked before copying input; each
/// additional step is reserved before scanning it. The default quota is one million,
/// zero disables only that quota. Output forest, runtime sort/DSU storage, capacities
/// and allocator bytes are excluded: this is not peak RAM. Failure returns no partial
/// trace. Copies own input/steps/result. No callbacks or replay are involved.
///
///
/// @par Example
/// @code
/// const auto t=dm::algorithms::kruskal_trace(2,{{0,1,3}});
/// std::print("{} {} {}\n",t.input_edges.size(),t.steps.size(),t.captured_entries); // 1 1 2
/// @endcode
///
/// @see kruskal_trace(), euclidean_minimum_spanning_tree_trace()
struct KruskalTrace {
    std::vector<WeightedEdge> input_edges; ///< Owned validated original edges, in original order.
    std::vector<KruskalStep> steps; ///< Actual visited sorted edges, ending as soon as a tree completes.
    KruskalOutcome outcome; ///< Actual output and work, independently owned.
    std::size_t captured_entries=0; ///< Logical input-edge plus step record count.
};

/// @brief Return a stable minimum spanning forest of a finite undirected weighted graph.
/// Stable ascending weight order keeps ties in input order. Union-find uses path
/// halving and union by rank. An accepted edge joins different components; self-loops
/// and cycle edges are skipped. Stops after n-1 accepted edges, otherwise returns a
/// minimum forest of all components. Empty/singleton output has finite total zero.
/// O(n+M log(M+1)) time and O(n+M) runtime storage; ordinary/work retain no optional
/// history or capture quota. Inputs are validated in full before sorting/access.
/// Independent calls share no mutable state; caller mutation of borrowed input is unsafe.
///
/// @param n Vertex count; valid endpoints lie in [0,n). Empty and singleton graphs are connected.
/// @param edges Borrowed undirected edges; finite signed weights, parallel edges and self-loops are allowed.
/// @return Owned forest; disconnected input has is_connected=false.
/// @throws std::invalid_argument If an endpoint is out of range or any weight is nonfinite.
/// @throws std::length_error If vertex/edge/container/count geometry is not representable.
/// @throws std::overflow_error If an intermediate accepted-edge total is not finite.
/// @throws std::bad_alloc If required storage cannot be allocated.
///
/// @par Example
/// @code
/// const auto r=dm::algorithms::kruskal(3,{{0,1,1},{1,2,2},{0,2,3}});
/// std::print("{} {}\n",r.edges.size(),r.total_weight); // 2 3
/// @endcode
///
/// @par References
/// - J. B. Kruskal, Jr. (1956). On the shortest spanning subtree of a graph and the traveling salesman problem. Proceedings of the AMS 7(1), 48-50. https://doi.org/10.1090/S0002-9939-1956-0078686-7
///
/// @see kruskal_work(), kruskal_trace()
MinimumSpanningTree kruskal(std::size_t n,const std::vector<WeightedEdge>& edges);

/// @brief Return the forest and logical counts from the same stable Kruskal execution.
/// Stable ascending weight order keeps ties in input order. Union-find uses path
/// halving and union by rank. An accepted edge joins different components; self-loops
/// and cycle edges are skipped. Stops after n-1 accepted edges, otherwise returns a
/// minimum forest of all components. Empty/singleton output has finite total zero.
/// O(n+M log(M+1)) time and O(n+M) runtime storage; ordinary/work retain no optional
/// history or capture quota. Inputs are validated in full before sorting/access.
/// Independent calls share no mutable state; caller mutation of borrowed input is unsafe.
///
/// @param n Vertex count; valid endpoints lie in [0,n). Empty and singleton graphs are connected.
/// @param edges Borrowed undirected edges; finite signed weights, parallel edges and self-loops are allowed.
/// @return Owned forest and actual counters, without input/history copies.
/// @throws std::invalid_argument If an endpoint is out of range or any weight is nonfinite.
/// @throws std::length_error If vertex/edge/container/count geometry is not representable.
/// @throws std::overflow_error If an intermediate accepted-edge total is not finite.
/// @throws std::bad_alloc If required storage cannot be allocated.
///
/// @par Example
/// @code
/// const auto r=dm::algorithms::kruskal_work(3,{{0,1,1},{1,2,2}});
/// std::print("{} {}\n",r.work.accepted_edges,r.work.cycle_rejections); // 2 0
/// @endcode
///
/// @par References
/// - J. B. Kruskal, Jr. (1956). On the shortest spanning subtree of a graph and the traveling salesman problem. Proceedings of the AMS 7(1), 48-50. https://doi.org/10.1090/S0002-9939-1956-0078686-7
///
/// @see kruskal(), kruskal_trace()
KruskalOutcome kruskal_work(std::size_t n,const std::vector<WeightedEdge>& edges);

/// @brief Capture actual stable Kruskal decisions with a checked retained-record quota.
/// Stable ascending weight order keeps ties in input order. Union-find uses path
/// halving and union by rank. An accepted edge joins different components; self-loops
/// and cycle edges are skipped. Stops after n-1 accepted edges, otherwise returns a
/// minimum forest of all components. Empty/singleton output has finite total zero.
/// O(n+M log(M+1)) time and O(n+M) runtime storage; ordinary/work retain no optional
/// history or capture quota. Inputs are validated in full before sorting/access.
/// Independent calls share no mutable state; caller mutation of borrowed input is unsafe.
/// The quota C=M+S follows KruskalTrace, including ignored input edges but only
/// scanned decisions. Steps identify the original edge, actual roots and component
/// count, ending when the same ordinary kernel stops.
///
/// @param n Vertex count; valid endpoints lie in [0,n). Empty and singleton graphs are connected.
/// @param edges Borrowed undirected edges; finite signed weights, parallel edges and self-loops are allowed.
/// @param max_entries Maximum logical captured records (default 1000000); zero disables only quota.
/// @return Owned inputs, steps and outcome; no partial trace escapes an exception.
/// @throws std::invalid_argument If an endpoint is out of range or any weight is nonfinite.
/// @throws std::length_error If vertex/edge/container/count geometry is not representable.
/// @throws std::overflow_error If an intermediate accepted-edge total is not finite.
/// @throws std::bad_alloc If required storage cannot be allocated.
/// @throws std::length_error If initial input capture or the next step would exceed max_entries.
///
/// @par Example
/// @code
/// const auto t=dm::algorithms::kruskal_trace(3,{{0,1,1},{1,2,2},{0,2,3}},5);
/// std::print("{} {}\n",t.steps.size(),t.captured_entries); // 2 5
/// @endcode
///
/// @par References
/// - J. B. Kruskal, Jr. (1956). On the shortest spanning subtree of a graph and the traveling salesman problem. Proceedings of the AMS 7(1), 48-50. https://doi.org/10.1090/S0002-9939-1956-0078686-7
///
/// @see KruskalTrace, kruskal_work()
KruskalTrace kruskal_trace(std::size_t n,const std::vector<WeightedEdge>& edges,std::size_t max_entries=1000000);

} // namespace datamunge::algorithms
