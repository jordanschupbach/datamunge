#pragma once
/// @file gabow_scc.hpp
/// @brief Validated iterative path-based SCC with actual open-stack/boundary events.
#include <cstddef>
#include <string>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Directed adjacency rows with full-width neighbor positions.
/// Rows correspond to vertex IDs0..n-1. Endpoint IDs are int; duplicate edges, loops and neighbor order are retained. No symmetry or simple-graph condition.
/// @par Example
/// @code
/// dm::algorithms::SccAdjacency a{{1},{0}};
/// std::cout << a.size(); // 2
/// @endcode
/// @see gabow_scc(), gabow_scc_trace()
using SccAdjacency = std::vector<std::vector<int>>;
/// @brief One actual suspended DFS frame: vertex and next neighbor index.
/// The first field is the vertex ID; the second is the next size_t adjacency position. Frames are bottom-to-top. No call-stack recursion.
/// @par Example
/// @code
/// dm::algorithms::GabowDfsFrame f{2,3};
/// std::cout << f.second; // 3
/// @endcode
/// @see gabow_scc(), gabow_scc_trace()
using GabowDfsFrame = std::pair<int, std::size_t>;
/// @brief Actual traversal operations and peak stack lengths.
/// Excludes validation, history copies, allocation, rendering and partition summaries. Each vertex enters/finishes once, is assigned once and is pushed/popped once on each stack. Individual edges can collapse many boundaries; total root pops is N. Event count is E+4N, including root-close pops and collapse pops.
/// @par Example
/// @code
/// dm::algorithms::GabowWork w;
/// std::cout << w.edges_examined; // 0
/// @endcode
/// @see gabow_scc(), gabow_scc_trace()
struct GabowWork {
    std::size_t visited_vertices{}; ///< Actual first visits.
    std::size_t edges_examined{}; ///< Actual adjacency entries consumed, including duplicates and loops.
    std::size_t tree_edges{}; ///< Edges to undiscovered vertices.
    std::size_t open_edges{}; ///< Edges to already discovered, unassigned vertices.
    std::size_t closed_edges{}; ///< Edges to assigned vertices.
    std::size_t root_pushes{}; ///< Candidate-root pushes.
    std::size_t root_pops{}; ///< All candidate-root pops, collapse plus component closure.
    std::size_t root_collapses{}; ///< Root pops caused by an open edge.
    std::size_t component_closures{}; ///< Completed components, in raw ID assignment order.
    std::size_t assignments{}; ///< Vertex component assignments.
    std::size_t dfs_pushes{}; ///< Heap DFS frame pushes.
    std::size_t dfs_pops{}; ///< Heap DFS frame finishes/pops.
    std::size_t open_pushes{}; ///< Open-vertex S pushes.
    std::size_t open_pops{}; ///< Open-vertex S pops on assignment.
    std::size_t peak_dfs{}; ///< Maximum active DFS frames.
    std::size_t peak_open{}; ///< Maximum discovered-unassigned S length.
    std::size_t peak_roots{}; ///< Maximum candidate boundary stack length.
    std::size_t event_count{}; ///< Actual observed state transitions, whether captured or not.
};
/// @brief Raw component IDs and actual work from one SCC execution.
/// IDs are contiguous0..C-1, assigned in sink-first closure order. For a cross-component edge u->v, id[u]>id[v]. Neighbor order can change labels for incomparable components; equality of partitions is the stable meaning.
/// @par Example
/// @code
/// dm::algorithms::GabowResult r;
/// std::cout << r.components.size(); // 0
/// @endcode
/// @see gabow_scc(), gabow_scc_trace()
struct GabowResult {
    std::vector<int> components{}; ///< One raw component ID per vertex.
    GabowWork work{}; ///< Actual traversal counts and peaks.
};
/// @brief Owned actual state after one traversal primitive, without replay.
/// Kinds are enter,edge,collapse,close-root,assign,finish. Enter follows first preorder/S/B pushes; edge follows neighbor-index increment before edge action; collapse/close-root follow one B pop; assign follows one S pop and component write; finish follows frame pop and any component counter increment. Intermediate assignment states need not satisfy quiescent boundary invariants. S contains every discovered-unassigned vertex, including finished DFS calls, and can exceed the active DFS path.
/// @par Example
/// @code
/// dm::algorithms::GabowEvent e;
/// std::cout << e.u; // -1
/// @endcode
/// @see gabow_scc(), gabow_scc_trace()
struct GabowEvent {
    std::string kind{}; ///< One of the six transition names.
    int u{-1}; ///< Current vertex; finished vertex for finish.
    int v{-1}; ///< Examined target for edge/collapse; assigned vertex for assign; otherwise-1.
    int counter{}; ///< Actual1-based preorder counter.
    int ncomp{}; ///< Actual component counter; increases after all assignments in one component.
    std::vector<int> comp{}; ///< Owned actual labels; -1 denotes unassigned.
    std::vector<int> preorder{}; ///< Owned actual discovery numbers;0 denotes undiscovered.
    std::vector<int> path{}; ///< Owned actual discovered-unassigned S, bottom-to-top.
    std::vector<int> roots{}; ///< Owned actual candidate root vertex IDs B, bottom-to-top.
    std::vector<GabowDfsFrame> stack{}; ///< Owned actual active DFS frames, bottom-to-top.
};
/// @brief Owned graph and actual primitive snapshots from one kernel execution.
/// Whole-request conservative admission precedes graph copy and state allocation. No replay, shared state, caller mutation, truncation or partial output. Captured entry accounting excludes ordinary final labels/work/scalar root metadata, working state and allocator bytes.
/// @par Example
/// @code
/// dm::algorithms::GabowTrace t;
/// std::cout << t.captured_entries; // 0
/// @endcode
/// @see gabow_scc(), gabow_scc_trace()
struct GabowTrace : public GabowResult {
    SccAdjacency adjacency{}; ///< Owned raw rows, preserving loops/duplicates/order.
    std::vector<GabowEvent> events{}; ///< Actual E+4N transitions, no synthetic initial event.
    std::size_t captured_entries{}; ///< Actual input row/neighbor entries plus retained snapshot records/elements.
};
/// @brief One labeled partition group with ascending member vertex IDs.
/// Groups are sorted by nonnegative raw label, which need not be contiguous. Vertex IDs are original row positions. This container does not itself certify strong connectivity.
/// @par Example
/// @code
/// dm::algorithms::SccComponent c;
/// std::cout << c.vertices.size(); // 0
/// @endcode
/// @see gabow_scc(), gabow_scc_trace()
struct SccComponent {
    int label{}; ///< Raw nonnegative label.
    std::vector<int> vertices{}; ///< Owned original vertex IDs in ascending order.
};
/// @brief Owned groups and the deduplicated quotient graph for supplied labels.
/// Dense quotient indices follow ascending raw labels, distinct from those labels. Internal edges are omitted from adjacency; cross edges are sorted/deduplicated. For arbitrary labels the quotient may contain cycles: no SCC correctness certificate is computed. Gabow labels make reverse_topological true, which proves acyclicity by strict label decrease.
/// @par Example
/// @code
/// dm::algorithms::SccSummary q;
/// std::cout << q.reverse_topological; // 1
/// @endcode
/// @see gabow_scc(), gabow_scc_trace()
struct SccSummary {
    std::vector<SccComponent> groups{}; ///< Groups in ascending raw-label order.
    SccAdjacency adjacency{}; ///< Quotient adjacency indexed by dense group position.
    std::size_t internal_edges{}; ///< Raw entries within a supplied group, including loops/duplicates.
    std::size_t cross_edges{}; ///< Raw entries across supplied groups, including duplicates.
    std::size_t unique_cross_edges{}; ///< Number of deduplicated quotient arcs.
    bool reverse_topological{true}; ///< Every raw cross arc strictly decreases its supplied label.
};
/// @brief Find strong components with validated iterative path-based DFS.
/// Validate n, exact row count and every endpoint before empty returns, state allocation or capture admission. Preserve loops, duplicate entries, neighbor order and ascending start-vertex order. Empty(0,{}) returns empty. One explicit heap DFS avoids call-stack recursion. S holds all discovered-unassigned vertices, not only active frames. Raw labels follow component closure order, with cross edges strictly decreasing IDs. Ordinary traversal is O(N+E) time/O(N) auxiliary state excluding input/result; no mutation or shared state.
/// @param n Vertex count, nonnegative.
/// @param adj Exactly n rows with endpoints in0..n-1.
/// @return Raw0-based contiguous component ID for each vertex.
/// @throws std::invalid_argument Negative n, wrong adjacency row count or invalid endpoint.
/// @throws std::length_error Traversal count or container limit overflow.
/// @throws std::bad_alloc Allocation of owned or working storage fails.
/// @par Example
/// @code
/// const auto c=dm::algorithms::gabow_scc(3,{{1,2},{0},{1}});
/// std::cout << c[0] << c[1] << c[2]; // 000
/// @endcode
/// @par References
/// - H.N. Gabow (2000), Path-based depth-first search for strong and biconnected components, Information Processing Letters74,107-114. doi:10.1016/S0020-0190(00)00051-X. Section2/Theorem2.1. Explicit DFS frames replace the recursive calls in the paper; candidate vertex IDs and preorder replace stack-index boundaries.
/// @see gabow_scc(), gabow_scc_trace(), scc_same_partition()
std::vector<int> gabow_scc(int n, const SccAdjacency& adj);
/// @brief Find strong components and count actual traversal operations.
/// Validate n, exact row count and every endpoint before empty returns, state allocation or capture admission. Preserve loops, duplicate entries, neighbor order and ascending start-vertex order. Empty(0,{}) returns empty. One explicit heap DFS avoids call-stack recursion. S holds all discovered-unassigned vertices, not only active frames. Raw labels follow component closure order, with cross edges strictly decreasing IDs. Ordinary traversal is O(N+E) time/O(N) auxiliary state excluding input/result; no mutation or shared state. Counts exclude history, validation and summary work; no replay.
/// @param n Vertex count, nonnegative.
/// @param adj Exactly n rows with endpoints in0..n-1.
/// @return Raw component labels and actual work.
/// @throws std::invalid_argument Negative n, wrong adjacency row count or invalid endpoint.
/// @throws std::length_error Traversal count or container limit overflow.
/// @throws std::bad_alloc Allocation of owned or working storage fails.
/// @par Example
/// @code
/// const auto r=dm::algorithms::gabow_scc_work(2,{{1},{0}});
/// std::cout << r.work.root_collapses; // 1
/// @endcode
/// @par References
/// - H.N. Gabow (2000), Path-based depth-first search for strong and biconnected components, Information Processing Letters74,107-114. doi:10.1016/S0020-0190(00)00051-X. Section2/Theorem2.1. Explicit DFS frames replace the recursive calls in the paper; candidate vertex IDs and preorder replace stack-index boundaries.
/// @see gabow_scc(), gabow_scc_trace(), scc_same_partition()
GabowResult gabow_scc_work(int n, const SccAdjacency& adj);
/// @brief Forecast whole retained-history entries from virtual graph dimensions.
/// Require n>=0; n0 permits only edges0. Actual transition count is E+4N. Input rows+neighbors cost N+E; each snapshot has one metadata record plus full comp/preorder arrays and up to N entries in each S, B and active-frame vector. Return (N+E)+(E+4N)*(1+5N), not bytes or peak RAM. Each DFS frame pair counts as one record. Excludes ordinary final labels/work/scalar root metadata/working state/allocator bytes. Bound all snapshots conservatively; check count/vector capacities without allocation.
/// @param n Virtual vertex count, nonnegative.
/// @param edges Raw adjacency entries, including loops/duplicates.
/// @return Conservative whole-request entry bound.
/// @throws std::invalid_argument Negative n, or n0 with positive edges.
/// @throws std::length_error Virtual count or vector-capacity bound overflow.
/// @par Example
/// @code
/// std::cout << dm::algorithms::gabow_scc_trace_entries(3,4); // 263
/// @endcode
/// @par References
/// - H.N. Gabow (2000), Path-based depth-first search for strong and biconnected components, Information Processing Letters74,107-114. doi:10.1016/S0020-0190(00)00051-X. Section2/Theorem2.1. Explicit DFS frames replace the recursive calls in the paper; candidate vertex IDs and preorder replace stack-index boundaries.
/// @see gabow_scc(), gabow_scc_trace(), scc_same_partition()
std::size_t gabow_scc_trace_entries(int n, std::size_t edges);
/// @brief Capture actual S/B/DFS states with whole-request admission.
/// Validate n, exact row count and every endpoint before empty returns, state allocation or capture admission. Preserve loops, duplicate entries, neighbor order and ascending start-vertex order. Empty(0,{}) returns empty. One explicit heap DFS avoids call-stack recursion. S holds all discovered-unassigned vertices, not only active frames. Raw labels follow component closure order, with cross edges strictly decreasing IDs. Ordinary traversal is O(N+E) time/O(N) auxiliary state excluding input/result; no mutation or shared state. Uses gabow_scc_trace_entries() before graph copies and state allocation. Default quota1m;0 disables only optional cap. Count/container checks remain. Snapshots are actual primitive states with owned storage; no partial trace or truncation. Snapshot copies add O((N+E)*N) time and retained storage in the worst case.
/// @param n Vertex count, nonnegative.
/// @param adj Exactly n rows with endpoints in0..n-1.
/// @param max_entries Whole conservative retained-entry quota;0 disables only the cap. Default 1000000.
/// @return Owned raw graph, actual transitions, labels and work.
/// @throws std::invalid_argument Negative n, wrong adjacency row count or invalid endpoint.
/// @throws std::length_error Count/container overflow or exceeded whole-history quota.
/// @throws std::bad_alloc Allocation of owned or working storage fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::gabow_scc_trace(3,{{1,2},{0},{1}},263);
/// std::cout << t.events.size(); // 16
/// @endcode
/// @par References
/// - H.N. Gabow (2000), Path-based depth-first search for strong and biconnected components, Information Processing Letters74,107-114. doi:10.1016/S0020-0190(00)00051-X. Section2/Theorem2.1. Explicit DFS frames replace the recursive calls in the paper; candidate vertex IDs and preorder replace stack-index boundaries.
/// @see gabow_scc(), gabow_scc_trace(), scc_same_partition()
GabowTrace gabow_scc_trace(int n, const SccAdjacency& adj, std::size_t max_entries = 1000000);
/// @brief Compare labeled partitions independently of their numeric IDs.
/// Validate every nonnegative label before a length mismatch returns false. Labels may be noncontiguous or INT_MAX. Two directional maps establish a label bijection in O(N log C) time/O(C) state; empty partitions agree. No mutation, relabeling or connectivity test.
/// @param a First complete nonnegative labeling.
/// @param b Second complete nonnegative labeling.
/// @return True iff both vectors describe the same equivalence relation.
/// @throws std::invalid_argument A negative label in either input, even when lengths differ.
/// @throws std::bad_alloc Allocation of owned or working storage fails.
/// @par Example
/// @code
/// std::cout << dm::algorithms::scc_same_partition({3,3,8},{0,0,1}); // 1
/// @endcode
/// @par References
/// - H.N. Gabow (2000), Path-based depth-first search for strong and biconnected components, Information Processing Letters74,107-114. doi:10.1016/S0020-0190(00)00051-X. Section2/Theorem2.1. Explicit DFS frames replace the recursive calls in the paper; candidate vertex IDs and preorder replace stack-index boundaries.
/// @see gabow_scc(), gabow_scc_trace(), scc_same_partition()
bool scc_same_partition(const std::vector<int>& a, const std::vector<int>& b);
/// @brief Group supplied labels and form a sorted deduplicated quotient graph.
/// Validate every adjacency row/endpoint and nonnegative label, with labels.size()==adj.size(), before grouping. Groups sort by raw label; quotient IDs index groups. This summarizes a supplied partition without proving its classes are SCCs; an arbitrary quotient may be cyclic. Costs O(N log C+E log C+sum degree*log degree), with O(N+E) owned/scratch storage. Input loops/duplicates count in raw metrics, but internal arcs are omitted and cross arcs deduplicated. No input mutation.
/// @param adj Directed graph with vertex IDs0..rows-1; row count fits int.
/// @param components One nonnegative raw label per vertex, not necessarily contiguous.
/// @return Owned groups, quotient adjacency and raw/unique edge counts.
/// @throws std::invalid_argument Invalid endpoint, negative label or graph/label row-count mismatch.
/// @throws std::length_error Vertex IDs exceed int, or count/container limits overflow.
/// @throws std::bad_alloc Allocation of owned or working storage fails.
/// @par Example
/// @code
/// const auto q=dm::algorithms::scc_summary({{1},{0,2},{}},{1,1,0});
/// std::cout << q.groups.size() << " " << q.unique_cross_edges; // 2 1
/// @endcode
/// @par References
/// - H.N. Gabow (2000), Path-based depth-first search for strong and biconnected components, Information Processing Letters74,107-114. doi:10.1016/S0020-0190(00)00051-X. Section2/Theorem2.1. Explicit DFS frames replace the recursive calls in the paper; candidate vertex IDs and preorder replace stack-index boundaries.
/// @see gabow_scc(), gabow_scc_trace(), scc_same_partition()
SccSummary scc_summary(const SccAdjacency& adj, const std::vector<int>& components);
/// @brief Construct the original eight-vertex cycle-chain fixture.
/// Owned fixed graph: cycles0<->1,2->3->4->2,5<->6 and sink7, linked by1->2,4->5,6->7. No shared state.
/// @return Eight raw adjacency rows in original neighbor order.
/// @throws std::bad_alloc Allocation of owned or working storage fails.
/// @par Example
/// @code
/// std::cout << dm::algorithms::gabow_scc_example_graph().size(); // 8
/// @endcode
/// @par References
/// - H.N. Gabow (2000), Path-based depth-first search for strong and biconnected components, Information Processing Letters74,107-114. doi:10.1016/S0020-0190(00)00051-X. Section2/Theorem2.1. Explicit DFS frames replace the recursive calls in the paper; candidate vertex IDs and preorder replace stack-index boundaries.
/// @see gabow_scc(), gabow_scc_trace(), scc_same_partition()
SccAdjacency gabow_scc_example_graph();
} // namespace datamunge::algorithms
