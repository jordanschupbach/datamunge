#pragma once
/// @file
/// @brief Checked DAG longest paths, actual topological/relaxation work and edge-identified witnesses.
#include <datamunge/algorithms/boruvka.hpp>
#include <cstddef>
#include <limits>
#include <vector>

namespace datamunge::algorithms {
/// @brief Owned single-source longest distances and predecessor vertices, retaining the original layout.
/// distance[source]=0 uses the empty path; unreachable vertices have negative
/// infinity. predecessor is n at source and unreachable vertices, otherwise an
/// original vertex id. Finite negative distances are reachable, not missing.
/// Values are ordinary rounded double sums; equal stored candidates keep the
/// first improvement. The record alone cannot identify parallel-edge choices.
/// Public arrays own their data and contain no caller references.
/// @par Example
/// @code
/// const auto r=dm::algorithms::dag_longest_paths(2,{{0,1,-2}},0);
/// std::print("{} {}\n",r.distance[1],r.predecessor[1]); // -2 0
/// @endcode
/// @see dag_longest_paths(), DagLongestPathTrace
struct LongestPathResult {
  std::vector<double> distance; ///< n distances, with negative infinity for unreachable vertices.
  std::vector<std::size_t> predecessor; ///< n parent ids; n is the source/unreachable sentinel.
};

/// @brief Actual operation counts for the canonical DAG ordering and relaxation phases.
/// Topological work includes the entire graph, even unreachable components.
/// Relaxation work includes only edges leaving source-reachable vertices.
/// Counts exclude input validation, adjacency construction, output copying,
/// allocation and later path recovery. They count logical visits, not elapsed
/// instructions or wall time. No shared state or borrowed storage.
/// @par Example
/// @code
/// const auto t=dm::algorithms::dag_longest_paths_trace(3,{{0,1,2},{1,2,-1}},0);
/// std::print("{} {} {}\n",t.work.topological_pops,t.work.topological_edge_visits,t.work.relaxation_attempts); // 3 2 2
/// @endcode
/// @see dag_longest_paths_trace()
struct DagLongestPathWork {
  std::size_t topological_pops{0}; ///< Exactly n FIFO removals on a valid DAG.
  std::size_t topological_edge_visits{0}; ///< Exactly m indegree decrements, including parallel edges.
  std::size_t reachable_vertices{0}; ///< Finite-distance vertices processed, including source.
  std::size_t relaxation_attempts{0}; ///< Actual outgoing edge candidates from reachable vertices.
  std::size_t improvements{0}; ///< Actual strict increases in stored distances.
};

/// @brief One actual candidate evaluated from a finalized reachable vertex.
/// edge_index identifies the original directed edge. from_distance is final at
/// evaluation; previous_distance may be negative infinity on first discovery.
/// candidate is its checked finite double sum. improved uses strict > without
/// a tolerance, so equal stored values preserve the earlier witness.
/// @par Example
/// @code
/// const auto t=dm::algorithms::dag_longest_paths_trace(2,{{0,1,1},{0,1,3}},0);
/// const auto& e=t.relaxations[1];
/// std::print("{} {} {}\n",e.previous_distance,e.candidate,e.improved); // 1 3 true
/// @endcode
/// @see DagLongestPathTrace
struct DagRelaxation {
  std::size_t edge_index{0}; ///< Original edge index; parallel edges remain distinct.
  double from_distance{0}; ///< Finalized stored distance of the edge's tail.
  double previous_distance{-std::numeric_limits<double>::infinity()}; ///< Destination distance before this attempt.
  double candidate{0}; ///< Checked finite sum from_distance+edge_weight.
  bool improved{false}; ///< Whether this candidate strictly raised the destination.
};

/// @brief Owned input edges, actual topological order, final paths and every reachable-edge attempt.
/// topological_order covers all n vertices. Initial zero-indegree vertices are
/// queued in ascending id; newly ready vertices are appended in original outgoing
/// edge order. This FIFO policy is deterministic for fixed input, not a global
/// smallest-id order. Original edge order and ready timing determine first ties.
/// predecessor_edge uses m as the source/unreachable sentinel; result's vertex
/// sentinel remains n. No input references or shared mutable state survive.
/// Public edits are possible; witness recovery validates local chains rather
/// than certifying arbitrary edited graphs for optimality or acyclicity.
/// @par Example
/// @code
/// const auto t=dm::algorithms::dag_longest_paths_trace(3,{{0,1,2}},0);
/// std::print("{} {}\n",t.topological_order,t.predecessor_edge); // [0, 2, 1] [1, 0, 1]
/// @endcode
/// @see dag_longest_paths_trace(), dag_longest_path_to()
struct DagLongestPathTrace {
  std::size_t source{0}; ///< Original source vertex id.
  std::vector<WeightedEdge> edges; ///< Owned directed input edges in original order.
  LongestPathResult result; ///< Final distances and predecessor vertices in the original layout.
  std::vector<std::size_t> predecessor_edge; ///< Chosen original edge for each reached non-source; m otherwise.
  std::vector<std::size_t> topological_order; ///< Actual whole-graph FIFO removal order.
  std::vector<DagRelaxation> relaxations; ///< Complete actual reachable-edge attempts, in execution order.
  DagLongestPathWork work; ///< Actual checked ordering and relaxation counters.
};

/// @brief Owned source-to-target vertex/edge witness with its stored rounded total.
/// Unreachable paths have found=false, empty arrays and negative-infinite weight.
/// The source's empty-edge path has one vertex and weight zero. Edge indices
/// select the exact original parallel edges. No input references survive.
/// @par Example
/// @code
/// const auto t=dm::algorithms::dag_longest_paths_trace(2,{{0,1,1},{0,1,3}},0);
/// const auto p=dm::algorithms::dag_longest_path_to(t,1);
/// std::print("{} {} {}\n",p.vertices,p.edge_indices,p.weight); // [0, 1] [1] 3
/// @endcode
/// @see dag_longest_path_to()
struct DagPath {
  bool found{false}; ///< True for any reached target, including source itself.
  std::vector<std::size_t> vertices; ///< Original vertex ids from source to target.
  std::vector<std::size_t> edge_indices; ///< Original directed edge ids between consecutive vertices.
  double weight{-std::numeric_limits<double>::infinity()}; ///< Stored target distance, or negative infinity when unreachable.
};

/// @brief Find longest paths from one source by whole-graph topological dynamic programming.
/// Vertices are [0,n). All finite negative/zero/positive weights and parallel
/// directed edges are allowed; any cycle, including an unreachable cycle or
/// self-loop, rejects. The original signature and two-vector result layout are
/// retained. Full endpoint/weight/source validation precedes allocation. Kahn's
/// FIFO order starts with ascending zero-indegree ids, and scans outgoing edges
/// in input order. Strict stored > keeps the first tied candidate. No optional
/// size cap, mutation of inputs or shared mutable state; O(n+m) time/storage.
/// In exact arithmetic the DAG recurrence is exact. Ordinary double additions
/// and comparisons can collapse distinct real path totals into ties; no exact-
/// real, interval or compensated-summation guarantee is made. Any reachable
/// candidate addition producing infinity rejects, even if it would lose to an
/// existing route. Unreachable edges are validated but never added.
/// @param n Positive vertex count, including isolated vertices.
/// @param edges Read-only directed (u,v,finite weight) tuples in original tie order.
/// @param source Valid source id in [0,n); n=0 has no valid source.
/// @return Owned final distances and predecessor vertices; unreachable distance is -infinity.
/// @throws std::invalid_argument Invalid source/endpoint, nonfinite weight or any cycle.
/// @throws std::overflow_error Any reachable candidate sum is not a finite double.
/// @throws std::length_error Requested container capacity exceeded.
/// @throws std::bad_alloc Graph/order/output allocation failure.
/// @par Example
/// @code
/// const auto r=dm::algorithms::dag_longest_paths(4,{{0,1,2},{0,2,1},{1,3,3},{2,3,8}},0);
/// std::print("{}\n",r.distance[3]); // 9
/// @endcode
/// @par References
/// - R. Sedgewick and K. Wayne, Algorithms, 4th ed., section 4.4, AcyclicLP: https://algs4.cs.princeton.edu/code/javadoc/edu/princeton/cs/algs4/AcyclicLP.html .
/// @see dag_longest_paths_trace(), dag_longest_path_to()
LongestPathResult dag_longest_paths(std::size_t n,const std::vector<WeightedEdge>& edges,std::size_t source);

/// @brief Capture the complete actual longest-path run using the same canonical kernel.
/// Same directed graph, numeric, tie, cycle and sentinel semantics as
/// dag_longest_paths(). Full endpoint/weight/source validation precedes complete
/// optional n/m caps and graph/order/trace allocation. Cycle detection follows
/// graph allocation, so an oversized cyclic graph can hit a size cap first.
/// Acyclicity is established before numerical relaxation. At most m events are
/// captured, including unsuccessful reachable candidates, without replay or per-
/// vertex snapshots. O(n+m) time/storage. Zero disables only its corresponding
/// optional cap. Results never silently truncate and own all inputs/results.
/// @param n Positive vertex count, including unreachable/isolated vertices.
/// @param edges Read-only directed finite-weight edges; parallel edges retained.
/// @param source Source id in [0,n).
/// @param max_vertices Complete vertex cap, default 100000; zero disables.
/// @param max_edges Complete input-edge/event cap, default 1000000; zero disables.
/// @return Owned edges, actual order/events, original result and exact predecessor-edge ids.
/// @throws std::invalid_argument Invalid source/endpoint/weight or any graph cycle.
/// @throws std::overflow_error Reachable nonfinite candidate sum or work-counter overflow.
/// @throws std::length_error Optional cap or container capacity exceeded.
/// @throws std::bad_alloc Graph/order/result/history allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::dag_longest_paths_trace(2,{{0,1,-2}},0,2,1);
/// std::print("{} {}\n",t.result.distance[1],t.work.improvements); // -2 1
/// @endcode
/// @par References
/// - R. Sedgewick and K. Wayne, Algorithms, 4th ed., section 4.4, https://algs4.cs.princeton.edu/44sp/ .
/// @see DagLongestPathTrace, dag_longest_paths()
DagLongestPathTrace dag_longest_paths_trace(std::size_t n,const std::vector<WeightedEdge>& edges,std::size_t source,std::size_t max_vertices=100000,std::size_t max_edges=1000000);

/// @brief Recover the stored source-to-target path without rerunning the graph algorithm.
/// Checks array sizes, source/sentinels and the visited predecessor chain, its
/// original edge endpoints, finite weights and exact stored double recurrence.
/// Unreachable target markers must be consistent. A length guard rejects edited
/// cycles without recursion. Checks do not certify an edited graph's global
/// acyclicity or optimality. O(n) worst-case time and O(path length) owned output
/// space; no caller mutation or retained views. Unrelated trace events/order are
/// not required for local recovery. The returned weight is the stored distance.
/// @param trace Read-only native or locally consistent public trace.
/// @param target Vertex in [0,n), with n derived from result.distance.size().
/// @return Owned chosen path, empty unreachable record, or singleton source path.
/// @throws std::out_of_range Target lies outside the result.
/// @throws std::invalid_argument Malformed shapes, markers or visited predecessor/edge recurrence.
/// @throws std::length_error Output container capacity exceeded.
/// @throws std::bad_alloc Path allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::dag_longest_paths_trace(3,{{0,1,2}},0);
/// std::print("{} {}\n",dm::algorithms::dag_longest_path_to(t,0).vertices,dm::algorithms::dag_longest_path_to(t,2).found); // [0] false
/// @endcode
/// @see DagPath, dag_longest_paths_trace()
DagPath dag_longest_path_to(const DagLongestPathTrace& trace,std::size_t target);

/// @brief Generate a forward DAG band, joining u to u+d for 1<=d<=width.
/// Output is ordered by ascending u then positive offset d. Width beyond n-1
/// is clamped, yielding the complete forward DAG; width zero or n<=1 yields no
/// edges. Finite weights of either sign are valid, even for empty output.
/// Checked edge count w*(n-w)+w*(w-1)/2 precedes allocation, with factors
/// cancelled before multiplication. Optional complete edge cap defaults to
/// 1000000; zero disables it. No random/shared state, caller mutation or borrowed
/// storage. O(n+m) time and O(m) owned output, with an immediate empty return.
/// @param n Vertex count, including zero; vertex arrays are not allocated here.
/// @param width Maximum positive forward id difference; any size_t accepted.
/// @param weight Common finite edge weight, default 1.
/// @param max_edges Complete output cap, default 1000000; zero disables.
/// @return Owned directed weighted edges, acyclic by increasing vertex id.
/// @throws std::invalid_argument Nonfinite weight, including for empty output.
/// @throws std::length_error Checked edge count, optional cap or container capacity exceeded.
/// @throws std::bad_alloc Edge allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::dag_forward_band_edges(3,1,2)); // [(0, 1, 2), (1, 2, 2)]
/// @endcode
/// @see dag_longest_paths_trace()
std::vector<WeightedEdge> dag_forward_band_edges(std::size_t n,std::size_t width,double weight=1,std::size_t max_edges=1000000);
} // namespace datamunge::algorithms
