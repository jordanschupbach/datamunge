#pragma once
/// @file edmonds_karp.hpp
/// @brief Checked shortest-augmenting-path flow, complete actual BFS evidence and residual updates.
#include <cstddef>
#include <tuple>
#include <vector>
namespace datamunge::algorithms {

/// @brief Owned flow value, per-input-edge flows and a source-side cut witness.
/// Shared by maximum-flow algorithms; the original three-field layout is retained.
/// Flow/cut agreement is numerical for double solvers, rather than an exact-real
/// guarantee. Default construction is an uncomputed placeholder. No input views survive.
/// @par Example
/// @code
/// const auto r=dm::algorithms::edmonds_karp(2,{{0,1,7.5}},0,1);
/// std::print("{} {}\n",r.max_flow,r.edge_flow.size()); // 7.5 1
/// @endcode
/// @par References
/// - Ford and Fulkerson (1956), Maximal flow through a network. doi:10.4153/CJM-1956-045-5.
/// @see edmonds_karp(), max_flow_certificate()
struct MaxFlowResult {
    double max_flow{0.0}; ///< Reported net source-to-sink flow value.
    std::vector<double> edge_flow; ///< Flow of each original edge, in input order.
    std::vector<std::size_t> min_cut_source_side; ///< Final residual source-reachable vertices.
};

/// @brief Checked logical operations from the actual complete-BFS Edmonds-Karp run.
/// Counts exclude validation, allocation, copying, certificate checks and CPU time.
/// Each BFS scans every outgoing residual arc of every reached vertex, including
/// zero, reverse and self arcs. The final failed search is included.
/// @par Example
/// @code
/// const auto r=dm::algorithms::edmonds_karp_work(2,{{0,1,7.5}},0,1);
/// std::print("{} {} {}\n",r.work.bfs_calls,r.work.bfs_arc_inspections,r.work.augmentations); // 2 3 1
/// @endcode
/// @see edmonds_karp_work(), EdmondsKarpTrace
struct EdmondsKarpWork {
    std::size_t bfs_calls=0; ///< Complete searches, including termination.
    std::size_t bfs_arc_inspections=0; ///< Total residual adjacency entries examined.
    std::size_t augmentations=0; ///< Positive shortest-path pushes.
    std::size_t path_arc_updates=0; ///< Sum of pushed path lengths, one per oriented pair update.
    std::size_t max_path_edges=0; ///< Longest successful source-to-sink path, zero if none.
};

/// @brief Ordinary owned flow result and work, without retained search/path history.
/// Inputs remain unchanged and independent calls share no mutable state. No optional
/// capture quota applies; ordinary working storage is O(V+E).
/// @par Example
/// @code
/// const auto r=dm::algorithms::edmonds_karp_work(2,{{0,1,3}},0,1);
/// std::print("{} {}\n",r.result.max_flow,r.work.path_arc_updates); // 3 1
/// @endcode
/// @see edmonds_karp_work()
struct EdmondsKarpResult {
    MaxFlowResult result; ///< Owned value, per-edge flows and source-side cut.
    EdmondsKarpWork work; ///< Counters of the same run.
};

/// @brief Before/after values of an actual oriented residual pair on a pushed path.
/// Input edge i owns arc ids 2*i (forward) and 2*i+1 (cancellation).
/// Reverse use decreases that edge's original flow, preserving independent identities
/// for parallel, antiparallel and self edges. Values are copied during the actual push.
/// @par Example
/// @code
/// const auto t=dm::algorithms::edmonds_karp_trace(2,{{0,1,3}},0,1);
/// const auto& a=t.augmentations[0].path[0];
/// std::print("{} {} {}\n",a.input_edge,a.reverse,a.residual_after); // 0 false 0
/// @endcode
/// @see EdmondsKarpAugmentation
struct EdmondsKarpPathArc {
    std::size_t input_edge=0; ///< Original input edge index.
    std::size_t from=0; ///< Oriented residual tail.
    std::size_t to=0; ///< Oriented residual head.
    bool reverse=false; ///< Whether the cancellation arc was used.
    double residual_before=0; ///< Used residual capacity before this push.
    double residual_after=0; ///< Used residual capacity after this push.
    double paired_before=0; ///< Opposite residual capacity before this push.
    double paired_after=0; ///< Opposite residual capacity after this push.
};

/// @brief Full actual BFS levels and first-discovery parents before a path push.
/// Unreachable level is vertex_count; source and unreachable parent ids are SIZE_MAX.
/// Every successful search has exactly one augmentation at the same vector index.
/// The last search fails to reach the sink and supplies the returned source cut.
/// @par Example
/// @code
/// const auto t=dm::algorithms::edmonds_karp_trace(2,{{0,1,3}},0,1);
/// std::print("{} {}\n",t.searches[0].levels[1],t.searches.back().levels[1]); // 1 2
/// @endcode
/// @see EdmondsKarpTrace
struct EdmondsKarpSearch {
    std::vector<std::size_t> levels; ///< Complete shortest residual distances from source.
    std::vector<std::size_t> parents; ///< First-discovery oriented arc ids, or SIZE_MAX.
    double flow_before=0; ///< Total flow before this search.
    std::size_t bfs_arc_inspections=0; ///< Outgoing residual entries examined in this search.
};

/// @brief One actual bottleneck push along a recorded BFS parent path.
/// path runs from source to sink. search identifies its complete pre-push BFS;
/// amount is the minimum positive residual on that path. Copies own all records.
/// @par Example
/// @code
/// const auto t=dm::algorithms::edmonds_karp_trace(2,{{0,1,3}},0,1);
/// std::print("{} {} {}\n",t.augmentations[0].search,t.augmentations[0].amount,t.augmentations[0].flow_after); // 0 3 3
/// @endcode
/// @see EdmondsKarpPathArc, edmonds_karp_trace()
struct EdmondsKarpAugmentation {
    std::size_t search=0; ///< Index of the BFS selecting this path.
    double amount=0; ///< Positive bottleneck.
    double flow_before=0; ///< Total before the push.
    double flow_after=0; ///< Total after the push.
    std::vector<EdmondsKarpPathArc> path; ///< Actual oriented updates in source-to-sink order.
};

/// @brief Owned original network and search/path evidence captured without replay.
/// Logical capture C=E+1+P*(1+2V)+A+L: E input records, one network header,
/// P full BFS headers/levels/parents (including failure), A augmentation headers
/// and L path-arc records. Result copies, runtime arrays, scalars, vector capacity
/// and allocator bytes are excluded; this is not a peak-RAM limit. Histories add
/// O(C) work/storage. Mutable records are inspectable data, not self-certifying proofs.
/// @par Example
/// @code
/// const auto t=dm::algorithms::edmonds_karp_trace(2,{{0,1,3}},0,1);
/// std::print("{} {} {}\n",t.searches.size(),t.augmentations.size(),t.captured_entries); // 2 1 14
/// @endcode
/// @see edmonds_karp_trace(), edmonds_karp_trace_base_entries()
struct EdmondsKarpTrace {
    std::size_t vertex_count=0; ///< Includes isolated vertices.
    std::vector<std::tuple<std::size_t,std::size_t,double>> edges; ///< Exact owned original input order.
    std::size_t source=0; ///< Source id.
    std::size_t sink=1; ///< Distinct sink id.
    MaxFlowResult result; ///< Ordinary result from the captured run.
    EdmondsKarpWork work; ///< Same run's checked counters.
    std::vector<EdmondsKarpSearch> searches; ///< Complete actual BFS records, including failure.
    std::vector<EdmondsKarpAugmentation> augmentations; ///< Actual positive parent-path pushes.
    std::size_t captured_entries=0; ///< Exact retained logical-entry count.
};

/// @brief Compute maximum flow by shortest residual augmenting paths.
///
/// Validates the entire finite nonnegative graph and checked geometry before
/// adjacency allocation. Parallel/antiparallel edges and self loops have independent
/// residual pairs; per-edge flows come from cancellation residuals, not subtraction
/// of nearly equal large capacities. Uses strictly positive residuals, no epsilon.
/// Complete BFS preserves first-discovery input-order paths and supplies final cut
/// reachability. Each positive subtraction/addition and total must make finite,
/// representable progress; otherwise the call fails without a partial result.
/// This is checked double arithmetic, not an exact-real or arbitrary-precision solver.
/// Exact-arithmetic bound O(V+E+VE^2), working storage O(V+E); the initial/final
/// BFS also matters for empty graphs and isolated vertices. Inputs are unchanged,
/// outputs own their buffers, independent calls share no mutable state. No history
/// or optional capture quota is retained.
/// @param n Vertex count; source/sink must be distinct and in [0,n).
/// @param edges Borrowed directed (tail,head,capacity) records in input order.
/// @param source Source id.
/// @param sink Sink id, distinct from source.
/// @return Owned value, per-input-edge flows and ascending final source-reachable cut ids.
/// @throws std::invalid_argument Invalid terminals/endpoints or nonfinite/negative capacity.
/// @throws std::length_error Residual, vector, count or capture geometry exceeds size limits.
/// @throws std::overflow_error Residual or total-flow arithmetic becomes nonfinite.
/// @throws std::runtime_error A positive push loses representable residual or total progress.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto r=dm::algorithms::edmonds_karp(2,{{0,1,3}},0,1);
/// std::print("{} {}\n",r.max_flow,r.min_cut_source_side.size()); // 3 1
/// @endcode
/// @par References
/// - J. Edmonds and R. M. Karp (1972), Theoretical Improvements in Algorithmic Efficiency for Network Flow Problems, JACM 19(2), 248–264. doi:10.1145/321694.321699.
/// @see edmonds_karp_work(), edmonds_karp_trace(), max_flow_certificate()
MaxFlowResult edmonds_karp(std::size_t n,
    const std::vector<std::tuple<std::size_t,std::size_t,double>>& edges,
    std::size_t source,std::size_t sink);

/// @brief Return the ordinary flow and checked actual BFS/path work.
///
/// Same kernel, validation, numerical limits and O(V+E) working storage as
/// edmonds_karp(). Retains no optional input/search/path history or capture quota.
/// Complete searches inspect arcs even after discovering the sink.
/// @param n Vertex count; source/sink must be distinct and in [0,n).
/// @param edges Borrowed directed (tail,head,capacity) records in input order.
/// @param source Source id.
/// @param sink Sink id, distinct from source.
/// @return Owned flow result and counters from that run.
/// @throws std::invalid_argument Invalid terminals/endpoints or nonfinite/negative capacity.
/// @throws std::length_error Residual, vector, count or capture geometry exceeds size limits.
/// @throws std::overflow_error Residual or total-flow arithmetic becomes nonfinite.
/// @throws std::runtime_error A positive push loses representable residual or total progress.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto r=dm::algorithms::edmonds_karp_work(2,{{0,1,3},{0,1,2}},0,1);
/// std::print("{} {}\n",r.result.max_flow,r.work.bfs_arc_inspections); // 5 10
/// @endcode
/// @par References
/// - J. Edmonds and R. M. Karp (1972), Theoretical Improvements in Algorithmic Efficiency for Network Flow Problems, JACM 19(2), 248–264. doi:10.1145/321694.321699.
/// @see EdmondsKarpWork, edmonds_karp()
EdmondsKarpResult edmonds_karp_work(std::size_t n,
    const std::vector<std::tuple<std::size_t,std::size_t,double>>& edges,
    std::size_t source,std::size_t sink);

/// @brief Calculate minimum trace entries including input and the first complete BFS.
///
/// Pure checked geometry pass; does not allocate history or validate endpoints
/// and capacities. E+2+2V counts E input records, network/BFS headers and initial
/// levels/parents. Actual capture additionally needs later searches and pushes.
/// @param n Vertex count, at least two.
/// @param edge_count Number of original input edges.
/// @return Checked minimum retained-entry count E+2+2V.
/// @throws std::invalid_argument If n<2.
/// @throws std::length_error On arithmetic, residual-pool or vector geometry limits.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::edmonds_karp_trace_base_entries(2,1)); // 7
/// @endcode
/// @see edmonds_karp_trace()
std::size_t edmonds_karp_trace_base_entries(std::size_t n,std::size_t edge_count);

/// @brief Capture complete actual BFS parents/levels and oriented path updates.
///
/// Same kernel and numerical behavior as edmonds_karp(). Complete domain and
/// geometry validation precedes the optional quota. The minimum including the first
/// BFS is checked before owned copies or adjacency allocation; subsequent BFS and
/// path records reserve their full cost before traversal/history copying or pushes.
/// Default one million entries; zero disables only quota, not arithmetic checks.
/// See EdmondsKarpTrace for counted/excluded memory. No partial trace escapes failure.
/// @param n Vertex count; source/sink must be distinct and in [0,n).
/// @param edges Borrowed directed (tail,head,capacity) records in input order.
/// @param source Source id.
/// @param sink Sink id, distinct from source.
/// @param max_entries Logical retained-history cap, default 1000000; zero disables it.
/// @return Owned network, ordinary result, work and actual execution evidence.
/// @throws std::invalid_argument Invalid terminals/endpoints or nonfinite/negative capacity.
/// @throws std::length_error Residual, vector, count or capture geometry exceeds size limits.
/// @throws std::overflow_error Residual or total-flow arithmetic becomes nonfinite.
/// @throws std::runtime_error A positive push loses representable residual or total progress.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::edmonds_karp_trace(2,{{0,1,3}},0,1,14);
/// std::print("{} {}\n",t.result.max_flow,t.captured_entries); // 3 14
/// @endcode
/// @par References
/// - J. Edmonds and R. M. Karp (1972), Theoretical Improvements in Algorithmic Efficiency for Network Flow Problems, JACM 19(2), 248–264. doi:10.1145/321694.321699.
/// @see EdmondsKarpTrace, edmonds_karp_trace_base_entries(), edmonds_karp_work()
EdmondsKarpTrace edmonds_karp_trace(std::size_t n,
    const std::vector<std::tuple<std::size_t,std::size_t,double>>& edges,
    std::size_t source,std::size_t sink,std::size_t max_entries=1000000);

} // namespace datamunge::algorithms
