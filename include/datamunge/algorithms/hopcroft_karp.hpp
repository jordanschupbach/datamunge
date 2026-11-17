#pragma once
/// @file hopcroft_karp.hpp
/// @brief Shortest-layer bipartite matching with actual path histories and minimum-cover certificates.
#include <cstddef>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief SIZE_MAX sentinel for an unmatched vertex; never a valid retained index.
inline constexpr std::size_t kBipartiteUnmatched = static_cast<std::size_t>(-1);
/// @brief Ordered left/right endpoint pair; repeated edges and input order are retained.
/// @par Example
/// @code
/// const dm::algorithms::HopcroftKarpEdge e{0,1};std::print("{} {}\n",e.first,e.second); // 0 1
/// @endcode
using HopcroftKarpEdge = std::pair<std::size_t, std::size_t>;
/// @brief Owned maximum-cardinality matching in both orientations, with preserved field order.
/// Unmatched entries use kBipartiteUnmatched. A matching is not generally unique;
/// left roots and supplied adjacency order determine deterministic choices.
/// @par Example
/// @code
/// const auto r=dm::algorithms::hopcroft_karp(1,1,{{0,0}});
/// std::print("{} {}\n",r.size,r.match_left[0]); // 1 0
/// @endcode
struct BipartiteMatchingResult {
  std::vector<std::size_t>
      match_left; ///< Right partner for every left vertex, or unmatched sentinel.
  std::vector<std::size_t>
              match_right; ///< Left partner for every right vertex, or unmatched sentinel.
  std::size_t size{0};     ///< Actual matched-pair count, at most either side size.
};
/// @brief Complete graph and online logical-history policies, each zero disabling its cap.
/// Input caps count all supplied vertices/edge entries, including isolated or
/// repeated entries. Trace counts records and retained vertex/distance values,
/// not bytes/capacity/peak memory. No sampling or truncation.
/// @par Example
/// @code
/// const dm::algorithms::HopcroftKarpOptions o;std::print("{}\n",o.max_vertices); // 1000000
/// @endcode
struct HopcroftKarpOptions {
  std::size_t max_vertices{1000000};      ///< Complete left-plus-right vertex cap.
  std::size_t max_edges{1000000};         ///< Complete supplied edge-entry cap.
  std::size_t max_trace_entries{1000000}; ///< Online record/value cap, ignored without capture.
};
/// @brief Selected actual producing visits, excluding validation/copy/allocation/certificate/capture costs.
/// Duplicate edges count repeatedly when visited. Pair writes count matched-pair
/// assignments along paths, each of which writes both orientation arrays.
/// These counters are not timing, FLOPs or complete memory traffic.
/// @par Example
/// @code
/// const dm::algorithms::HopcroftKarpWork w;std::print("{}\n",w.bfs_calls); // 0
/// @endcode
struct HopcroftKarpWork {
  std::size_t adjacency_edges{0};  ///< Actual edge entries inserted into left adjacency.
  std::size_t bfs_calls{0};        ///< Actual BFS calls, including the final unsuccessful one.
  std::size_t initialized_left{0}; ///< Actual left entries initialized by all BFS calls.
  std::size_t queue_pops{
      0}; ///< Actual BFS queue removals, including skipped cutoff-frontier entries.
  std::size_t bfs_edges{0}; ///< Actual adjacency entries examined by BFS below its current cutoff.
  std::size_t dfs_roots{0}; ///< Actual free-left DFS root attempts during augmenting phases.
  std::size_t dfs_edges{
      0}; ///< Actual adjacency entries examined through phase-local current-edge cursors.
  std::size_t path_pairs_written{0}; ///< Actual path pairs assigned across all augmentations.
  std::size_t max_path_left_vertices{
      0}; ///< Actual high-water explicit DFS left-stack size, zero if unused.
};
/// @brief Owned actual matching, terminal-BFS minimum vertex cover and selected work.
/// cover_left/cover_right use their own side indices in ascending order. They
/// cover every supplied edge and their combined size equals matching size,
/// certifying optimality independently of tie choices. No references retained.
/// @par Example
/// @code
/// const auto r=dm::algorithms::hopcroft_karp_run(1,1,{{0,0}});
/// std::print("{} {}\n",r.result.size,r.cover_left.size()+r.cover_right.size()); // 1 1
/// @endcode
struct HopcroftKarpRun {
  BipartiteMatchingResult  result;     ///< Actual maximum matching, same historical layout.
  std::vector<std::size_t> cover_left; ///< Left vertices outside terminal alternating reachability.
  std::vector<std::size_t>
              cover_right; ///< Right vertices inside terminal alternating reachability.
  std::size_t augmenting_phases{
      0};                ///< Actual successful BFS/blocking-path phases, excluding terminal BFS.
  HopcroftKarpWork work; ///< Actual selected visits and stack high-water size.
};
/// @brief Owned actual shortest augmenting path immediately before its producing flip.
/// left[i]->right[i] are the path's forward unmatched edges. Before the flip,
/// right[i] is matched to left[i+1] except for the final free right vertex.
/// The root is free and all path vertices are distinct. k pairs mean2k-1 graph
/// edges; flipping gains one matched pair. No replay or input references.
/// @par Example
/// @code
/// const dm::algorithms::HopcroftKarpPath p;std::print("{}\n",p.left.size()); // 0
/// @endcode
struct HopcroftKarpPath {
  std::vector<std::size_t> left; ///< Actual left stack from free root to final left vertex.
  std::vector<std::size_t>
      right; ///< Actual chosen forward right endpoint for each left stack vertex.
};
/// @brief Owned actual BFS labels and subsequent shortest-path flips, including terminal BFS.
/// left_distance counts matched-right-to-left hops from free left roots, before
/// DFS dead-end pruning; SIZE_MAX means undiscovered. BFS stops expanding labels
/// at/above the shortest free-right distance, though already discovered frontier
/// labels can remain. Every augmentation has shortest_pairs forward pairs.
/// Terminal BFS has shortest_pairs=0 and no augmentations; its complete reachable
/// sets produce the run's minimum cover. Records own all arrays.
/// @par Example
/// @code
/// const dm::algorithms::HopcroftKarpPhase p;std::print("{}\n",p.shortest_pairs); // 0
/// @endcode
struct HopcroftKarpPhase {
  std::size_t bfs_index{0}; ///< One-based actual BFS invocation, zero for a default record.
  std::vector<std::size_t> left_distance; ///< Actual pre-DFS labels; SIZE_MAX is undiscovered.
  std::size_t              shortest_pairs{
      0}; ///< Shortest free-right distance in forward pairs, zero for terminal failure.
  std::vector<HopcroftKarpPath>
      augmentations; ///< Actual chronological complete paths flipped in this phase.
};
/// @brief Owned producing BFS/path history from an initially empty matching.
/// Each BFS admits1+n_left record/distance units before its work; each known path
/// admits1+2k record/vertex units before copies and flipping. Working buffers,
/// ordinary output, metadata/capacity/byte peak memory are excluded. No partial
/// result, truncation, post-hoc reconstruction or retained references.
/// @par Example
/// @code
/// const auto t=dm::algorithms::hopcroft_karp_trace(0,0,{});
/// std::print("{} {}\n",t.phases.size(),t.captured_entries); // 1 1
/// @endcode
struct HopcroftKarpTrace {
  HopcroftKarpRun                run;    ///< Actual matching/cover/selected work.
  std::vector<HopcroftKarpPhase> phases; ///< Actual BFS order, including final unsuccessful BFS.
  std::size_t captured_entries{0};       ///< Retained logical record/value units, not bytes.
};
/// @brief Compute maximum-cardinality bipartite matching using shortest-layer blocking paths.
/// Full endpoint validation precedes allocation. Duplicates/order are retained;
/// roots ascend and adjacency follows supplied order. BFS bounds free-right
/// distance, and an explicit DFS stack with phase-local edge cursors extracts
/// shortest augmenting paths only, avoiding recursive stack exhaustion. Empty
/// sides/edge lists are valid and leave unmatched sentinels. Tie choices can
/// differ from older code that allowed longer paths in a shortest-layer phase;
/// cardinality/validity and the original public signature/layout are preserved.
/// For V>=1, O((E+V)*sqrt(V)) time and O(E+V) working/output storage, with
/// supplied duplicate entries counted in E. No optional graph/history caps;
/// count/container representation checks still apply. No input mutation,
/// retained references or shared mutable state.
/// @param n_left Complete number of left vertices, zero allowed.
/// @param n_right Complete number of right vertices, zero allowed.
/// @param edges Borrowed ordered (left,right) endpoints, each strictly in range.
/// @return Owned maximum matching in both orientations and actual cardinality.
/// @throws std::invalid_argument On an invalid endpoint before policy/allocation checks.
/// @throws std::length_error On count/container representation limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::hopcroft_karp(2,2,{{0,0},{0,1},{1,0}}).size); // 2
/// @endcode
/// @par References
/// - J. E. Hopcroft and R. M. Karp (1973), An n^(5/2) Algorithm for Maximum
///   Matchings in Bipartite Graphs, doi:10.1137/0202019 . Uses shortest blocking
///   phases with an iterative current-edge DFS representation.
/// @see hopcroft_karp_run(), hopcroft_karp_trace()
BipartiteMatchingResult
hopcroft_karp(std::size_t n_left, std::size_t n_right, const std::vector<HopcroftKarpEdge>& edges);
/// @brief Run the shared shortest-layer kernel with graph policies and terminal minimum cover.
/// Same domains/order/storage as hopcroft_karp(). Validates every endpoint before
/// combined vertex/edge caps or adjacency allocation. Capture cap ignored.
/// Cover extraction uses actual terminal BFS reachability with no extra edge walk.
/// @param n_left Complete left-side size, zero allowed.
/// @param n_right Complete right-side size, zero allowed.
/// @param edges Borrowed ordered endpoint pairs.
/// @param options Combined vertex/edge caps, default1m each; zero disables each policy.
/// @return Owned actual matching, minimum cover, successful phase count and selected work.
/// @throws std::invalid_argument On invalid endpoints before policies.
/// @throws std::length_error On complete graph/count/container limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto r=dm::algorithms::hopcroft_karp_run(2,2,{{0,0},{0,1},{1,0}});
/// std::print("{} {}\n",r.result.size,r.augmenting_phases); // 2 2
/// @endcode
/// @see HopcroftKarpRun, HopcroftKarpOptions
HopcroftKarpRun hopcroft_karp_run(
    std::size_t                          n_left,
    std::size_t                          n_right,
    const std::vector<HopcroftKarpEdge>& edges,
    HopcroftKarpOptions                  options = {});
/// @brief Capture actual BFS labels and shortest path stacks before their producing flips.
/// Same contracts as hopcroft_karp_run(). Online complete BFS/path units are
/// admitted before work/copy/flip as described in HopcroftKarpTrace, with no
/// sampling, truncation or replay. Added storage is O(retained logical units)
/// plus metadata, beyond ordinary O(E+V) buffers/output.
/// @param n_left Complete left-side size, zero allowed.
/// @param n_right Complete right-side size, zero allowed.
/// @param edges Borrowed ordered endpoint pairs.
/// @param options Complete graph and online capture caps, default1m each; zero disables each.
/// @return Owned actual complete BFS/path history and producing run/certificate.
/// @throws std::invalid_argument On invalid endpoints before policies.
/// @throws std::length_error On graph/online history/count/container limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::hopcroft_karp_trace(2,2,{{0,0},{0,1},{1,0}});
/// std::print("{} {}\n",t.phases[1].shortest_pairs,t.phases[1].augmentations[0].left.size()); // 2 2
/// @endcode
/// @see HopcroftKarpTrace
HopcroftKarpTrace hopcroft_karp_trace(
    std::size_t                          n_left,
    std::size_t                          n_right,
    const std::vector<HopcroftKarpEdge>& edges,
    HopcroftKarpOptions                  options = {});
} // namespace datamunge::algorithms
