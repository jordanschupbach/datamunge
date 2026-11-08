#pragma once
/// @file blossom.hpp
/// @brief Checked maximum-cardinality general-graph matching and canonical blossom/augmentation records.
#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::algorithms {
/// @brief Sentinel for an unmatched vertex; never a valid vertex index.
/// Guard this value before indexing a mate vector or printing an edge: the
/// comparison vertex<mate[vertex] alone also succeeds for unmatched vertices.
/// @par Example
/// @code
/// const auto matching=dm::algorithms::blossom_maximum_matching(1,{});
/// std::print("{}\n",matching.mate[0]==dm::algorithms::kGeneralMatchingUnmatched); // true
/// @endcode
/// @see GeneralMatchingResult
inline constexpr std::size_t kGeneralMatchingUnmatched=static_cast<std::size_t>(-1);

/// @brief Owned maximum-cardinality matching, with symmetric mates and an explicit unmatched sentinel.
/// mate has one entry per input vertex. Matched endpoints refer to each other;
/// size counts pairs, not matched vertices. A maximum need not be unique and
/// its selected edges can depend on input order. No graph storage is borrowed.
/// @par Example
/// @code
/// const auto matching=dm::algorithms::blossom_maximum_matching(4,{{0,1},{1,2},{2,0},{1,3}});
/// std::print("{}\n",matching.size); // 2
/// @endcode
/// @see blossom_maximum_matching(), kGeneralMatchingUnmatched
struct GeneralMatchingResult {
  std::vector<std::size_t> mate; ///< Partner indices or kGeneralMatchingUnmatched.
  std::size_t size{0}; ///< Number of disjoint graph edges selected.
};

/// @brief One actual odd-cycle contraction, expanded to original vertex identities.
/// members includes the base's existing group and every merged group, sorted by
/// vertex index. It is an odd set, not a cyclic vertex ordering. Nested blossoms
/// therefore appear as larger sets; contractions reset at the next root search.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::blossom_maximum_matching_trace(4,{{0,1},{1,2},{2,0},{1,3}});
/// std::print("{} {}\n",trace.contractions.front().base,trace.contractions.front().members); // 2 [0, 1, 2]
/// @endcode
/// @see GeneralMatchingTrace, MatchingAugmentation
struct BlossomContraction {
  std::size_t search_index{0}; ///< Zero-based root search in the shared execution timeline.
  std::size_t root{0}; ///< Free root of that search.
  std::size_t base{0}; ///< Original vertex used as the merged representative.
  std::size_t first{0}; ///< First endpoint of the actual edge closing the odd cycle.
  std::size_t second{0}; ///< Second endpoint of that edge.
  std::vector<std::size_t> members; ///< All original vertices of the newly contracted blossom.
};

/// @brief One lifted augmenting path and the matching immediately after flipping it.
/// path uses original vertices, begins/ends free before this event and alternates
/// previously unmatched/matched edges. Its even vertex count gives one extra
/// selected edge. The snapshot is owned and independent of later augmentations.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::blossom_maximum_matching_trace(4,{{0,1},{1,2},{2,0},{1,3}});
/// std::print("{}\n",trace.augmentations.back().path); // [2, 0, 1, 3]
/// @endcode
/// @see GeneralMatchingTrace, GeneralMatchingResult
struct MatchingAugmentation {
  std::size_t search_index{0}; ///< Root search that found this path.
  std::vector<std::size_t> path; ///< Lifted path in root-to-free-endpoint order.
  GeneralMatchingResult matching; ///< Complete mate map and cardinality after the flip.
};

/// @brief Owned final matching, actual contraction sets, lifted paths and search work counters.
/// Root searches are numbered chronologically; contraction/augmentation vectors
/// each preserve their own execution order. Edge scans count visited adjacency
/// entries, including entries then skipped as matched or already contracted.
/// Ignored self-loops add no adjacency entries; duplicates do add entries.
/// Recording can require O(V^3) storage for expanded contraction sets, in addition
/// to O(V+E) graph/work storage. The unrecorded function allocates no event records.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::blossom_maximum_matching_trace(4,{{0,1},{1,2},{2,0},{1,3}});
/// std::print("{} {}\n",trace.result.size,trace.augmentations.size()); // 2 2
/// @endcode
/// @see blossom_maximum_matching_trace(), BlossomContraction, MatchingAugmentation
struct GeneralMatchingTrace {
  GeneralMatchingResult result; ///< Maximum matching from the canonical search.
  std::vector<BlossomContraction> contractions; ///< Odd-set contractions in chronological order.
  std::vector<MatchingAugmentation> augmentations; ///< Lifted paths and complete post-flip states.
  std::size_t searches{0}; ///< Number of BFS root searches, successful or unsuccessful.
  std::size_t edge_scans{0}; ///< Number of adjacency entries inspected across those searches.
};

/// @brief Find a maximum-cardinality matching with Edmonds' unweighted blossom search.
/// Vertices are 0,...,vertex_count-1. Self-loops are ignored after endpoint
/// validation; duplicate undirected edges and input adjacency order are preserved.
/// No caller input or shared state is mutated. Empty graphs are supported.
/// For V vertices and E supplied edges, time is O(V^3+V*E), storage O(V+E).
/// For simple graphs E=O(V^2), giving the usual O(V^3) bound. This is cardinality
/// matching, not weighted matching or a guarantee of any particular tied solution.
/// @param vertex_count Number of vertices, at most INT_MAX (internal indices are int).
/// @param edges Borrowed undirected endpoint pairs, including optional loops/duplicates.
/// @return Owned symmetric mates, unmatched sentinels and maximum pair count.
/// @throws std::length_error If vertex_count exceeds INT_MAX, before allocating graph storage.
/// @throws std::invalid_argument If any endpoint is outside [0,vertex_count).
/// @par Example
/// @code
/// const auto matching=dm::algorithms::blossom_maximum_matching(6,{{0,1},{1,2},{2,0},{1,3},{3,4},{2,5}});
/// std::print("{}\n",matching.size); // 3
/// @endcode
/// @par References
/// - J. Edmonds (1965), Paths, trees, and flowers, Canadian Journal of Mathematics
///   17, 449-467, doi:10.4153/CJM-1965-045-4. Array-based unweighted search here.
/// @see blossom_maximum_matching_trace(), GeneralMatchingResult
GeneralMatchingResult blossom_maximum_matching(std::size_t vertex_count,
    const std::vector<std::pair<std::size_t,std::size_t>>& edges);

/// @brief Run the same blossom search while recording actual contractions and lifted augmentations.
/// Uses the same canonical kernel as blossom_maximum_matching(), with identical
/// adjacency order and final result. Each snapshot contains original vertex IDs;
/// no report-side reconstruction of blossom lifting is necessary. Recording
/// keeps the O(V^3+V*E) time bound but can use O(V^3+E) storage. No input mutation
/// or shared writes; copies in the returned records own their data.
/// @param vertex_count Number of vertices, in [0,INT_MAX].
/// @param edges Undirected endpoint pairs; loops ignored, duplicates retained.
/// @return Final result, odd-set events, true augmenting paths and logical work counters.
/// @throws std::length_error If vertex_count exceeds INT_MAX, before allocation.
/// @throws std::invalid_argument If an endpoint is outside [0,vertex_count).
/// @par Example
/// @code
/// const auto trace=dm::algorithms::blossom_maximum_matching_trace(4,{{0,1},{1,2},{2,0},{1,3}});
/// std::print("{} {}\n",trace.contractions.size(),trace.augmentations.back().matching.size); // 1 2
/// @endcode
/// @par References
/// - J. Edmonds (1965), Paths, trees, and flowers, doi:10.4153/CJM-1965-045-4.
/// @see blossom_maximum_matching(), GeneralMatchingTrace
GeneralMatchingTrace blossom_maximum_matching_trace(std::size_t vertex_count,
    const std::vector<std::pair<std::size_t,std::size_t>>& edges);
} // namespace datamunge::algorithms
