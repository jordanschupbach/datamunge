#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// @brief Sentinel for an unmatched vertex in a BipartiteMatchingResult.
inline constexpr std::size_t kBipartiteUnmatched = static_cast<std::size_t>(-1);

struct BipartiteMatchingResult {
    /// @brief match_left[l] is the right vertex matched to left vertex l (or kBipartiteUnmatched).
    std::vector<std::size_t> match_left;
    /// @brief match_right[r] is the left vertex matched to right vertex r (or kBipartiteUnmatched).
    std::vector<std::size_t> match_right;
    /// @brief Cardinality of the matching (number of matched pairs).
    std::size_t size{0};
};

/// @brief The Hopcroft-Karp algorithm (Hopcroft & Karp 1973) for *maximum-cardinality matching*
///        in a bipartite graph. Given a left vertex set \(\{0,\dots,n_{left}-1\}\), a right vertex
///        set \(\{0,\dots,n_{right}-1\}\), and a set of undirected edges between them, it returns a
///        largest possible set of edges no two of which share an endpoint. It improves on the
///        simple augmenting-path method (Kuhn's algorithm) by working in *phases*: each phase runs
///        one breadth-first search that computes the length of a shortest augmenting path and
///        stratifies the graph into distance layers, then a single depth-first search that greedily
///        extracts a *maximal set of vertex-disjoint shortest* augmenting paths and augments along
///        all of them at once. Because the shortest augmenting-path length strictly increases from
///        phase to phase, only \(O(\sqrt{V})\) phases are needed, giving an overall running time of
///        \(O(E\sqrt{V})\).
///
/// @param n_left number of left vertices.
/// @param n_right number of right vertices.
/// @param edges undirected edges as (left_index, right_index) pairs; duplicates and any ordering
///        are tolerated. Every endpoint must be in range or std::invalid_argument is thrown.
/// @return the maximum-cardinality matching, in both orientations plus its size.
BipartiteMatchingResult hopcroft_karp(std::size_t n_left, std::size_t n_right,
                                      const std::vector<std::pair<std::size_t, std::size_t>>& edges);

} // namespace datamunge::algorithms
