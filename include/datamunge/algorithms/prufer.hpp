#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// @brief Encodes a labeled tree as its Prüfer sequence (Prüfer 1918). A tree on the \c n
///        labeled vertices \c 0..n-1 is turned into a sequence of \c n-2 vertex labels by
///        repeatedly deleting the smallest-labeled *leaf* and recording its unique neighbor,
///        until only two vertices remain. The map is a *bijection* between labeled trees on
///        \c n vertices and sequences in \c {0,...,n-1}^{n-2}; since there are exactly
///        \c n^(n-2) such sequences, this proves Cayley's formula for the number of labeled
///        trees.
///
/// @param n the number of vertices; the tree is over labels \c 0..n-1. Must be at least 2.
/// @param tree_edges the \c n-1 undirected edges of the tree, each a pair of endpoints in
///        \c [0, n).
/// @return the Prüfer sequence, of length \c n-2 (empty when \c n == 2).
/// @throws std::invalid_argument if \c n < 2, if the edge count is not exactly \c n-1, or if
///         any endpoint is \c >= n.
std::vector<std::size_t> tree_to_prufer(std::size_t n,
                                        const std::vector<std::pair<std::size_t, std::size_t>>& tree_edges);

/// @brief Decodes a Prüfer sequence back into its labeled tree -- the inverse of
///        @ref tree_to_prufer. The number of vertices is \c n = sequence.size() + 2. Using the
///        standard degree-count method, each sequence entry consumes the current smallest-labeled
///        leaf, joining it to that entry; the two vertices left with degree one at the end form
///        the final edge.
///
/// @param sequence a Prüfer sequence; every entry must be a valid vertex label, i.e. \c < n where
///        \c n = sequence.size() + 2. Any such sequence is a valid Prüfer code.
/// @return the \c n-1 edges of the reconstructed tree.
/// @throws std::invalid_argument if any entry is \c >= n.
std::vector<std::pair<std::size_t, std::size_t>> prufer_to_tree(const std::vector<std::size_t>& sequence);

} // namespace datamunge::algorithms
