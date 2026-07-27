#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// @brief Tarjan's off-line lowest-common-ancestors algorithm (Tarjan 1979). Given a tree on
///        @p n nodes rooted at @p root and a batch of node pairs, it reports, for each pair
///        \((a,b)\), their *lowest common ancestor* (LCA): the deepest node that is an ancestor
///        of both. It is *off-line* -- every query is known before any is answered -- which is
///        exactly what lets a single depth-first search settle them all. The DFS maintains a
///        disjoint-set (union-find) structure: on finishing a node's subtree, that subtree is
///        merged into its parent's set and the set's *representative ancestor* is reset to the
///        parent. A node is coloured *black* once its subtree is fully explored; when finishing a
///        node @p u, any query \((u,v)\) whose other endpoint @p v is already black is answered by
///        =find_ancestor(v)= -- the representative of @p v's set is precisely their LCA. With
///        path-compressed union-find the whole batch costs \(O((n+q)\,\alpha(n))\), effectively
///        linear.
///
/// @param n          number of nodes, labelled \(0..n-1\).
/// @param root       the node the tree is rooted at (must be \(< n\)).
/// @param tree_edges the \(n-1\) undirected edges of the tree; each endpoint must be \(< n\).
/// @param queries    the pairs whose LCA is requested; each endpoint must be \(< n\).
/// @return a vector @c lca of the same length as @p queries, with @c lca[k] the lowest common
///         ancestor of @c queries[k].first and @c queries[k].second in the tree rooted at @p root.
/// @throws std::invalid_argument if @p root, any edge endpoint, or any query endpoint is \(\ge n\),
///         or if @p tree_edges does not contain exactly \(n-1\) edges.
[[nodiscard]] std::vector<std::size_t>
tarjan_offline_lca(std::size_t n, std::size_t root,
                   const std::vector<std::pair<std::size_t, std::size_t>>& tree_edges,
                   const std::vector<std::pair<std::size_t, std::size_t>>& queries);

} // namespace datamunge::algorithms
