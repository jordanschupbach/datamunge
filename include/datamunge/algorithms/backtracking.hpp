#pragma once

/// @file backtracking.hpp
/// @brief Backtracking over simple paths: exhaustive s-t path counting with search-tree statistics,
///        Hamiltonian-path search with branch ordering and pruning, and small graph generators.
///
/// backtracking_search() in search_batch.hpp stops at the first path it finds. The functions here
/// explore the whole tree (count_simple_paths()) or search for a path through every vertex
/// (hamiltonian_path()), and report how many search nodes that took.

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// @brief Edges of the rows x cols grid graph (4-neighbour lattice); vertex (r, c) is r * cols + c.
std::vector<std::pair<std::size_t, std::size_t>> grid_graph_edges(std::size_t rows, std::size_t cols);

/// @brief Edges of the knight's-move graph on a rows x cols board; square (r, c) is r * cols + c.
std::vector<std::pair<std::size_t, std::size_t>> knight_graph_edges(std::size_t rows, std::size_t cols);

/// @brief Edges of the complete graph on @p n vertices.
std::vector<std::pair<std::size_t, std::size_t>> complete_graph_edges(std::size_t n);

/// Totals from exhausting the backtracking tree of simple paths.
struct SimplePathCount {
    std::uint64_t paths     = 0;  ///< simple paths from start to goal.
    std::uint64_t nodes     = 0;  ///< search-tree nodes (calls), one per simple path from start not through goal.
    std::uint64_t dead_ends = 0;  ///< nodes other than the goal with no unvisited neighbour.
    std::size_t   max_depth = 0;  ///< most edges on any path explored.
};

/// @brief Counts every simple path from @p start to @p goal by backtracking.
///
/// The search extends a simple path one unvisited neighbour at a time, records a path on reaching
/// @p goal (and does not continue past it), and undoes the last step when no extension remains.
/// The work is exponential in general; @p node_limit (0 for none) stops it early.
/// @throws std::invalid_argument if @p start or @p goal is out of range.
SimplePathCount count_simple_paths(std::size_t n, const std::vector<std::pair<std::size_t, std::size_t>>& edges,
                                   std::size_t start, std::size_t goal, bool directed = false,
                                   std::uint64_t node_limit = 0);

/// How hamiltonian_path() orders the neighbours it tries.
enum class BranchOrder {
    Index,        ///< ascending vertex index.
    FewestOnward  ///< fewest unvisited neighbours first (Warnsdorff's rule), ties by index.
};

/// Result of a Hamiltonian-path search.
struct HamiltonianSearch {
    bool                     found     = false;  ///< whether a path through every vertex was found.
    bool                     hit_limit = false;  ///< whether the node limit stopped the search.
    std::vector<std::size_t> path;               ///< the Hamiltonian path, when found.
    std::uint64_t            nodes      = 0;     ///< search-tree nodes visited.
    std::uint64_t            backtracks = 0;     ///< times a vertex was removed from the path.
    std::uint64_t            pruned     = 0;     ///< subtrees cut by the connectivity test.
};

/// @brief Searches for a Hamiltonian path that starts at @p start, by backtracking.
///
/// @param order      the order in which neighbours are tried.
/// @param prune      when true, abandon a partial path whose unvisited vertices are not all
///                   reachable from its end through unvisited vertices (no completion exists).
/// @param node_limit the most search nodes to visit (0 for no limit).
/// @throws std::invalid_argument if @p start is out of range.
HamiltonianSearch hamiltonian_path(std::size_t n, const std::vector<std::pair<std::size_t, std::size_t>>& edges,
                                   std::size_t start, BranchOrder order = BranchOrder::Index, bool prune = false,
                                   std::uint64_t node_limit = 0);

}  // namespace datamunge::algorithms
