#pragma once

/// @file proximity_graph_search.hpp
/// @brief Best-first nearest-neighbour search on a proximity graph, the search loop of navigable
///        small-world graphs and HNSW (Malkov et al. 2014, Malkov and Yashunin 2018).
///
/// The points are the vertices of a graph whose edges join nearby points. To find the neighbours
/// of a query, start at an entry vertex and repeatedly expand the closest unexpanded vertex found
/// so far, measuring the distance from the query to each of its unvisited neighbours. A result
/// list keeps the `ef` closest vertices seen; the search stops when the closest unexpanded vertex
/// is farther than all of them. `ef = 1` is greedy routing, which stops at the first local minimum
/// of the distance; larger `ef` keeps going past local minima.

#include <cstddef>
#include <vector>

namespace datamunge::geometry {

/// @brief Result of proximity_graph_search().
struct GraphSearchResult {
    std::vector<int>    indices;      ///< The k closest vertices found, nearest first.
    std::vector<double> distances;    ///< Their distances to the query.
    int                 evaluations = 0;  ///< Distance computations (vertices visited).
    int                 expansions  = 0;  ///< Vertices whose neighbour lists were scanned.
};

/// @brief Best-first search of a proximity graph for the k nearest vertices to @p query.
///
/// @param points The vertices' coordinates, all of one dimension.
/// @param adjacency Neighbour lists, one per vertex (for example from stats::KnnGraph).
/// @param query The query point.
/// @param entry The vertex the search starts from.
/// @param k How many neighbours to return, at least 1.
/// @param ef Size of the result list that controls stopping, raised to at least @p k.
/// @return The neighbours found and the work done.
/// @throws std::invalid_argument if the sizes disagree, @p entry is out of range or k < 1.
///
/// @par Example
/// @code
/// const auto hit = dm::geometry::proximity_graph_search(points, adjacency, query, 0, 1, 32);
/// // hit.indices[0] is (usually) the exact nearest neighbour; hit.evaluations its cost
/// @endcode
/// @see BestBinFirstKd, knn_brute_force()
GraphSearchResult proximity_graph_search(const std::vector<std::vector<double>>& points,
                                         const std::vector<std::vector<std::size_t>>& adjacency,
                                         const std::vector<double>& query, std::size_t entry, int k, int ef);

}  // namespace datamunge::geometry
