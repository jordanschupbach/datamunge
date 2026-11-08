#pragma once
/// @file mst_result.hpp
/// @brief Shared owned edge and minimum-spanning-result types.

#include <cstddef>
#include <tuple>
#include <vector>

namespace datamunge::algorithms {

/// @brief Edge endpoints and weight; direction and weight requirements follow the consuming algorithm.
/// @par Example
/// @code
/// const dm::algorithms::WeightedEdge edge{0,1,2.5};
/// std::print("{}\n",std::get<2>(edge)); // 2.5
/// @endcode
/// @see boruvka(), MinimumSpanningTree
using WeightedEdge = std::tuple<std::size_t,std::size_t,double>;

/// @brief Owned selected edges, their total weight and a whole-graph connectivity flag.
/// Boruvka and Kruskal return a minimum spanning forest of every input component.
/// Prim returns only the tree reachable from its starting vertex. Edge order and
/// tie rules follow the producing algorithm; no graph storage is borrowed.
/// @par Example
/// @code
/// const auto tree=dm::algorithms::boruvka(2,{{0,1,2.5}});
/// std::print("{} {}\n",tree.edges.size(),tree.total_weight); // 1 2.5
/// @endcode
/// @see boruvka(), kruskal(), prim()
struct MinimumSpanningTree {
    std::vector<WeightedEdge> edges; ///< Selected (u,v,weight) tuples; orientations follow the producer.
    double total_weight{0.0}; ///< Sum of selected weights, with the producer's accumulation policy.
    bool is_connected{false}; ///< Whether the producer reached the whole input graph; default false.
};

} // namespace datamunge::algorithms
