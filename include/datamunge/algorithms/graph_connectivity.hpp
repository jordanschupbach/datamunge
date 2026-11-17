/// @file graph_connectivity.hpp
/// @brief Compatibility aggregation of focused checked graph and sparse families.
#pragma once
#include <datamunge/algorithms/tarjan_scc.hpp>

// Graph structure algorithms:
//   - kosaraju_scc / tarjan_scc : strongly connected components of a directed
//     graph, by two different classic methods (they agree on the partition)
//   - bron_kerbosch             : all maximal cliques of an undirected graph
//
// Graphs are given as adjacency lists over vertices 0..n-1.

#include <algorithm>
#include <datamunge/algorithms/bron_kerbosch.hpp>
#include <datamunge/algorithms/kosaraju.hpp>
#include <vector>

namespace datamunge::algorithms {} // namespace datamunge::algorithms
