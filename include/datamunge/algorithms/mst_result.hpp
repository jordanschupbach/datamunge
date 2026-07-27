#pragma once

#include <cstddef>
#include <tuple>
#include <vector>

namespace datamunge::algorithms {

/// @brief A minimum spanning tree (or forest), shared by Kruskal's and Prim's algorithms: the
///        chosen edges as (u, v, weight) tuples in the order the algorithm accepts them, their
///        total weight, and whether the input graph was connected (if not, @ref edges is a
///        minimum spanning *forest* -- an MST of each connected component).
struct MinimumSpanningTree {
    std::vector<std::tuple<std::size_t, std::size_t, double>> edges;
    double total_weight{0.0};
    bool is_connected{false};
};

} // namespace datamunge::algorithms
