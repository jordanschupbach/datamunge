#pragma once

#include <cstddef>
#include <tuple>
#include <vector>

namespace datamunge::algorithms {

/// @brief The outcome of a maximum-flow computation on a directed capacitated network.
struct MaxFlowResult {
    /// @brief The value of a maximum flow: the net flow leaving the source (equivalently, the
    ///        net flow entering the sink), which by max-flow min-cut equals the minimum cut
    ///        capacity.
    double max_flow{0.0};
    /// @brief edge_flow[i] is the flow assigned to input edge i, in the SAME order as the
    ///        `edges` argument. It always satisfies 0 <= edge_flow[i] <= capacity[i] and flow
    ///        conservation at every non-terminal vertex.
    std::vector<double> edge_flow;
    /// @brief The vertices reachable from the source in the FINAL residual graph -- the source
    ///        side S of a minimum s-t cut. Its complement is the sink side; the capacity of the
    ///        edges crossing S -> complement equals @ref max_flow.
    std::vector<std::size_t> min_cut_source_side;
};

/// @brief The Edmonds-Karp maximum-flow algorithm (Edmonds & Karp 1972): the Ford-Fulkerson
///        method specialized to always augment along a *shortest* (fewest-edge) augmenting path,
///        found by breadth-first search in the residual graph. Given a directed network with
///        non-negative edge capacities, a source, and a sink, it computes a maximum feasible
///        flow. Because BFS chooses shortest augmenting paths, the shortest-path distance from
///        the source to any vertex never decreases across augmentations, which bounds the number
///        of augmentations by O(V E) and the total running time by O(V E^2) -- independent of the
///        capacity magnitudes, so unlike generic Ford-Fulkerson it terminates even on irrational
///        capacities. By the max-flow min-cut theorem the maximum flow value equals the minimum
///        cut capacity, and the set of vertices still reachable from the source in the residual
///        graph at termination is exactly the source side of such a minimum cut.
///
///        Parallel edges (same endpoints) and antiparallel edges (u->v alongside v->u) are
///        handled correctly: each input edge keeps its own residual arc, so per-edge flows are
///        recovered unambiguously.
///
/// @param n      the number of vertices; valid vertex indices are 0..n-1.
/// @param edges  the directed capacity edges as (from, to, capacity) tuples with capacity >= 0.
/// @param source the source vertex s (must differ from @p sink and be < n).
/// @param sink   the sink vertex t (must be < n).
/// @return the maximum flow value, the per-edge flows, and the source side of a minimum cut.
/// @throws std::invalid_argument if source == sink, any endpoint is >= n, or any capacity < 0.
MaxFlowResult edmonds_karp(std::size_t n,
                           const std::vector<std::tuple<std::size_t, std::size_t, double>>& edges,
                           std::size_t source, std::size_t sink);

} // namespace datamunge::algorithms
