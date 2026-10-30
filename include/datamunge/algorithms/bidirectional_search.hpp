#pragma once

/// @file bidirectional_search.hpp
/// @brief Bidirectional breadth-first search and bidirectional Dijkstra, with one-directional
///        baselines that report the same work counters.
///
/// A bidirectional search grows one search tree from the start and one from the goal (along
/// reversed edges) and stops when the two can no longer improve the best path found where they
/// touch. If balls in the graph grow like \f$r^D\f$, a one-directional search to distance d
/// explores about \f$d^D\f$ vertices and the two half-searches about \f$2(d/2)^D\f$, a saving of
/// \f$2^{D-1}\f$; with exponential growth \f$b^r\f$ the saving is \f$b^{d/2}/2\f$.

#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// @brief Result of a single-pair search, with work counters.
struct PairSearchResult {
    std::vector<std::size_t> path;                ///< Vertices from start to goal; empty if unreachable.
    bool                     found{false};        ///< Whether the goal is reachable.
    double                   distance{0.0};       ///< Path length (edge count or total weight); +inf if not found.
    std::size_t              forward_expanded{0}; ///< Vertices expanded by the search from the start.
    std::size_t              backward_expanded{0};///< Vertices expanded by the search from the goal (0 for one-directional).
    std::size_t              edges_scanned{0};    ///< Edge relaxations performed in both directions.
    std::size_t              meeting_vertex{0};   ///< Vertex where the two searches joined (the goal for one-directional).

    /// @brief Total vertices expanded in both directions.
    [[nodiscard]] std::size_t expanded() const noexcept { return forward_expanded + backward_expanded; }
};

/// @brief Breadth-first search from @p start that stops once @p goal is dequeued.
///
/// @param n Number of vertices.
/// @param edges Edges u -> v (both directions when @p directed is false).
/// @param start,goal Terminal vertices.
/// @param directed Whether edges are one-way.
/// @return A shortest path in edges and the work done.
/// @throws std::invalid_argument for an out-of-range vertex.
PairSearchResult bfs_pair_search(std::size_t n, const std::vector<std::pair<std::size_t, std::size_t>>& edges,
                                 std::size_t start, std::size_t goal, bool directed = false);

/// @brief Bidirectional breadth-first search.
///
/// Each round expands one complete layer of the side whose frontier is smaller, recording every
/// edge that reaches a vertex already labelled by the other side. The search stops after the
/// first layer that finds such an edge; the best edge found in that layer gives a shortest path.
/// (Stopping at the first such edge without finishing the layer can miss a shorter path.)
///
/// @param n Number of vertices.
/// @param edges Edges u -> v; the backward search follows them in reverse.
/// @param start,goal Terminal vertices.
/// @param directed Whether edges are one-way.
/// @return A shortest path in edges and the work done on each side.
/// @throws std::invalid_argument for an out-of-range vertex.
///
/// @par Example
/// @code
/// const auto r = dm::algorithms::bidirectional_bfs(4, {{0, 1}, {1, 2}, {2, 3}}, 0, 3);
/// // r.path == {0, 1, 2, 3}, r.distance == 3
/// @endcode
PairSearchResult bidirectional_bfs(std::size_t n, const std::vector<std::pair<std::size_t, std::size_t>>& edges,
                                   std::size_t start, std::size_t goal, bool directed = false);

/// @brief Dijkstra's algorithm from @p start that stops once @p goal is settled.
///
/// @param n Number of vertices.
/// @param edges Directed edges (u, v, weight) with non-negative weights.
/// @param start,goal Terminal vertices.
/// @return A shortest path and the number of vertices settled.
/// @throws std::invalid_argument for an out-of-range vertex or a negative weight.
PairSearchResult dijkstra_pair_search(std::size_t n, const std::vector<std::tuple<std::size_t, std::size_t, double>>& edges,
                                      std::size_t start, std::size_t goal);

/// @brief Bidirectional Dijkstra (Pohl 1971).
///
/// A forward search from @p start on the edges and a backward search from @p goal on the reversed
/// edges alternate, each settling the vertex at the top of its own heap; the side with the
/// smaller top key goes next. Every scanned edge u -> v whose far end is labelled by the other
/// side offers a path of length \f$d_f(u)+w+d_b(v)\f$, and the best such value \f$\mu\f$ is kept.
/// The search stops when the two top keys satisfy \f$k_f+k_b\ge\mu\f$: any path not yet seen
/// must pass through a vertex unsettled on both sides and so is at least that long. The path
/// through the first vertex settled by both sides is *not* necessarily shortest.
///
/// @param n Number of vertices.
/// @param edges Directed edges (u, v, weight) with non-negative weights.
/// @param start,goal Terminal vertices.
/// @return A shortest path and the vertices settled on each side.
/// @throws std::invalid_argument for an out-of-range vertex or a negative weight.
PairSearchResult bidirectional_dijkstra(std::size_t n, const std::vector<std::tuple<std::size_t, std::size_t, double>>& edges,
                                        std::size_t start, std::size_t goal);

/// @brief Sizes of the balls around a vertex: element r is the number of vertices within r edges.
///
/// @param n Number of vertices.
/// @param edges Edges u -> v (both directions when @p directed is false).
/// @param source Centre vertex.
/// @param max_radius Largest radius reported.
/// @param directed Whether edges are one-way.
/// @return max_radius + 1 counts, nondecreasing.
/// @throws std::invalid_argument for an out-of-range vertex.
std::vector<std::size_t> ball_sizes(std::size_t n, const std::vector<std::pair<std::size_t, std::size_t>>& edges,
                                    std::size_t source, std::size_t max_radius, bool directed = false);

} // namespace datamunge::algorithms
