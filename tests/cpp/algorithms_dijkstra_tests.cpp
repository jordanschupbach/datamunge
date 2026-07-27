#include <gtest/gtest.h>

#include <datamunge/algorithms/dijkstra.hpp>

#include <limits>
#include <random>
#include <tuple>
#include <vector>

using datamunge::algorithms::dijkstra;
using datamunge::algorithms::kNoPredecessor;
using datamunge::algorithms::reconstruct_path;
using datamunge::algorithms::ShortestPathResult;

namespace {

using Edge = std::tuple<std::size_t, std::size_t, double>;
constexpr double kInf = std::numeric_limits<double>::infinity();

// A classic five-vertex directed instance with hand-verifiable shortest paths from vertex 0.
const std::vector<Edge> kSampleEdges = {
    {0, 1, 10.0}, {0, 4, 5.0}, {4, 1, 3.0}, {4, 2, 9.0}, {4, 3, 2.0},
    {1, 2, 1.0},  {1, 4, 2.0}, {2, 3, 4.0}, {3, 0, 7.0}, {3, 2, 6.0},
};

// Inline Bellman-Ford reference (non-negative weights): relax all edges n-1 times. Used to
// cross-check dijkstra's distance vector on random graphs.
std::vector<double> bellman_ford(std::size_t n, const std::vector<Edge>& edges, std::size_t source) {
    std::vector<double> dist(n, kInf);
    dist[source] = 0.0;
    for (std::size_t pass = 0; pass + 1 < n; ++pass)
        for (const auto& [u, v, w] : edges)
            if (dist[u] != kInf && dist[u] + w < dist[v]) dist[v] = dist[u] + w;
    return dist;
}

} // namespace

TEST(Dijkstra, KnownGraphHandComputedDistances) {
    // Six vertices: 0..4 as above, plus an isolated vertex 5 unreachable from 0.
    const auto r = dijkstra(6, kSampleEdges, 0);
    EXPECT_DOUBLE_EQ(r.distance[0], 0.0);
    EXPECT_DOUBLE_EQ(r.distance[1], 8.0); // 0->4->1
    EXPECT_DOUBLE_EQ(r.distance[2], 9.0); // 0->4->1->2
    EXPECT_DOUBLE_EQ(r.distance[3], 7.0); // 0->4->3
    EXPECT_DOUBLE_EQ(r.distance[4], 5.0); // 0->4
    EXPECT_EQ(r.distance[5], kInf);       // unreachable
}

TEST(Dijkstra, UnreachableVerticesAreInfiniteWithNoPredecessor) {
    const auto r = dijkstra(6, kSampleEdges, 0);
    EXPECT_EQ(r.distance[5], kInf);
    EXPECT_EQ(r.predecessor[5], kNoPredecessor);
    EXPECT_EQ(r.predecessor[0], kNoPredecessor); // the source has no predecessor either
    EXPECT_TRUE(reconstruct_path(r, 5).empty());
}

TEST(Dijkstra, ReconstructsAShortestPath) {
    const auto r = dijkstra(6, kSampleEdges, 0);
    const std::vector<std::size_t> expected = {0, 4, 1, 2}; // length 5 + 3 + 1 = 9
    EXPECT_EQ(reconstruct_path(r, 2), expected);
    EXPECT_EQ(reconstruct_path(r, 0), (std::vector<std::size_t>{0})); // source-to-source is trivial
    EXPECT_EQ(reconstruct_path(r, 4), (std::vector<std::size_t>{0, 4}));
}

TEST(Dijkstra, MatchesBellmanFordOnRandomGraphs) {
    std::mt19937_64 rng(20240927);
    for (int trial = 0; trial < 500; ++trial) {
        const std::size_t n = 2 + rng() % 12;
        std::uniform_int_distribution<std::size_t> vertex(0, n - 1);
        std::uniform_real_distribution<double> weight(0.0, 20.0);
        const std::size_t m = rng() % (2 * n + 1);
        std::vector<Edge> edges;
        for (std::size_t e = 0; e < m; ++e)
            edges.emplace_back(vertex(rng), vertex(rng), weight(rng));
        const std::size_t source = vertex(rng);

        const auto got = dijkstra(n, edges, source);
        const auto reference = bellman_ford(n, edges, source);
        for (std::size_t v = 0; v < n; ++v)
            EXPECT_DOUBLE_EQ(got.distance[v], reference[v]) << "trial " << trial << " vertex " << v;
    }
}

TEST(Dijkstra, UndirectedViaBothDirections) {
    // Path graph 0 - 1 - 2 with unit weights, encoded undirected by adding both directions.
    const std::vector<Edge> edges = {{0, 1, 1.0}, {1, 0, 1.0}, {1, 2, 1.0}, {2, 1, 1.0}};
    const auto r = dijkstra(3, edges, 0);
    EXPECT_DOUBLE_EQ(r.distance[0], 0.0);
    EXPECT_DOUBLE_EQ(r.distance[1], 1.0);
    EXPECT_DOUBLE_EQ(r.distance[2], 2.0);
}

TEST(Dijkstra, RejectsNegativeWeights) {
    const std::vector<Edge> edges = {{0, 1, 1.0}, {1, 2, -3.0}};
    EXPECT_THROW(dijkstra(3, edges, 0), std::invalid_argument);
}

TEST(Dijkstra, RejectsOutOfRangeEndpointsAndSource) {
    const std::vector<Edge> ok = {{0, 1, 1.0}};
    EXPECT_THROW(dijkstra(2, ok, 5), std::invalid_argument);              // source >= n
    EXPECT_THROW(dijkstra(2, {{0, 9, 1.0}}, 0), std::invalid_argument);   // 'to' endpoint >= n
    EXPECT_THROW(dijkstra(2, {{7, 1, 1.0}}, 0), std::invalid_argument);   // 'from' endpoint >= n
}
