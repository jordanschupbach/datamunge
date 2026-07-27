#include <gtest/gtest.h>

#include <datamunge/algorithms/floyd_warshall.hpp>

#include <limits>
#include <random>
#include <tuple>
#include <vector>

using datamunge::algorithms::AllPairsShortestPaths;
using datamunge::algorithms::floyd_warshall;
using datamunge::algorithms::kNoVertex;
using datamunge::algorithms::reconstruct_path;

namespace {

using Edge = std::tuple<std::size_t, std::size_t, double>;
constexpr double kInf = std::numeric_limits<double>::infinity();

// The classic 5-vertex directed graph (CLRS Fig. 25.1), 0-indexed, with one negative edge (0->4)
// but no negative cycle. Its full all-pairs shortest-path distance matrix is known by hand.
std::vector<Edge> clrs_graph() {
    return {{0, 1, 3.0}, {0, 2, 8.0}, {0, 4, -4.0}, {1, 3, 1.0}, {1, 4, 7.0},
            {2, 1, 4.0}, {3, 0, 2.0}, {3, 2, -5.0}, {4, 3, 6.0}};
}

// Reference single-source shortest paths by Bellman-Ford (handles negative edges).
std::vector<double> bellman_ford(std::size_t n, const std::vector<Edge>& edges, std::size_t src) {
    std::vector<double> dist(n, kInf);
    dist[src] = 0.0;
    for (std::size_t iter = 0; iter + 1 < n; ++iter)
        for (const auto& [u, v, w] : edges)
            if (dist[u] != kInf && dist[u] + w < dist[v]) dist[v] = dist[u] + w;
    return dist;
}

} // namespace

TEST(FloydWarshall, KnownGraphMatchesHandComputedMatrix) {
    const auto apsp = floyd_warshall(5, clrs_graph());
    EXPECT_FALSE(apsp.has_negative_cycle);

    // Hand-computed shortest-path distance matrix for the CLRS graph.
    const std::vector<std::vector<double>> expected = {
        {0, 1, -3, 2, -4}, {3, 0, -4, 1, -1}, {7, 4, 0, 5, 3}, {2, -1, -5, 0, -2}, {8, 5, 1, 6, 0}};

    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 5; ++j)
            EXPECT_EQ(apsp.distance[i][j], expected[i][j]) << "at (" << i << "," << j << ")";
}

TEST(FloydWarshall, ReconstructsShortestPaths) {
    const auto apsp = floyd_warshall(5, clrs_graph());

    // The cheapest 0->2 route is 0 -> 4 -> 3 -> 2 with total weight -4 + 6 - 5 = -3.
    const std::vector<std::size_t> expected = {0, 4, 3, 2};
    EXPECT_EQ(reconstruct_path(apsp, 0, 2), expected);
    EXPECT_EQ(apsp.distance[0][2], -3.0);

    // A trivial self path is just the single endpoint.
    EXPECT_EQ(reconstruct_path(apsp, 3, 3), (std::vector<std::size_t>{3}));

    // Every reconstructed path must start at i, end at j, and have total weight == distance[i][j].
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 5; ++j) {
            const auto path = reconstruct_path(apsp, i, j);
            ASSERT_FALSE(path.empty()) << "(" << i << "," << j << ")"; // this graph is strongly connected
            EXPECT_EQ(path.front(), i);
            EXPECT_EQ(path.back(), j);
            double total = 0.0;
            for (std::size_t s = 0; s + 1 < path.size(); ++s) {
                double best = kInf; // lightest edge from path[s] to path[s+1]
                for (const auto& [u, v, w] : clrs_graph())
                    if (u == path[s] && v == path[s + 1] && w < best) best = w;
                ASSERT_NE(best, kInf) << "path uses a nonexistent edge";
                total += best;
            }
            EXPECT_EQ(total, apsp.distance[i][j]) << "(" << i << "," << j << ")";
        }
}

TEST(FloydWarshall, UnreachablePairsAreInfiniteWithNoPath) {
    // 0 -> 1 -> 2 is a one-way chain; nothing points back, so 1 and 2 cannot reach 0.
    const std::vector<Edge> chain = {{0, 1, 2.0}, {1, 2, 3.0}};
    const auto apsp = floyd_warshall(3, chain);

    EXPECT_EQ(apsp.distance[0][2], 5.0);
    EXPECT_EQ(apsp.distance[2][0], kInf);
    EXPECT_TRUE(reconstruct_path(apsp, 2, 0).empty());
    EXPECT_EQ(reconstruct_path(apsp, 0, 2), (std::vector<std::size_t>{0, 1, 2}));
}

TEST(FloydWarshall, MatchesBellmanFordOnRandomGraphs) {
    std::mt19937_64 rng(20240927);
    for (int trial = 0; trial < 400; ++trial) {
        const std::size_t n = 2 + rng() % 7; // 2..8 vertices
        std::vector<Edge> edges;
        std::uniform_int_distribution<int> present(0, 2);   // ~1/3 of ordered pairs get an edge
        std::uniform_int_distribution<int> weight(0, 9);    // nonnegative -> no negative cycle possible
        for (std::size_t u = 0; u < n; ++u)
            for (std::size_t v = 0; v < n; ++v)
                if (u != v && present(rng) == 0)
                    edges.emplace_back(u, v, static_cast<double>(weight(rng)));

        const auto apsp = floyd_warshall(n, edges);
        ASSERT_FALSE(apsp.has_negative_cycle) << "trial " << trial;
        for (std::size_t src = 0; src < n; ++src) {
            const auto ref = bellman_ford(n, edges, src);
            for (std::size_t j = 0; j < n; ++j)
                EXPECT_EQ(apsp.distance[src][j], ref[j]) << "trial " << trial << " (" << src << "," << j << ")";
        }
    }
}

TEST(FloydWarshall, MatchesBellmanFordOnNegativeWeightDags) {
    // Edges only go from a lower to a higher index -> an acyclic graph, so negative weights are
    // safe (no cycle can be negative) and the distances stay well-defined.
    std::mt19937_64 rng(7);
    for (int trial = 0; trial < 300; ++trial) {
        const std::size_t n = 2 + rng() % 7;
        std::vector<Edge> edges;
        std::uniform_int_distribution<int> present(0, 1);   // ~1/2 of forward pairs get an edge
        std::uniform_int_distribution<int> weight(-4, 9);   // allow negative weights
        for (std::size_t u = 0; u < n; ++u)
            for (std::size_t v = u + 1; v < n; ++v)
                if (present(rng) == 0) edges.emplace_back(u, v, static_cast<double>(weight(rng)));

        const auto apsp = floyd_warshall(n, edges);
        ASSERT_FALSE(apsp.has_negative_cycle) << "trial " << trial;
        for (std::size_t src = 0; src < n; ++src) {
            const auto ref = bellman_ford(n, edges, src);
            for (std::size_t j = 0; j < n; ++j)
                EXPECT_EQ(apsp.distance[src][j], ref[j]) << "trial " << trial << " (" << src << "," << j << ")";
        }
    }
}

TEST(FloydWarshall, DetectsNegativeCycle) {
    // 0 -> 1 -> 0 with total weight 1 + (-3) = -2 is a negative cycle.
    const std::vector<Edge> two_cycle = {{0, 1, 1.0}, {1, 0, -3.0}};
    EXPECT_TRUE(floyd_warshall(2, two_cycle).has_negative_cycle);

    // A negative self-loop is a length-1 negative cycle.
    const std::vector<Edge> self_loop = {{0, 0, -1.0}};
    EXPECT_TRUE(floyd_warshall(1, self_loop).has_negative_cycle);

    // A positive self-loop is not.
    const std::vector<Edge> positive_self = {{0, 0, 4.0}};
    EXPECT_FALSE(floyd_warshall(1, positive_self).has_negative_cycle);
}

TEST(FloydWarshall, RejectsOutOfRange) {
    // Edge endpoint >= n.
    EXPECT_THROW(floyd_warshall(2, {{0, 2, 1.0}}), std::invalid_argument);
    EXPECT_THROW(floyd_warshall(2, {{5, 1, 1.0}}), std::invalid_argument);

    // reconstruct_path endpoints must be < n.
    const auto apsp = floyd_warshall(3, {{0, 1, 1.0}});
    EXPECT_THROW((void)reconstruct_path(apsp, 0, 3), std::invalid_argument);
    EXPECT_THROW((void)reconstruct_path(apsp, 9, 0), std::invalid_argument);
}
