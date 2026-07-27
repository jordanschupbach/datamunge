#include <gtest/gtest.h>

#include <datamunge/algorithms/bellman_ford.hpp>

#include <cstddef>
#include <limits>
#include <queue>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

using datamunge::algorithms::bellman_ford;
using datamunge::algorithms::BellmanFordResult;
using datamunge::algorithms::kNoPredecessor;

namespace {

using Edge  = std::tuple<std::size_t, std::size_t, double>;
using Edges = std::vector<Edge>;

constexpr double kInf = std::numeric_limits<double>::infinity();

// Independent reference shortest paths for graphs with NON-NEGATIVE weights only: Dijkstra with a
// binary heap. Used to cross-check Bellman-Ford, which must agree with it on such graphs.
std::vector<double> dijkstra(std::size_t n, const Edges& edges, std::size_t source) {
    std::vector<std::vector<std::pair<std::size_t, double>>> adj(n);
    for (const auto& [u, v, w] : edges) adj[u].push_back({v, w});

    std::vector<double> dist(n, kInf);
    dist[source] = 0.0;
    using QItem = std::pair<double, std::size_t>; // (distance, vertex)
    std::priority_queue<QItem, std::vector<QItem>, std::greater<QItem>> pq;
    pq.push({0.0, source});
    while (!pq.empty()) {
        const auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[u]) continue; // stale heap entry
        for (const auto& [v, w] : adj[u])
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
    }
    return dist;
}

} // namespace

TEST(BellmanFord, KnownGraphWithNegativeEdges) {
    // CLRS Figure 24.4: vertices s=0, t=1, x=2, y=3, z=4. Several edges are negative but there is
    // no negative cycle, so shortest paths are well-defined and hand-computable.
    const std::size_t n = 5;
    const Edges edges = {
        {0, 1, 6}, {0, 3, 7}, {1, 2, 5}, {1, 3, 8}, {1, 4, -4},
        {2, 1, -2}, {3, 2, -3}, {3, 4, 9}, {4, 0, 2}, {4, 2, 7},
    };
    const auto r = bellman_ford(n, edges, 0);

    EXPECT_FALSE(r.has_negative_cycle);
    const std::vector<double> expected_dist = {0, 2, 4, 7, -2};
    EXPECT_EQ(r.distance, expected_dist);
    // Predecessor tree: t<-x, x<-y, y<-s, z<-t; s has none.
    EXPECT_EQ(r.predecessor[0], kNoPredecessor);
    EXPECT_EQ(r.predecessor[1], 2u);
    EXPECT_EQ(r.predecessor[2], 3u);
    EXPECT_EQ(r.predecessor[3], 0u);
    EXPECT_EQ(r.predecessor[4], 1u);
}

TEST(BellmanFord, ReachableNegativeCycleDetected) {
    // 0 -> 1, then the cycle 1 -> 2 -> 1 has total weight -2 and is reachable from source 0.
    const Edges edges = {{0, 1, 1}, {1, 2, -1}, {2, 1, -1}};
    const auto r = bellman_ford(3, edges, 0);
    EXPECT_TRUE(r.has_negative_cycle);
}

TEST(BellmanFord, UnreachableNegativeCycleNotDetected) {
    // Negative cycle 1 <-> 2 (weight -2) exists but is unreachable from source 0, which can only
    // reach vertex 3. Bellman-Ford must NOT flag it.
    const Edges edges = {{1, 2, -1}, {2, 1, -1}, {0, 3, 5}};
    const auto r = bellman_ford(4, edges, 0);
    EXPECT_FALSE(r.has_negative_cycle);
    EXPECT_EQ(r.distance[0], 0.0);
    EXPECT_EQ(r.distance[3], 5.0);
    EXPECT_EQ(r.distance[1], kInf); // unreachable
    EXPECT_EQ(r.distance[2], kInf); // unreachable
}

TEST(BellmanFord, UnreachableVerticesAreInfinity) {
    // Vertices 3 and 4 have no incoming edges from the source's component.
    const Edges edges = {{0, 1, 2}, {1, 2, 3}, {3, 4, 1}};
    const auto r = bellman_ford(5, edges, 0);
    EXPECT_FALSE(r.has_negative_cycle);
    EXPECT_EQ(r.distance[0], 0.0);
    EXPECT_EQ(r.distance[1], 2.0);
    EXPECT_EQ(r.distance[2], 5.0);
    EXPECT_EQ(r.distance[3], kInf);
    EXPECT_EQ(r.distance[4], kInf);
    EXPECT_EQ(r.predecessor[3], kNoPredecessor);
    EXPECT_EQ(r.predecessor[4], kNoPredecessor);
}

TEST(BellmanFord, MatchesDijkstraOnRandomNonNegativeGraphs) {
    // On graphs with non-negative weights, Bellman-Ford and Dijkstra must return identical
    // distances (including +infinity for unreachable vertices).
    std::mt19937_64 rng(20240927);
    for (int trial = 0; trial < 500; ++trial) {
        const std::size_t n = 2 + rng() % 11; // 2..12 vertices
        const std::size_t m = rng() % (2 * n + 1);
        Edges edges;
        std::uniform_int_distribution<std::size_t> vtx(0, n - 1);
        std::uniform_real_distribution<double> wt(0.0, 10.0);
        for (std::size_t e = 0; e < m; ++e)
            edges.push_back({vtx(rng), vtx(rng), wt(rng)});
        const std::size_t source = vtx(rng);

        const auto bf  = bellman_ford(n, edges, source);
        const auto ref = dijkstra(n, edges, source);
        ASSERT_FALSE(bf.has_negative_cycle) << "trial " << trial; // non-negative weights: impossible
        for (std::size_t v = 0; v < n; ++v) {
            if (ref[v] == kInf) {
                EXPECT_EQ(bf.distance[v], kInf) << "trial " << trial << " vertex " << v;
            } else {
                EXPECT_NEAR(bf.distance[v], ref[v], 1e-9) << "trial " << trial << " vertex " << v;
            }
        }
    }
}

TEST(BellmanFord, RejectsOutOfRangeInput) {
    EXPECT_THROW(bellman_ford(3, {}, 3), std::invalid_argument);            // source == n
    EXPECT_THROW(bellman_ford(3, {}, 99), std::invalid_argument);           // source > n
    EXPECT_THROW(bellman_ford(2, {{0, 5, 1.0}}, 0), std::invalid_argument); // 'to' endpoint >= n
    EXPECT_THROW(bellman_ford(2, {{7, 1, 1.0}}, 0), std::invalid_argument); // 'from' endpoint >= n
    EXPECT_THROW(bellman_ford(0, {}, 0), std::invalid_argument);            // empty graph, no valid source
}
