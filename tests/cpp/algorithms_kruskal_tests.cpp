#include <gtest/gtest.h>

#include <datamunge/algorithms/kruskal.hpp>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

using datamunge::algorithms::kruskal;
using datamunge::algorithms::MinimumSpanningTree;

namespace {

using Edge = std::tuple<std::size_t, std::size_t, double>;

// A minimal union-find, used by the tests to independently certify that a returned edge set is a
// valid forest (no accepted edge closes a cycle) and to count connected components.
struct UnionFind {
    std::vector<std::size_t> parent;
    explicit UnionFind(std::size_t n) : parent(n) { std::iota(parent.begin(), parent.end(), 0); }
    std::size_t find(std::size_t x) {
        while (parent[x] != x) x = parent[x] = parent[parent[x]];
        return x;
    }
    bool unite(std::size_t a, std::size_t b) {
        a = find(a);
        b = find(b);
        if (a == b) return false;
        parent[a] = b;
        return true;
    }
};

// Independent reference: Prim's algorithm (lazy, priority-queue). Returns the total MST weight of a
// connected graph, computed by a completely different method than Kruskal.
double prim_mst_weight(std::size_t n, const std::vector<Edge>& edges) {
    std::vector<std::vector<std::pair<std::size_t, double>>> adj(n);
    for (const auto& [u, v, w] : edges) {
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    std::vector<char> in_tree(n, 0);
    using Item = std::pair<double, std::size_t>; // (weight to reach, vertex)
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;
    pq.push({0.0, 0});
    double total = 0.0;
    std::size_t added = 0;
    while (!pq.empty()) {
        const auto [w, u] = pq.top();
        pq.pop();
        if (in_tree[u]) continue;
        in_tree[u] = 1;
        total += w;
        ++added;
        for (const auto& [v, ew] : adj[u])
            if (!in_tree[v]) pq.push({ew, v});
    }
    EXPECT_EQ(added, n) << "prim reference expected a connected graph";
    return total;
}

// Normalize an undirected edge to (min, max, weight) so edge sets compare regardless of endpoint order.
std::set<std::tuple<std::size_t, std::size_t, double>> edge_set(const std::vector<Edge>& edges) {
    std::set<std::tuple<std::size_t, std::size_t, double>> s;
    for (const auto& [u, v, w] : edges) s.insert({std::min(u, v), std::max(u, v), w});
    return s;
}

// Assert `mst` is a valid spanning tree of a connected n-vertex graph: n-1 edges, acyclic, spanning.
void expect_valid_spanning_tree(std::size_t n, const MinimumSpanningTree& mst) {
    ASSERT_TRUE(mst.is_connected);
    ASSERT_EQ(mst.edges.size(), n - 1); // a tree on n vertices has exactly n-1 edges
    UnionFind uf(n);
    for (const auto& [u, v, w] : mst.edges) {
        (void)w;
        EXPECT_TRUE(uf.unite(u, v)) << "MST edge " << u << "-" << v << " closes a cycle";
    }
    std::size_t components = 0; // after uniting all tree edges, a spanning tree is one component
    for (std::size_t i = 0; i < n; ++i)
        if (uf.find(i) == i) ++components;
    EXPECT_EQ(components, 1u);
}

// Build a random connected weighted graph on n vertices: a random spanning tree guarantees
// connectivity, then extra random edges are layered on top.
std::vector<Edge> random_connected_graph(std::size_t n, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> weight(0.1, 100.0);
    std::vector<Edge> edges;
    for (std::size_t i = 1; i < n; ++i) {
        const std::size_t j = rng() % i; // attach vertex i to some earlier vertex
        edges.push_back({i, j, weight(rng)});
    }
    const std::size_t extra = rng() % (2 * n + 1);
    for (std::size_t e = 0; e < extra; ++e) {
        const std::size_t u = rng() % n;
        const std::size_t v = rng() % n;
        if (u != v) edges.push_back({u, v, weight(rng)});
    }
    std::shuffle(edges.begin(), edges.end(), rng); // so input order does not favor Kruskal's sort
    return edges;
}

} // namespace

TEST(Kruskal, KnownSmallGraphUniqueMst) {
    // Six vertices, nine edges, all-distinct weights => the MST is unique. Hand computation:
    // accept (4,5,2),(3,4,3),(2,5,4),(0,3,5),(1,4,6); reject (0,1,7),(1,2,8),(1,3,9),(2,4,10) as
    // cycle-closing. Total weight = 2+3+4+5+6 = 20.
    const std::size_t n = 6;
    const std::vector<Edge> edges = {{0, 1, 7.0}, {0, 3, 5.0}, {1, 2, 8.0}, {1, 3, 9.0}, {1, 4, 6.0},
                                     {2, 4, 10.0}, {2, 5, 4.0}, {3, 4, 3.0}, {4, 5, 2.0}};
    const auto mst = kruskal(n, edges);

    EXPECT_TRUE(mst.is_connected);
    EXPECT_DOUBLE_EQ(mst.total_weight, 20.0);
    EXPECT_EQ(mst.edges.size(), 5u);

    const std::vector<Edge> expected = {{4, 5, 2.0}, {3, 4, 3.0}, {2, 5, 4.0}, {0, 3, 5.0}, {1, 4, 6.0}};
    EXPECT_EQ(edge_set(mst.edges), edge_set(expected));
    // Distinct weights => Kruskal accepts strictly in ascending weight order.
    EXPECT_EQ(mst.edges, expected);
    expect_valid_spanning_tree(n, mst);
}

TEST(Kruskal, DisconnectedGraphYieldsMinimumSpanningForest) {
    // Component {0,1,2}: MST edges (0,1,1)+(1,2,2)=3, rejecting (0,2,5). Component {3,4}: (3,4,4)=4.
    // Forest weight 3+4 = 7 over 3 edges; a spanning tree of 5 vertices would need 4 edges, so the
    // graph is reported disconnected.
    const std::size_t n = 5;
    const std::vector<Edge> edges = {{0, 1, 1.0}, {1, 2, 2.0}, {0, 2, 5.0}, {3, 4, 4.0}};
    const auto mst = kruskal(n, edges);

    EXPECT_FALSE(mst.is_connected);
    EXPECT_DOUBLE_EQ(mst.total_weight, 7.0);
    EXPECT_EQ(mst.edges.size(), 3u);
    const std::vector<Edge> expected = {{0, 1, 1.0}, {1, 2, 2.0}, {3, 4, 4.0}};
    EXPECT_EQ(edge_set(mst.edges), edge_set(expected));

    // The forest is acyclic and leaves exactly two components (one per connected piece).
    UnionFind uf(n);
    for (const auto& [u, v, w] : mst.edges) {
        (void)w;
        EXPECT_TRUE(uf.unite(u, v));
    }
    std::size_t components = 0;
    for (std::size_t i = 0; i < n; ++i)
        if (uf.find(i) == i) ++components;
    EXPECT_EQ(components, 2u);
}

TEST(Kruskal, SingleVertexIsTriviallyConnected) {
    const auto mst = kruskal(1, {});
    EXPECT_TRUE(mst.is_connected); // one vertex, zero edges, is a valid (trivial) spanning tree
    EXPECT_TRUE(mst.edges.empty());
    EXPECT_DOUBLE_EQ(mst.total_weight, 0.0);
}

TEST(Kruskal, NoEdgesAcrossSeveralVertices) {
    const auto mst = kruskal(3, {});
    EXPECT_FALSE(mst.is_connected); // three isolated vertices cannot form a spanning tree
    EXPECT_TRUE(mst.edges.empty());
    EXPECT_DOUBLE_EQ(mst.total_weight, 0.0);
}

TEST(Kruskal, EmptyGraphIsConnectedByConvention) {
    const auto mst = kruskal(0, {});
    EXPECT_TRUE(mst.is_connected);
    EXPECT_TRUE(mst.edges.empty());
    EXPECT_DOUBLE_EQ(mst.total_weight, 0.0);
}

TEST(Kruskal, MatchesPrimOnManyRandomConnectedGraphs) {
    std::mt19937_64 rng(20240927);
    for (int trial = 0; trial < 400; ++trial) {
        const std::size_t n = 2 + rng() % 30;
        const auto edges = random_connected_graph(n, rng);
        const auto mst = kruskal(n, edges);

        // Same weight as the independent Prim reference (MST weight is unique even under ties).
        EXPECT_NEAR(mst.total_weight, prim_mst_weight(n, edges), 1e-9) << "trial " << trial;
        // And the returned edges really are a spanning tree.
        expect_valid_spanning_tree(n, mst);

        // The total weight equals the sum of the returned edges' weights (bookkeeping check).
        double summed = 0.0;
        for (const auto& [u, v, w] : mst.edges) {
            (void)u;
            (void)v;
            summed += w;
        }
        EXPECT_NEAR(mst.total_weight, summed, 1e-9) << "trial " << trial;
    }
}

TEST(Kruskal, AcceptsEdgesInNonDecreasingWeightOrder) {
    std::mt19937_64 rng(7);
    for (int trial = 0; trial < 100; ++trial) {
        const std::size_t n = 3 + rng() % 20;
        const auto edges = random_connected_graph(n, rng);
        const auto mst = kruskal(n, edges);
        for (std::size_t i = 1; i < mst.edges.size(); ++i)
            EXPECT_LE(std::get<2>(mst.edges[i - 1]), std::get<2>(mst.edges[i])) << "trial " << trial;
    }
}

TEST(Kruskal, RejectsOutOfRangeEndpoints) {
    EXPECT_THROW(kruskal(3, {{0, 3, 1.0}}), std::invalid_argument); // v = 3 not in [0, 3)
    EXPECT_THROW(kruskal(3, {{3, 0, 1.0}}), std::invalid_argument); // u = 3 not in [0, 3)
    EXPECT_THROW(kruskal(0, {{0, 0, 1.0}}), std::invalid_argument); // no valid vertices at all
    EXPECT_NO_THROW(kruskal(3, {{0, 2, 1.0}, {1, 2, 2.0}}));        // in range
}
