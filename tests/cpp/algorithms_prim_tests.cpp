#include <gtest/gtest.h>

#include <datamunge/algorithms/prim.hpp>

#include <algorithm>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

using datamunge::algorithms::MinimumSpanningTree;
using datamunge::algorithms::prim;

namespace {

using Edge = std::tuple<std::size_t, std::size_t, double>;

// A tiny disjoint-set (union-find), used both by the inline Kruskal reference and to certify that
// the edges Prim returns form an acyclic spanning tree.
struct DisjointSet {
    std::vector<std::size_t> parent;
    explicit DisjointSet(std::size_t n) : parent(n) { std::iota(parent.begin(), parent.end(), 0); }
    std::size_t find(std::size_t x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    }
    bool unite(std::size_t a, std::size_t b) { // returns false if a, b were already connected
        a = find(a);
        b = find(b);
        if (a == b) return false;
        parent[a] = b;
        return true;
    }
};

// Independent minimum-spanning-tree reference (Kruskal): sort edges ascending by weight and add
// each one whose endpoints are not yet connected, skipping the rest with union-find.
double kruskal_weight(std::size_t n, std::vector<Edge> edges) {
    std::sort(edges.begin(), edges.end(),
              [](const Edge& a, const Edge& b) { return std::get<2>(a) < std::get<2>(b); });
    DisjointSet ds(n);
    double total = 0.0;
    for (const auto& [u, v, w] : edges)
        if (ds.unite(u, v)) total += w;
    return total;
}

// A random *connected* weighted graph on n vertices: start from a random spanning tree (which
// guarantees connectivity), then sprinkle extra random edges on top.
std::vector<Edge> random_connected_graph(std::size_t n, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> weight(0.1, 100.0);
    std::vector<Edge> edges;
    std::vector<std::size_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    for (std::size_t i = 1; i < n; ++i) {
        const std::size_t parent = perm[rng() % i]; // attach perm[i] to an earlier vertex
        edges.emplace_back(perm[i], parent, weight(rng));
    }
    const std::size_t extra = rng() % (2 * n + 1);
    for (std::size_t k = 0; k < extra; ++k) {
        const std::size_t a = rng() % n;
        const std::size_t b = rng() % n;
        if (a != b) edges.emplace_back(a, b, weight(rng));
    }
    return edges;
}

} // namespace

TEST(Prim, KnownSmallGraphMatchesHandComputedMst) {
    // Six vertices, all-distinct edge weights => the MST is unique. Growing from vertex 0 Prim
    // attaches 1 (via 0-1, w3), then 3 (0-3, w4), then 2 (1-2, w5), then 4 (2-4, w2), then 5
    // (4-5, w1); total weight 3+4+5+2+1 = 15.
    const std::size_t n = 6;
    const std::vector<Edge> graph = {
        {0, 1, 3.0}, {0, 3, 4.0}, {1, 2, 5.0}, {1, 3, 6.0}, {1, 4, 7.0},
        {2, 4, 2.0}, {2, 5, 8.0}, {3, 4, 9.0}, {4, 5, 1.0}};

    const auto mst = prim(n, graph, 0);

    EXPECT_TRUE(mst.is_connected);
    EXPECT_DOUBLE_EQ(mst.total_weight, 15.0);
    const std::vector<Edge> expected = {
        {0, 1, 3.0}, {0, 3, 4.0}, {1, 2, 5.0}, {2, 4, 2.0}, {4, 5, 1.0}};
    EXPECT_EQ(mst.edges, expected);
    EXPECT_DOUBLE_EQ(mst.total_weight, kruskal_weight(n, graph));
}

TEST(Prim, SingleVertexHasEmptyTree) {
    const auto mst = prim(1, {}, 0);
    EXPECT_TRUE(mst.is_connected); // the lone vertex is trivially spanned
    EXPECT_TRUE(mst.edges.empty());
    EXPECT_DOUBLE_EQ(mst.total_weight, 0.0);
}

TEST(Prim, DisconnectedGraphSpansOnlyStartComponent) {
    // Two components: {0,1,2} and {3,4}. Growing from 0 spans the first only.
    const std::size_t n = 5;
    const std::vector<Edge> graph = {
        {0, 1, 1.0}, {1, 2, 2.0}, {0, 2, 3.0}, // component A
        {3, 4, 4.0}};                           // component B
    const auto mst = prim(n, graph, 0);
    EXPECT_FALSE(mst.is_connected);
    EXPECT_EQ(mst.edges.size(), 2u); // spans the 3-vertex component A
    EXPECT_DOUBLE_EQ(mst.total_weight, 3.0); // edges 0-1 (1) and 1-2 (2)
    for (const auto& [u, v, w] : mst.edges) {
        (void)w;
        EXPECT_LT(u, 3u);
        EXPECT_LT(v, 3u); // never reaches component B
    }

    // Starting inside component B spans only B.
    const auto mst_b = prim(n, graph, 3);
    EXPECT_FALSE(mst_b.is_connected);
    EXPECT_EQ(mst_b.edges.size(), 1u);
    EXPECT_DOUBLE_EQ(mst_b.total_weight, 4.0);
}

TEST(Prim, MatchesKruskalAndIsAValidSpanningTreeOnRandomGraphs) {
    std::mt19937_64 rng(20240927);
    for (int trial = 0; trial < 400; ++trial) {
        const std::size_t n = 1 + rng() % 30;
        const auto graph = random_connected_graph(n, rng);
        const std::size_t start = rng() % n;
        const auto mst = prim(n, graph, start);

        // The graph is connected by construction.
        ASSERT_TRUE(mst.is_connected) << "trial " << trial;
        ASSERT_EQ(mst.edges.size(), n - 1) << "trial " << trial; // a spanning tree has n-1 edges

        // The returned edges are acyclic and connect all n vertices (a valid spanning tree), and
        // every reported edge is a real edge orientation with u already-in-tree.
        DisjointSet ds(n);
        double sum = 0.0;
        for (const auto& [u, v, w] : mst.edges) {
            ASSERT_LT(u, n);
            ASSERT_LT(v, n);
            ASSERT_TRUE(ds.unite(u, v)) << "trial " << trial << ": tree contains a cycle";
            sum += w;
        }
        for (std::size_t v = 1; v < n; ++v)
            ASSERT_EQ(ds.find(v), ds.find(0)) << "trial " << trial << ": tree is not spanning";

        // Prim's total weight equals the independent Kruskal reference and its own edge sum.
        EXPECT_NEAR(mst.total_weight, sum, 1e-9) << "trial " << trial;
        EXPECT_NEAR(mst.total_weight, kruskal_weight(n, graph), 1e-6) << "trial " << trial;
    }
}

TEST(Prim, RejectsOutOfRange) {
    EXPECT_THROW(prim(3, {}, 3), std::invalid_argument);              // start == n
    EXPECT_THROW(prim(0, {}, 0), std::invalid_argument);             // start >= n (empty graph)
    EXPECT_THROW(prim(2, {{0, 2, 1.0}}, 0), std::invalid_argument);  // endpoint out of range
    EXPECT_THROW(prim(2, {{2, 0, 1.0}}, 0), std::invalid_argument);  // endpoint out of range
}
