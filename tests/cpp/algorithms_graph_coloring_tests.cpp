#include <gtest/gtest.h>

#include <datamunge/algorithms/graph_coloring.hpp>

#include <algorithm>
#include <random>
#include <utility>
#include <vector>

using datamunge::algorithms::ColoringResult;
using datamunge::algorithms::greedy_coloring;
using datamunge::algorithms::is_proper_coloring;

namespace {

using Edge = std::pair<std::size_t, std::size_t>;
using EdgeList = std::vector<Edge>;

EdgeList complete_graph(std::size_t n) {
    EdgeList e;
    for (std::size_t a = 0; a < n; ++a)
        for (std::size_t b = a + 1; b < n; ++b) e.emplace_back(a, b);
    return e;
}

EdgeList cycle_graph(std::size_t n) {
    EdgeList e;
    for (std::size_t i = 0; i < n; ++i) e.emplace_back(i, (i + 1) % n);
    return e;
}

std::size_t max_degree(std::size_t n, const EdgeList& edges) {
    std::vector<std::size_t> deg(n, 0);
    for (const auto& [u, v] : edges)
        if (u != v) { ++deg[u]; ++deg[v]; }
    return deg.empty() ? 0 : *std::max_element(deg.begin(), deg.end());
}

} // namespace

TEST(GraphColoring, FiveCycleNeedsThreeColors) {
    // The 5-cycle is an odd cycle: chi(C5) = 3, and Welsh-Powell attains it.
    const std::size_t n = 5;
    const auto edges = cycle_graph(n);
    const auto r = greedy_coloring(n, edges);
    EXPECT_TRUE(is_proper_coloring(n, edges, r.color));
    EXPECT_EQ(r.num_colors, 3u);
    // Exact hand-computed assignment (order 0,1,2,3,4, all degree 2).
    const std::vector<std::size_t> expected = {0, 1, 0, 1, 2};
    EXPECT_EQ(r.color, expected);
}

TEST(GraphColoring, EvenCycleIsBipartite) {
    // Even cycles are bipartite: chi(C6) = 2.
    const std::size_t n = 6;
    const auto edges = cycle_graph(n);
    const auto r = greedy_coloring(n, edges);
    EXPECT_TRUE(is_proper_coloring(n, edges, r.color));
    EXPECT_EQ(r.num_colors, 2u);
}

TEST(GraphColoring, EdgelessGraphUsesOneColor) {
    const std::size_t n = 7;
    const EdgeList edges;
    const auto r = greedy_coloring(n, edges);
    EXPECT_TRUE(is_proper_coloring(n, edges, r.color));
    EXPECT_EQ(r.num_colors, 1u);
    for (std::size_t v = 0; v < n; ++v) EXPECT_EQ(r.color[v], 0u);
}

TEST(GraphColoring, CompleteGraphNeedsNColors) {
    // chi(K_n) = n, and the greedy is forced to use a fresh color for every vertex.
    for (std::size_t n = 1; n <= 8; ++n) {
        const auto edges = complete_graph(n);
        const auto r = greedy_coloring(n, edges);
        EXPECT_TRUE(is_proper_coloring(n, edges, r.color)) << "K_" << n;
        EXPECT_EQ(r.num_colors, n) << "K_" << n;
        // All colors distinct.
        std::vector<std::size_t> sorted = r.color;
        std::sort(sorted.begin(), sorted.end());
        EXPECT_TRUE(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end()) << "K_" << n;
    }
}

TEST(GraphColoring, SelfLoopsAreIgnored) {
    // A self-loop on vertex 0 must not prevent a proper 2-coloring of the single edge {0,1}.
    const std::size_t n = 2;
    const EdgeList edges = {{0, 0}, {0, 1}};
    const auto r = greedy_coloring(n, edges);
    EXPECT_TRUE(is_proper_coloring(n, edges, r.color));
    EXPECT_EQ(r.num_colors, 2u);
    EXPECT_NE(r.color[0], r.color[1]);
}

TEST(GraphColoring, AlwaysProperAndWithinDeltaPlusOneOnRandomGraphs) {
    std::mt19937_64 rng(20240927);
    for (int trial = 0; trial < 2000; ++trial) {
        const std::size_t n = 1 + rng() % 24;
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        const double p = unit(rng); // varied edge density, from sparse to near-complete
        EdgeList edges;
        for (std::size_t a = 0; a < n; ++a)
            for (std::size_t b = a + 1; b < n; ++b)
                if (unit(rng) < p) edges.emplace_back(a, b);

        const auto r = greedy_coloring(n, edges);
        ASSERT_TRUE(is_proper_coloring(n, edges, r.color)) << "trial " << trial << " n=" << n;
        EXPECT_LE(r.num_colors, max_degree(n, edges) + 1) << "trial " << trial; // Delta + 1 bound
        EXPECT_LE(r.num_colors, n) << "trial " << trial;
        EXPECT_GE(r.num_colors, 1u) << "trial " << trial;
        for (std::size_t v = 0; v < n; ++v) EXPECT_LT(r.color[v], r.num_colors) << "trial " << trial;
    }
}

TEST(GraphColoring, IsProperColoringDetectsAConflict) {
    // The path 0-1-2 with a coloring that paints 1 and 2 alike must be rejected.
    const std::size_t n = 3;
    const EdgeList edges = {{0, 1}, {1, 2}};
    const std::vector<std::size_t> bad = {0, 1, 1}; // edge {1,2} joins two color-1 vertices
    EXPECT_FALSE(is_proper_coloring(n, edges, bad));
    const std::vector<std::size_t> good = {0, 1, 0};
    EXPECT_TRUE(is_proper_coloring(n, edges, good));
}

TEST(GraphColoring, RejectsOutOfRangeEndpoints) {
    EXPECT_THROW(greedy_coloring(2, {{0, 2}}), std::invalid_argument); // 2 >= n
    EXPECT_THROW(greedy_coloring(0, {{0, 0}}), std::invalid_argument); // no valid vertices
    EXPECT_THROW(greedy_coloring(3, {{1, 5}}), std::invalid_argument);
}
