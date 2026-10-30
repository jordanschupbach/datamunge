#include <gtest/gtest.h>

#include <datamunge/algorithms/bidirectional_search.hpp>
#include <datamunge/algorithms/search_batch.hpp>

#include <cmath>
#include <random>
#include <set>

using namespace datamunge::algorithms;

namespace {

using Edges = std::vector<std::pair<std::size_t, std::size_t>>;
using WEdges = std::vector<std::tuple<std::size_t, std::size_t, double>>;

bool valid_path(const std::vector<std::size_t>& path, const Edges& edges, bool directed) {
    std::set<std::pair<std::size_t, std::size_t>> set(edges.begin(), edges.end());
    for (std::size_t i = 0; i + 1 < path.size(); ++i) {
        const auto a = path[i], b = path[i + 1];
        if (!set.count({a, b}) && (directed || !set.count({b, a}))) return false;
    }
    return true;
}

double path_weight(const std::vector<std::size_t>& path, const WEdges& edges) {
    double total = 0;
    for (std::size_t i = 0; i + 1 < path.size(); ++i) {
        double best = INFINITY;
        for (const auto& [u, v, w] : edges)
            if (u == path[i] && v == path[i + 1]) best = std::min(best, w);
        total += best;
    }
    return total;
}

} // namespace

TEST(BidirectionalBfs, MatchesBfsOnRandomGraphs) {
    std::mt19937 rng(7);
    for (int trial = 0; trial < 3000; ++trial) {
        const std::size_t n = 2 + rng() % 40;
        const bool directed = trial % 2 == 1;
        Edges edges;
        const std::size_t m = n + rng() % (2 * n);
        for (std::size_t k = 0; k < m; ++k) {
            const std::size_t u = rng() % n, v = rng() % n;
            if (u != v) edges.emplace_back(u, v);
        }
        const std::size_t s = rng() % n, t = rng() % n;
        const auto one = bfs_pair_search(n, edges, s, t, directed);
        const auto two = bidirectional_bfs(n, edges, s, t, directed);
        ASSERT_EQ(one.found, two.found) << trial;
        if (!one.found) continue;
        ASSERT_EQ(one.distance, two.distance) << trial;
        ASSERT_EQ(two.path.size(), static_cast<std::size_t>(two.distance) + 1);
        ASSERT_EQ(two.path.front(), s);
        ASSERT_EQ(two.path.back(), t);
        ASSERT_TRUE(valid_path(two.path, edges, directed)) << trial;
        ASSERT_TRUE(valid_path(one.path, edges, directed)) << trial;
        if (!directed) {
            const auto legacy = bidirectional_search(n, edges, s, t);
            ASSERT_EQ(legacy.path, two.path);
        }
    }
}

TEST(BidirectionalDijkstra, MatchesDijkstraOnRandomGraphs) {
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> weight(0.0, 5.0);
    for (int trial = 0; trial < 2000; ++trial) {
        const std::size_t n = 2 + rng() % 40;
        WEdges edges;
        const std::size_t m = 2 * n + rng() % (3 * n);
        for (std::size_t k = 0; k < m; ++k) {
            const std::size_t u = rng() % n, v = rng() % n;
            edges.emplace_back(u, v, trial % 3 == 0 ? std::round(weight(rng)) : weight(rng));  // ties too
        }
        const std::size_t s = rng() % n, t = rng() % n;
        const auto one = dijkstra_pair_search(n, edges, s, t);
        const auto two = bidirectional_dijkstra(n, edges, s, t);
        ASSERT_EQ(one.found, two.found) << trial;
        if (!one.found) continue;
        ASSERT_NEAR(one.distance, two.distance, 1e-9) << trial;
        ASSERT_EQ(two.path.front(), s);
        ASSERT_EQ(two.path.back(), t);
        ASSERT_NEAR(path_weight(two.path, edges), one.distance, 1e-9) << trial;
        ASSERT_NEAR(path_weight(one.path, edges), one.distance, 1e-9) << trial;
    }
}

TEST(BidirectionalDijkstra, FirstDoublySettledVertexIsNotTheAnswer) {
    // s=0, t=1, x=2: s-x-t costs 6 and x is settled by both sides first; the direct edge costs 5.
    const WEdges edges = {{0, 2, 3}, {2, 0, 3}, {2, 1, 3}, {1, 2, 3}, {0, 1, 5}, {1, 0, 5}};
    const auto r = bidirectional_dijkstra(3, edges, 0, 1);
    EXPECT_DOUBLE_EQ(r.distance, 5.0);
    EXPECT_EQ(r.path, (std::vector<std::size_t>{0, 1}));
}

TEST(BidirectionalSearch, SavesWorkOnAGridAndHandlesEdgeCases) {
    const std::size_t side = 41;
    Edges grid;
    for (std::size_t i = 0; i < side; ++i)
        for (std::size_t j = 0; j < side; ++j) {
            if (i + 1 < side) grid.emplace_back(i * side + j, (i + 1) * side + j);
            if (j + 1 < side) grid.emplace_back(i * side + j, i * side + j + 1);
        }
    const std::size_t s = 20 * side + 0, t = 20 * side + 40;
    const auto one = bfs_pair_search(side * side, grid, s, t);
    const auto two = bidirectional_bfs(side * side, grid, s, t);
    EXPECT_EQ(one.distance, 40.0);
    EXPECT_EQ(two.distance, 40.0);
    EXPECT_LT(two.expanded(), one.expanded());
    EXPECT_GT(two.forward_expanded, 0u);
    EXPECT_GT(two.backward_expanded, 0u);

    const auto balls = ball_sizes(side * side, grid, 20 * side + 20, 3);
    EXPECT_EQ(balls, (std::vector<std::size_t>{1, 5, 13, 25}));  // 2r^2 + 2r + 1

    const auto same = bidirectional_bfs(3, {{0, 1}}, 2, 2);
    EXPECT_TRUE(same.found);
    EXPECT_EQ(same.path, std::vector<std::size_t>{2});
    EXPECT_FALSE(bidirectional_bfs(3, {{0, 1}}, 0, 2).found);
    EXPECT_TRUE(std::isinf(bidirectional_dijkstra(3, {{0, 1, 1.0}}, 0, 2).distance));
    EXPECT_FALSE(bidirectional_bfs(2, {{0, 1}}, 1, 0, true).found);
    EXPECT_THROW(bidirectional_bfs(2, {{0, 5}}, 0, 1), std::invalid_argument);
    EXPECT_THROW(bidirectional_dijkstra(2, {{0, 1, -1.0}}, 0, 1), std::invalid_argument);
}
