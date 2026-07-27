#include <gtest/gtest.h>

#include <datamunge/algorithms/topological_sort.hpp>

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

using datamunge::algorithms::is_topological_order;
using datamunge::algorithms::topological_sort;
using datamunge::algorithms::TopologicalSortResult;

namespace {

using Edges = std::vector<std::pair<std::size_t, std::size_t>>;

} // namespace

TEST(TopologicalSort, OrdersASmallDag) {
    // A course-prerequisite DAG (edge = prereq -> course):
    //   0 -> 2, 0 -> 3, 1 -> 2, 1 -> 4, 2 -> 4, 2 -> 5, 3 -> 5.
    const std::size_t n = 6;
    const Edges edges = {{0, 2}, {0, 3}, {1, 2}, {1, 4}, {2, 4}, {2, 5}, {3, 5}};
    const auto r = topological_sort(n, edges);
    EXPECT_TRUE(r.is_dag);
    EXPECT_EQ(r.order.size(), n);
    EXPECT_TRUE(is_topological_order(n, edges, r.order));
    // Kahn with a FIFO queue and ascending source seeding makes the output deterministic.
    EXPECT_EQ(r.order, (std::vector<std::size_t>{0, 1, 3, 2, 4, 5}));
}

TEST(TopologicalSort, LinearChainIsItsOwnOrder) {
    const std::size_t n = 5;
    const Edges edges = {{0, 1}, {1, 2}, {2, 3}, {3, 4}};
    const auto r = topological_sort(n, edges);
    EXPECT_TRUE(r.is_dag);
    EXPECT_EQ(r.order, (std::vector<std::size_t>{0, 1, 2, 3, 4}));
    EXPECT_TRUE(is_topological_order(n, edges, r.order));
}

TEST(TopologicalSort, DetectsACycle) {
    const std::size_t n = 3;
    const Edges edges = {{0, 1}, {1, 2}, {2, 0}}; // a 3-cycle: no valid ordering exists
    const auto r = topological_sort(n, edges);
    EXPECT_FALSE(r.is_dag);
    EXPECT_TRUE(r.order.empty());
    EXPECT_FALSE(is_topological_order(n, edges, r.order));
}

TEST(TopologicalSort, NoEdgesKeepsEveryVertex) {
    const std::size_t n = 4;
    const auto r = topological_sort(n, {});
    EXPECT_TRUE(r.is_dag);
    EXPECT_EQ(r.order, (std::vector<std::size_t>{0, 1, 2, 3}));
}

TEST(TopologicalSort, ValidOrderOnRandomDags) {
    std::mt19937_64 rng(2024);
    for (int trial = 0; trial < 300; ++trial) {
        const std::size_t n = 2 + rng() % 30;
        // A random permutation fixes a valid vertex order; every edge is drawn to point *forward*
        // along it (perm[a] -> perm[b] with a < b), so the graph is acyclic by construction.
        std::vector<std::size_t> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        Edges edges;
        for (std::size_t a = 0; a < n; ++a)
            for (std::size_t b = a + 1; b < n; ++b)
                if (rng() % 4 == 0) edges.push_back({perm[a], perm[b]});
        std::shuffle(edges.begin(), edges.end(), rng); // edge order must not matter

        const auto r = topological_sort(n, edges);
        ASSERT_TRUE(r.is_dag) << "trial " << trial;
        ASSERT_EQ(r.order.size(), n) << "trial " << trial;
        ASSERT_TRUE(is_topological_order(n, edges, r.order)) << "trial " << trial;
    }
}

TEST(TopologicalSort, RejectsOutOfRangeEdges) {
    EXPECT_THROW(topological_sort(2, Edges{{0, 2}}), std::invalid_argument); // v == n
    EXPECT_THROW(topological_sort(2, Edges{{3, 1}}), std::invalid_argument); // u > n
    EXPECT_THROW(topological_sort(0, Edges{{0, 0}}), std::invalid_argument); // any edge when n == 0
}
