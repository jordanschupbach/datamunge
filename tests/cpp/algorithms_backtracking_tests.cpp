#include <datamunge/algorithms/backtracking.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <set>

namespace alg = datamunge::algorithms;

TEST(Backtracking, CompleteGraphPathCountIsFloorEFactorial) {
    // In K_n the simple paths from s to t number sum_{k=0}^{n-2} (n-2)!/k! = floor(e (n-2)!).
    for (std::size_t n = 3; n <= 9; ++n) {
        const auto   c = alg::count_simple_paths(n, alg::complete_graph_edges(n), 0, n - 1);
        double       f = 1;
        for (std::size_t k = 2; k <= n - 2; ++k) f *= static_cast<double>(k);
        EXPECT_EQ(c.paths, static_cast<std::uint64_t>(std::floor(std::exp(1.0) * f))) << n;
        EXPECT_EQ(c.max_depth, n - 1);
    }
}

TEST(Backtracking, GridCornerPathsMatchKnownSequence) {
    // OEIS A007764: corner-to-corner self-avoiding paths on an n x n grid of vertices.
    const std::uint64_t expected[] = {1, 2, 12, 184, 8512};
    for (std::size_t k = 1; k <= 5; ++k) {
        const auto c = alg::count_simple_paths(k * k, alg::grid_graph_edges(k, k), 0, k * k - 1);
        EXPECT_EQ(c.paths, expected[k - 1]) << k;
    }
    const auto limited = alg::count_simple_paths(25, alg::grid_graph_edges(5, 5), 0, 24, false, 100);
    EXPECT_EQ(limited.nodes, 100u);
}

TEST(Backtracking, KnightGraphDegrees) {
    const auto e = alg::knight_graph_edges(8, 8);
    EXPECT_EQ(e.size(), 168u);  // 336 moves / 2
}

TEST(Backtracking, HamiltonianPathIsValid) {
    const std::size_t rows = 5, cols = 5;
    const auto        e    = alg::knight_graph_edges(rows, cols);
    std::set<std::pair<std::size_t, std::size_t>> edge_set;
    for (auto [u, v] : e) {
        edge_set.insert({u, v});
        edge_set.insert({v, u});
    }
    for (auto order : {alg::BranchOrder::Index, alg::BranchOrder::FewestOnward})
        for (bool prune : {false, true}) {
            const auto h = alg::hamiltonian_path(rows * cols, e, 0, order, prune);
            ASSERT_TRUE(h.found);
            ASSERT_EQ(h.path.size(), rows * cols);
            EXPECT_EQ(std::set<std::size_t>(h.path.begin(), h.path.end()).size(), rows * cols);
            for (std::size_t k = 1; k < h.path.size(); ++k) EXPECT_TRUE(edge_set.count({h.path[k - 1], h.path[k]}));
        }
    // A knight's path from a square next to the corner of a 5x5 board does not exist (colour argument).
    const auto none = alg::hamiltonian_path(25, e, 1, alg::BranchOrder::FewestOnward, true);
    EXPECT_FALSE(none.found);
    EXPECT_FALSE(none.hit_limit);
    const auto cut = alg::hamiltonian_path(25, e, 1, alg::BranchOrder::Index, false, 50);
    EXPECT_TRUE(cut.hit_limit);
    EXPECT_THROW(alg::hamiltonian_path(3, {}, 5), std::invalid_argument);
}
