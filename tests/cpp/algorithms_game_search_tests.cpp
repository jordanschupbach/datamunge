#include <gtest/gtest.h>

#include <datamunge/algorithms/game_search.hpp>

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace alg = datamunge::algorithms;

TEST(GameSearch, BestFirstOrderingReachesKnuthMooreMinimalTree) {
    for (int b = 2; b <= 4; ++b)
        for (int d = 0; d <= 6; ++d) {
            const auto tree    = alg::random_game_tree(d, b, 100 + 10 * b + d);
            const auto ordered = alg::order_moves(tree, true, alg::MoveOrder::BestFirst);
            const auto ab      = alg::alpha_beta(ordered, true);
            EXPECT_EQ(ab.leaves_evaluated, alg::minimal_tree_leaves(b, d)) << "b=" << b << " d=" << d;
            EXPECT_DOUBLE_EQ(ab.value, alg::minimax(tree, true).value);
        }
}

TEST(GameSearch, WorstFirstPrunesLessThanAsGiven) {
    long long worst_total = 0, given_total = 0;
    for (std::uint64_t seed = 0; seed < 20; ++seed) {
        const auto tree  = alg::random_game_tree(6, 3, seed);
        const auto worst = alg::order_moves(tree, true, alg::MoveOrder::WorstFirst);
        const auto ab    = alg::alpha_beta(worst, true);
        EXPECT_LE(ab.leaves_evaluated, 729);
        EXPECT_DOUBLE_EQ(ab.value, alg::minimax(tree, true).value);
        worst_total += ab.leaves_evaluated;
        given_total += alg::alpha_beta(tree, true).leaves_evaluated;
    }
    EXPECT_GT(worst_total, given_total);
}

TEST(GameSearch, WindowGivesBoundsOutsideAndExactInside) {
    const auto   tree = alg::random_game_tree(6, 3, 11);
    const double v    = alg::minimax(tree, true).value;
    const auto   in   = alg::alpha_beta(tree, true, v - 0.01, v + 0.01);
    EXPECT_DOUBLE_EQ(in.value, v);
    const auto low = alg::alpha_beta(tree, true, v + 0.01, v + 0.02);  // fail low
    EXPECT_LE(low.value, v + 0.01);
    EXPECT_GE(low.value, v);
    const auto high = alg::alpha_beta(tree, true, v - 0.02, v - 0.01);  // fail high
    EXPECT_GE(high.value, v - 0.01);
    EXPECT_LE(high.value, v);
}

TEST(GameSearch, PearlBranchingFactor) {
    EXPECT_DOUBLE_EQ(alg::alpha_beta_branching_factor(1), 1.0);
    // b = 2: x^2 + x - 1 = 0 gives the golden-ratio conjugate, and xi/(1-xi) = golden ratio.
    EXPECT_NEAR(alg::alpha_beta_branching_factor(2), (1 + std::sqrt(5.0)) / 2, 1e-12);
    const double r3 = alg::alpha_beta_branching_factor(3);
    const double xi = r3 / (1 + r3);
    EXPECT_NEAR(xi * xi * xi + xi - 1, 0.0, 1e-12);
    EXPECT_THROW(alg::alpha_beta_branching_factor(0), std::invalid_argument);
}

TEST(GameSearch, TicTacToeIsADraw) {
    const auto tree = alg::tictactoe::game_tree();
    const auto mm   = alg::minimax(tree, true);
    EXPECT_EQ(mm.leaves_evaluated, 255168);
    EXPECT_EQ(mm.nodes_visited, 549946);
    EXPECT_DOUBLE_EQ(mm.value, 0.0);
    const auto ab = alg::alpha_beta(tree, true);
    EXPECT_DOUBLE_EQ(ab.value, 0.0);
    EXPECT_LT(ab.leaves_evaluated, mm.leaves_evaluated);
    EXPECT_EQ(alg::tictactoe::winner("XXXOO...."), 'X');
    EXPECT_EQ(alg::tictactoe::winner("XO.XO.X.O"), 'X');
    EXPECT_EQ(alg::tictactoe::winner("........."), '.');
    EXPECT_FALSE(alg::tictactoe::x_to_move("X........"));
    // X to move can win at once on square 2.
    const auto r = alg::minimax(alg::tictactoe::game_tree("XX.OO...."), true);
    EXPECT_DOUBLE_EQ(r.value, 1.0);
    EXPECT_EQ(r.best_child, 0);
    EXPECT_THROW(alg::tictactoe::game_tree("XXX......"), std::invalid_argument);
    EXPECT_THROW(alg::tictactoe::winner("abc"), std::invalid_argument);
}
