#include <gtest/gtest.h>

#include <datamunge/algorithms/hungarian.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

using datamunge::algorithms::AssignmentResult;
using datamunge::algorithms::hungarian;

namespace {

// Brute-force minimum-cost assignment over all n! permutations, as an independent reference.
double brute_force_min(const std::vector<std::vector<double>>& cost) {
    const std::size_t n = cost.size();
    std::vector<std::size_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    double best = std::numeric_limits<double>::infinity();
    do {
        double c = 0.0;
        for (std::size_t i = 0; i < n; ++i) c += cost[i][perm[i]];
        best = std::min(best, c);
    } while (std::next_permutation(perm.begin(), perm.end()));
    return best;
}

// Asserts that `assignment` is a permutation of 0..n-1 and that its summed cost equals `reported`.
void expect_valid_permutation(const AssignmentResult& r, const std::vector<std::vector<double>>& cost) {
    const std::size_t n = cost.size();
    ASSERT_EQ(r.assignment.size(), n);
    std::vector<char> seen(n, 0);
    double summed = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        ASSERT_LT(r.assignment[i], n);
        EXPECT_EQ(seen[r.assignment[i]], 0) << "column " << r.assignment[i] << " assigned twice";
        seen[r.assignment[i]] = 1;
        summed += cost[i][r.assignment[i]];
    }
    EXPECT_NEAR(summed, r.cost, 1e-9);
}

} // namespace

TEST(Hungarian, KnownSmallMatrixHasHandComputedOptimum) {
    // Workers (rows) x tasks (cols). The unique optimum is row0->col1, row1->col0, row2->col2:
    // 2 + 6 + 1 = 9. The next-best assignment costs 10, so the optimizer's choice is unambiguous.
    const std::vector<std::vector<double>> cost = {
        {9, 2, 7},
        {6, 4, 3},
        {5, 8, 1},
    };
    const auto r = hungarian(cost);
    EXPECT_EQ(r.assignment, (std::vector<std::size_t>{1, 0, 2}));
    EXPECT_NEAR(r.cost, 9.0, 1e-9);
    expect_valid_permutation(r, cost);
    EXPECT_NEAR(r.cost, brute_force_min(cost), 1e-9);
}

TEST(Hungarian, DiagonalMatrixPicksTheIdentity) {
    // A cheap diagonal against an expensive off-diagonal forces the identity assignment.
    const std::size_t n = 6;
    std::vector<std::vector<double>> cost(n, std::vector<double>(n, 100.0));
    for (std::size_t i = 0; i < n; ++i) cost[i][i] = 1.0;
    const auto r = hungarian(cost);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(r.assignment[i], i);
    EXPECT_NEAR(r.cost, static_cast<double>(n), 1e-9);
    expect_valid_permutation(r, cost);
}

TEST(Hungarian, SingletonMatrix) {
    const auto r = hungarian({{42.0}});
    EXPECT_EQ(r.assignment, (std::vector<std::size_t>{0}));
    EXPECT_NEAR(r.cost, 42.0, 1e-9);
}

TEST(Hungarian, HandlesNegativeCosts) {
    const std::vector<std::vector<double>> cost = {
        {-3.0, -1.0},
        {-2.0, -4.0},
    };
    const auto r = hungarian(cost);
    EXPECT_EQ(r.assignment, (std::vector<std::size_t>{0, 1})); // -3 + -4 = -7 beats -1 + -2 = -3
    EXPECT_NEAR(r.cost, -7.0, 1e-9);
    EXPECT_NEAR(r.cost, brute_force_min(cost), 1e-9);
}

TEST(Hungarian, MatchesBruteForceOnRandomInstances) {
    std::mt19937_64 rng(20240927);
    std::uniform_real_distribution<double> value(-5.0, 5.0);
    for (int trial = 0; trial < 400; ++trial) {
        const std::size_t n = 1 + rng() % 7; // 1..7 -> at most 7! = 5040 permutations
        std::vector<std::vector<double>> cost(n, std::vector<double>(n));
        for (auto& row : cost)
            for (auto& c : row) c = value(rng);
        const auto r = hungarian(cost);
        expect_valid_permutation(r, cost);
        EXPECT_NEAR(r.cost, brute_force_min(cost), 1e-9) << "trial " << trial << " n " << n;
    }
}

TEST(Hungarian, RejectsMalformedInput) {
    EXPECT_THROW(hungarian({}), std::invalid_argument);                 // empty
    EXPECT_THROW(hungarian({{1.0, 2.0}}), std::invalid_argument);       // 1x2, not square
    EXPECT_THROW(hungarian({{1.0, 2.0}, {3.0}}), std::invalid_argument); // ragged rows
}
