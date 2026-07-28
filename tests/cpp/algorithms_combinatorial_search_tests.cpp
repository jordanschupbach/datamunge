#include <gtest/gtest.h>

#include <datamunge/algorithms/branch_and_bound.hpp>
#include <datamunge/algorithms/dpll.hpp>
#include <datamunge/algorithms/game_search.hpp>
#include <datamunge/algorithms/matrix_chain.hpp>

#include <cstdint>
#include <limits>
#include <random>
#include <vector>

using namespace datamunge::algorithms;

namespace {

// Brute-force minimum matrix-chain cost by trying every split (exponential).
long long brute_chain(const std::vector<int>& dims, std::size_t i, std::size_t j) {
    if (i == j) return 0;
    long long best = std::numeric_limits<long long>::max();
    for (std::size_t k = i; k < j; ++k) {
        const long long c = brute_chain(dims, i, k) + brute_chain(dims, k + 1, j) +
                            static_cast<long long>(dims[i - 1]) * dims[k] * dims[j];
        best = std::min(best, c);
    }
    return best;
}

} // namespace

TEST(Combinatorial, MatrixChainMatchesBruteForce) {
    std::mt19937                       rng(1);
    std::uniform_int_distribution<int> dim(1, 20), count(2, 7);
    for (int t = 0; t < 300; ++t) {
        const int          n = count(rng);
        std::vector<int>   dims(n + 1);
        for (int& d : dims) d = dim(rng);
        EXPECT_EQ(matrix_chain_order(dims).cost, brute_chain(dims, 1, n)) << "n=" << n;
    }
    // Textbook example (CLRS): dims 30x35x15x5x10x20x25 -> 15125.
    EXPECT_EQ(matrix_chain_order({30, 35, 15, 5, 10, 20, 25}).cost, 15125);
}

TEST(Combinatorial, KnapsackBranchAndBoundMatchesDP) {
    std::mt19937                       rng(2);
    std::uniform_int_distribution<int> vd(1, 30), wd(1, 15), nd(1, 14);
    for (int t = 0; t < 400; ++t) {
        const int        n = nd(rng);
        std::vector<int> values(n), weights(n);
        for (int& v : values) v = vd(rng);
        for (int& w : weights) w = wd(rng);
        const int cap = 30;

        // DP reference.
        std::vector<int> dp(cap + 1, 0);
        for (int i = 0; i < n; ++i)
            for (int c = cap; c >= weights[i]; --c) dp[c] = std::max(dp[c], dp[c - weights[i]] + values[i]);
        const int dp_best = dp[cap];

        const KnapsackResult r = knapsack_branch_and_bound(values, weights, cap);
        EXPECT_EQ(r.value, dp_best);

        // The reported set must be feasible and achieve the value.
        int w = 0, v = 0;
        for (int i = 0; i < n; ++i)
            if (r.chosen[i]) { w += weights[i]; v += values[i]; }
        EXPECT_LE(w, cap);
        EXPECT_EQ(v, r.value);
    }
}

namespace {
// Build a random game tree of given depth/branching; leaves get random values.
GameNode random_tree(std::mt19937& rng, int depth, int branching) {
    GameNode n;
    if (depth == 0) {
        n.leaf = true;
        n.value = std::uniform_int_distribution<int>(-50, 50)(rng);
        return n;
    }
    for (int i = 0; i < branching; ++i) n.children.push_back(random_tree(rng, depth - 1, branching));
    return n;
}
} // namespace

TEST(Combinatorial, AlphaBetaEqualsMinimaxWithFewerLeaves) {
    std::mt19937 rng(3);
    for (int t = 0; t < 400; ++t) {
        const GameNode root = random_tree(rng, 4, 3); // depth 4, branching 3 -> 81 leaves
        const GameResult mm = minimax(root, true);
        const GameResult ab = alpha_beta(root, true);
        EXPECT_DOUBLE_EQ(ab.value, mm.value);
        EXPECT_LE(ab.leaves_evaluated, mm.leaves_evaluated); // pruning never evaluates more
    }
}

namespace {
// Brute-force satisfiability over all 2^n assignments.
bool brute_sat(int n, const std::vector<std::vector<int>>& clauses) {
    for (std::uint32_t mask = 0; mask < (1u << n); ++mask) {
        bool ok = true;
        for (const auto& clause : clauses) {
            bool c = false;
            for (int lit : clause) {
                const int v = std::abs(lit);
                const bool val = (mask >> (v - 1)) & 1u;
                if ((lit > 0 && val) || (lit < 0 && !val)) { c = true; break; }
            }
            if (!c) { ok = false; break; }
        }
        if (ok) return true;
    }
    return false;
}
} // namespace

TEST(Combinatorial, DpllMatchesBruteForceSat) {
    std::mt19937                       rng(4);
    std::uniform_int_distribution<int> nvars(2, 8), nclauses(1, 20);
    for (int t = 0; t < 500; ++t) {
        const int n = nvars(rng), m = nclauses(rng);
        std::uniform_int_distribution<int> vpick(1, n);
        std::vector<std::vector<int>>      clauses;
        for (int c = 0; c < m; ++c) {
            std::vector<int> clause;
            const int        k = 1 + rng() % 3; // 1..3 literals
            for (int j = 0; j < k; ++j) {
                const int v = vpick(rng);
                clause.push_back((rng() & 1) ? v : -v);
            }
            clauses.push_back(clause);
        }
        const SatResult r = dpll_sat(n, clauses);
        EXPECT_EQ(r.satisfiable, brute_sat(n, clauses));

        if (r.satisfiable) {
            // Verify the returned assignment satisfies every clause.
            for (const auto& clause : clauses) {
                bool c = false;
                for (int lit : clause) {
                    const int v = std::abs(lit);
                    if ((lit > 0 && r.assignment[v] == 1) || (lit < 0 && r.assignment[v] == -1)) { c = true; break; }
                }
                EXPECT_TRUE(c);
            }
        }
    }
}
