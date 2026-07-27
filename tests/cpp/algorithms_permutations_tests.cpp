#include <gtest/gtest.h>

#include <datamunge/algorithms/permutations.hpp>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <set>
#include <vector>

using datamunge::algorithms::fisher_yates_shuffle;
using datamunge::algorithms::heap_permutations;
using datamunge::algorithms::rsk_insert;
using datamunge::algorithms::sjt_permutations;
using datamunge::algorithms::YoungTableaux;

namespace {

// The ground-truth set of all permutations of `base`, via the standard library.
std::set<std::vector<int>> all_permutations(std::vector<int> base) {
    std::sort(base.begin(), base.end());
    std::set<std::vector<int>> s;
    do {
        s.insert(base);
    } while (std::next_permutation(base.begin(), base.end()));
    return s;
}

std::size_t factorial(std::size_t n) {
    std::size_t f = 1;
    for (std::size_t i = 2; i <= n; ++i) f *= i;
    return f;
}

// Length of the longest strictly increasing subsequence (patience sorting, O(n log n)).
std::size_t lis_length(const std::vector<int>& a) {
    std::vector<int> tails;
    for (int x : a) {
        auto it = std::lower_bound(tails.begin(), tails.end(), x);
        if (it == tails.end())
            tails.push_back(x);
        else
            *it = x;
    }
    return tails.size();
}

// Positions at which two equal-length vectors differ.
std::vector<std::size_t> diff_positions(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<std::size_t> d;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) d.push_back(i);
    return d;
}

} // namespace

// ------------------------------------------------------------------------------------------------
// Fisher-Yates
// ------------------------------------------------------------------------------------------------

TEST(FisherYates, PermutesAndIsDeterministic) {
    std::vector<int> items(12);
    std::iota(items.begin(), items.end(), 0);
    const std::vector<int> sorted = items;

    for (std::uint64_t seed : {1u, 2u, 42u, 1000u}) {
        const std::vector<int> s = fisher_yates_shuffle(items, seed);
        std::vector<int>       t = s;
        std::sort(t.begin(), t.end());
        EXPECT_EQ(t, sorted);                                  // a permutation of the input
        EXPECT_EQ(s, fisher_yates_shuffle(items, seed));       // deterministic for a fixed seed
    }
}

TEST(FisherYates, EmptyAndSingleton) {
    EXPECT_TRUE(fisher_yates_shuffle({}, 5).empty());
    EXPECT_EQ(fisher_yates_shuffle({7}, 5), std::vector<int>({7}));
}

TEST(FisherYates, ApproximatelyUniformFirstPosition) {
    // Over many seeds, each value should land in position 0 a roughly equal share of the time.
    const int        n = 5;
    std::vector<int> items(n);
    std::iota(items.begin(), items.end(), 0);
    std::vector<int> first_counts(n, 0);
    const int        trials = 60000;
    for (int seed = 0; seed < trials; ++seed)
        ++first_counts[static_cast<std::size_t>(fisher_yates_shuffle(items, static_cast<std::uint64_t>(seed))[0])];
    for (int c : first_counts) {
        EXPECT_GT(c, trials / n * 0.9); // within +-10% of the uniform expectation
        EXPECT_LT(c, trials / n * 1.1);
    }
}

// ------------------------------------------------------------------------------------------------
// Heap's algorithm
// ------------------------------------------------------------------------------------------------

TEST(HeapPermutations, EnumeratesAllExactlyOnceBySingleSwaps) {
    for (int n = 0; n <= 6; ++n) {
        std::vector<int> items(static_cast<std::size_t>(n));
        std::iota(items.begin(), items.end(), 1);
        const auto perms = heap_permutations(items);

        EXPECT_EQ(perms.size(), factorial(static_cast<std::size_t>(n)));
        const std::set<std::vector<int>> unique(perms.begin(), perms.end());
        EXPECT_EQ(unique.size(), perms.size());                    // all distinct
        if (n >= 1) EXPECT_EQ(unique, all_permutations(items));    // exactly the full set

        for (std::size_t k = 1; k < perms.size(); ++k)
            EXPECT_EQ(diff_positions(perms[k - 1], perms[k]).size(), 2u); // consecutive: one swap
    }
}

// ------------------------------------------------------------------------------------------------
// Steinhaus-Johnson-Trotter
// ------------------------------------------------------------------------------------------------

TEST(SteinhausJohnsonTrotter, EnumeratesAllByAdjacentTranspositions) {
    for (std::size_t n = 0; n <= 6; ++n) {
        const auto perms = sjt_permutations(n);
        EXPECT_EQ(perms.size(), factorial(n));

        std::vector<int> base(n);
        std::iota(base.begin(), base.end(), 1);
        const std::set<std::vector<int>> unique(perms.begin(), perms.end());
        EXPECT_EQ(unique.size(), perms.size());
        if (n >= 1) EXPECT_EQ(unique, all_permutations(base));

        for (std::size_t k = 1; k < perms.size(); ++k) {
            const auto d = diff_positions(perms[k - 1], perms[k]);
            ASSERT_EQ(d.size(), 2u);
            EXPECT_EQ(d[1], d[0] + 1); // the two changed positions are adjacent
            EXPECT_EQ(perms[k - 1][d[0]], perms[k][d[1]]); // and their values are swapped
            EXPECT_EQ(perms[k - 1][d[1]], perms[k][d[0]]);
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Schensted / RSK
// ------------------------------------------------------------------------------------------------

TEST(Schensted, KnownExample) {
    // Row-inserting 4 1 3 2 builds P = {{1,2},{3},{4}} with first row length 2 (the LIS 1,3 or 1,2).
    const YoungTableaux t = rsk_insert({4, 1, 3, 2});
    ASSERT_EQ(t.p.size(), 3u);
    EXPECT_EQ(t.p[0], std::vector<int>({1, 2}));
    EXPECT_EQ(t.p[1], std::vector<int>({3}));
    EXPECT_EQ(t.p[2], std::vector<int>({4}));
    EXPECT_EQ(t.p[0].size(), lis_length({4, 1, 3, 2}));
}

TEST(Schensted, TableauShapeValidityAndSchenstedTheorem) {
    std::mt19937 rng(2025);
    for (int n = 1; n <= 8; ++n) {
        for (int trial = 0; trial < 400; ++trial) {
            std::vector<int> perm(static_cast<std::size_t>(n));
            std::iota(perm.begin(), perm.end(), 1);
            std::shuffle(perm.begin(), perm.end(), rng);

            const YoungTableaux t = rsk_insert(perm);
            ASSERT_EQ(t.p.size(), t.q.size());

            std::set<int> p_values, q_values;
            for (std::size_t r = 0; r < t.p.size(); ++r) {
                ASSERT_EQ(t.p[r].size(), t.q[r].size());                       // same shape
                if (r + 1 < t.p.size())
                    EXPECT_GE(t.p[r].size(), t.p[r + 1].size());               // weakly decreasing rows
                for (std::size_t c = 0; c < t.p[r].size(); ++c) {
                    if (c + 1 < t.p[r].size()) {
                        EXPECT_LT(t.p[r][c], t.p[r][c + 1]);                    // P rows increase
                        EXPECT_LT(t.q[r][c], t.q[r][c + 1]);                    // Q rows increase
                    }
                    if (r + 1 < t.p.size() && c < t.p[r + 1].size()) {
                        EXPECT_LT(t.p[r][c], t.p[r + 1][c]);                    // P columns increase
                        EXPECT_LT(t.q[r][c], t.q[r + 1][c]);                    // Q columns increase
                    }
                    p_values.insert(t.p[r][c]);
                    q_values.insert(t.q[r][c]);
                }
            }
            // P holds the permutation's values; Q is standard on 1..n.
            std::set<int> expected;
            for (int v = 1; v <= n; ++v) expected.insert(v);
            EXPECT_EQ(p_values, expected);
            EXPECT_EQ(q_values, expected);

            // Schensted's theorem: first-row length == length of longest increasing subsequence.
            EXPECT_EQ(t.p[0].size(), lis_length(perm));
        }
    }
}
