#include <gtest/gtest.h>

#include <datamunge/algorithms/trigram_search.hpp>
#include <datamunge/algorithms/uniform_binary_search.hpp>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

using namespace datamunge::algorithms;

// ---------------- Uniform binary search ----------------

TEST(UniformBinarySearch, MatchesReferenceRandom) {
    std::mt19937_64 rng(1);
    for (int t = 0; t < 3000; ++t) {
        const int        n = static_cast<int>(rng() % 200);
        std::vector<int> a(n);
        for (auto& x : a) x = static_cast<int>(rng() % 400) - 200;
        std::sort(a.begin(), a.end());
        a.erase(std::unique(a.begin(), a.end()), a.end());
        const auto table = uniform_search_table(static_cast<int>(a.size()));

        for (int q = -210; q <= 210; q += 7) {
            const std::size_t got = uniform_binary_search(a, q, table);
            const bool        present = std::binary_search(a.begin(), a.end(), q);
            if (present) {
                ASSERT_NE(got, static_cast<std::size_t>(-1)) << "missed " << q;
                EXPECT_EQ(a[got], q);
            } else {
                EXPECT_EQ(got, static_cast<std::size_t>(-1)) << "false hit " << q;
            }
        }
    }
}

TEST(UniformBinarySearch, KnownCases) {
    std::vector<int> a = {1, 3, 5, 7, 9, 11, 13};
    const auto       tbl = uniform_search_table(static_cast<int>(a.size()));
    EXPECT_EQ(uniform_binary_search(a, 7, tbl), 3u);
    EXPECT_EQ(uniform_binary_search(a, 1, tbl), 0u);
    EXPECT_EQ(uniform_binary_search(a, 13, tbl), 6u);
    EXPECT_EQ(uniform_binary_search(a, 8, tbl), static_cast<std::size_t>(-1));
    EXPECT_EQ(uniform_binary_search(a, 100, tbl), static_cast<std::size_t>(-1));
    EXPECT_EQ(uniform_binary_search(std::vector<int>{}, 5, uniform_search_table(0)),
              static_cast<std::size_t>(-1));
}

// ---------------- Trigram search ----------------

TEST(TrigramSearch, IdentityAndDisjoint) {
    EXPECT_DOUBLE_EQ(trigram_similarity("hello", "hello"), 1.0);
    EXPECT_DOUBLE_EQ(trigram_similarity("HeLLo", "hello"), 1.0); // case-insensitive
    EXPECT_DOUBLE_EQ(trigram_similarity("abcdef", "xyzuvw"), 0.0);
    // Similar strings score between 0 and 1, higher for closer strings.
    const double near = trigram_similarity("hello", "hallo");
    const double far  = trigram_similarity("hello", "world");
    EXPECT_GT(near, 0.0);
    EXPECT_GT(near, far);
}

TEST(TrigramSearch, RanksBestMatchFirst) {
    const std::vector<std::string> cands = {"orange", "apple", "grape", "pineapple", "apricot"};
    const auto                     ranked = trigram_search(cands, "aple");
    // "apple" should rank at or near the top for the misspelling "aple".
    EXPECT_EQ(cands[ranked.front().second], "apple");

    // An exact query ranks its exact match first with similarity 1.
    const auto exact = trigram_search(cands, "grape");
    EXPECT_EQ(cands[exact.front().second], "grape");
    EXPECT_DOUBLE_EQ(exact.front().first, 1.0);

    // Ranking is sorted descending.
    for (std::size_t i = 1; i < ranked.size(); ++i)
        EXPECT_GE(ranked[i - 1].first, ranked[i].first);
}
