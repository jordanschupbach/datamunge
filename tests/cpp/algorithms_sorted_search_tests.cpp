#include <gtest/gtest.h>

#include <datamunge/algorithms/sorted_search.hpp>

#include <algorithm>
#include <random>
#include <vector>

using datamunge::algorithms::binary_search;
using datamunge::algorithms::eytzinger_layout;
using datamunge::algorithms::eytzinger_search;
using datamunge::algorithms::fibonacci_search;
using datamunge::algorithms::interpolation_search;
using datamunge::algorithms::jump_search;
using datamunge::algorithms::SearchResult;

namespace {

// Every array searcher: consistent found/index with the standard library, and index valid.
void check_array_searcher(SearchResult (*search)(const std::vector<int>&, int),
                          const std::vector<int>& a, int key) {
    const bool         truth = std::binary_search(a.begin(), a.end(), key);
    const SearchResult r     = search(a, key);
    EXPECT_EQ(r.found, truth) << "key=" << key;
    if (r.found) {
        ASSERT_GE(r.index, 0);
        ASSERT_LT(static_cast<std::size_t>(r.index), a.size());
        EXPECT_EQ(a[static_cast<std::size_t>(r.index)], key);
    } else {
        EXPECT_EQ(r.index, -1);
    }
}

} // namespace

TEST(SortedSearch, KnownSmallArray) {
    const std::vector<int> a = {1, 3, 5, 7, 9, 11, 13};
    for (auto* f : {&binary_search, &jump_search, &interpolation_search, &fibonacci_search}) {
        EXPECT_TRUE(f(a, 1).found);
        EXPECT_TRUE(f(a, 13).found);
        EXPECT_TRUE(f(a, 7).found);
        EXPECT_FALSE(f(a, 0).found);   // below range
        EXPECT_FALSE(f(a, 8).found);   // gap
        EXPECT_FALSE(f(a, 14).found);  // above range
        EXPECT_EQ(f(a, 7).index, 3);
    }
}

TEST(SortedSearch, EmptyAndSingleton) {
    const std::vector<int> empty;
    const std::vector<int> one = {42};
    for (auto* f : {&binary_search, &jump_search, &interpolation_search, &fibonacci_search}) {
        EXPECT_FALSE(f(empty, 5).found);
        EXPECT_TRUE(f(one, 42).found);
        EXPECT_EQ(f(one, 42).index, 0);
        EXPECT_FALSE(f(one, 41).found);
    }
}

TEST(SortedSearch, RandomAgainstStdBinarySearch) {
    std::mt19937                       rng(2024);
    std::uniform_int_distribution<int> len(0, 40);
    std::uniform_int_distribution<int> val(-20, 40); // small range -> lots of duplicates & gaps
    for (int trial = 0; trial < 4000; ++trial) {
        std::vector<int> a(static_cast<std::size_t>(len(rng)));
        for (int& x : a) x = val(rng);
        std::sort(a.begin(), a.end());

        for (int key = -25; key <= 45; ++key) {
            check_array_searcher(&binary_search, a, key);
            check_array_searcher(&jump_search, a, key);
            check_array_searcher(&interpolation_search, a, key);
            check_array_searcher(&fibonacci_search, a, key);
        }
    }
}

TEST(SortedSearch, ProbeCountsAreLogarithmicForBinary) {
    // Binary search should never probe more than ceil(log2(n)) + 1 elements.
    std::mt19937 rng(7);
    for (int n = 1; n <= 4096; n *= 2) {
        std::vector<int> a(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) a[static_cast<std::size_t>(i)] = 2 * i;
        std::uniform_int_distribution<int> pick(0, n - 1);
        for (int t = 0; t < 50; ++t) {
            const int          key = 2 * pick(rng);
            const SearchResult r   = binary_search(a, key);
            EXPECT_TRUE(r.found);
            std::size_t bound = 1;
            while ((static_cast<std::size_t>(1) << bound) <= static_cast<std::size_t>(n)) ++bound;
            EXPECT_LE(r.probes, bound + 1);
        }
    }
}

TEST(Eytzinger, LayoutIsPermutationAndSearches) {
    std::mt19937                       rng(909);
    std::uniform_int_distribution<int> len(0, 40);
    std::uniform_int_distribution<int> val(-20, 40);
    for (int trial = 0; trial < 3000; ++trial) {
        std::vector<int> a(static_cast<std::size_t>(len(rng)));
        for (int& x : a) x = val(rng);
        std::sort(a.begin(), a.end());

        const std::vector<int> layout = eytzinger_layout(a);
        ASSERT_EQ(layout.size(), a.size());
        std::vector<int> sorted_layout = layout;
        std::sort(sorted_layout.begin(), sorted_layout.end());
        EXPECT_EQ(sorted_layout, a); // same multiset of values

        for (int key = -25; key <= 45; ++key) {
            const bool         truth = std::binary_search(a.begin(), a.end(), key);
            const SearchResult r     = eytzinger_search(layout, key);
            EXPECT_EQ(r.found, truth) << "key=" << key;
            if (r.found) {
                ASSERT_GE(r.index, 0);
                ASSERT_LT(static_cast<std::size_t>(r.index), layout.size());
                EXPECT_EQ(layout[static_cast<std::size_t>(r.index)], key);
            }
        }
    }
}
