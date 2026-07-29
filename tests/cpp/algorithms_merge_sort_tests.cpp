#include <gtest/gtest.h>

#include <datamunge/algorithms/merge_sort.hpp>

#include <algorithm>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

using datamunge::algorithms::merge_sort;
using datamunge::algorithms::merge_sort_pairs;

TEST(MergeSort, EmptyStaysEmpty) {
    std::vector<std::int64_t> v;
    merge_sort(v);
    EXPECT_TRUE(v.empty());
}

TEST(MergeSort, Singleton) {
    std::vector<std::int64_t> v{42};
    merge_sort(v);
    EXPECT_EQ(v, (std::vector<std::int64_t>{42}));
}

TEST(MergeSort, AlreadySorted) {
    std::vector<std::int64_t> v{1, 2, 3, 4, 5};
    merge_sort(v);
    EXPECT_EQ(v, (std::vector<std::int64_t>{1, 2, 3, 4, 5}));
}

TEST(MergeSort, Reversed) {
    std::vector<std::int64_t> v{5, 4, 3, 2, 1};
    merge_sort(v);
    EXPECT_EQ(v, (std::vector<std::int64_t>{1, 2, 3, 4, 5}));
}

TEST(MergeSort, AllEqual) {
    std::vector<std::int64_t> v(7, 3);
    merge_sort(v);
    EXPECT_EQ(v, std::vector<std::int64_t>(7, 3));
}

TEST(MergeSort, Duplicates) {
    std::vector<std::int64_t> v{3, 1, 2, 3, 1, 2, 3};
    merge_sort(v);
    EXPECT_EQ(v, (std::vector<std::int64_t>{1, 1, 2, 2, 3, 3, 3}));
}

TEST(MergeSort, NegativesAndZero) {
    std::vector<std::int64_t> v{0, -5, 3, -1, 3, -5};
    merge_sort(v);
    EXPECT_EQ(v, (std::vector<std::int64_t>{-5, -5, -1, 0, 3, 3}));
}

TEST(MergeSort, WorkedEightElementExample) {
    // The exact array hand-traced in examples/org/merge_sort_analysis.org.
    std::vector<std::int64_t> v{4, 1, 3, 2, 6, 5, 2, 6};
    merge_sort(v);
    EXPECT_EQ(v, (std::vector<std::int64_t>{1, 2, 2, 3, 4, 5, 6, 6}));
}

TEST(MergeSort, RandomMatchesStdSort) {
    std::mt19937_64 rng(2024);
    std::uniform_int_distribution<std::int64_t> value(-50, 50); // narrow range forces many ties
    for (int trial = 0; trial < 500; ++trial) {
        const std::size_t n = rng() % 200;
        std::vector<std::int64_t> v(n);
        for (auto& x : v) x = value(rng);
        auto expected = v;
        std::sort(expected.begin(), expected.end());
        merge_sort(v);
        ASSERT_EQ(v, expected) << "trial " << trial << " n=" << n;
    }
}

TEST(MergeSort, StableOnKeyedRecordsVsStableSort) {
    // pair.first is the sort key; pair.second is the record's original position. A stable sort by
    // key must leave equal-key records in ascending .second order -- exactly what std::stable_sort
    // produces, so we cross-check against it.
    std::mt19937_64 rng(7);
    std::uniform_int_distribution<int> key(0, 5); // few distinct keys -> lots of ties to expose
    for (int trial = 0; trial < 300; ++trial) {
        const std::size_t n = rng() % 100;
        std::vector<std::pair<int, int>> v(n);
        for (std::size_t i = 0; i < n; ++i) v[i] = {key(rng), static_cast<int>(i)};

        auto expected = v;
        std::stable_sort(expected.begin(), expected.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });

        merge_sort_pairs(v);
        ASSERT_EQ(v, expected) << "trial " << trial;

        // Direct stability check: equal keys retain ascending original index.
        for (std::size_t i = 1; i < v.size(); ++i)
            if (v[i - 1].first == v[i].first)
                EXPECT_LT(v[i - 1].second, v[i].second) << "trial " << trial << " at " << i;
    }
}
