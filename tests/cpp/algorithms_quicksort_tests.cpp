#include <gtest/gtest.h>

#include <datamunge/algorithms/quicksort.hpp>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

using datamunge::algorithms::quicksort;

namespace {

std::vector<std::int64_t> sorted_copy(std::vector<std::int64_t> v) {
    std::sort(v.begin(), v.end());
    return v;
}

} // namespace

TEST(Quicksort, HandlesEmptyAndSingleton) {
    std::vector<std::int64_t> empty;
    quicksort(empty);
    EXPECT_TRUE(empty.empty());

    std::vector<std::int64_t> one = {42};
    quicksort(one);
    EXPECT_EQ(one, (std::vector<std::int64_t>{42}));
}

TEST(Quicksort, SortsKnownArrays) {
    std::vector<std::int64_t> data = {9, 3, 7, 1, 8, 5, 2};
    quicksort(data);
    EXPECT_EQ(data, (std::vector<std::int64_t>{1, 2, 3, 5, 7, 8, 9}));

    std::vector<std::int64_t> two = {2, 1};
    quicksort(two);
    EXPECT_EQ(two, (std::vector<std::int64_t>{1, 2}));
}

TEST(Quicksort, HandlesAlreadySortedAndReversed) {
    // These are exactly the inputs that break a naive first/last-element pivot; median-of-three
    // must keep them O(n log n) and, of course, correct.
    std::vector<std::int64_t> ascending(64);
    std::iota(ascending.begin(), ascending.end(), -10);
    const auto asc_expected = ascending;
    quicksort(ascending);
    EXPECT_EQ(ascending, asc_expected);

    std::vector<std::int64_t> descending(64);
    for (std::size_t i = 0; i < descending.size(); ++i)
        descending[i] = static_cast<std::int64_t>(descending.size() - i);
    const auto desc_expected = sorted_copy(descending);
    quicksort(descending);
    EXPECT_EQ(descending, desc_expected);
}

TEST(Quicksort, HandlesAllEqual) {
    std::vector<std::int64_t> equal(2000, 7);
    quicksort(equal);
    EXPECT_EQ(equal, std::vector<std::int64_t>(2000, 7));
}

TEST(Quicksort, HandlesDuplicatesAndNegatives) {
    std::vector<std::int64_t> dups = {4, 1, 4, 2, 1, 4, 2, 2, 1, 3, 3, 1};
    quicksort(dups);
    EXPECT_EQ(dups, (std::vector<std::int64_t>{1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4}));

    std::vector<std::int64_t> data = {0, -5, 7, -1, std::int64_t{1} << 40, -(std::int64_t{1} << 40), 7, -5};
    const auto expected = sorted_copy(data);
    quicksort(data);
    EXPECT_EQ(data, expected);
}

TEST(Quicksort, ReportsComparisonCount) {
    // The instrumented overload reports comparisons; the sort must still be correct, and an empty
    // input performs no comparisons.
    std::vector<std::int64_t> empty;
    std::uint64_t cmps = 12345;
    quicksort(empty, cmps);
    EXPECT_EQ(cmps, 0u);

    std::vector<std::int64_t> data(5000);
    std::mt19937_64 rng(99);
    for (auto& x : data) x = static_cast<std::int64_t>(rng() % 100000);
    const auto expected = sorted_copy(data);
    quicksort(data, cmps);
    EXPECT_EQ(data, expected);
    EXPECT_GT(cmps, 0u); // a nontrivial input requires comparisons
}

TEST(Quicksort, SortsLargeArrayWithHeavyDuplication) {
    std::vector<std::int64_t> data(50000);
    std::mt19937_64 rng(12345);
    for (auto& x : data) x = static_cast<std::int64_t>(rng() % 100); // only 100 distinct values -> many ties
    const auto expected = sorted_copy(data);
    quicksort(data);
    EXPECT_EQ(data, expected);
}

TEST(Quicksort, RandomCrossCheckAgainstStdSort) {
    std::mt19937_64 rng(2024);
    // Vary both size and value range so some trials are nearly-unique and others are dominated by
    // duplicates; every trial must agree element-for-element with std::sort.
    const std::int64_t ranges[] = {2, 8, 64, 1000, 1000000};
    for (int trial = 0; trial < 1000; ++trial) {
        const std::size_t n = rng() % 400;
        const std::int64_t range = ranges[rng() % (sizeof(ranges) / sizeof(ranges[0]))];
        std::vector<std::int64_t> data(n);
        for (auto& x : data)
            x = static_cast<std::int64_t>(rng() % static_cast<std::uint64_t>(range)) -
                static_cast<std::int64_t>(range / 2); // include negatives
        const auto expected = sorted_copy(data);
        quicksort(data);
        ASSERT_EQ(data, expected) << "trial " << trial << " n=" << n << " range=" << range;
    }
}
