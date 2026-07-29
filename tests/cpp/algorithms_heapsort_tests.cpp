#include <gtest/gtest.h>

#include <datamunge/algorithms/heapsort.hpp>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

using datamunge::algorithms::build_max_heap;
using datamunge::algorithms::heapsort;
using datamunge::algorithms::is_max_heap;

namespace {

std::vector<std::int64_t> sorted_copy(std::vector<std::int64_t> v) {
    std::sort(v.begin(), v.end());
    return v;
}

} // namespace

TEST(Heapsort, HandlesEmptyAndSingleton) {
    std::vector<std::int64_t> empty;
    heapsort(empty);
    EXPECT_TRUE(empty.empty());

    std::vector<std::int64_t> one = {42};
    heapsort(one);
    EXPECT_EQ(one, (std::vector<std::int64_t>{42}));
}

TEST(Heapsort, SortsKnownArrays) {
    std::vector<std::int64_t> data = {3, 7, 1, 9, 2, 8};
    heapsort(data);
    EXPECT_EQ(data, (std::vector<std::int64_t>{1, 2, 3, 7, 8, 9}));
}

TEST(Heapsort, HandlesAlreadySortedAndReversed) {
    std::vector<std::int64_t> ascending = {1, 2, 3, 4, 5, 6, 7};
    heapsort(ascending);
    EXPECT_EQ(ascending, (std::vector<std::int64_t>{1, 2, 3, 4, 5, 6, 7}));

    std::vector<std::int64_t> descending = {7, 6, 5, 4, 3, 2, 1};
    heapsort(descending);
    EXPECT_EQ(descending, (std::vector<std::int64_t>{1, 2, 3, 4, 5, 6, 7}));
}

TEST(Heapsort, HandlesAllEqualAndDuplicates) {
    std::vector<std::int64_t> equal(9, 5);
    heapsort(equal);
    EXPECT_EQ(equal, std::vector<std::int64_t>(9, 5));

    std::vector<std::int64_t> dups = {4, 1, 4, 2, 1, 4, 2, 2, 1};
    heapsort(dups);
    EXPECT_EQ(dups, (std::vector<std::int64_t>{1, 1, 1, 2, 2, 2, 4, 4, 4}));
}

TEST(Heapsort, HandlesNegativesAndExtremes) {
    std::vector<std::int64_t> data = {0, -5, 7, -1, std::int64_t{1} << 40, -(std::int64_t{1} << 40)};
    const auto expected = sorted_copy(data);
    heapsort(data);
    EXPECT_EQ(data, expected);
}

TEST(Heapsort, BuildMaxHeapProducesValidHeap) {
    // The intermediate produced by build_max_heap must satisfy the sift-down invariant.
    std::vector<std::int64_t> data = {3, 7, 1, 9, 2, 8};
    build_max_heap(data);
    EXPECT_TRUE(is_max_heap(data));
    // For this deterministic input the bottom-up build yields exactly this heap.
    EXPECT_EQ(data, (std::vector<std::int64_t>{9, 7, 8, 3, 2, 1}));
    // Heapifying must be a permutation -- no values invented or lost.
    EXPECT_EQ(sorted_copy(data), (std::vector<std::int64_t>{1, 2, 3, 7, 8, 9}));
}

TEST(Heapsort, IsMaxHeapRejectsAViolatingArray) {
    EXPECT_FALSE(is_max_heap(std::vector<std::int64_t>{1, 2, 3})); // children larger than root
    EXPECT_TRUE(is_max_heap(std::vector<std::int64_t>{3, 2, 1}));
    EXPECT_TRUE(is_max_heap(std::vector<std::int64_t>{}));         // trivially a heap
    EXPECT_TRUE(is_max_heap(std::vector<std::int64_t>{5}));
}

TEST(Heapsort, SortsLargeArray) {
    std::vector<std::int64_t> data(10000);
    std::mt19937_64 rng(12345);
    for (auto& x : data) x = static_cast<std::int64_t>(rng()) % 1000000 - 500000;
    const auto expected = sorted_copy(data);
    heapsort(data);
    EXPECT_EQ(data, expected);
}

TEST(Heapsort, RandomCrossCheckAgainstStdSort) {
    std::mt19937_64 rng(2024);
    for (int trial = 0; trial < 500; ++trial) {
        const std::size_t n = rng() % 200;
        std::vector<std::int64_t> data(n);
        for (auto& x : data) x = static_cast<std::int64_t>(rng() % 50); // small range -> many ties
        const auto expected = sorted_copy(data);
        heapsort(data);
        ASSERT_EQ(data, expected) << "trial " << trial << " n=" << n;
    }
}
