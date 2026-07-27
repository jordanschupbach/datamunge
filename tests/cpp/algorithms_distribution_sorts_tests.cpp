#include <gtest/gtest.h>

#include <datamunge/algorithms/distribution_sorts.hpp>

#include <algorithm>
#include <functional>
#include <random>
#include <vector>

using datamunge::algorithms::bucket_sort;
using datamunge::algorithms::cycle_sort;
using datamunge::algorithms::pigeonhole_sort;
using datamunge::algorithms::tree_sort;

namespace {

std::vector<std::function<std::vector<int>(std::vector<int>)>> all_sorts() {
    return {bucket_sort, pigeonhole_sort, cycle_sort, tree_sort};
}
std::vector<std::function<std::vector<int>(std::vector<int>)>> wide_range_sorts() {
    return {bucket_sort, cycle_sort, tree_sort}; // pigeonhole needs a bounded value range
}

} // namespace

TEST(DistributionSorts, KnownAndEdgeCases) {
    for (const auto& sort : all_sorts()) {
        EXPECT_EQ(sort({}), std::vector<int>{});
        EXPECT_EQ(sort({42}), std::vector<int>({42}));
        EXPECT_EQ(sort({5, 3, 8, 1, 9, 2, 7}), std::vector<int>({1, 2, 3, 5, 7, 8, 9}));
        EXPECT_EQ(sort({3, 1, 2, 3, 1, 2}), std::vector<int>({1, 1, 2, 2, 3, 3}));  // duplicates
        EXPECT_EQ(sort({1, 2, 3, 4, 5}), std::vector<int>({1, 2, 3, 4, 5}));        // already sorted
        EXPECT_EQ(sort({5, 4, 3, 2, 1}), std::vector<int>({1, 2, 3, 4, 5}));        // reversed
        EXPECT_EQ(sort({-3, 5, -1, 0, -3, 2}), std::vector<int>({-3, -3, -1, 0, 2, 5})); // negatives
        EXPECT_EQ(sort({7, 7, 7, 7}), std::vector<int>({7, 7, 7, 7}));              // all equal
    }
}

TEST(DistributionSorts, MatchStdSortModerateRange) {
    std::mt19937                       rng(2024);
    std::uniform_int_distribution<int> len(0, 60);
    std::uniform_int_distribution<int> val(-40, 40);
    for (int t = 0; t < 8000; ++t) {
        std::vector<int> a(static_cast<std::size_t>(len(rng)));
        for (int& x : a) x = val(rng);
        std::vector<int> expected = a;
        std::sort(expected.begin(), expected.end());
        for (const auto& sort : all_sorts()) EXPECT_EQ(sort(a), expected);
    }
}

TEST(DistributionSorts, WideRangeAndLargerArrays) {
    std::mt19937 rng(99);
    std::vector<int> a(2000);
    for (int& x : a) x = static_cast<int>(rng()) / 2; // full-ish int range
    std::vector<int> expected = a;
    std::sort(expected.begin(), expected.end());
    for (const auto& sort : wide_range_sorts()) EXPECT_EQ(sort(a), expected);
}

TEST(DistributionSorts, TreeSortHandlesDegenerateSortedInput) {
    // Already-sorted input makes an unbalanced BST a linked list; the iterative traversal must
    // still return the correct order without stack overflow.
    std::vector<int> a(5000);
    for (int i = 0; i < 5000; ++i) a[static_cast<std::size_t>(i)] = i;
    EXPECT_EQ(tree_sort(a), a);
}
