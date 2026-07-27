#include <gtest/gtest.h>

#include <datamunge/algorithms/comparison_sorts.hpp>

#include <algorithm>
#include <functional>
#include <random>
#include <vector>

using datamunge::algorithms::comb_sort;
using datamunge::algorithms::insertion_sort;
using datamunge::algorithms::selection_sort;
using datamunge::algorithms::shell_sort;

namespace {

std::vector<std::function<std::vector<int>(std::vector<int>)>> all_sorts() {
    return {insertion_sort, selection_sort, shell_sort, comb_sort};
}

} // namespace

TEST(ComparisonSorts, KnownAndEdgeCases) {
    for (const auto& sort : all_sorts()) {
        EXPECT_EQ(sort({}), std::vector<int>{});
        EXPECT_EQ(sort({42}), std::vector<int>({42}));
        EXPECT_EQ(sort({5, 3, 8, 1, 9, 2, 7}), std::vector<int>({1, 2, 3, 5, 7, 8, 9}));
        EXPECT_EQ(sort({3, 1, 2, 3, 1, 2}), std::vector<int>({1, 1, 2, 2, 3, 3})); // duplicates
        EXPECT_EQ(sort({1, 2, 3, 4, 5}), std::vector<int>({1, 2, 3, 4, 5}));       // already sorted
        EXPECT_EQ(sort({5, 4, 3, 2, 1}), std::vector<int>({1, 2, 3, 4, 5}));       // reversed
    }
}

TEST(ComparisonSorts, MatchStdSortOnRandomArrays) {
    std::mt19937                       rng(12345);
    std::uniform_int_distribution<int> len(0, 60);
    std::uniform_int_distribution<int> val(-50, 50); // small range -> many duplicates
    for (int t = 0; t < 8000; ++t) {
        std::vector<int> a(static_cast<std::size_t>(len(rng)));
        for (int& x : a) x = val(rng);
        std::vector<int> expected = a;
        std::sort(expected.begin(), expected.end());
        for (const auto& sort : all_sorts()) EXPECT_EQ(sort(a), expected);
    }
}

TEST(ComparisonSorts, HandlesLargerArrays) {
    std::mt19937 rng(7);
    std::vector<int> a(2000);
    for (int& x : a) x = static_cast<int>(rng());
    std::vector<int> expected = a;
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(insertion_sort(a), expected);
    EXPECT_EQ(selection_sort(a), expected);
    EXPECT_EQ(shell_sort(a), expected);
    EXPECT_EQ(comb_sort(a), expected);
}
