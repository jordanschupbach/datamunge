#include <gtest/gtest.h>

#include <datamunge/algorithms/exchange_sorts.hpp>

#include <algorithm>
#include <functional>
#include <random>
#include <vector>

using datamunge::algorithms::cocktail_shaker_sort;
using datamunge::algorithms::gnome_sort;
using datamunge::algorithms::odd_even_sort;
using datamunge::algorithms::pancake_sort;
using datamunge::algorithms::PancakeResult;

namespace {

std::vector<std::function<std::vector<int>(std::vector<int>)>> vanilla_sorts() {
    return {gnome_sort, cocktail_shaker_sort, odd_even_sort};
}

} // namespace

TEST(ExchangeSorts, KnownAndEdgeCases) {
    for (const auto& sort : vanilla_sorts()) {
        EXPECT_EQ(sort({}), std::vector<int>{});
        EXPECT_EQ(sort({42}), std::vector<int>({42}));
        EXPECT_EQ(sort({5, 3, 8, 1, 9, 2, 7}), std::vector<int>({1, 2, 3, 5, 7, 8, 9}));
        EXPECT_EQ(sort({3, 1, 2, 3, 1, 2}), std::vector<int>({1, 1, 2, 2, 3, 3}));
        EXPECT_EQ(sort({1, 2, 3, 4, 5}), std::vector<int>({1, 2, 3, 4, 5}));
        EXPECT_EQ(sort({5, 4, 3, 2, 1}), std::vector<int>({1, 2, 3, 4, 5}));
        EXPECT_EQ(sort({-3, 5, -1, 0, -3, 2}), std::vector<int>({-3, -3, -1, 0, 2, 5}));
    }
}

TEST(ExchangeSorts, MatchStdSortOnRandomArrays) {
    std::mt19937                       rng(2024);
    std::uniform_int_distribution<int> len(0, 60);
    std::uniform_int_distribution<int> val(-40, 40);
    for (int t = 0; t < 6000; ++t) {
        std::vector<int> a(static_cast<std::size_t>(len(rng)));
        for (int& x : a) x = val(rng);
        std::vector<int> expected = a;
        std::sort(expected.begin(), expected.end());
        for (const auto& sort : vanilla_sorts()) EXPECT_EQ(sort(a), expected);
        EXPECT_EQ(pancake_sort(a).sorted, expected);
    }
}

TEST(PancakeSort, FlipsReplayToSortedAndAreBounded) {
    std::mt19937                       rng(7);
    std::uniform_int_distribution<int> len(1, 40);
    std::uniform_int_distribution<int> val(-20, 20);
    for (int t = 0; t < 4000; ++t) {
        std::vector<int> a(static_cast<std::size_t>(len(rng)));
        for (int& x : a) x = val(rng);
        const std::vector<int> original = a;

        const PancakeResult r = pancake_sort(a);
        std::vector<int>     expected = original;
        std::sort(expected.begin(), expected.end());
        EXPECT_EQ(r.sorted, expected);

        // Replaying the flips on the original must reproduce the sorted array.
        std::vector<int> replay = original;
        for (std::size_t k : r.flips) {
            ASSERT_GE(k, 2u);
            ASSERT_LE(k, original.size());
            std::reverse(replay.begin(), replay.begin() + static_cast<std::ptrdiff_t>(k));
        }
        EXPECT_EQ(replay, expected);
        EXPECT_LE(r.flips.size(), 2 * (original.size() - (original.empty() ? 0 : 1)));
    }
}
