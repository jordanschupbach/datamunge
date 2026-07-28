#include <gtest/gtest.h>

#include <datamunge/algorithms/sorting_extra.hpp>

#include <algorithm>
#include <random>
#include <vector>

using namespace datamunge::algorithms;

namespace {

template <typename F>
void check_sorter(F sorter, std::mt19937_64& rng) {
    for (int t = 0; t < 2000; ++t) {
        const int        n = static_cast<int>(rng() % 130);
        std::vector<int> a(n);
        for (auto& x : a) x = static_cast<int>(rng() % 200) - 100;
        std::vector<int> want = a;
        std::sort(want.begin(), want.end());
        sorter(a);
        ASSERT_EQ(a, want) << "n=" << n;
    }
}

} // namespace

TEST(SortingExtra, BitonicMatchesStdSort) {
    std::mt19937_64 rng(1);
    check_sorter([](std::vector<int>& v) { bitonic_sort(v); }, rng);
}

TEST(SortingExtra, StrandMatchesStdSort) {
    std::mt19937_64 rng(2);
    check_sorter([](std::vector<int>& v) { strand_sort(v); }, rng);
}

TEST(SortingExtra, PatienceMatchesStdSort) {
    std::mt19937_64 rng(3);
    check_sorter([](std::vector<int>& v) { patience_sort(v); }, rng);
}

TEST(SortingExtra, FlashsortMatchesStdSort) {
    std::mt19937_64 rng(4);
    check_sorter([](std::vector<int>& v) { flashsort(v); }, rng);
}

TEST(SortingExtra, EdgeCasesAndDuplicates) {
    for (auto sorter : {+[](std::vector<int>& v) { bitonic_sort(v); },
                        +[](std::vector<int>& v) { strand_sort(v); },
                        +[](std::vector<int>& v) { patience_sort(v); },
                        +[](std::vector<int>& v) { flashsort(v); }}) {
        std::vector<int> empty;
        sorter(empty);
        EXPECT_TRUE(empty.empty());

        std::vector<int> one = {42};
        sorter(one);
        EXPECT_EQ(one, (std::vector<int>{42}));

        std::vector<int> same = {7, 7, 7, 7, 7};
        sorter(same);
        EXPECT_EQ(same, (std::vector<int>{7, 7, 7, 7, 7}));

        std::vector<int> dup = {5, 1, 3, 3, 1, 5, 2, 2, 4};
        std::vector<int> want = dup;
        std::sort(want.begin(), want.end());
        sorter(dup);
        EXPECT_EQ(dup, want);

        std::vector<int> rev = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
        sorter(rev);
        EXPECT_TRUE(std::is_sorted(rev.begin(), rev.end()));
    }
}

TEST(SortingExtra, FlashsortDoubles) {
    std::mt19937_64                        rng(5);
    std::uniform_real_distribution<double> d(-1000, 1000);
    for (int t = 0; t < 500; ++t) {
        std::vector<double> a(static_cast<std::size_t>(rng() % 100));
        for (auto& x : a) x = d(rng);
        std::vector<double> want = a;
        std::sort(want.begin(), want.end());
        flashsort(a);
        EXPECT_EQ(a, want);
    }
}
