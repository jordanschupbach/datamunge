#include <gtest/gtest.h>

#include <datamunge/algorithms/sorting_dist.hpp>

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

using namespace datamunge::algorithms;

TEST(SortingDist, SamplesortMatchesStdSort) {
    std::mt19937_64 rng(1);
    for (int t = 0; t < 2000; ++t) {
        std::vector<int> a(static_cast<std::size_t>(rng() % 500));
        for (auto& x : a) x = static_cast<int>(rng() % 2000) - 1000;
        std::vector<int> want = a;
        std::sort(want.begin(), want.end());
        samplesort(a);
        ASSERT_EQ(a, want) << "n=" << a.size();
    }
    // Adversarial: already sorted / reversed / all equal.
    std::vector<int> sorted(300); for (int i = 0; i < 300; ++i) sorted[i] = i;
    auto s2 = sorted; samplesort(sorted); EXPECT_EQ(sorted, s2);
    std::vector<int> rev(300); for (int i = 0; i < 300; ++i) rev[i] = 300 - i;
    samplesort(rev); EXPECT_TRUE(std::is_sorted(rev.begin(), rev.end()));
    std::vector<int> same(300, 42); samplesort(same); EXPECT_EQ(same, std::vector<int>(300, 42));
}

TEST(SortingDist, PostmanSortMatchesStdSort) {
    std::mt19937_64 rng(2);
    for (int t = 0; t < 2000; ++t) {
        std::vector<std::uint32_t> a(static_cast<std::size_t>(rng() % 500));
        for (auto& x : a) x = static_cast<std::uint32_t>(rng());
        std::vector<std::uint32_t> want = a;
        std::sort(want.begin(), want.end());
        postman_sort(a);
        ASSERT_EQ(a, want) << "n=" << a.size();
    }
    // Small-range keys (many shared high bytes) exercise deep recursion.
    for (int t = 0; t < 200; ++t) {
        std::vector<std::uint32_t> a(static_cast<std::size_t>(rng() % 300));
        for (auto& x : a) x = static_cast<std::uint32_t>(rng() % 50);
        std::vector<std::uint32_t> want = a;
        std::sort(want.begin(), want.end());
        postman_sort(a);
        EXPECT_EQ(a, want);
    }
}

TEST(SortingDist, BurstsortMatchesStdSort) {
    std::mt19937_64 rng(3);
    const std::string alpha = "abcde";
    for (int t = 0; t < 2000; ++t) {
        std::vector<std::string> a(static_cast<std::size_t>(rng() % 200));
        for (auto& s : a) {
            const int len = static_cast<int>(rng() % 8);
            for (int i = 0; i < len; ++i) s.push_back(alpha[rng() % alpha.size()]);
        }
        std::vector<std::string> want = a;
        std::sort(want.begin(), want.end());
        burstsort(a);
        ASSERT_EQ(a, want) << "n=" << a.size();
    }
    // Shared prefixes and empty/duplicate strings.
    std::vector<std::string> pre = {"apple", "app", "apply", "app", "apple", "a", "", "banana", "ban", ""};
    std::vector<std::string> want = pre;
    std::sort(want.begin(), want.end());
    burstsort(pre);
    EXPECT_EQ(pre, want);
}
