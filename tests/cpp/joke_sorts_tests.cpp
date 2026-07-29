#include <gtest/gtest.h>

#include <datamunge/algorithms/joke_sorts.hpp>

#include <algorithm>
#include <random>
#include <vector>

using namespace datamunge::algorithms;

TEST(Bogosort, SortsTinyArrays) {
    // Bogosort is only feasible for very small n.
    for (int n = 0; n <= 6; ++n) {
        std::mt19937 rng(100 + n);
        std::vector<int> a(n);
        for (int& x : a) x = static_cast<int>(rng() % 20);
        auto ref = a; std::sort(ref.begin(), ref.end());
        EXPECT_EQ(bogosort(a, 1, 5000000), ref) << "n=" << n;
    }
}

TEST(Slowsort, MatchesStdSort) {
    std::mt19937 rng(7);
    for (int t = 0; t < 200; ++t) {
        const int n = static_cast<int>(rng() % 40);
        std::vector<int> a(n);
        for (int& x : a) x = static_cast<int>(rng() % 100) - 50;
        auto ref = a; std::sort(ref.begin(), ref.end());
        EXPECT_EQ(slowsort(a), ref) << "n=" << n;
    }
}
