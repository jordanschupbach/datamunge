#include <gtest/gtest.h>

#include <datamunge/algorithms/sorting_hybrid.hpp>

#include <algorithm>
#include <random>
#include <vector>

using namespace datamunge::algorithms;

TEST(SortingHybrid, TimsortMatchesStdSort) {
    std::mt19937_64 rng(1);
    for (int t = 0; t < 3000; ++t) {
        const int        n = static_cast<int>(rng() % 500);
        std::vector<int> a(n);
        for (auto& x : a) x = static_cast<int>(rng() % 1000) - 500;
        std::vector<int> want = a;
        std::sort(want.begin(), want.end());
        timsort(a);
        ASSERT_EQ(a, want) << "n=" << n;
    }
}

TEST(SortingHybrid, TimsortStableAndStructured) {
    // Partly-sorted data (runs) is where Timsort shines; verify correctness.
    std::mt19937_64 rng(2);
    for (int t = 0; t < 500; ++t) {
        std::vector<int> a;
        for (int r = 0; r < 10; ++r) { // build ascending and descending runs
            int start = static_cast<int>(rng() % 100);
            int len   = 1 + static_cast<int>(rng() % 20);
            if (rng() & 1) for (int i = 0; i < len; ++i) a.push_back(start + i);
            else for (int i = 0; i < len; ++i) a.push_back(start - i);
        }
        std::vector<int> want = a;
        std::sort(want.begin(), want.end());
        timsort(a);
        EXPECT_EQ(a, want);
    }
    // Empty and single.
    std::vector<int> e;
    timsort(e);
    EXPECT_TRUE(e.empty());
    std::vector<int> one = {5};
    timsort(one);
    EXPECT_EQ(one, (std::vector<int>{5}));
}

TEST(SortingHybrid, BeadSortNonNegative) {
    std::mt19937_64 rng(3);
    for (int t = 0; t < 1500; ++t) {
        const int             n = static_cast<int>(rng() % 60);
        std::vector<unsigned> a(n);
        for (auto& x : a) x = static_cast<unsigned>(rng() % 100);
        std::vector<unsigned> want = a;
        std::sort(want.begin(), want.end());
        bead_sort(a);
        ASSERT_EQ(a, want) << "n=" << n;
    }
    std::vector<unsigned> zeros = {0, 0, 0};
    bead_sort(zeros);
    EXPECT_EQ(zeros, (std::vector<unsigned>{0, 0, 0}));
    std::vector<unsigned> dup = {3, 1, 3, 0, 2, 1};
    bead_sort(dup);
    EXPECT_EQ(dup, (std::vector<unsigned>{0, 1, 1, 2, 3, 3}));
}

TEST(SortingHybrid, StoogeSortMatchesStdSort) {
    std::mt19937_64 rng(4);
    for (int t = 0; t < 2000; ++t) {
        const int        n = static_cast<int>(rng() % 40); // small: O(n^2.71)
        std::vector<int> a(n);
        for (auto& x : a) x = static_cast<int>(rng() % 100) - 50;
        std::vector<int> want = a;
        std::sort(want.begin(), want.end());
        stooge_sort(a);
        ASSERT_EQ(a, want) << "n=" << n;
    }
}

TEST(BeadSortPartitions, ConjugationAndGravity) {
    using namespace datamunge::algorithms;
    // 3 + 1 + 4 + 1 + 5: column heights #{a > j} for j = 0..4.
    const std::vector<unsigned> a{3, 1, 4, 1, 5};
    EXPECT_EQ(bead_columns(a), (std::vector<unsigned>{5, 3, 3, 2, 1}));
    EXPECT_EQ(conjugate_partition(conjugate_partition(a)), (std::vector<unsigned>{5, 4, 3, 1, 1}));
    EXPECT_TRUE(conjugate_partition({0, 0}).empty());

    const auto fall = bead_sort_gravity(a);
    ASSERT_EQ(fall.after.size(), 5U);
    // After the fall, row r (top first) holds the (r+1)-th smallest value.
    for (std::size_t r = 0; r < 5; ++r) {
        unsigned beads = 0;
        for (const auto c : fall.after[r]) beads += c;
        EXPECT_EQ(beads, (std::vector<unsigned>{1, 1, 3, 4, 5})[r]);
    }
    long long total = 0;
    for (const long long m : fall.moves_per_tick) total += m;
    EXPECT_EQ(total, fall.bead_moves);
    EXPECT_LE(fall.ticks, 4);
    // Column 4 has one bead starting in the bottom row: no fall. Column 2's beads start in rows 0, 2, 4.
    EXPECT_GT(fall.ticks, 0);
}

TEST(BeadSortPartitions, MatchesStdSortOnRandomArrays) {
    std::mt19937 rng(7);
    for (int trial = 0; trial < 200; ++trial) {
        std::vector<unsigned> v(rng() % 50);
        for (auto& x : v) x = rng() % 40;
        auto expected = v;
        std::sort(expected.begin(), expected.end());
        datamunge::algorithms::bead_sort(v);
        EXPECT_EQ(v, expected);
    }
}
