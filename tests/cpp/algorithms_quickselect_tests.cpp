#include <gtest/gtest.h>

#include <datamunge/algorithms/quickselect.hpp>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

using datamunge::algorithms::quickselect;
using datamunge::algorithms::quickselect_mom;

namespace {

// Brute-force reference: the k-th smallest via a full sort.
std::int64_t sorted_kth(std::vector<std::int64_t> v, std::size_t k) {
    std::sort(v.begin(), v.end());
    return v[k];
}

} // namespace

TEST(Quickselect, KnownSelectionsOnSmallArray) {
    // Sorted: 1 2 3 4 5 6 7 8 9  ->  min=1, median=5, max=9.
    const std::vector<std::int64_t> data = {9, 3, 7, 1, 8, 2, 6, 5, 4};
    EXPECT_EQ(quickselect(data, 0), 1);     // minimum
    EXPECT_EQ(quickselect(data, 4), 5);     // median (5th smallest)
    EXPECT_EQ(quickselect(data, 8), 9);     // maximum
    EXPECT_EQ(quickselect_mom(data, 0), 1); // median-of-medians agrees
    EXPECT_EQ(quickselect_mom(data, 4), 5);
    EXPECT_EQ(quickselect_mom(data, 8), 9);
    // Every rank matches the sorted reference for both variants.
    for (std::size_t k = 0; k < data.size(); ++k) {
        EXPECT_EQ(quickselect(data, k), sorted_kth(data, k)) << "k=" << k;
        EXPECT_EQ(quickselect_mom(data, k), sorted_kth(data, k)) << "k=" << k;
    }
}

TEST(Quickselect, SingleElement) {
    const std::vector<std::int64_t> data = {42};
    EXPECT_EQ(quickselect(data, 0), 42);
    EXPECT_EQ(quickselect_mom(data, 0), 42);
}

TEST(Quickselect, OutOfRangeThrows) {
    const std::vector<std::int64_t> data = {1, 2, 3};
    EXPECT_THROW((void)quickselect(data, 3), std::invalid_argument);         // k == n
    EXPECT_THROW((void)quickselect(data, 100), std::invalid_argument);       // k > n
    EXPECT_THROW((void)quickselect_mom(data, 3), std::invalid_argument);
    EXPECT_THROW((void)quickselect(std::vector<std::int64_t>{}, 0), std::invalid_argument); // empty
    EXPECT_THROW((void)quickselect_mom(std::vector<std::int64_t>{}, 0), std::invalid_argument);
}

TEST(Quickselect, DoesNotModifyCallerArgument) {
    // The array is taken by value, so the caller's copy must be untouched even for a mid rank.
    std::vector<std::int64_t> data = {5, 1, 4, 2, 8, 0, 7};
    const std::vector<std::int64_t> before = data;
    (void)quickselect(data, 3);
    (void)quickselect_mom(data, 3);
    EXPECT_EQ(data, before);
}

TEST(Quickselect, BothVariantsAgreeWithSortOverManyRandomTrials) {
    std::mt19937_64 rng(20240727);
    for (int trial = 0; trial < 4000; ++trial) {
        const std::size_t n = 1 + rng() % 200;
        // Narrow value range on purpose so plenty of duplicates appear.
        std::uniform_int_distribution<std::int64_t> vals(-15, 15);
        std::vector<std::int64_t> data(n);
        for (auto& x : data) x = vals(rng);
        const std::size_t k = rng() % n;
        const std::int64_t expected = sorted_kth(data, k);
        const std::int64_t qs = quickselect(data, k);
        const std::int64_t mom = quickselect_mom(data, k);
        ASSERT_EQ(qs, expected) << "trial " << trial << " n=" << n << " k=" << k;
        ASSERT_EQ(mom, expected) << "trial " << trial << " n=" << n << " k=" << k;
        ASSERT_EQ(qs, mom) << "variants disagree, trial " << trial;
    }
}

TEST(Quickselect, AllDuplicatesEqualElement) {
    const std::vector<std::int64_t> data(50, 7); // every element is 7
    for (std::size_t k = 0; k < data.size(); k += 7) {
        EXPECT_EQ(quickselect(data, k), 7);
        EXPECT_EQ(quickselect_mom(data, k), 7);
    }
}

TEST(Quickselect, AdversarialSortedAndReverseInputs) {
    // Already-sorted and reverse-sorted inputs are exactly where a naive first/last-element pivot
    // degrades to O(n^2). Both variants must still return the correct order statistic. The
    // median-of-medians variant additionally guarantees O(n) worst case on these.
    for (std::size_t n : {1u, 2u, 5u, 6u, 11u, 25u, 64u, 129u, 500u}) {
        std::vector<std::int64_t> ascending(n), descending(n);
        std::iota(ascending.begin(), ascending.end(), 0);
        for (std::size_t i = 0; i < n; ++i) descending[i] = static_cast<std::int64_t>(n - 1 - i);
        for (std::size_t k = 0; k < n; k += (n / 8 + 1)) {
            const auto ek = static_cast<std::int64_t>(k); // sorted[k] == k for 0..n-1
            EXPECT_EQ(quickselect(ascending, k), ek) << "asc n=" << n << " k=" << k;
            EXPECT_EQ(quickselect_mom(ascending, k), ek) << "asc mom n=" << n << " k=" << k;
            EXPECT_EQ(quickselect(descending, k), ek) << "desc n=" << n << " k=" << k;
            EXPECT_EQ(quickselect_mom(descending, k), ek) << "desc mom n=" << n << " k=" << k;
        }
        EXPECT_EQ(quickselect(descending, n - 1), static_cast<std::int64_t>(n - 1)); // max
        EXPECT_EQ(quickselect_mom(ascending, 0), 0);                                 // min
    }
}

TEST(Quickselect, NegativeValuesAndMixedSigns) {
    const std::vector<std::int64_t> data = {-5, 3, -1, 0, -100, 42, -1, 7};
    for (std::size_t k = 0; k < data.size(); ++k) {
        const std::int64_t expected = sorted_kth(data, k);
        EXPECT_EQ(quickselect(data, k), expected) << "k=" << k;
        EXPECT_EQ(quickselect_mom(data, k), expected) << "k=" << k;
    }
}
