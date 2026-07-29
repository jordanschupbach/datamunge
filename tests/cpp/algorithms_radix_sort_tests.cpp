#include <gtest/gtest.h>

#include <datamunge/algorithms/radix_sort.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

using datamunge::algorithms::radix_sort;

namespace {

constexpr std::int64_t kMin = std::numeric_limits<std::int64_t>::min();
constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();

} // namespace

TEST(RadixSort, SortsAKnownArray) {
    std::vector<std::int64_t> v = {170, 45, 75, 90, 2, 802, 24, 66};
    radix_sort(v);
    EXPECT_EQ(v, (std::vector<std::int64_t>{2, 24, 45, 66, 75, 90, 170, 802}));
}

TEST(RadixSort, HandlesEmpty) {
    std::vector<std::int64_t> v;
    radix_sort(v);
    EXPECT_TRUE(v.empty());
}

TEST(RadixSort, HandlesSingleton) {
    std::vector<std::int64_t> v = {-42};
    radix_sort(v);
    EXPECT_EQ(v, (std::vector<std::int64_t>{-42}));
}

TEST(RadixSort, HandlesAllEqual) {
    std::vector<std::int64_t> v(17, 7);
    radix_sort(v);
    EXPECT_EQ(v, std::vector<std::int64_t>(17, 7));
}

TEST(RadixSort, HandlesDuplicates) {
    std::vector<std::int64_t> v = {5, 3, 5, 1, 3, 5, 1, 0, 3};
    radix_sort(v);
    EXPECT_EQ(v, (std::vector<std::int64_t>{0, 1, 1, 3, 3, 3, 5, 5, 5}));
}

TEST(RadixSort, SortsNegativesZeroAndPositivesInSignedOrder) {
    // The key correctness point: signed ordering must place all negatives below zero and the
    // positives, and order the negatives correctly among themselves (more-negative first).
    std::vector<std::int64_t> v = {3, -1, -128, 127, 0, -1, 42, -42, -256, 256};
    radix_sort(v);
    EXPECT_EQ(v, (std::vector<std::int64_t>{-256, -128, -42, -1, -1, 0, 3, 42, 127, 256}));
    EXPECT_TRUE(std::is_sorted(v.begin(), v.end()));
}

TEST(RadixSort, SpansTheFullInt64Range) {
    std::vector<std::int64_t> v = {kMax, kMin, 0, -1, 1, kMin + 1, kMax - 1, -kMax, kMin / 2, kMax / 2};
    auto expected = v;
    std::sort(expected.begin(), expected.end());
    radix_sort(v);
    EXPECT_EQ(v, expected);
    EXPECT_EQ(v.front(), kMin);
    EXPECT_EQ(v.back(), kMax);
}

TEST(RadixSort, AlreadySortedAndReverseSorted) {
    std::vector<std::int64_t> asc = {-3, -2, -1, 0, 1, 2, 3};
    auto asc_expected = asc;
    radix_sort(asc);
    EXPECT_EQ(asc, asc_expected);

    std::vector<std::int64_t> desc = {3, 2, 1, 0, -1, -2, -3};
    radix_sort(desc);
    EXPECT_EQ(desc, (std::vector<std::int64_t>{-3, -2, -1, 0, 1, 2, 3}));
}

TEST(RadixSort, MatchesStdSortOnManyRandomTrials) {
    std::mt19937_64 rng(20240927);
    for (int trial = 0; trial < 1000; ++trial) {
        std::vector<std::int64_t> v(rng() % 400); // includes the empty case
        for (auto& x : v) {
            const std::uint64_t r = rng();
            switch (r % 8) {
                case 0: x = kMin; break;
                case 1: x = kMax; break;
                case 2: x = kMin + static_cast<std::int64_t>(rng() % 1000); break; // near INT64_MIN
                case 3: x = kMax - static_cast<std::int64_t>(rng() % 1000); break; // near INT64_MAX
                case 4: x = -static_cast<std::int64_t>(rng() % 100000); break;      // negatives
                case 5: x = static_cast<std::int64_t>(rng() % 100000); break;       // small positives
                default: x = static_cast<std::int64_t>(rng()); break;               // full-range mix
            }
        }
        auto expected = v;
        std::sort(expected.begin(), expected.end());
        radix_sort(v);
        ASSERT_EQ(v, expected) << "trial " << trial << " (n = " << v.size() << ")";
    }
}

TEST(RadixSort, LargeArrayStaysSorted) {
    std::mt19937_64 rng(11);
    std::vector<std::int64_t> v(200000);
    for (auto& x : v) x = static_cast<std::int64_t>(rng());
    auto expected = v;
    std::sort(expected.begin(), expected.end());
    radix_sort(v);
    EXPECT_EQ(v, expected);
    EXPECT_TRUE(std::is_sorted(v.begin(), v.end()));
}
