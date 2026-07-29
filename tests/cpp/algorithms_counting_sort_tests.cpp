#include <gtest/gtest.h>

#include <datamunge/algorithms/counting_sort.hpp>

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

using datamunge::algorithms::counting_sort;

namespace {

std::vector<std::int64_t> sorted_copy(std::vector<std::int64_t> v) {
    std::sort(v.begin(), v.end());
    return v;
}

} // namespace

TEST(CountingSort, SortsWithNegativesAndZeroAutoDetectingRange) {
    std::vector<std::int64_t> data = {3, -2, 0, -5, 7, -2, 4, 0, -5};
    counting_sort(data);
    EXPECT_EQ(data, (std::vector<std::int64_t>{-5, -5, -2, -2, 0, 0, 3, 4, 7}));
    EXPECT_TRUE(std::is_sorted(data.begin(), data.end()));
}

TEST(CountingSort, EmptyIsANoOp) {
    std::vector<std::int64_t> data;
    counting_sort(data);
    EXPECT_TRUE(data.empty());
    counting_sort(data, -3, 3); // explicit range must also tolerate empty input
    EXPECT_TRUE(data.empty());
}

TEST(CountingSort, Singleton) {
    std::vector<std::int64_t> data = {42};
    counting_sort(data);
    EXPECT_EQ(data, (std::vector<std::int64_t>{42}));
}

TEST(CountingSort, AllEqual) {
    std::vector<std::int64_t> data(6, 9);
    counting_sort(data);
    EXPECT_EQ(data, std::vector<std::int64_t>(6, 9));
}

TEST(CountingSort, HeavyDuplicates) {
    std::vector<std::int64_t> data = {2, 2, 1, 1, 2, 1, 2, 1, 2, 1};
    counting_sort(data);
    EXPECT_EQ(data, (std::vector<std::int64_t>{1, 1, 1, 1, 1, 2, 2, 2, 2, 2}));
}

TEST(CountingSort, ExplicitRangeSortsInRangeInput) {
    std::vector<std::int64_t> data = {5, 1, 3, 1, 5, 2};
    counting_sort(data, 0, 5);
    EXPECT_EQ(data, (std::vector<std::int64_t>{1, 1, 2, 3, 5, 5}));
}

TEST(CountingSort, ExplicitRangeRejectsOutOfRangeAndInvertedRange) {
    std::vector<std::int64_t> above = {1, 2, 3};
    EXPECT_THROW(counting_sort(above, 0, 2), std::invalid_argument); // 3 is above max
    std::vector<std::int64_t> below = {1, 2, 3};
    EXPECT_THROW(counting_sort(below, 2, 5), std::invalid_argument); // 1 is below min
    std::vector<std::int64_t> any = {1};
    EXPECT_THROW(counting_sort(any, 5, 3), std::invalid_argument);   // max < min
}

TEST(CountingSort, RejectsAstronomicalRange) {
    std::vector<std::int64_t> data = {0, 1};
    EXPECT_THROW(counting_sort(data, 0, datamunge::algorithms::kCountingSortMaxRange),
                 std::length_error);
}

TEST(CountingSort, IsStableForKeyedRecords) {
    // Records share keys but carry distinct payloads. We pack the key into the high part and the
    // original arrival index into the low part, so an ascending numeric sort orders by key and
    // breaks ties by original position -- exactly what a stable sort must produce. We then check
    // counting_sort agrees, element for element, with std::stable_sort keyed on the key alone.
    struct Rec {
        std::int64_t key;
        int          tag;
    };
    const std::vector<Rec> records = {{1, 0}, {0, 1}, {2, 2}, {1, 3}, {0, 4},
                                      {1, 5}, {2, 6}, {0, 7}, {2, 8}, {1, 9}};
    constexpr std::int64_t base = 100; // > number of records, so low part never collides

    std::vector<std::int64_t> packed;
    packed.reserve(records.size());
    for (const auto& r : records) packed.push_back(r.key * base + r.tag);
    counting_sort(packed);

    std::vector<Rec> reference = records;
    std::stable_sort(reference.begin(), reference.end(),
                     [](const Rec& a, const Rec& b) { return a.key < b.key; });

    ASSERT_EQ(packed.size(), reference.size());
    for (std::size_t i = 0; i < packed.size(); ++i) {
        EXPECT_EQ(packed[i] / base, reference[i].key) << "at position " << i;
        EXPECT_EQ(static_cast<int>(packed[i] % base), reference[i].tag) << "at position " << i;
    }
}

TEST(CountingSort, RandomCrossCheckAgainstStdSort) {
    std::mt19937_64 rng(20240927);
    std::uniform_int_distribution<std::int64_t> value(-50, 50);
    for (int trial = 0; trial < 500; ++trial) {
        const std::size_t n = rng() % 200;
        std::vector<std::int64_t> data(n);
        for (auto& x : data) x = value(rng);

        auto by_counting = data;
        counting_sort(by_counting);
        EXPECT_EQ(by_counting, sorted_copy(data)) << "auto-detect trial " << trial;

        auto by_explicit = data;
        counting_sort(by_explicit, -50, 50); // bounded range keeps k small
        EXPECT_EQ(by_explicit, sorted_copy(data)) << "explicit-range trial " << trial;
    }
}
