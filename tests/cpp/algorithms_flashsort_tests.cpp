#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <climits>
#include <datamunge/algorithms/bucket_sort.hpp>
#include <datamunge/algorithms/flashsort.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <type_traits>
namespace {
using namespace datamunge::algorithms;
template<class T>
void sorted_oracle(std::vector<T> a) {
  auto want = a;
  std::sort(want.begin(), want.end());
  auto plain = a;
  flashsort(plain);
  const auto t    = flashsort_trace(a);
  auto       work = a;
  const auto w    = flashsort_work(work);
  EXPECT_EQ(plain, want);
  EXPECT_EQ(t.output, want);
  EXPECT_EQ(work, want);
  EXPECT_EQ(t.input, a);
  EXPECT_EQ(w.insertion_shifts, t.work.insertion_shifts);
  EXPECT_EQ(w.permutation_writes, t.work.permutation_writes);
}
TEST(FlashsortNative, TinyActualCarryAndOriginalIntegers) {
  const auto t = flashsort_trace(std::vector<int>{2, 0, 1}, 19);
  EXPECT_EQ(t.output, (std::vector<int>{0, 1, 2}));
  EXPECT_EQ(t.classified, t.output);
  EXPECT_EQ(t.class_counts, (std::vector<std::size_t>{2, 1}));
  EXPECT_EQ(t.class_ends, (std::vector<std::size_t>{2, 3}));
  EXPECT_EQ(t.class_of_input, (std::vector<std::size_t>{1, 0, 0}));
  ASSERT_EQ(t.moves.size(), 3);
  EXPECT_EQ(t.moves[0].destination, 2);
  EXPECT_EQ(t.moves[0].carried, 2);
  EXPECT_EQ(t.moves[0].displaced, 1);
  EXPECT_EQ(t.work.permutation_writes, 3);
  EXPECT_EQ(t.work.classifications, 6);
  EXPECT_EQ(t.work.insertion_comparisons, 2);
  EXPECT_EQ(t.work.insertion_shifts, 0);
  EXPECT_EQ(t.work.insertion_placements, 2);
  sorted_oracle<int>({29, 4, 17, 88, 41, 6, 73, 50, 12, 95, 33, 61, 8, 24, 77});
}
TEST(FlashsortNative, EverySmallPermutationAndDuplicateArray) {
  for (int n = 0; n <= 8; ++n) {
    std::vector<int> a(n);
    std::iota(a.begin(), a.end(), 0);
    do {
      auto b = a;
      flashsort(b);
      EXPECT_TRUE(std::is_sorted(b.begin(), b.end()));
      auto ref = a;
      std::sort(ref.begin(), ref.end());
      EXPECT_EQ(b, ref);
    } while (std::next_permutation(a.begin(), a.end()));
  }
  for (int mask = 0; mask < 6561; ++mask) {
    int              digits = mask;
    std::vector<int> a(8);
    for (auto& v : a) {
      v = digits % 3 - 1;
      digits /= 3;
    }
    sorted_oracle(a);
  }
}
TEST(FlashsortNative, FullSignedAndUnsignedArithmeticDomains) {
  std::mt19937_64 rng(113);
  for (int trial = 0; trial < 1600; ++trial) {
    const auto                 n = 2 + rng() % 70;
    std::vector<std::int64_t>  a(n);
    std::vector<std::uint64_t> b(n);
    for (std::size_t i = 0; i < n; ++i) {
      a[i] = std::bit_cast<std::int64_t>(rng());
      b[i] = rng();
    }
    a[0] = INT64_MIN;
    a[1] = INT64_MAX;
    b[0] = 0;
    b[1] = UINT64_MAX;
    sorted_oracle(a);
    sorted_oracle(b);
  }
  sorted_oracle<int>({INT_MIN, 0, INT_MAX, -1, 1, INT_MIN});
  sorted_oracle<signed char>({SCHAR_MIN, 0, SCHAR_MAX, -1, 1});
  sorted_oracle<short>({SHRT_MIN, 0, SHRT_MAX});
  sorted_oracle<unsigned char>({0, 255, 1, 254, 128});
}
TEST(FlashsortNative, FiniteFloatingExtremesSubnormalsAndCloseWideValues) {
  for (auto a : std::vector<std::vector<double>>{
           {-std::numeric_limits<double>::max(), 0, std::numeric_limits<double>::max()},
           {0,
            std::numeric_limits<double>::denorm_min(),
            2 * std::numeric_limits<double>::denorm_min()},
           {-0., 0., -1., 1., -0.},
           {1e300,
            std::nextafter(1e300, 0.),
            std::nextafter(1e300, std::numeric_limits<double>::infinity())}})
    sorted_oracle(a);
  sorted_oracle<float>(
      {-std::numeric_limits<float>::max(), 0, std::numeric_limits<float>::max(), 1});
  sorted_oracle<long double>(
      {-std::numeric_limits<long double>::max(), 0, std::numeric_limits<long double>::max(), 1});
  sorted_oracle<long double>(
      {0,
       std::numeric_limits<long double>::denorm_min(),
       2 * std::numeric_limits<long double>::denorm_min()});
  EXPECT_TRUE(flashsort_trace(
                  std::vector<double>{
                      -std::numeric_limits<double>::max(), 0, std::numeric_limits<double>::max()})
                  .work.scaled_classification);
  EXPECT_TRUE(flashsort_trace(
                  std::vector<double>{
                      0,
                      std::numeric_limits<double>::denorm_min(),
                      2 * std::numeric_limits<double>::denorm_min()})
                  .work.scaled_classification);
  std::mt19937_64 rng(7113);
  for (int trial = 0; trial < 1600; ++trial) {
    std::vector<double> a(2 + rng() % 100);
    for (auto& v : a) {
      const int exp = int(rng() % 2098) - 1074;
      v             = std::ldexp(.5 + double(rng() % 100) / 200, exp);
      if (rng() & 1)
        v = -v;
    }
    sorted_oracle(a);
  }
}
TEST(FlashsortNative, IndependentClassFloorAndMonotoneRoundedMaps) {
  for (int k = 0; k <= 128; ++k) {
    const double v = -16 + k * .25;
    EXPECT_EQ(flashsort_class_index(v, -16., 16., 9), static_cast<std::size_t>(k / 16));
  }
  for (int n = 2; n <= 100; ++n) {
    auto a       = flashsort_uniform_sample(n, 11, 0, 1024);
    a[0]         = 0;
    a[1]         = 1024;
    const auto t = flashsort_trace(a);
    for (std::size_t i = 0; i < a.size(); ++i)
      EXPECT_EQ(
          t.class_of_input[i],
          static_cast<std::size_t>(
              (static_cast<std::uint64_t>(t.work.class_count - 1) * a[i]) / 1024));
  }
  std::vector<std::int64_t> a{
      INT64_MIN, INT64_MIN + 1, -100, -1, 0, 1, 100, INT64_MAX - 1, INT64_MAX};
  std::size_t old = 0;
  for (auto v : a) {
    auto k = flashsort_class_index(v, INT64_MIN, INT64_MAX, 101);
    EXPECT_GE(k, old);
    EXPECT_LT(k, 101);
    old = k;
  }
  EXPECT_EQ(flashsort_class_index(4, 4, 4, 5), 0);
  EXPECT_EQ(flashsort_class_index(9, 0, 10, 1), 0);
}
TEST(FlashsortNative, ActualCarryReplayBoundariesAndInversionOracle) {
  std::mt19937 rng(39113);
  for (int fixture = 0; fixture < 400; ++fixture) {
    std::vector<int> a(2 + rng() % 80);
    for (auto& v : a)
      v = int(rng() % 1025);
    a[0]                     = 0;
    a[1]                     = 1024;
    const auto        t      = flashsort_trace(a);
    auto              replay = a;
    auto              ends   = t.class_ends;
    std::vector<bool> written(a.size());
    int               carry    = 0;
    std::size_t       previous = a.size();
    for (const auto& s : t.moves) {
      if (s.cycle_start != previous) {
        carry    = replay[s.cycle_start];
        previous = s.cycle_start;
      }
      EXPECT_EQ(s.carried, carry);
      EXPECT_EQ(s.displaced, replay[s.destination]);
      EXPECT_EQ(s.class_index, flashsort_class_index(carry, 0, 1024, t.work.class_count));
      EXPECT_EQ(s.destination, --ends[s.class_index]);
      EXPECT_FALSE(written[s.destination]);
      written[s.destination] = true;
      std::swap(carry, replay[s.destination]);
    }
    EXPECT_EQ(replay, t.classified);
    EXPECT_EQ(t.moves.size(), t.work.permutation_writes);
    EXPECT_LE(t.moves.size(), a.size());
    std::uint64_t inversions = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
      for (std::size_t j = i + 1; j < a.size(); ++j)
        if (t.classified[i] > t.classified[j])
          ++inversions;
    EXPECT_EQ(inversions, t.work.insertion_shifts);
    EXPECT_GE(t.work.insertion_comparisons, inversions);
    EXPECT_LE(t.work.insertion_comparisons, inversions + a.size() - 1);
    std::size_t begin = 0;
    for (std::size_t k = 0; k < t.class_ends.size(); ++k) {
      EXPECT_EQ(t.class_ends[k] - begin, t.class_counts[k]);
      for (std::size_t i = begin; i < t.class_ends[k]; ++i)
        EXPECT_EQ(flashsort_class_index(t.classified[i], 0, 1024, t.work.class_count), k);
      begin = t.class_ends[k];
    }
    EXPECT_EQ(begin, a.size());
  }
}
TEST(FlashsortNative, SameOccupancyDifferentQuadraticInsertionTail) {
  for (std::size_t m : {1, 2, 3, 4, 16, 64, 128}) {
    const auto hard = flashsort_trace(bucket_sort_collision_input(m, true)),
               easy = flashsort_trace(bucket_sort_collision_input(m, false));
    EXPECT_EQ(hard.class_counts, easy.class_counts);
    EXPECT_EQ(hard.work.occupancy_square_sum, (m + 1) * (m + 1) + 1);
    EXPECT_EQ(hard.work.insertion_shifts, m * (m + 1) / 2);
    EXPECT_EQ(easy.work.insertion_shifts, m);
    EXPECT_EQ(hard.work.permutation_writes, m + 1);
    EXPECT_EQ(easy.work.permutation_writes, m + 1);
  }
}
TEST(FlashsortNative, ShortcutsPreserveBitsAndInvalidInputsDoNotMutate) {
  for (auto a : std::vector<std::vector<double>>{{}, {7}, {7, 7, 7}, {-0., 0., -0.}}) {
    const auto t = flashsort_trace(a, 2 * a.size());
    EXPECT_TRUE(t.work.shortcut);
    EXPECT_EQ(t.work.class_count, 0);
    EXPECT_TRUE(t.moves.empty());
    EXPECT_TRUE(t.class_counts.empty());
    EXPECT_TRUE(t.classified.empty());
    EXPECT_EQ(t.work.insertion_shifts, 0);
    for (std::size_t i = 0; i < a.size(); ++i)
      EXPECT_EQ(std::bit_cast<std::uint64_t>(a[i]), std::bit_cast<std::uint64_t>(t.output[i]));
  }
  for (double bad :
       {std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()}) {
    std::vector<double> a{3, bad, 1};
    const auto          before = a;
    EXPECT_THROW(flashsort(a), std::invalid_argument);
    for (std::size_t i = 0; i < a.size(); ++i)
      EXPECT_EQ(std::bit_cast<std::uint64_t>(a[i]), std::bit_cast<std::uint64_t>(before[i]));
    EXPECT_THROW(flashsort_trace(std::vector<double>{bad}), std::invalid_argument);
  }
}
TEST(FlashsortNative, WholeCaptureCapsAndOwnedRecords) {
  const std::vector<int> a{2, 0, 1};
  const auto             t = flashsort_trace(a, 19);
  EXPECT_THROW(flashsort_trace(a, 18), std::length_error);
  EXPECT_EQ(flashsort_trace(a, 0).output, t.output);
  auto copy              = t;
  copy.input[0]          = 99;
  copy.output[0]         = 99;
  copy.classified[0]     = 99;
  copy.class_counts[0]   = 99;
  copy.class_ends[0]     = 99;
  copy.class_of_input[0] = 99;
  copy.moves[0].carried  = 99;
  EXPECT_EQ(t.input, a);
  EXPECT_NE(copy.classified, t.classified);
  EXPECT_NE(copy.moves[0].carried, t.moves[0].carried);
  EXPECT_THROW(flashsort_trace(std::vector<int>{7, 7}, 3), std::length_error);
  EXPECT_TRUE(flashsort_trace(std::vector<int>{7, 7}, 4).work.shortcut);
  EXPECT_EQ(flashsort_class_count(2000), 860);
  EXPECT_THROW(flashsort_class_count(static_cast<std::size_t>(INT_MAX) + 1), std::length_error);
  EXPECT_THROW(flashsort_class_index(1, 2, 0, 3), std::invalid_argument);
  EXPECT_THROW(flashsort_class_index(4, 0, 3, 3), std::invalid_argument);
  EXPECT_THROW(flashsort_class_index(1, 0, 2, 0), std::invalid_argument);
  EXPECT_THROW(
      flashsort_class_index(1, 0, 2, static_cast<std::size_t>(INT_MAX) + 1), std::length_error);
}
TEST(FlashsortNative, ActualClassGroupingAndWholeCap) {
  const auto a = flashsort_uniform_sample();
  const auto t = flashsort_trace(a);
  const auto g = flashsort_class_groups(t.class_counts, 20, 20);
  ASSERT_EQ(g.size(), 20);
  std::uint64_t sum = 0;
  for (std::size_t i = 0; i < g.size(); ++i) {
    EXPECT_EQ(g[i].begin, 43 * i);
    EXPECT_EQ(g[i].end, 43 * (i + 1));
    std::uint64_t exact = 0;
    for (auto k = g[i].begin; k < g[i].end; ++k)
      exact += t.class_counts[k];
    EXPECT_EQ(exact, g[i].count);
    sum += g[i].count;
  }
  EXPECT_EQ(sum, 2000);
  EXPECT_THROW(flashsort_class_groups(t.class_counts, 20, 19), std::length_error);
  EXPECT_EQ(flashsort_class_groups(t.class_counts, 20, 0).size(), 20);
  EXPECT_THROW(flashsort_class_groups({}, 0), std::invalid_argument);
  EXPECT_EQ(flashsort_class_groups({}, 3)[2].count, 0);
  EXPECT_EQ(flashsort_class_groups({2, 3}, 4)[3].count, 3);
  if constexpr (sizeof(std::size_t) == 8)
    EXPECT_THROW(
        flashsort_class_groups({std::numeric_limits<std::size_t>::max(), 1}, 1),
        std::overflow_error);
}
TEST(FlashsortNative, OriginalSampleGeneratorAndLegacyTemplateSignature) {
  using Old = void (*)(std::vector<int>&);
  static_assert(std::is_same_v<decltype(&flashsort<int>), Old>);
  std::mt19937_64                    rng(11);
  std::uniform_int_distribution<int> d(0, 9999);
  const auto                         a = flashsort_uniform_sample();
  ASSERT_EQ(a.size(), 2000);
  for (auto value : a)
    EXPECT_EQ(value, d(rng));
  EXPECT_EQ(a, flashsort_uniform_sample());
  EXPECT_TRUE(flashsort_uniform_sample(0).empty());
  EXPECT_EQ(flashsort_uniform_sample(3, 1, 4, 4), (std::vector<int>{4, 4, 4}));
  EXPECT_THROW(flashsort_uniform_sample(2, 1, 3, 2), std::invalid_argument);
  EXPECT_THROW(
      flashsort_uniform_sample(std::numeric_limits<std::size_t>::max()), std::length_error);
}
} // namespace
