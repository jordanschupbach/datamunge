#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/algorithms/insertion_sort.hpp>
#include <limits>
#include <numeric>
using namespace datamunge::algorithms;
namespace {
void check(const std::vector<int>& a) {
  auto t        = insertion_sort_trace(a);
  auto expected = a;
  std::sort(expected.begin(), expected.end());
  EXPECT_EQ(t.run.values, expected);
  EXPECT_EQ(insertion_sort(a), expected);
  EXPECT_EQ(insertion_sort_run(a).values, expected);
  std::vector<std::size_t> ids(a.size());
  std::iota(ids.begin(), ids.end(), 0);
  std::stable_sort(ids.begin(), ids.end(), [&](auto x, auto y) { return a[x] < a[y]; });
  EXPECT_EQ(t.run.identities, ids);
  std::size_t inversions = 0, records = 0;
  for (std::size_t i = 1; i < a.size(); ++i) {
    bool record = true;
    for (std::size_t j = 0; j < i; ++j) {
      inversions += a[j] > a[i];
      record = record && a[i] < a[j];
    }
    records += record;
  }
  const auto insertions = a.empty() ? 0 : a.size() - 1;
  EXPECT_EQ(t.run.work.shifts, inversions);
  EXPECT_EQ(t.run.work.placements, insertions);
  EXPECT_EQ(t.run.work.comparisons, inversions + insertions - records);
  EXPECT_EQ(t.steps.size(), insertions);
  std::size_t comps = 0, shifts = 0;
  for (const auto& s : t.steps) {
    ASSERT_LT(s.index, a.size());
    EXPECT_EQ(s.key, a[s.key_identity]);
    EXPECT_EQ(s.shifts, s.index - s.destination);
    comps += s.comparisons;
    shifts += s.shifts;
    std::vector<std::size_t> prefix(s.index + 1);
    std::iota(prefix.begin(), prefix.end(), 0);
    std::stable_sort(prefix.begin(), prefix.end(), [&](auto x, auto y) { return a[x] < a[y]; });
    for (std::size_t j = 0; j <= s.index; ++j) {
      EXPECT_EQ(s.identities[j], prefix[j]);
      EXPECT_EQ(s.values[j], a[prefix[j]]);
    }
    for (std::size_t j = s.index + 1; j < a.size(); ++j) {
      EXPECT_EQ(s.identities[j], j);
      EXPECT_EQ(s.values[j], a[j]);
    }
  }
  EXPECT_EQ(comps, t.run.work.comparisons);
  EXPECT_EQ(shifts, t.run.work.shifts);
}
} // namespace
TEST(InsertionSortTrace, AllTernaryInputsAndEverySmallPermutation) {
  for (unsigned n = 0; n <= 9; ++n) {
    std::size_t total = 1;
    for (unsigned i = 0; i < n; ++i)
      total *= 3;
    for (std::size_t code = 0; code < total; ++code) {
      auto             x = code;
      std::vector<int> a(n);
      for (auto& v : a) {
        v = int(x % 3) - 1;
        x /= 3;
      }
      check(a);
    }
  }
  std::vector<int> p(8);
  std::iota(p.begin(), p.end(), 0);
  do {
    check(p);
  } while (std::next_permutation(p.begin(), p.end()));
}
TEST(InsertionSortTrace, OriginalStatesExtremesAndCosts) {
  check({5, 3, 8, 1, 9, 2, 7});
  auto t = insertion_sort_trace({5, 3, 8, 1, 9, 2, 7});
  EXPECT_EQ(t.steps[0].values, (std::vector<int>{3, 5, 8, 1, 9, 2, 7}));
  EXPECT_EQ(t.steps.back().values, (std::vector<int>{1, 2, 3, 5, 7, 8, 9}));
  check(
      {std::numeric_limits<int>::max(),
       0,
       std::numeric_limits<int>::min(),
       std::numeric_limits<int>::min()});
  for (unsigned n = 0; n < 100; ++n) {
    std::vector<int> a(n);
    std::iota(a.begin(), a.end(), 0);
    auto r = insertion_sort_run(a);
    EXPECT_EQ(r.work.shifts, 0);
    EXPECT_EQ(r.work.comparisons, n ? n - 1 : 0);
    std::reverse(a.begin(), a.end());
    r = insertion_sort_run(a);
    EXPECT_EQ(r.work.shifts, n * (n - 1ull) / 2);
    EXPECT_EQ(r.work.comparisons, r.work.shifts);
  }
}
TEST(InsertionSortTrace, CompleteAdmissionAndOwnedSnapshots) {
  EXPECT_NO_THROW(insertion_sort_trace({3, 1, 2}, {3, 14}));
  EXPECT_THROW(insertion_sort_trace({3, 1, 2}, {3, 13}), std::length_error);
  EXPECT_THROW(insertion_sort_trace({3, 1, 2}, {2, 0}), std::length_error);
  EXPECT_NO_THROW(insertion_sort_run({3, 1, 2}, {3, 1}));
  EXPECT_NO_THROW(insertion_sort_trace({3, 1, 2}, {0, 0}));
  EXPECT_NO_THROW(insertion_sort_trace({}, {1, 1}));
  EXPECT_NO_THROW(insertion_sort_trace({1}, {1, 1}));
  std::vector<int> a{3, 1, 2};
  auto             t = insertion_sort_trace(a);
  a[0]               = 99;
  EXPECT_EQ(t.original[0], 3);
  t.steps[0].values[0] = 77;
  EXPECT_EQ(t.steps[1].values[0], 1);
  EXPECT_EQ(t.run.values[0], 1);
}

#include <datamunge/algorithms/insertion_sort_analysis.hpp>
TEST(InsertionSortTrace, OriginalProfileAndNativeDiagram) {
  auto p = insertion_sort_example_profile();
  ASSERT_EQ(p.size(), 40);
  EXPECT_EQ(p.back().n, 200);
  EXPECT_EQ(p.back().sorted.comparisons, 199);
  EXPECT_EQ(p.back().reversed.comparisons, 19900);
  std::mt19937 rng(1);
  for (const auto& r : p) {
    std::vector<int> a(r.n);
    std::iota(a.begin(), a.end(), 0);
    std::shuffle(a.begin(), a.end(), rng);
    EXPECT_EQ(a, r.shuffled);
    check(a);
  }
  EXPECT_NO_THROW(insertion_sort_example_profile(200, 5, 1, 4100));
  EXPECT_THROW(insertion_sort_example_profile(200, 5, 1, 4099), std::length_error);
  EXPECT_THROW(insertion_sort_example_profile(20, 0), std::invalid_argument);
  EXPECT_TRUE(insertion_sort_example_profile(0).empty());
  auto t = insertion_sort_trace({3, 1, 2});
  EXPECT_FALSE(insertion_sort_step_diagram(t).to_svg().empty());
  EXPECT_THROW(insertion_sort_step_diagram(t, 2), std::out_of_range);
  EXPECT_THROW(insertion_sort_step_diagram(t, 0, 2), std::length_error);
  t.steps[0].identities[0] = 99;
  EXPECT_THROW(insertion_sort_step_diagram(t), std::invalid_argument);
}
