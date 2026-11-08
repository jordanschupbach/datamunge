#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/algorithms/gnome_sort.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

#include "../oracles/gnome_sort_exact.hpp"

using namespace datamunge::algorithms;
namespace {
std::size_t inversions(const std::vector<int>& a) {
  std::size_t result = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    for (std::size_t j = i + 1; j < a.size(); ++j)
      result += a[i] > a[j];
  return result;
}
void expect_work(const GnomeSortRun& r, std::size_t n, std::size_t inv, std::size_t boundary) {
  EXPECT_EQ(r.work.visits, n + 2 * inv);
  EXPECT_EQ(r.work.swaps, inv);
  EXPECT_EQ(r.work.backward_steps, inv);
  EXPECT_EQ(r.work.forward_steps, n + inv);
  EXPECT_EQ(r.work.boundary_visits, boundary);
  EXPECT_EQ(r.work.key_comparisons, n + 2 * inv - boundary);
}
} // namespace
TEST(GnomeSortTrace, ExactIndependentStableRankAndInversionFixtures) {
  for (const auto& c : gnome_sort_exact_fixture::cases) {
    const auto*            first = gnome_sort_exact_fixture::values + c.offset;
    const std::vector<int> a(first, first + c.size);
    const auto             r   = gnome_sort_run(a);
    const auto*            ids = gnome_sort_exact_fixture::original_indices + c.offset;
    EXPECT_EQ(r.original_indices, (std::vector<std::size_t>(ids, ids + c.size)));
    ASSERT_EQ(r.output.size(), a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
      EXPECT_EQ(r.output[i], a[ids[i]]);
    EXPECT_EQ(gnome_sort(a), r.output);
    expect_work(r, c.size, c.inversions, c.boundary_visits);
  }
}
TEST(GnomeSortTrace, ActualPointerPairsIdentitiesPrefixAndPotentialCertificates) {
  std::mt19937_64 rng(136);
  for (int trial = 0; trial < 600; ++trial) {
    std::vector<int> a(rng() % 17);
    for (auto& x : a)
      x = int(rng() % 17) - 8;
    const auto t = gnome_sort_trace(a, 0);
    const auto n = a.size();
    ASSERT_GE(t.events.size(), 2u);
    EXPECT_EQ(t.events.front().kind, GnomeSortEventKind::initial);
    EXPECT_EQ(t.events.back().kind, GnomeSortEventKind::stop);
    EXPECT_EQ(t.events.front().index_before, 0u);
    EXPECT_EQ(t.events.front().index_after, 0u);
    EXPECT_EQ(t.events.back().index_before, n);
    EXPECT_EQ(t.events.back().index_after, n);
    auto                     values = a;
    std::vector<std::size_t> ids(n);
    std::iota(ids.begin(), ids.end(), 0);
    std::size_t position = 0, visits = 0, compared = 0, swaps = 0, forward = 0, backward = 0,
                boundary = 0;
    for (const auto& e : t.events) {
      if (e.kind != GnomeSortEventKind::step) {
        EXPECT_EQ(e.ordinal, 0u);
        EXPECT_FALSE(e.compared);
        EXPECT_FALSE(e.swapped);
        EXPECT_EQ(e.left, 0);
        EXPECT_EQ(e.right, 0);
        EXPECT_EQ(e.left_original_index, 0u);
        EXPECT_EQ(e.right_original_index, 0u);
        continue;
      }
      ASSERT_LT(position, n);
      EXPECT_EQ(e.index_before, position);
      EXPECT_EQ(e.ordinal, visits++);
      EXPECT_TRUE(std::is_sorted(values.begin(), values.begin() + position));
      const auto potential = n - position + 2 * inversions(values);
      EXPECT_EQ(e.compared, position != 0);
      if (e.compared) {
        ++compared;
        EXPECT_EQ(e.left, values[position - 1]);
        EXPECT_EQ(e.right, values[position]);
        EXPECT_EQ(e.left_original_index, ids[position - 1]);
        EXPECT_EQ(e.right_original_index, ids[position]);
        EXPECT_EQ(e.swapped, e.left > e.right);
      } else {
        ++boundary;
        EXPECT_FALSE(e.swapped);
        EXPECT_EQ(e.left, 0);
        EXPECT_EQ(e.right, 0);
        EXPECT_EQ(e.left_original_index, 0u);
        EXPECT_EQ(e.right_original_index, 0u);
      }
      if (e.swapped) {
        ++swaps;
        ++backward;
        std::swap(values[position - 1], values[position]);
        std::swap(ids[position - 1], ids[position]);
        --position;
      } else {
        ++position;
        ++forward;
      }
      EXPECT_EQ(e.index_after, position);
      EXPECT_EQ(n - position + 2 * inversions(values) + 1, potential);
      EXPECT_TRUE(std::is_sorted(values.begin(), values.begin() + position));
    }
    EXPECT_EQ(position, n);
    EXPECT_EQ(values, t.run.output);
    EXPECT_EQ(ids, t.run.original_indices);
    std::vector<std::size_t> expected(n);
    std::iota(expected.begin(), expected.end(), 0);
    std::stable_sort(expected.begin(), expected.end(), [&](auto i, auto j) { return a[i] < a[j]; });
    EXPECT_EQ(ids, expected);
    EXPECT_EQ(t.events.size(), visits + 2);
    EXPECT_EQ(t.run.work.visits, visits);
    EXPECT_EQ(t.run.work.key_comparisons, compared);
    EXPECT_EQ(t.run.work.swaps, swaps);
    EXPECT_EQ(t.run.work.forward_steps, forward);
    EXPECT_EQ(t.run.work.backward_steps, backward);
    EXPECT_EQ(t.run.work.boundary_visits, boundary);
  }
}
TEST(GnomeSortTrace, OriginalWalkAndBoundaryDegeneracies) {
  const auto t = gnome_sort_trace({5, 3, 8, 1, 9, 2, 7});
  EXPECT_EQ(t.run.output, (std::vector<int>{1, 2, 3, 5, 7, 8, 9}));
  EXPECT_EQ(t.run.work.visits, 27u);
  EXPECT_EQ(t.run.work.swaps, 10u);
  EXPECT_EQ(t.events.size(), 29u);
  const auto               tiny = gnome_sort_trace({3, 1, 2});
  std::vector<std::size_t> walk;
  for (const auto& e : tiny.events)
    if (e.kind == GnomeSortEventKind::step)
      walk.push_back(e.index_before);
  EXPECT_EQ(walk, (std::vector<std::size_t>{0, 1, 0, 1, 2, 1, 2}));
  expect_work(gnome_sort_run({}), 0, 0, 0);
  expect_work(gnome_sort_run({42}), 1, 0, 1);
  expect_work(gnome_sort_run({7, 7, 7, 7}), 4, 0, 1);
  const int  lo = std::numeric_limits<int>::min(), hi = std::numeric_limits<int>::max();
  const auto r = gnome_sort_run({hi, lo, hi, lo});
  EXPECT_EQ(r.output, (std::vector<int>{lo, lo, hi, hi}));
  EXPECT_EQ(r.original_indices, (std::vector<std::size_t>{1, 3, 0, 2}));
}
TEST(GnomeSortTrace, CompleteOnlineRecordQuotaIncludesFinalAndEmptyMetadata) {
  EXPECT_THROW(gnome_sort_trace({2, 1}, 5), std::length_error);
  EXPECT_EQ(gnome_sort_trace({2, 1}, 6).events.size(), 6u);
  EXPECT_EQ(gnome_sort_trace({2, 1}, 0).events.size(), 6u);
  EXPECT_THROW(gnome_sort_trace({}, 1), std::length_error);
  EXPECT_EQ(gnome_sort_trace({}, 2).events.size(), 2u);
  EXPECT_EQ(gnome_sort_trace({}, 0).events.size(), 2u);
  EXPECT_THROW(gnome_sort_trace({1}, 2), std::length_error);
  EXPECT_EQ(gnome_sort_trace({1}, 3).events.size(), 3u);
  const auto t = gnome_sort_trace({4, 1, 3, 1}, 0);
  const auto h = t.events.size();
  EXPECT_THROW(gnome_sort_trace({4, 1, 3, 1}, h - 1), std::length_error);
  EXPECT_EQ(gnome_sort_trace({4, 1, 3, 1}, h).events.size(), h);
}
TEST(GnomeSortTrace, IndependentOwnedOutputsAndNoCallerLvalueMutation) {
  std::vector<int> a{2, 1, 2, 1};
  const auto       original = a;
  auto             t        = gnome_sort_trace(a);
  EXPECT_EQ(a, original);
  auto copy                    = t;
  copy.events[0].index_before  = 999;
  copy.run.output[0]           = 999;
  copy.run.original_indices[0] = 999;
  EXPECT_NE(t.events[0].index_before, 999u);
  EXPECT_NE(t.run.output[0], 999);
  EXPECT_NE(t.run.original_indices[0], 999u);
  EXPECT_THROW(gnome_sort_trace(a, 1), std::length_error);
  EXPECT_EQ(a, original);
  const auto r = gnome_sort_run(a);
  EXPECT_EQ(a, original);
  EXPECT_EQ(r.output, t.run.output);
  EXPECT_EQ(gnome_sort(a), r.output);
  EXPECT_EQ(a, original);
}
TEST(GnomeSortTrace, NativeControlledFamiliesHaveIndependentKnownInversionCounts) {
  for (std::size_t n = 0; n <= 64; ++n) {
    const auto f = adjacent_sort_inputs(n, 4 * n);
    for (const auto* a : {&f.ascending, &f.descending, &f.adjacent_inversion, &f.last_minimum}) {
      EXPECT_EQ(a->size(), n);
      auto ordered = *a;
      std::sort(ordered.begin(), ordered.end());
      EXPECT_EQ(ordered, f.ascending);
    }
    EXPECT_EQ(inversions(f.ascending), 0u);
    EXPECT_EQ(inversions(f.descending), n * (n - (n ? 1 : 0)) / 2);
    EXPECT_EQ(inversions(f.adjacent_inversion), n >= 2 ? 1u : 0u);
    EXPECT_EQ(inversions(f.last_minimum), n ? n - 1 : 0);
    EXPECT_EQ(gnome_sort_run(f.ascending).work.visits, n);
    EXPECT_EQ(gnome_sort_run(f.descending).work.visits, n * n);
    EXPECT_EQ(gnome_sort_run(f.adjacent_inversion).work.visits, n + (n >= 2 ? 2 : 0));
    EXPECT_EQ(gnome_sort_run(f.last_minimum).work.visits, n + 2 * (n ? n - 1 : 0));
  }
}
TEST(GnomeSortTrace, WholeFamilyCountCapacityAndOwnershipAdmission) {
  EXPECT_THROW(adjacent_sort_inputs(4, 15), std::length_error);
  const auto f = adjacent_sort_inputs(4, 16);
  EXPECT_EQ(f.last_minimum, (std::vector<int>{1, 2, 3, 0}));
  EXPECT_EQ(adjacent_sort_inputs(4, 0).descending, f.descending);
  EXPECT_TRUE(adjacent_sort_inputs(0, 1).ascending.empty());
  EXPECT_THROW(adjacent_sort_inputs(std::numeric_limits<std::size_t>::max(), 0), std::length_error);
  if (std::cmp_greater(std::numeric_limits<std::size_t>::max(), std::numeric_limits<int>::max()))
    EXPECT_THROW(
        adjacent_sort_inputs(std::size_t(std::numeric_limits<int>::max()) + 2, 1),
        std::length_error);
  auto copy          = f;
  copy.ascending[0]  = 999;
  copy.descending[0] = 999;
  EXPECT_NE(f.ascending[0], 999);
  EXPECT_NE(f.descending[0], 999);
  auto independent         = f;
  independent.ascending[0] = 888;
  EXPECT_NE(independent.adjacent_inversion[0], 888);
}
