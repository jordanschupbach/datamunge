#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <climits>
#include <datamunge/algorithms/sorted_search.hpp>
#include <functional>
#include <queue>
#include <random>
#include <stdexcept>
#include <tuple>
namespace {
using namespace datamunge::algorithms;
// Independent complete-tree rank formula and BFS work queue, not in-order recursion.
std::size_t left_count(std::size_t n) {
  if (n < 2)
    return 0;
  const auto full = std::bit_floor(n);
  return full / 2 - 1 + std::min(n - (full - 1), full / 2);
}
std::vector<std::size_t> ranks(std::size_t n) {
  std::vector<std::size_t> out(n);
  if (!n)
    return out;
  std::queue<std::tuple<std::size_t, std::size_t, std::size_t>> q;
  q.emplace(0, 0, n);
  while (!q.empty()) {
    auto [i, first, count] = q.front();
    q.pop();
    auto left = left_count(count);
    out[i]    = first + left;
    if (left)
      q.emplace(2 * i + 1, first, left);
    if (count > left + 1)
      q.emplace(2 * i + 2, first + left + 1, count - left - 1);
  }
  return out;
}
void same(const SearchResult& a, const SearchResult& b) {
  EXPECT_EQ(a.found, b.found);
  EXPECT_EQ(a.index, b.index);
  EXPECT_EQ(a.probes, b.probes);
}
void check_layout(const std::vector<int>& a) {
  auto t      = eytzinger_layout_trace(a, 0);
  auto oracle = ranks(a.size());
  ASSERT_EQ(t.layout.size(), a.size());
  ASSERT_EQ(t.placements.size(), a.size());
  EXPECT_EQ(t.captured_entries, a.size());
  EXPECT_EQ(eytzinger_layout(a), t.layout);
  std::vector<bool> seen(a.size());
  for (const auto& p : t.placements) {
    ASSERT_LT(p.layout_index, a.size());
    ASSERT_LT(p.sorted_index, a.size());
    EXPECT_FALSE(seen[p.layout_index]);
    seen[p.layout_index] = true;
    EXPECT_EQ(oracle[p.layout_index], p.sorted_index);
    EXPECT_EQ(p.value, a[p.sorted_index]);
    EXPECT_EQ(t.layout[p.layout_index], p.value);
  }
  for (std::size_t i = 0; i < t.placements.size(); ++i)
    EXPECT_EQ(t.placements[i].sorted_index, i);
}
void check_search(const std::vector<int>& a, int key) {
  const auto l = eytzinger_layout(a);
  auto       t = eytzinger_search_trace(l, key, 0);
  same(t.result, eytzinger_search(l, key));
  EXPECT_EQ(t.key, key);
  EXPECT_EQ(t.layout_size, l.size());
  EXPECT_EQ(t.result.found, std::binary_search(a.begin(), a.end(), key));
  EXPECT_EQ(t.captured_entries, t.probes.size());
  EXPECT_EQ(t.result.probes, t.probes.size());
  EXPECT_LE(t.probes.size(), std::bit_width(a.size()));
  if (a.empty()) {
    EXPECT_TRUE(t.probes.empty());
    EXPECT_EQ(t.result.index, -1);
    return;
  }
  ASSERT_FALSE(t.probes.empty());
  EXPECT_EQ(t.probes.front().index, 0u);
  for (std::size_t j = 0; j < t.probes.size(); ++j) {
    const auto& p = t.probes[j];
    ASSERT_LT(p.index, l.size());
    EXPECT_EQ(p.value, l[p.index]);
    const int direction = key == p.value ? 0 : (key < p.value ? -1 : 1);
    EXPECT_EQ(p.direction, direction);
    const std::size_t next = direction > 0 ? 2 * p.index + 2 : 2 * p.index + 1;
    const auto        expected_next =
        direction && next < l.size() ? static_cast<std::ptrdiff_t>(next) : -1;
    EXPECT_EQ(p.next_index, expected_next);
    if (j + 1 < t.probes.size()) {
      ASSERT_GE(p.next_index, 0);
      EXPECT_EQ(t.probes[j + 1].index, static_cast<std::size_t>(p.next_index));
    } else {
      EXPECT_EQ(p.next_index, -1);
      if (t.result.found) {
        EXPECT_EQ(p.direction, 0);
        EXPECT_EQ(t.result.index, static_cast<std::ptrdiff_t>(p.index));
        EXPECT_EQ(p.value, key);
      } else {
        EXPECT_NE(p.direction, 0);
        EXPECT_EQ(t.result.index, -1);
      }
    }
  }
}
} // namespace
TEST(EytzingerTrace, OriginalExactLayoutAndWalk) {
  const std::vector<int> a{1, 3, 5, 7, 9, 11, 13, 15, 17, 19}, l{13, 7, 17, 3, 11, 15, 19, 1, 5, 9};
  EXPECT_EQ(eytzinger_layout(a), l);
  auto t = eytzinger_search_trace(l, 9);
  ASSERT_EQ(t.probes.size(), 4u);
  const std::vector<std::size_t> path{0, 1, 4, 9};
  for (std::size_t i = 0; i < path.size(); ++i)
    EXPECT_EQ(t.probes[i].index, path[i]);
  const std::vector<int>            keys{13, 9, 1, 19, 8};
  const std::vector<std::ptrdiff_t> positions{0, 9, 7, 6, -1};
  const std::vector<std::size_t>    counts{1, 4, 4, 3, 4};
  for (std::size_t i = 0; i < keys.size(); ++i) {
    auto r = eytzinger_search(l, keys[i]);
    EXPECT_EQ(r.index, positions[i]);
    EXPECT_EQ(r.probes, counts[i]);
    check_search(a, keys[i]);
  }
}
TEST(EytzingerTrace, ExhaustiveNondecreasingArrays) {
  std::size_t cases = 0;
  for (int n = 0; n <= 8; ++n) {
    std::vector<int>              a(n);
    std::function<void(int, int)> visit = [&](int i, int lo) {
      if (i == n) {
        ++cases;
        check_layout(a);
        for (int key = -2; key <= 2; ++key)
          check_search(a, key);
        return;
      }
      for (int v = lo; v <= 1; ++v) {
        a[i] = v;
        visit(i + 1, v);
      }
    };
    visit(0, -1);
  }
  EXPECT_EQ(cases, 165u);
}
TEST(EytzingerTrace, EverySmallLengthAndIncompleteRootRank) {
  for (std::size_t n = 0; n <= 1024; ++n) {
    std::vector<int> a(n);
    for (std::size_t i = 0; i < n; ++i)
      a[i] = static_cast<int>(2 * i) - 1024;
    check_layout(a);
    if (n) {
      auto t = eytzinger_layout_trace(a);
      EXPECT_EQ(t.layout[0], a[left_count(n)]);
      for (int key : {a.front(), a.back(), a[n / 2], -1025, 1025})
        check_search(a, key);
    }
  }
  auto t = eytzinger_layout_trace({1, 3, 5, 7, 9, 11, 13, 15, 17, 19});
  EXPECT_EQ(t.layout[0], 13);
  EXPECT_EQ(left_count(10), 6u);
}
TEST(EytzingerTrace, SeededWideAndRepeatedValues) {
  std::mt19937_64 rng(909);
  for (int trial = 0; trial < 600; ++trial) {
    std::vector<int> a(rng() % 2049);
    for (auto& v : a)
      v = trial % 2 ? static_cast<int>(rng() % 19) - 9
                    : std::bit_cast<int>(static_cast<unsigned>(rng()));
    std::sort(a.begin(), a.end());
    check_layout(a);
    for (int j = 0; j < 12; ++j)
      check_search(a, std::bit_cast<int>(static_cast<unsigned>(rng())));
    if (!a.empty()) {
      check_search(a, a.front());
      check_search(a, a.back());
      check_search(a, a[rng() % a.size()]);
    }
  }
}
TEST(EytzingerTrace, EmptySingletonAndExtremeKeys) {
  for (const auto& a : std::vector<std::vector<int>>{
           {},
           {0},
           {INT_MIN},
           {INT_MAX},
           {INT_MIN, -1, 0, 1, INT_MAX},
           {INT_MIN, INT_MIN, INT_MAX, INT_MAX}}) {
    check_layout(a);
    for (int key : {INT_MIN, INT_MIN + 1, -1, 0, 1, INT_MAX - 1, INT_MAX})
      check_search(a, key);
  }
}
TEST(EytzingerTrace, DuplicateMatchIsFirstEncounterNotSortedOccurrence) {
  auto l = eytzinger_layout({1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  auto r = eytzinger_search_trace(l, 1);
  EXPECT_EQ(r.result.index, 0);
  EXPECT_EQ(r.result.probes, 1u);
  EXPECT_EQ(r.probes.front().direction, 0);
  EXPECT_EQ(r.probes.front().next_index, -1);
  auto t = eytzinger_layout_trace({1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  EXPECT_EQ(t.placements[6].layout_index, 0u);
  check_search({1, 1, 1, 2, 2, 2, 3, 3, 3, 3}, 2);
}
TEST(EytzingerTrace, ExactShortDisabledAndEmptyQuotas) {
  const std::vector<int> a{1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
  auto                   t = eytzinger_layout_trace(a, 10);
  EXPECT_THROW(eytzinger_layout_trace(a, 9), std::length_error);
  EXPECT_EQ(eytzinger_layout_trace(a, 0).layout, t.layout);
  auto s = eytzinger_search_trace(t.layout, 9, 4);
  EXPECT_THROW(eytzinger_search_trace(t.layout, 9, 3), std::length_error);
  same(eytzinger_search_trace(t.layout, 9, 0).result, s.result);
  EXPECT_NO_THROW(eytzinger_layout_trace({}, 1));
  EXPECT_NO_THROW(eytzinger_search_trace({}, 9, 1));
  EXPECT_NO_THROW(eytzinger_layout_trace({5}, 1));
  EXPECT_NO_THROW(eytzinger_search_trace({5}, 5, 1));
  EXPECT_THROW(eytzinger_search_trace(t.layout, 8, 3), std::length_error);
  EXPECT_EQ(a, (std::vector<int>{1, 3, 5, 7, 9, 11, 13, 15, 17, 19}));
}
TEST(EytzingerTrace, OwnedSnapshotsSurviveInputsAndCopies) {
  EytzingerLayoutTrace t;
  EytzingerSearchTrace s;
  {
    std::vector<int> a{1, 3, 5};
    t = eytzinger_layout_trace(a);
    s = eytzinger_search_trace(t.layout, 5);
    a.assign(20, 99);
  }
  auto tc               = t;
  auto sc               = s;
  t.layout[0]           = 99;
  t.placements[0].value = 99;
  s.probes[0].value     = 99;
  EXPECT_EQ(tc.layout, (std::vector<int>{3, 1, 5}));
  EXPECT_EQ(tc.placements.front().value, 1);
  EXPECT_EQ(sc.probes.front().value, 3);
  EXPECT_EQ(sc.result.index, 2);
  EXPECT_EQ(sc.key, 5);
  EXPECT_EQ(sc.layout_size, 3u);
}
TEST(EytzingerTrace, CompleteTreeDepthSumAndMissBounds) {
  for (unsigned h = 1; h <= 11; ++h) {
    std::size_t      n = (std::size_t{1} << h) - 1;
    std::vector<int> a(n);
    for (std::size_t i = 0; i < n; ++i)
      a[i] = 2 * static_cast<int>(i);
    auto        l     = eytzinger_layout(a);
    std::size_t total = 0;
    for (std::size_t i = 0; i < n; ++i) {
      auto r = eytzinger_search(l, l[i]);
      EXPECT_EQ(r.index, static_cast<std::ptrdiff_t>(i));
      EXPECT_EQ(r.probes, std::bit_width(i + 1));
      total += r.probes;
    }
    EXPECT_EQ(total, (h - 1) * (std::size_t{1} << h) + 1);
    EXPECT_EQ(eytzinger_search(l, -1).probes, h);
    EXPECT_EQ(eytzinger_search(l, 2 * static_cast<int>(n)).probes, h);
  }
}
TEST(EytzingerTrace, OrdinaryLayoutIndependentOfOptionalMillionRecordQuota) {
  std::vector<int> a(1000001, 7);
  auto             l = eytzinger_layout(a);
  EXPECT_EQ(l.size(), a.size());
  EXPECT_EQ(eytzinger_search(l, 7).probes, 1u);
  EXPECT_THROW(eytzinger_layout_trace(a), std::length_error);
  auto t = eytzinger_layout_trace(a, 0);
  EXPECT_EQ(t.layout, l);
  EXPECT_EQ(t.captured_entries, a.size());
  EXPECT_EQ(t.placements.front().sorted_index, 0u);
  EXPECT_EQ(t.placements.back().sorted_index, a.size() - 1);
}
