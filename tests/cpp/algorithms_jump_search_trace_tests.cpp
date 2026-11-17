#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/jump_search.hpp>
#include <limits>
#include <random>
using namespace datamunge::algorithms;
namespace {
void compare(const std::vector<int>& a, int key) {
  const auto t     = jump_search_trace(a, key);
  const auto r     = jump_search(a, key);
  const auto z     = jump_search_analyze(a, key);
  const auto it    = std::lower_bound(a.begin(), a.end(), key);
  const bool found = it != a.end() && *it == key;
  EXPECT_EQ(r.found, found);
  EXPECT_EQ(r.index, found ? it - a.begin() : -1);
  EXPECT_EQ(r.index, t.analysis.result.index);
  EXPECT_EQ(r.probes, t.probes.size());
  EXPECT_EQ(r.probes, z.result.probes);
  EXPECT_EQ(r.probes, t.analysis.work.block_reads + t.analysis.work.scan_reads);
  EXPECT_LE(r.probes, t.analysis.schedule.max_probes);
  std::size_t blocks = 0, scans = 0, greater = 0, begin = 0,
              end = a.empty() ? 0 : std::min(a.size(), t.analysis.schedule.block_size);
  bool scan = false, done = false;
  for (const auto& e : t.probes) {
    ASSERT_FALSE(done);
    EXPECT_EQ(e.block_begin, begin);
    EXPECT_EQ(e.block_end, end);
    ASSERT_LT(e.index, a.size());
    EXPECT_EQ(e.value, a[e.index]);
    if (!scan) {
      ++blocks;
      EXPECT_EQ(e.index, end - 1);
      const bool select = e.value >= key;
      EXPECT_EQ(e.action, select ? JumpSearchAction::Select : JumpSearchAction::Advance);
      if (select)
        scan = true;
      else {
        begin = end;
        if (begin < a.size())
          end = std::min(a.size(), begin + t.analysis.schedule.block_size);
      }
    } else {
      EXPECT_EQ(e.index, begin + scans);
      ++scans;
      if (e.value == key) {
        EXPECT_EQ(e.action, JumpSearchAction::Found);
        done = true;
      } else {
        ++greater;
        EXPECT_EQ(e.action, e.value > key ? JumpSearchAction::Passed : JumpSearchAction::Continue);
        done = e.value > key;
      }
    }
  }
  EXPECT_EQ(blocks, t.analysis.work.block_reads);
  EXPECT_EQ(scans, t.analysis.work.scan_reads);
  EXPECT_EQ(greater, t.analysis.work.greater_tests);
}
} // namespace
TEST(JumpSearchTrace, OriginalOrderAndBackwardsRevisit) {
  const std::vector<int>   a = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
  const auto               t = jump_search_trace(a, 13);
  std::vector<std::size_t> indices;
  for (const auto& e : t.probes)
    indices.push_back(e.index);
  EXPECT_EQ(indices, (std::vector<std::size_t>{2, 5, 8, 6}));
  for (int key : {13, 1, 19, 8, 20})
    compare(a, key);
}
TEST(JumpSearchTrace, AllSmallSortedTernaryInputs) {
  for (std::size_t n = 0; n <= 8; ++n) {
    std::size_t count = 1;
    for (std::size_t i = 0; i < n; ++i)
      count *= 3;
    for (std::size_t code = 0; code < count; ++code) {
      auto             state = code;
      std::vector<int> a(n);
      for (auto& v : a) {
        v = int(state % 3) - 1;
        state /= 3;
      }
      if (!std::is_sorted(a.begin(), a.end()))
        continue;
      for (int key = -2; key <= 2; ++key)
        compare(a, key);
    }
  }
}
TEST(JumpSearchTrace, IntegerScheduleAndWideBoundaries) {
  for (std::size_t n = 0; n <= 100000; ++n) {
    const auto s = jump_search_schedule(n);
    if (!n) {
      EXPECT_EQ(s.block_size, 0U);
      continue;
    }
    EXPECT_LE(s.block_size, n / s.block_size);
    EXPECT_GT(s.block_size + 1, n / (s.block_size + 1));
    EXPECT_EQ(s.block_count, n / s.block_size + (n % s.block_size != 0));
  }
  const auto maximum = std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
  for (auto n : {maximum, maximum - 1, maximum / 2}) {
    const auto s = jump_search_schedule(n);
    EXPECT_LE(s.block_size, n / s.block_size);
    EXPECT_GT(s.block_size + 1, n / (s.block_size + 1));
  }
  EXPECT_THROW((void)jump_search_schedule(maximum + 1), std::length_error);
}
TEST(JumpSearchTrace, DuplicatesExtremeKeysAndRandomInputs) {
  std::mt19937 rng(47);
  for (int i = 0; i < 1000; ++i) {
    std::vector<int> a(rng() % 300);
    for (auto& v : a)
      v = int(rng() % 21) - 10;
    std::sort(a.begin(), a.end());
    compare(a, int(rng() % 25) - 12);
  }
  for (int key : {std::numeric_limits<int>::min(), 0, std::numeric_limits<int>::max()})
    compare(
        {std::numeric_limits<int>::min(),
         std::numeric_limits<int>::min(),
         0,
         std::numeric_limits<int>::max()},
        key);
}
TEST(JumpSearchTrace, CompleteReadQuotaOwnedSnapshotsAndSingleton) {
  std::vector<int> a = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
  const auto       t = jump_search_trace(a, 13, 4);
  EXPECT_THROW((void)jump_search_trace(a, 13, 3), std::length_error);
  EXPECT_NO_THROW((void)jump_search_trace(a, 13, 0));
  a[2] = 999;
  EXPECT_EQ(t.probes[0].value, 5);
  const auto one = jump_search_trace({1}, 1);
  ASSERT_EQ(one.probes.size(), 2U);
  EXPECT_EQ(one.probes[0].index, one.probes[1].index);
  compare({}, 0);
  compare({1}, 0);
  compare({1}, 1);
  compare({1}, 2);
}
