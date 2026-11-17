#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/algorithms/kadane.hpp>
#include <limits>
using namespace datamunge::algorithms;
namespace {
void compare(const std::vector<int>& a) {
  const auto t = kadane_trace(a);
  const auto r = kadane(a);
  EXPECT_EQ(r.sum, t.result.sum);
  EXPECT_EQ(r.begin, t.result.begin);
  EXPECT_EQ(r.end, t.result.end);
  EXPECT_EQ(t.steps.size(), a.size());
  if (a.empty()) {
    EXPECT_EQ(r.sum, 0);
    return;
  }
  std::int64_t best = std::numeric_limits<std::int64_t>::min();
  std::size_t  bb = 0, be = 0;
  for (std::size_t end = 1; end <= a.size(); ++end)
    for (std::size_t begin = 0; begin < end; ++begin) {
      std::int64_t sum = 0;
      for (auto i = begin; i < end; ++i)
        sum += a[i];
      if (sum > best) {
        best = sum;
        bb   = begin;
        be   = end;
      }
    }
  EXPECT_EQ(r.sum, best);
  EXPECT_EQ(r.begin, bb);
  EXPECT_EQ(r.end, be);
  for (std::size_t i = 0; i < a.size(); ++i) {
    const auto& s = t.steps[i];
    EXPECT_EQ(s.index, i);
    EXPECT_EQ(s.value, a[i]);
    std::int64_t actual = 0;
    for (auto j = s.running_begin; j <= i; ++j)
      actual += a[j];
    EXPECT_EQ(s.running_after, actual);
    if (i)
      EXPECT_EQ(s.running_before, t.steps[i - 1].running_after);
    std::int64_t ending = std::numeric_limits<std::int64_t>::min();
    for (std::size_t begin = 0; begin <= i; ++begin) {
      std::int64_t sum = 0;
      for (auto j = begin; j <= i; ++j)
        sum += a[j];
      ending = std::max(ending, sum);
    }
    EXPECT_EQ(s.running_after, ending);
  }
}
} // namespace
TEST(KadaneTrace, ExhaustiveSignedSmallArrays) {
  for (std::size_t n = 0; n <= 7; ++n) {
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
      compare(a);
    }
  }
}
TEST(KadaneTrace, OriginalExtremesAndTies) {
  compare({-2, 1, -3, 4, -1, 2, 1, -5, 4});
  compare({-3, -1, -2});
  compare({0, 0, 0});
  compare({1, -1, 1});
  compare(
      {std::numeric_limits<int>::max(),
       std::numeric_limits<int>::max(),
       std::numeric_limits<int>::min()});
}
TEST(KadaneTrace, CompleteQuotaAndOwnedState) {
  std::vector<int> a = {-2, 3, -1};
  auto             t = kadane_trace(a, 3);
  EXPECT_THROW((void)kadane_trace(a, 2), std::length_error);
  EXPECT_NO_THROW((void)kadane_trace(a, 0));
  a[0] = 99;
  EXPECT_EQ(t.steps[0].value, -2);
}

TEST(KadaneTrace,LargeIntegerDecisionDoesNotLoseFinalImprovement){std::vector<int>a(4194305,std::numeric_limits<int>::max());a.push_back(1);const auto r=kadane(a);EXPECT_EQ(r.begin,0U);EXPECT_EQ(r.end,a.size());}
