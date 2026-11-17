#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <cmath>
#include <datamunge/algorithms/kahan.hpp>
#include <limits>
using namespace datamunge::algorithms;
TEST(KahanTrace, ActualRecurrenceAndStreamingParity) {
  const std::vector<double> v   = {1e16, 1, 1, -1e16, .1, .1, .1};
  const auto                t   = kahan_trace(v);
  double                    sum = 0, c = 0;
  for (std::size_t i = 0; i < v.size(); ++i) {
    const auto& e = t.steps[i];
    EXPECT_EQ(e.index, i);
    EXPECT_EQ(e.value, v[i]);
    EXPECT_EQ(e.sum_before, sum);
    EXPECT_EQ(e.correction_before, c);
    const double y = v[i] - c, total = sum + y, next = (total - sum) - y;
    EXPECT_EQ(e.adjusted, y);
    EXPECT_EQ(e.total, total);
    EXPECT_EQ(e.correction_after, next);
    sum = total;
    c   = next;
  }
  EXPECT_EQ(t.sum, sum);
  EXPECT_EQ(t.sum, kahan_sum(v));
  KahanAccumulator a;
  a.add_repeated(.1, 100000);
  std::vector<double> repeated(100000, .1);
  EXPECT_EQ(
      std::bit_cast<std::uint64_t>(a.sum()), std::bit_cast<std::uint64_t>(kahan_sum(repeated)));
  EXPECT_EQ(a.count(), 100000U);
}
TEST(KahanTrace, CompensationIsNotExactOrOrderIndependent) {
  EXPECT_EQ(kahan_sum({1e16, 1, -1e16}), 0);
  EXPECT_EQ(kahan_sum({1e16, -1e16, 1}), 1);
  EXPECT_EQ(kahan_sum({}), 0);
  EXPECT_TRUE(std::isinf(kahan_sum({std::numeric_limits<double>::infinity()})));
  EXPECT_TRUE(std::isnan(kahan_sum({std::numeric_limits<double>::infinity(), 1})));
}
TEST(KahanTrace, CompleteQuotaAndSnapshots) {
  std::vector<double> v = {1, 2, 3};
  auto                t = kahan_trace(v, 3);
  EXPECT_THROW((void)kahan_trace(v, 2), std::length_error);
  EXPECT_NO_THROW((void)kahan_trace(v, 0));
  v[0] = 99;
  EXPECT_EQ(t.steps[0].value, 1);
}

TEST(KahanTrace,RepeatCountOverflowIsAtomic){KahanAccumulator a;(void)a.add(1);EXPECT_THROW(a.add_repeated(1,std::numeric_limits<std::size_t>::max()),std::length_error);EXPECT_EQ(a.sum(),1);EXPECT_EQ(a.count(),1U);}
