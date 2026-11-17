#include <gtest/gtest.h>

#include <datamunge/algorithms/leaky_bucket.hpp>
#include <datamunge/algorithms/token_bucket.hpp>
#include <limits>
using namespace datamunge::algorithms;
TEST(LeakyBucketTrace, OriginalBurstAndSteadyEvents) {
  std::vector<LeakyBucketArrival> a(8, {0, 1});
  for (int i = 1; i <= 6; ++i)
    a.push_back({double(i), 1});
  auto t = leaky_bucket_trace(5, 1, a);
  EXPECT_EQ(t.accepted, 11);
  EXPECT_EQ(t.rejected, 3);
  for (std::size_t i = 0; i < 8; ++i)
    EXPECT_EQ(t.events[i].accepted, i < 5);
  for (std::size_t i = 8; i < 14; ++i) {
    EXPECT_EQ(t.events[i].leaked_level, 4);
    EXPECT_EQ(t.events[i].after, 5);
  }
  EXPECT_EQ(t.captured_entries, 14);
}
TEST(LeakyBucketTrace, BinaryExactTokenDuality) {
  for (double capacity : {1, 2, 5})
    for (double rate : {0., .5, 1., 2.}) {
      TokenBucket token(capacity, rate);
      LeakyBucket meter(capacity, rate);
      double      time = 0;
      for (int i = 0; i < 2000; ++i) {
        time += (i % 5) * .25;
        double amount = (i % 13) * .25;
        auto   e      = meter.add_detail(time, amount);
        EXPECT_EQ(e.accepted, token.allow(time, amount));
        EXPECT_EQ(e.after + token.stored_tokens(), capacity);
        EXPECT_GE(e.after, 0);
        EXPECT_LE(e.after, capacity);
      }
    }
}
TEST(LeakyBucketTrace, RejectionLeaksOversizeZeroAndIdle) {
  LeakyBucket m(5, 1);
  m.add(0, 5);
  auto e = m.add_detail(1, 9);
  EXPECT_FALSE(e.accepted);
  EXPECT_EQ(e.after, 4);
  EXPECT_EQ(m.last_time(), 1);
  EXPECT_EQ(m.level(10), 0);
  EXPECT_TRUE(m.add(10, 5));
  EXPECT_TRUE(m.add(10, 0));
  EXPECT_FALSE(m.add(10, 1));
  LeakyBucket z(2, 0);
  z.add(0, 2);
  EXPECT_FALSE(z.add(100, 1));
}
TEST(LeakyBucketTrace, GuardsAreAtomicAndCaptureIsOwned) {
  double q   = std::numeric_limits<double>::quiet_NaN(),
         inf = std::numeric_limits<double>::infinity();
  EXPECT_THROW(LeakyBucket(0, 1), std::invalid_argument);
  EXPECT_THROW(LeakyBucket(1, -1), std::invalid_argument);
  EXPECT_THROW(LeakyBucket(inf, 1), std::invalid_argument);
  LeakyBucket m(5, 1);
  m.add(2, 3);
  for (double now : {-1., 1., q, inf})
    EXPECT_THROW(m.add(now, 1), std::invalid_argument);
  for (double amount : {-1., q, inf})
    EXPECT_THROW(m.add(3, amount), std::invalid_argument);
  EXPECT_EQ(m.last_time(), 2);
  EXPECT_EQ(m.stored_level(), 3);
  EXPECT_THROW(leaky_bucket_trace(5, 1, {{0, 1}, {-1, 1}}), std::invalid_argument);
  EXPECT_THROW(leaky_bucket_trace(5, 1, {{0, 1}, {1, 1}}, 1), std::length_error);
  std::vector<LeakyBucketArrival> a{{0, 2}};
  auto                            t = leaky_bucket_trace(5, 1, a, 0);
  a[0].amount                       = 4;
  EXPECT_EQ(t.events[0].amount, 2);
  EXPECT_TRUE(leaky_bucket_trace(5, 1, {}).events.empty());
}
TEST(LeakyBucketTrace, OverflowingDrainSaturatesAtZero) {
  double      m = std::numeric_limits<double>::max();
  LeakyBucket b(m, m);
  b.add(0, m);
  auto e = b.add_detail(m, 1);
  EXPECT_TRUE(e.widened);
  EXPECT_EQ(e.leaked_level, 0);
  EXPECT_EQ(e.after, 1);
}
