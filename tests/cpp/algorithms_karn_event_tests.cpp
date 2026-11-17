#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/algorithms/karn.hpp>
#include <limits>
using namespace datamunge::algorithms;
TEST(KarnEvent, ActualOldMeanVarianceOrderAndIgnoredAck) {
  KarnEstimator k;
  const auto    a = k.ack(.2);
  EXPECT_EQ(a.action, KarnAction::FirstSample);
  EXPECT_DOUBLE_EQ(a.after.srtt, .2);
  EXPECT_DOUBLE_EQ(a.after.rttvar, .1);
  const auto b = k.ack(.4);
  EXPECT_DOUBLE_EQ(b.before.srtt, .2);
  EXPECT_DOUBLE_EQ(b.after.rttvar, .75 * .1 + .25 * std::abs(.2 - .4));
  EXPECT_DOUBLE_EQ(b.after.srtt, .875 * .2 + .125 * .4);
  const auto ignored = k.ack(std::numeric_limits<double>::quiet_NaN(), true);
  EXPECT_EQ(ignored.action, KarnAction::IgnoredAck);
  EXPECT_DOUBLE_EQ(ignored.before.rto, ignored.after.rto);
  EXPECT_DOUBLE_EQ(ignored.before.srtt, ignored.after.srtt);
  const auto timeout = k.timeout();
  EXPECT_DOUBLE_EQ(timeout.after.rto, 2 * timeout.before.rto);
  EXPECT_DOUBLE_EQ(timeout.before.rttvar, timeout.after.rttvar);
}
TEST(KarnEvent, InvalidAndOverflowTransitionsAreAtomic) {
  for (double x : {0., -1., std::numeric_limits<double>::infinity()})
    EXPECT_THROW((void)KarnEstimator(x), std::invalid_argument);
  KarnEstimator k;
  EXPECT_THROW((void)k.ack(-1), std::invalid_argument);
  EXPECT_THROW((void)k.ack(std::numeric_limits<double>::infinity()), std::invalid_argument);
  EXPECT_FALSE(k.initialized());
  EXPECT_EQ(k.rto(), 1);
  EXPECT_THROW((void)k.ack(std::numeric_limits<double>::max() / 2), std::overflow_error);
  EXPECT_FALSE(k.initialized());
  EXPECT_EQ(k.rto(), 1);
  KarnEstimator huge(std::numeric_limits<double>::max());
  EXPECT_THROW((void)huge.timeout(), std::overflow_error);
  EXPECT_EQ(huge.rto(), std::numeric_limits<double>::max());
}
TEST(KarnEvent, ConstantInputClosedRecurrenceAndCopyOwnership) {
  KarnEstimator k;
  for (int i = 0; i < 40; ++i) {
    const auto e = k.ack(.2);
    EXPECT_NEAR(e.after.rttvar, .1 * std::pow(.75, i), 1e-16);
    EXPECT_NEAR(e.after.srtt, .2, 1e-16);
  }
  auto copy = k;
  k.on_timeout();
  EXPECT_NE(copy.rto(), k.rto());
  KarnEstimator zero;
  zero.on_ack(0, false);
  EXPECT_EQ(zero.rto(), 0);
  EXPECT_TRUE(zero.initialized());
}
