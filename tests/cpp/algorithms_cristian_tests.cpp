#include <gtest/gtest.h>
#include <datamunge/algorithms/clock_synchronization.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;

TEST(Cristian, OriginalFixtureAndAllOriginalSweepRows) {
  EXPECT_DOUBLE_EQ(cristian_estimate(1000,40),1020);EXPECT_DOUBLE_EQ(cristian_error_bound(40,5),15);
  const auto a=cristian_analyze({1000,40,5,0});
  EXPECT_EQ(a.network_round_trip,40);EXPECT_EQ(a.lower_delay,5);EXPECT_EQ(a.estimate_delay,20);EXPECT_EQ(a.upper_delay,35);
  EXPECT_EQ(a.lower_time,1005);EXPECT_EQ(a.estimate,1020);EXPECT_EQ(a.upper_time,1035);EXPECT_EQ(a.error_bound,15);
  for(int r=10;r<=100;r+=10) EXPECT_DOUBLE_EQ(cristian_error_bound(r,5),(r-10)/2.);
}
TEST(Cristian, ExhaustiveIntegerPathsAttainBothBoundsAndProcessingCancels) {
  for(int minimum=0;minimum<=5;++minimum)for(int a=minimum;a<=35;++a)for(int b=minimum;b<=35;++b)for(int p:{0,1,10,100}) {
    const auto x=cristian_exchange(1000,a,p,b,minimum);
    ASSERT_EQ(x.analysis.probe.round_trip_time,a+p+b);ASSERT_EQ(x.analysis.network_round_trip,a+b);
    ASSERT_DOUBLE_EQ(x.analysis.estimate,1000+(a+b)/2.);
    ASSERT_DOUBLE_EQ(x.model_error,(a-b)/2.);ASSERT_DOUBLE_EQ(x.receipt_time,1000+b);
    ASSERT_LE(std::fabs(x.model_error),x.analysis.error_bound);
    ASSERT_LE(x.analysis.lower_time,x.receipt_time);ASSERT_GE(x.analysis.upper_time,x.receipt_time);
    if(a==minimum)ASSERT_DOUBLE_EQ(x.model_error,-x.analysis.error_bound);
    if(b==minimum)ASSERT_DOUBLE_EQ(x.model_error,x.analysis.error_bound);
  }
}
TEST(Cristian, FractionalPathsAndMinimumAreCheckedAgainstWidenedAlgebra) {
  std::mt19937_64 rng(20261107);
  for(int trial=0;trial<10000;++trial) {
    const double a=(rng()%10000)/8.,b=(rng()%10000)/8.,p=(rng()%10000)/8.;
    const double minimum=std::min(a,b)/2,server=(static_cast<int>(rng()%200000)-100000)/8.;
    const auto x=cristian_exchange(server,a,p,b,minimum);
    EXPECT_DOUBLE_EQ(x.analysis.network_round_trip,a+b);
    EXPECT_DOUBLE_EQ(x.analysis.error_bound,(a+b)/2-minimum);
    EXPECT_DOUBLE_EQ(x.model_error,(a-b)/2);
    EXPECT_LE(std::fabs(x.model_error),x.analysis.error_bound);
    EXPECT_EQ(x.analysis.probe.server_time,server);
  }
}
TEST(Cristian, AsymmetryAndFastestProbeNeedNotMinimizeActualError) {
  const auto slow=cristian_exchange(1000,20,0,20,5),fast=cristian_exchange(1000,5,0,15,5);
  EXPECT_LT(fast.analysis.error_bound,slow.analysis.error_bound);
  EXPECT_GT(std::fabs(fast.model_error),std::fabs(slow.model_error));
  const auto corrected=cristian_exchange(1000,20,10,20,5);
  EXPECT_EQ(corrected.analysis.estimate,1020);EXPECT_EQ(cristian_estimate(1000,50),1025);
  EXPECT_EQ(corrected.model_error,0);
}
TEST(Cristian, SelectionAccountsForAssumptionsAndRetainsFirstTies) {
  const std::vector<CristianProbe> probes{{1000,20,5,0},{2000,40,5,30},{3000,10,5,0}};
  const auto s=cristian_select(probes,3);ASSERT_TRUE(s.has_probe);EXPECT_EQ(s.index,1u);ASSERT_EQ(s.probes.size(),3u);
  EXPECT_EQ(s.probes[1].estimate,2005);EXPECT_EQ(s.probes[1].error_bound,0);
  EXPECT_EQ(s.probes[2].error_bound,0);EXPECT_NE(s.probes[1].estimate,s.probes[2].estimate);
  const auto empty=cristian_select({});EXPECT_FALSE(empty.has_probe);EXPECT_EQ(empty.index,0);EXPECT_TRUE(empty.probes.empty());
}
TEST(Cristian, CompleteSelectionCapsAndValidationOfEveryProbe) {
  EXPECT_NO_THROW(cristian_select({{0,2,0,0},{0,4,0,0}},2));
  EXPECT_THROW(cristian_select({{0,2,0,0},{0,4,0,0}},1),std::length_error);
  EXPECT_NO_THROW(cristian_select({{0,2,0,0},{0,4,0,0}},0));
  EXPECT_THROW(cristian_select({{0,2,0,0},{0,4,3,0}},1),std::invalid_argument);
  const auto high=std::numeric_limits<double>::max();
  EXPECT_THROW(cristian_select({{0,2,0,0},{high,high,0,0}},1),std::overflow_error);
}
TEST(Cristian, InvalidDomainsRejectRatherThanClampImpossibleBounds) {
  const auto infinity=std::numeric_limits<double>::infinity(),nan=std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(cristian_estimate(0,-1),std::invalid_argument);
  EXPECT_THROW(cristian_estimate(nan,1),std::invalid_argument);
  EXPECT_THROW(cristian_estimate(0,infinity),std::invalid_argument);
  EXPECT_THROW(cristian_error_bound(8,5),std::invalid_argument);
  EXPECT_THROW(cristian_error_bound(10,-1),std::invalid_argument);
  EXPECT_THROW(cristian_error_bound(nan,0),std::invalid_argument);
  EXPECT_THROW(cristian_error_bound(10,infinity),std::invalid_argument);
  EXPECT_THROW(cristian_analyze({0,10,0,11}),std::invalid_argument);
  EXPECT_THROW(cristian_analyze({0,10,0,-1}),std::invalid_argument);
  EXPECT_THROW(cristian_analyze({0,10,1,infinity}),std::invalid_argument);
  EXPECT_THROW(cristian_exchange(1000,4,0,6,5),std::invalid_argument);
  EXPECT_THROW(cristian_exchange(1000,5,-1,5,5),std::invalid_argument);
  EXPECT_THROW(cristian_exchange(1000,nan,0,5,0),std::invalid_argument);
  EXPECT_THROW(cristian_exchange(1000,5,0,infinity,0),std::invalid_argument);
}
TEST(Cristian, ZeroBoundaryEpochRoundingAndUnrepresentableArithmetic) {
  const auto zero=cristian_analyze({-5,0,0,0});EXPECT_EQ(zero.estimate,-5);EXPECT_EQ(zero.error_bound,0);
  const auto boundary=cristian_analyze({0,12,5,2});EXPECT_EQ(boundary.lower_time,5);EXPECT_EQ(boundary.upper_time,5);
  const auto huge=cristian_analyze({1e16,1,0,0});EXPECT_EQ(huge.lower_time,huge.upper_time);
  EXPECT_EQ(huge.error_bound,.5);EXPECT_EQ(huge.upper_delay,1);EXPECT_EQ(huge.estimate_delay,.5);
  const double small=std::ldexp(1.,-53)+std::ldexp(1.,-65);
  const auto point=cristian_analyze({1,2*small,small,0});
  EXPECT_EQ(point.error_bound,0);EXPECT_EQ(point.estimate,std::nextafter(1.,2.));
  EXPECT_EQ(point.lower_time,point.estimate);EXPECT_EQ(point.upper_time,point.estimate);
  EXPECT_EQ(cristian_exchange(1,small,0,small,small).receipt_time,point.estimate);
  const auto high=std::numeric_limits<double>::max(),low=std::numeric_limits<double>::denorm_min();
  EXPECT_THROW(cristian_estimate(high,high),std::overflow_error);
  EXPECT_THROW(cristian_analyze({high/2,high,0,0}),std::overflow_error);
  EXPECT_TRUE(std::isfinite(cristian_estimate(high/2,high)));
  EXPECT_THROW(cristian_estimate(0,low),std::overflow_error);
  EXPECT_THROW(cristian_error_bound(3*low,2*low),std::invalid_argument);
  EXPECT_THROW(cristian_error_bound(5*low,2*low),std::overflow_error);
  EXPECT_THROW(cristian_exchange(0,1,high,1,0),std::overflow_error);
  EXPECT_THROW(cristian_exchange(0,high,0,high,0),std::overflow_error);
  EXPECT_THROW(cristian_exchange(0,.3,1e12,.3,.3),std::overflow_error);
}
TEST(Cristian, OwnedAnalysesDoNotAliasInputsOrEachOther) {
  std::vector<CristianProbe> probes{{1000,40,5,0},{1000,20,5,0}};auto s=cristian_select(probes);
  probes[0].server_time=99;s.probes[0].probe.server_time=77;s.probes[0].estimate=55;
  EXPECT_EQ(s.probes[1].probe.server_time,1000);EXPECT_EQ(s.probes[1].estimate,1010);
  EXPECT_EQ(cristian_select({{1000,40,5,0}}).probes[0].estimate,1020);
}
