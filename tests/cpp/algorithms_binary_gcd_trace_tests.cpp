#include <datamunge/algorithms/binary_gcd.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <limits>
#include <random>

using namespace datamunge::algorithms;

namespace {
std::uint64_t euclid(std::uint64_t a,std::uint64_t b) {
  while (b) { const auto next=a%b; a=b; b=next; }
  return a;
}
}

TEST(BinaryGcdTrace,OriginalExampleRecordsTheCanonicalStates) {
  const auto t=binary_gcd_trace(1071,462);
  EXPECT_EQ(t.gcd,21);
  EXPECT_EQ(t.common_twos,0);
  EXPECT_EQ(t.initial_odd_a,1071);
  ASSERT_EQ(t.steps.size(),5);
  const std::vector<std::uint64_t> a{231,105,63,21,21},d{840,126,42,42,0};
  for (std::size_t i=0;i<a.size();++i) {
    EXPECT_EQ(t.steps[i].smaller_odd,a[i]);
    EXPECT_EQ(t.steps[i].difference,d[i]);
  }
}

TEST(BinaryGcdTrace,RestoresTheCommonPowerOfTwo) {
  const auto t=binary_gcd_trace(54,24);
  EXPECT_EQ(t.gcd,6);
  EXPECT_EQ(t.common_twos,1);
  EXPECT_EQ(t.initial_odd_a,27);
  ASSERT_EQ(t.steps.size(),2);
  EXPECT_EQ(t.steps[0].smaller_odd,3);
  EXPECT_EQ(t.steps[0].larger_odd,27);
  EXPECT_EQ(t.steps[0].removed_twos,3);
  EXPECT_EQ(t.steps[0].difference,24);
  EXPECT_EQ(t.steps[1].removed_twos,3);
}

TEST(BinaryGcdTrace,ZeroCasesHaveNoNormalisationOrSubtractions) {
  for (auto [a,b]:std::vector<std::pair<std::uint64_t,std::uint64_t>>{{0,0},{0,7},{9,0}}) {
    const auto t=binary_gcd_trace(a,b);
    EXPECT_EQ(t.gcd,euclid(a,b));
    EXPECT_TRUE(t.steps.empty());
    EXPECT_EQ(t.common_twos,0);
    EXPECT_EQ(t.initial_odd_a,0);
  }
}

TEST(BinaryGcdTrace,EntireByteDomainMatchesIndependentRemainderAlgorithm) {
  for (std::uint64_t a=0;a<256;++a) for (std::uint64_t b=0;b<256;++b) {
    const auto expected=euclid(a,b),t=binary_gcd_trace(a,b).gcd;
    EXPECT_EQ(binary_gcd(a,b),expected) << a << ',' << b;
    EXPECT_EQ(t,expected) << a << ',' << b;
  }
}

TEST(BinaryGcdTrace,FullWidthInputsAndEveryPowerOfTwoAreSafe) {
  const auto max=std::numeric_limits<std::uint64_t>::max();
  for (unsigned bit=0;bit<64;++bit) {
    const std::uint64_t p=std::uint64_t{1}<<bit;
    EXPECT_EQ(binary_gcd(p,p),p);
    EXPECT_EQ(binary_gcd(p,0),p);
    EXPECT_EQ(binary_gcd(p,max),1);
    EXPECT_EQ(binary_gcd_trace(p,p).common_twos,bit);
  }
  EXPECT_EQ(binary_gcd(max,max),max);
  EXPECT_EQ(binary_gcd(max,max-1),1);
  EXPECT_EQ(binary_gcd_trace(max,max).gcd,max);
}

TEST(BinaryGcdTrace,RandomSnapshotsPreserveGcdAndObeyTheRoundBound) {
  std::mt19937_64 rng(20261106);
  for (int trial=0;trial<20000;++trial) {
    const auto a=rng(),b=rng(),expected=euclid(a,b);
    const auto t=binary_gcd_trace(a,b);
    ASSERT_EQ(t.gcd,expected);
    EXPECT_EQ(binary_gcd(a,b),expected);
    EXPECT_EQ(binary_gcd(b,a),expected);
    EXPECT_LE(t.steps.size(),2*std::bit_width(std::max(a,b)));
    for (const auto& step:t.steps) {
      EXPECT_EQ(step.smaller_odd&1,1);
      EXPECT_EQ(step.larger_odd&1,1);
      EXPECT_LE(step.smaller_odd,step.larger_odd);
      EXPECT_EQ(step.difference,step.larger_odd-step.smaller_odd);
      EXPECT_EQ(step.difference&1,0);
      EXPECT_LT(step.removed_twos,64);
      EXPECT_EQ(euclid(step.smaller_odd,step.difference)<<t.common_twos,expected);
    }
    ASSERT_FALSE(t.steps.empty());
    EXPECT_EQ(t.steps.back().difference,0);
  }
}
