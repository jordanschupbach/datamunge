#include <datamunge/algorithms/booth_multiplication.hpp>
#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <limits>
#include <random>
using namespace datamunge::algorithms;
namespace {
void check_trace(std::int32_t a,std::int32_t b){
  const auto trace=booth_multiply_trace(a,b);ASSERT_EQ(trace.steps.size(),32);
  EXPECT_EQ(trace.multiplicand,a);EXPECT_EQ(trace.multiplier,b);
  const auto bits=static_cast<std::uint32_t>(b);int plus=0,minus=0;std::int64_t previous_sum=0;
  for(unsigned i=0;i<32;++i){
    const auto& step=trace.steps[i];const auto current=(bits>>i)&1u,previous=i?(bits>>(i-1))&1u:0;
    const auto weight=std::int64_t{1}<<i;
    const auto prefix=static_cast<std::int64_t>(std::uint64_t(bits)&((std::uint64_t{1}<<i)-1))-current*weight;
    EXPECT_EQ(step.bit_index,i);EXPECT_EQ(step.current,current);EXPECT_EQ(step.previous,previous);
    EXPECT_EQ(step.digit,int(previous)-int(current));EXPECT_EQ(step.scaled_multiplicand,std::int64_t(a)*weight);
    EXPECT_EQ(step.contribution,step.digit*step.scaled_multiplicand);
    EXPECT_EQ(step.accumulator,previous_sum+step.contribution);EXPECT_EQ(step.accumulator,std::int64_t(a)*prefix);
    plus+=step.digit==1;minus+=step.digit==-1;previous_sum=step.accumulator;
  }
  const auto plain=booth_multiply(a,b);EXPECT_EQ(trace.result.product,std::int64_t(a)*b);
  EXPECT_EQ(trace.result.product,plain.product);EXPECT_EQ(trace.result.additions,plus);EXPECT_EQ(trace.result.subtractions,minus);
  EXPECT_EQ(plus,plain.additions);EXPECT_EQ(minus,plain.subtractions);
  EXPECT_EQ(plus+minus,std::popcount(bits^(bits<<1)));EXPECT_EQ(minus-plus,int(bits>>31));
}
}
TEST(BoothTrace, EverySignedBytePairHasExactPartialAccumulatorInvariant){
  for(int a=-128;a<128;++a)for(int b=-128;b<128;++b)check_trace(a,b);
}
TEST(BoothTrace, FullWidthRandomProductsAndBoundaryGridAreExact){
  std::mt19937 rng(1951);
  for(int i=0;i<100000;++i){
    const auto a=std::bit_cast<std::int32_t>(std::uint32_t(rng()));
    const auto b=std::bit_cast<std::int32_t>(std::uint32_t(rng()));
    const auto result=booth_multiply(a,b);EXPECT_EQ(result.product,std::int64_t(a)*b);
    const auto bits=std::bit_cast<std::uint32_t>(b);
    EXPECT_EQ(result.additions+result.subtractions,std::popcount(bits^(bits<<1)));
    if(i<1000)check_trace(a,b);
  }
  const std::array<std::int32_t,9> corners{std::numeric_limits<std::int32_t>::min(),std::numeric_limits<std::int32_t>::max(),
      -1,0,1,-65536,65536,-1073741824,1073741824};
  for(auto a:corners)for(auto b:corners)check_trace(a,b);
}
TEST(BoothTrace, RunsSignExtensionAndAlternatingBitsHaveDifferentCosts){
  for(unsigned k=1;k<=31;++k){
    const auto q=static_cast<std::int32_t>((std::uint64_t{1}<<k)-1);
    const auto result=booth_multiply(1,q);EXPECT_EQ(result.product,q);
    EXPECT_EQ(result.additions,1);EXPECT_EQ(result.subtractions,1);
  }
  const auto negative=booth_multiply(1,-1);EXPECT_EQ(negative.additions,0);EXPECT_EQ(negative.subtractions,1);
  const auto minimum=booth_multiply(1,std::numeric_limits<std::int32_t>::min());
  EXPECT_EQ(minimum.additions,0);EXPECT_EQ(minimum.subtractions,1);
  const auto alternating=booth_multiply(1,0x55555555);EXPECT_EQ(alternating.additions,16);EXPECT_EQ(alternating.subtractions,16);
  const auto zero=booth_multiply(0,7);EXPECT_EQ(zero.product,0);EXPECT_EQ(zero.additions,1);EXPECT_EQ(zero.subtractions,1);
}
TEST(BoothTrace, SignedByteOperationCountsHaveExactBinomialDistribution){
  std::array<unsigned,9> counts{};unsigned sum=0;
  for(int b=-128;b<128;++b){const auto result=booth_multiply(1,b);const auto ops=result.additions+result.subtractions;
    ASSERT_LE(ops,8);++counts[ops];sum+=ops;}
  EXPECT_EQ(counts,(std::array<unsigned,9>{1,8,28,56,70,56,28,8,1}));EXPECT_EQ(sum,256*4);
}
TEST(BoothTrace, NegativeMultiplicandKeepsOperationAndContributionSignsDistinct){
  const auto t=booth_multiply_trace(-7,6);
  EXPECT_EQ(t.steps[1].digit,-1);EXPECT_EQ(t.steps[1].contribution,14);EXPECT_EQ(t.steps[1].accumulator,14);
  EXPECT_EQ(t.steps[3].digit,1);EXPECT_EQ(t.steps[3].contribution,-56);EXPECT_EQ(t.result.product,-42);
}
