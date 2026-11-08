#include <datamunge/random/blum_blum_shub_analysis.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
using namespace datamunge::random;
namespace {
// Independent modular multiplication: add/double without either wide products or overflow.
std::uint64_t product_by_addition(std::uint64_t a,std::uint64_t b,std::uint64_t modulus){
  auto add=[&](std::uint64_t x,std::uint64_t y){return x>=modulus-y?x-(modulus-y):x+y;};
  std::uint64_t result=0;
  while(b){if(b&1)result=add(result,a);b>>=1;if(b)a=add(a,a);}
  return result;
}
}
TEST(BbsAnalysis, OriginalCycleBitsAndPackedWordsAreExact){
  BlumBlumShub rng(3,7,11);EXPECT_EQ(rng.modulus(),77);EXPECT_EQ(rng.state(),9);
  const auto trace=blum_blum_shub_trace(rng,4);ASSERT_EQ(trace.size(),4);
  const std::uint64_t states[]{9,4,16,25,9},bits[]{0,0,1,1};
  for(std::size_t i=0;i<4;++i){EXPECT_EQ(trace[i].before,states[i]);EXPECT_EQ(trace[i].after,states[i+1]);EXPECT_EQ(trace[i].bit,bits[i]);}
  EXPECT_EQ(rng.state(),9);for(int i=0;i<5;++i)EXPECT_EQ(rng.next_u64(),0xccccccccccccccccull);
  const auto orbit=blum_blum_shub_orbit(rng,4);EXPECT_TRUE(orbit.closed);EXPECT_EQ(orbit.transient,0);EXPECT_EQ(orbit.period,4);
  EXPECT_EQ(orbit.states,(std::vector<std::uint64_t>{9,4,16,25,9}));
}
TEST(BbsAnalysis, FullWidthModularArithmeticMatchesOverflowFreeAddition){
  for(const auto& [p,q]:std::vector<std::pair<std::uint64_t,std::uint64_t>>{
      {7,11},{1000003,2001911},{4294967291ull,4294967279ull}}){
    const std::uint64_t modulus=p*q; // Both fixed reference factors are below 2^32.
    for(std::uint64_t seed:{1ull,3ull,0x123456789abcdef0ull,0xffffffffffffffffull}){
      if(std::gcd(seed,modulus)!=1)continue;
      BlumBlumShub words(seed,p,q),individual(seed,p,q),recorded(seed,p,q);
      std::uint64_t state=product_by_addition(seed%modulus,seed%modulus,modulus);
      EXPECT_EQ(words.modulus(),modulus);
      EXPECT_EQ(words.state(),state);
      for(int block=0;block<4;++block){
        const auto trace=blum_blum_shub_trace(recorded,64);std::uint64_t expected_word=0;
        for(unsigned bit=0;bit<64;++bit){
          EXPECT_EQ(trace[bit].before,state);
          state=product_by_addition(state,state,modulus);const auto expected=state&1;
          EXPECT_EQ(individual.next_bit(),expected);EXPECT_EQ(trace[bit].bit,expected);
          EXPECT_EQ(trace[bit].after,state);expected_word|=expected<<bit;
        }
        EXPECT_EQ(words.next_u64(),expected_word);EXPECT_EQ(words.state(),individual.state());EXPECT_EQ(words.state(),recorded.state());
      }
    }
  }
}
TEST(BbsAnalysis, AllAdmissibleSeedsHaveExpectedPureStatePeriodsFor77){
  std::map<std::size_t,std::size_t> periods;std::vector<std::uint64_t> initial;
  for(std::uint64_t seed=0;seed<77;++seed)if(std::gcd(seed,std::uint64_t{77})==1){
    BlumBlumShub rng(seed,7,11);initial.push_back(rng.state());
    const auto orbit=blum_blum_shub_orbit(rng,15);ASSERT_TRUE(orbit.closed);
    EXPECT_EQ(orbit.transient,0);EXPECT_EQ(orbit.states.front(),orbit.states.back());
    EXPECT_EQ(orbit.states.size(),orbit.period+1);++periods[orbit.period];
  }
  EXPECT_EQ(initial.size(),60);std::sort(initial.begin(),initial.end());initial.erase(std::unique(initial.begin(),initial.end()),initial.end());
  EXPECT_EQ(initial.size(),15);EXPECT_EQ(periods,(std::map<std::size_t,std::size_t>{{1,4},{2,8},{4,48}}));
}
TEST(BbsAnalysis, ExplicitBudgetsAndCompositeFactorTransientAreDistinguished){
  BlumBlumShub rng(3,7,11);
  const auto zero=blum_blum_shub_orbit(rng,0);EXPECT_FALSE(zero.closed);EXPECT_EQ(zero.states,(std::vector<std::uint64_t>{9}));
  const auto open=blum_blum_shub_orbit(rng,3);EXPECT_FALSE(open.closed);EXPECT_EQ(open.period,0);EXPECT_EQ(open.transient,0);
  EXPECT_EQ(open.states,(std::vector<std::uint64_t>{9,4,16,25}));EXPECT_EQ(rng.state(),9);
  // Structural validation intentionally does not claim primality validation.
  BlumBlumShub composite(2,15,7);const auto orbit=blum_blum_shub_orbit(composite,3);
  EXPECT_TRUE(orbit.closed);EXPECT_EQ(orbit.transient,1);EXPECT_EQ(orbit.period,2);
  EXPECT_EQ(orbit.states,(std::vector<std::uint64_t>{4,16,46,16}));
}
TEST(BbsAnalysis, InvalidInputsAndFailedTraceAllocationDoNotAdvanceState){
  EXPECT_THROW((BlumBlumShub(0,7,11)),std::invalid_argument);
  EXPECT_THROW((BlumBlumShub(7,7,11)),std::invalid_argument);
  EXPECT_THROW((BlumBlumShub(3,7,7)),std::invalid_argument);
  EXPECT_THROW((BlumBlumShub(3,5,11)),std::invalid_argument);
  EXPECT_THROW((BlumBlumShub(3,0,11)),std::invalid_argument);
  EXPECT_THROW((BlumBlumShub(1,std::numeric_limits<std::uint64_t>::max(),7)),std::invalid_argument);
  BlumBlumShub rng(3,7,11);EXPECT_TRUE(blum_blum_shub_trace(rng,0).empty());EXPECT_EQ(rng.state(),9);
  EXPECT_THROW(blum_blum_shub_trace(rng,std::numeric_limits<std::size_t>::max()),std::length_error);
  EXPECT_EQ(rng.state(),9);
}
TEST(BbsAnalysis, RawInspectionPreservesCopiesAndInheritedNormalCache){
  BlumBlumShub rng(1234);rng.normal();auto baseline=rng;const auto state=rng.state();
  blum_blum_shub_orbit(rng,100);EXPECT_EQ(rng.state(),state);
  EXPECT_EQ(rng.normal(),baseline.normal());
  rng.normal();baseline=rng;blum_blum_shub_trace(rng,7);
  EXPECT_EQ(rng.normal(),baseline.normal());
  auto copy=rng;EXPECT_EQ(rng.next_u64(),copy.next_u64());
}
