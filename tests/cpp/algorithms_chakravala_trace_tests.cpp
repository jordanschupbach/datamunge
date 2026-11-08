#include <datamunge/algorithms/chakravala.hpp>
#include <datamunge/algorithms/number_theory_advanced.hpp>
#include <gtest/gtest.h>
#include <array>
#include <limits>
#include <random>
#include <stdexcept>
namespace {
namespace a=datamunge::algorithms;
using Wide=unsigned __int128;
constexpr Wide limit=(Wide{1}<<127)-1;
// Independent continued-fraction oracle: period parity determines the exact
// convergent index; no Chakravala multipliers or equation checker are used.
a::PellSolution continued_fraction(std::uint64_t n) {
  const auto root=a::integer_sqrt_u64(n);
  if(Wide(root)*root==n)return {};
  std::uint64_t m=0,d=1,term=root;
  std::vector<std::uint64_t> period;
  do {
    m=d*term-m;d=static_cast<std::uint64_t>((Wide(n)-Wide(m)*m)/d);
    term=(root+m)/d;period.push_back(term);
  } while(term!=2*root);
  const auto length=period.size()*(period.size()%2?2:1);
  Wide p0=0,p1=1,q0=1,q1=0;
  for(std::size_t i=0;i<length;++i) {
    const auto t=i?period[(i-1)%period.size()]:root;
    if(p1>(limit-p0)/t || q1>(limit-q0)/t)throw std::overflow_error("reference coordinate range");
    const auto p=t*p1+p0,q=t*q1+q0;p0=p1;p1=p;q0=q1;q1=q;
  }
  return {static_cast<__int128>(p1),static_cast<__int128>(q1),true};
}
void compare(std::uint64_t n) {
  a::PellSolution expected;
  try {expected=continued_fraction(n);}
  catch(const std::overflow_error&) {
    EXPECT_THROW(a::chakravala(n),std::overflow_error) << n;
    EXPECT_THROW(a::chakravala_trace(n),std::overflow_error) << n;
    return;
  }
  const auto actual=a::chakravala(n);const auto trace=a::chakravala_trace(n);
  EXPECT_EQ(actual.found,expected.found)<<n;EXPECT_EQ(actual.x,expected.x)<<n;EXPECT_EQ(actual.y,expected.y)<<n;
  EXPECT_EQ(trace.solution.x,expected.x)<<n;EXPECT_EQ(trace.solution.y,expected.y)<<n;
  if(expected.found)EXPECT_TRUE(a::pell_is_solution(n,actual))<<n;
  EXPECT_EQ(trace.N,n);
  auto state=trace.initial;
  for(const auto& step:trace.steps) {
    EXPECT_EQ(step.before.a,state.a);EXPECT_EQ(step.before.b,state.b);EXPECT_EQ(step.before.k,state.k);
    const auto& c=step.choice;
    EXPECT_GT(c.modulus,0u);EXPECT_GT(c.chosen,0u);EXPECT_LT(c.residue,c.modulus);
    EXPECT_EQ(static_cast<Wide>(c.chosen)%c.modulus,c.residue);
    EXPECT_EQ((static_cast<Wide>(state.a)%c.modulus+(static_cast<Wide>(state.b)%c.modulus)*c.chosen)%c.modulus,0u);
    EXPECT_GT(c.above,a::integer_sqrt_u64(n));
    if(c.below) {
      EXPECT_LE(c.below,a::integer_sqrt_u64(n));EXPECT_EQ(c.above-c.below,c.modulus);
      EXPECT_EQ(c.below_error,Wide(n)-Wide(c.below)*c.below);
      EXPECT_EQ(c.chosen,c.below_error<=c.above_error?c.below:c.above);
    } else EXPECT_EQ(c.chosen,c.above);
    EXPECT_EQ(c.above_error,Wide(c.above)*c.above-n);
    state=step.after;
  }
  if(expected.found) {EXPECT_EQ(state.a,actual.x);EXPECT_EQ(state.b,actual.y);EXPECT_EQ(state.k,1);}
}
}
TEST(ChakravalaTrace, EveryCoefficientThroughTenThousandMatchesIndependentConvergents) {
  for(std::uint64_t n=0;n<=10000;++n)compare(n);
}
TEST(ChakravalaTrace, ThousandLargerRandomCoefficientsMatchIndependentConvergents) {
  std::mt19937_64 rng(1150);for(int i=0;i<1000;++i)compare(2+rng()%1000000);
}
TEST(ChakravalaTrace, ExactThirteenPathAndNegativeOneCorrection) {
  const auto t=a::chakravala_trace(13);ASSERT_EQ(t.steps.size(),6u);
  const std::array<long long,7> aa{3,4,7,18,137,256,649},bb{1,1,2,5,38,71,180},kk{-4,3,-3,-1,-3,3,1};
  const std::array<std::uint64_t,6> mm{1,2,4,4,2,4};
  for(std::size_t i=0;i<6;++i) {
    const auto& s=t.steps[i];EXPECT_EQ(s.before.a,aa[i]);EXPECT_EQ(s.before.b,bb[i]);EXPECT_EQ(s.before.k,kk[i]);
    EXPECT_EQ(s.choice.chosen,mm[i]);EXPECT_EQ(s.after.a,aa[i+1]);EXPECT_EQ(s.after.b,bb[i+1]);EXPECT_EQ(s.after.k,kk[i+1]);
  }
}
TEST(ChakravalaTrace, MinimumSquaredErrorDiffersFromNearestRootAndTiesChooseLower) {
  const auto seven=a::chakravala_trace(7).steps[0].choice;
  EXPECT_EQ(seven.below,1u);EXPECT_EQ(seven.above,4u);EXPECT_EQ(seven.below_error,6u);EXPECT_EQ(seven.above_error,9u);EXPECT_EQ(seven.chosen,1u);
  const auto thirteen=a::chakravala_trace(13).steps[0].choice;
  EXPECT_EQ(thirteen.below,1u);EXPECT_EQ(thirteen.above,5u);EXPECT_EQ(thirteen.below_error,thirteen.above_error);EXPECT_EQ(thirteen.chosen,1u);
}
TEST(ChakravalaTrace, OriginalFamousSolutionsAndLargeRepresentableAnswer) {
  for(const auto n:{2u,3u,5u,13u,29u,61u,109u,661u})compare(n);
  const auto p=a::chakravala(661);
  EXPECT_EQ(a::int128_text(p.x),"16421658242965910275055840472270471049");
  EXPECT_EQ(a::int128_text(p.y),"638728478116949861246791167518480580");
  EXPECT_TRUE(a::pell_is_solution(661,p));
}
TEST(ChakravalaTrace, AllUint64BoundarySquaresAndNeighboursStayExact) {
  for(std::uint64_t root:{1ULL,2ULL,3ULL,65535ULL,2147483648ULL,4294967295ULL}) {
    const auto square=root*root;compare(square);if(square>1)compare(square-1);
    // square+1 has a known fundamental pair (2*r*r+1,2*r).
    const auto p=a::chakravala(square+1);EXPECT_EQ(static_cast<Wide>(p.x),Wide(2)*square+1);EXPECT_EQ(static_cast<Wide>(p.y),Wide(2)*root);
    EXPECT_TRUE(a::pell_is_solution(square+1,p));
  }
  const auto last=a::chakravala(std::numeric_limits<std::uint64_t>::max());
  EXPECT_EQ(last.x,static_cast<__int128>(4294967296ULL));EXPECT_EQ(last.y,1);
  EXPECT_TRUE(a::pell_is_solution(std::numeric_limits<std::uint64_t>::max(),last));
}
TEST(ChakravalaTrace, ExactBudgetsThrowInsteadOfReturningPartialSolutions) {
  const auto full=a::chakravala_trace(61);
  ASSERT_EQ(full.steps.size(),14u);
  EXPECT_EQ(a::chakravala_trace(61,14).solution.x,full.solution.x);
  EXPECT_THROW(a::chakravala_trace(61,13),std::length_error);
  EXPECT_THROW(a::chakravala_trace(2,0),std::length_error);
  for(const auto n:{0u,1u,4u,49u}) {
    const auto t=a::chakravala_trace(n,0);EXPECT_FALSE(t.solution.found);EXPECT_TRUE(t.steps.empty());
    EXPECT_EQ(t.solution.x,0);EXPECT_EQ(t.solution.y,0);EXPECT_EQ(t.initial.k,0);
  }
}
TEST(ChakravalaTrace, RangeFailureIsExplicitAndDoesNotPoisonLaterCalls) {
  EXPECT_THROW(a::chakravala(1021),std::overflow_error);
  EXPECT_THROW(a::chakravala_trace(1021),std::overflow_error);
  EXPECT_EQ(a::chakravala(61).x,1766319049);
}
TEST(ChakravalaTrace, TraceRecordsAreOwnedAndCopiesAreIndependent) {
  const auto original=a::chakravala_trace(13);auto copy=original;
  copy.steps[0].after.a=999;copy.initial.a=0;
  EXPECT_EQ(original.steps[0].after.a,4);EXPECT_EQ(original.initial.a,3);
  EXPECT_EQ(a::chakravala_trace(13).steps[0].after.a,4);
}
TEST(PellCheck, ExactIdentityDoesNotImplyFundamentalOrTrustFlagsAndNegativeFields) {
  EXPECT_TRUE(a::pell_is_solution(2,{3,2,true}));EXPECT_TRUE(a::pell_is_solution(2,{17,12,true}));
  EXPECT_FALSE(a::pell_is_solution(2,{3,2,false}));EXPECT_FALSE(a::pell_is_solution(2,{3,-2,true}));
  EXPECT_FALSE(a::pell_is_solution(2,{-3,2,true}));EXPECT_FALSE(a::pell_is_solution(2,{1,0,true}));
  EXPECT_FALSE(a::pell_is_solution(1,{2,1,true}));EXPECT_FALSE(a::pell_is_solution(0,{2,1,true}));
  const auto maximum=static_cast<__int128>(limit),minimum=-maximum-1;
  EXPECT_FALSE(a::pell_is_solution(2,{maximum,maximum,true}));
  EXPECT_FALSE(a::pell_is_solution(2,{minimum,minimum,true}));
  EXPECT_FALSE(a::pell_is_solution(4,{maximum,maximum/2,true}));
  auto p=a::chakravala(661);++p.x;EXPECT_FALSE(a::pell_is_solution(661,p));--p.x;++p.y;EXPECT_FALSE(a::pell_is_solution(661,p));
}
TEST(PellCheck, SmallPositivePairsAgreeWithDirectWidenedProducts) {
  for(std::uint64_t n=0;n<20;++n)for(__int128 x=0;x<100;++x)for(__int128 y=0;y<100;++y)
    EXPECT_EQ(a::pell_is_solution(n,{x,y,true}),n>=2 && x>1 && y>0 && x*x-static_cast<__int128>(n)*y*y==1);
}
TEST(IntegerArithmetic, RootsSatisfyUniqueIntegerBracketAcrossFullRange) {
  std::mt19937_64 rng(64);
  for(int i=0;i<50000;++i) {
    const auto n=rng(),r=a::integer_sqrt_u64(n);EXPECT_LE(Wide(r)*r,n);EXPECT_GT(Wide(r+1)*(r+1),n);
  }
  for(std::uint64_t root=0;root<100000;++root) {
    const auto square=root*root;EXPECT_EQ(a::integer_sqrt_u64(square),root);
    if(root)EXPECT_EQ(a::integer_sqrt_u64(square-1),root-1);
  }
  EXPECT_EQ(a::integer_sqrt_u64(std::numeric_limits<std::uint64_t>::max()),4294967295ULL);
}
TEST(IntegerArithmetic, DecimalTextCoversZeroEverySignedEndpointAndAllUnsignedBits) {
  EXPECT_EQ(a::int128_text(0),"0");EXPECT_EQ(a::uint128_text(0),"0");
  const auto maximum=static_cast<__int128>(limit),minimum=-maximum-1;
  EXPECT_EQ(a::int128_text(maximum),"170141183460469231731687303715884105727");
  EXPECT_EQ(a::int128_text(minimum),"-170141183460469231731687303715884105728");
  EXPECT_EQ(a::uint128_text(~Wide{0}),"340282366920938463463374607431768211455");
  std::mt19937_64 rng(128);
  for(int i=0;i<1000;++i) {
    const Wide value=(Wide(rng())<<64)|rng();const auto text=a::uint128_text(value);Wide parsed=0;
    for(const auto c:text){ASSERT_GE(c,'0');ASSERT_LE(c,'9');parsed=parsed*10+(c-'0');}
    EXPECT_EQ(parsed,value);EXPECT_LE(text.size(),39u);
  }
}
