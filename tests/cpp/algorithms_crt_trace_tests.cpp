#include <datamunge/algorithms/chinese_remainder.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <random>
#include <vector>
namespace {
namespace a=datamunge::algorithms;using Wide=__int128;
constexpr auto maximum=std::numeric_limits<std::int64_t>::max();
constexpr auto minimum=std::numeric_limits<std::int64_t>::min();
std::int64_t norm(Wide x,std::int64_t m) {x%=m;return static_cast<std::int64_t>(x<0?x+m:x);}
std::size_t divisions(std::int64_t x,std::int64_t y) {std::size_t steps=0;while(y){const auto r=x%y;x=y;y=r;++steps;}return steps;}
a::CrtSolution brute(const std::vector<std::int64_t>& r,const std::vector<std::int64_t>& m) {
  std::int64_t period=1;for(const auto modulus:m)period=std::lcm(period,modulus);
  for(std::int64_t x=0;x<period;++x) {
    bool valid=true;for(std::size_t i=0;i<m.size();++i)if(x%m[i]!=norm(r[i],m[i])){valid=false;break;}
    if(valid)return {true,x,period};
  }
  return {false,0,0};
}
void verify_certificate(const a::CrtTrace& trace,const std::vector<std::int64_t>& r,const std::vector<std::int64_t>& m) {
  EXPECT_EQ(trace.work.validated_moduli,m.size());EXPECT_EQ(trace.work.attempted_merges,trace.steps.size());
  a::CrtSolution prefix{true,0,1};std::size_t complete=0,compatible=0,steps=0;
  for(std::size_t i=0;i<trace.steps.size();++i) {
    const auto& s=trace.steps[i];ASSERT_LT(i,m.size());EXPECT_EQ(s.index,i);EXPECT_TRUE(s.before.solvable);
    EXPECT_EQ(s.before.remainder,prefix.remainder);EXPECT_EQ(s.before.modulus,prefix.modulus);
    EXPECT_EQ(s.input_remainder,r[i]);EXPECT_EQ(s.input_modulus,m[i]);EXPECT_EQ(s.normalized_remainder,norm(r[i],m[i]));
    const auto gcd=std::gcd(prefix.modulus,m[i]);EXPECT_EQ(s.gcd,gcd);
    EXPECT_EQ(Wide(prefix.modulus)*s.bezout_first+Wide(m[i])*s.bezout_second,gcd);
    EXPECT_EQ(s.euclidean_steps,divisions(prefix.modulus,m[i]));steps+=s.euclidean_steps;
    EXPECT_EQ(s.difference,norm(r[i],m[i])-prefix.remainder);EXPECT_EQ(s.reduced_modulus,m[i]/gcd);
    if(s.difference%gcd) {
      EXPECT_EQ(s.status,a::CrtStatus::Inconsistent);EXPECT_EQ(s.candidate_modulus,0);EXPECT_EQ(s.candidate_remainder,0);EXPECT_EQ(s.correction_product,0);
      EXPECT_EQ(trace.status,a::CrtStatus::Inconsistent);EXPECT_EQ(i+1,trace.steps.size());break;
    }
    ++compatible;
    const Wide expected_period=Wide(prefix.modulus/gcd)*m[i];
    EXPECT_EQ(s.candidate_modulus,expected_period);EXPECT_GE(s.candidate_remainder,0);EXPECT_LT(s.candidate_remainder,expected_period);
    EXPECT_EQ(s.correction_product,(Wide(s.difference/gcd)%s.reduced_modulus)*(s.bezout_first%s.reduced_modulus));
    EXPECT_EQ(s.multiplier,norm(s.correction_product,s.reduced_modulus));EXPECT_GE(s.multiplier,0);EXPECT_LT(s.multiplier,s.reduced_modulus);
    EXPECT_EQ(s.candidate_remainder,Wide(prefix.remainder)+Wide(prefix.modulus)*s.multiplier);
    EXPECT_EQ(s.candidate_remainder%prefix.modulus,prefix.remainder);EXPECT_EQ(s.candidate_remainder%m[i],norm(r[i],m[i]));
    if(expected_period>maximum) {
      EXPECT_EQ(s.status,a::CrtStatus::Overflow);EXPECT_EQ(trace.status,a::CrtStatus::Overflow);EXPECT_EQ(i+1,trace.steps.size());break;
    }
    EXPECT_EQ(s.status,a::CrtStatus::Complete);prefix={true,static_cast<std::int64_t>(s.candidate_remainder),static_cast<std::int64_t>(expected_period)};++complete;
  }
  EXPECT_EQ(trace.prefix.remainder,prefix.remainder);EXPECT_EQ(trace.prefix.modulus,prefix.modulus);EXPECT_TRUE(trace.prefix.solvable);
  EXPECT_EQ(trace.work.completed_merges,complete);EXPECT_EQ(trace.work.euclidean_steps,steps);
  EXPECT_EQ(trace.work.correction_products,compatible);EXPECT_EQ(trace.work.period_products,compatible);EXPECT_EQ(trace.work.residue_products,compatible);
  if(trace.status==a::CrtStatus::Complete) {
    ASSERT_TRUE(trace.solution);EXPECT_TRUE(trace.solution->solvable);EXPECT_EQ(trace.steps.size(),m.size());
    EXPECT_EQ(trace.solution->remainder,prefix.remainder);EXPECT_EQ(trace.solution->modulus,prefix.modulus);
  } else if(trace.status==a::CrtStatus::Inconsistent) {
    ASSERT_TRUE(trace.solution);EXPECT_FALSE(trace.solution->solvable);EXPECT_EQ(trace.solution->remainder,0);EXPECT_EQ(trace.solution->modulus,0);
  } else {EXPECT_EQ(trace.status,a::CrtStatus::Overflow);EXPECT_FALSE(trace.solution);}
}
void verify_small(const std::vector<std::int64_t>& r,const std::vector<std::int64_t>& m) {
  const auto want=brute(r,m),got=a::chinese_remainder(r,m);const auto t=a::chinese_remainder_trace(r,m);verify_certificate(t,r,m);
  ASSERT_TRUE(t.solution);EXPECT_EQ(got.solvable,want.solvable);EXPECT_EQ(got.remainder,want.remainder);EXPECT_EQ(got.modulus,want.modulus);
  EXPECT_EQ(t.solution->solvable,want.solvable);EXPECT_EQ(t.solution->remainder,want.remainder);EXPECT_EQ(t.solution->modulus,want.modulus);
}
}
TEST(CrtTrace, ExhaustiveSmallPairsAndTriplesAgreeWithEnumeratedCompleteResidueClass) {
  std::size_t cases=0;
  for(std::int64_t m=1;m<=10;++m)for(std::int64_t n=1;n<=10;++n)
    for(std::int64_t r=0;r<m;++r)for(std::int64_t s=0;s<n;++s){verify_small({r,s},{m,n});++cases;}
  for(std::int64_t m=1;m<=6;++m)for(std::int64_t n=1;n<=6;++n)for(std::int64_t o=1;o<=6;++o)
    for(std::int64_t r=0;r<m;++r)for(std::int64_t s=0;s<n;++s)for(std::int64_t t=0;t<o;++t){verify_small({r,s,t},{m,n,o});++cases;}
  EXPECT_EQ(cases,12286u);
}
TEST(CrtTrace, ThousandSignedRandomSmallSystemsAgreeWithBruteForce) {
  std::mt19937_64 random(3);
  for(int trial=0;trial<1000;++trial) {
    const auto n=random()%7;std::vector<std::int64_t> r,m;
    for(std::size_t i=0;i<n;++i){r.push_back(static_cast<std::int64_t>(random()%2001)-1000);m.push_back(1+random()%8);}
    verify_small(r,m);
  }
}
TEST(CrtTrace, OriginalThreeSystemsAndActualSunTzuCertificates) {
  const std::vector<std::int64_t> r{2,3,2},m{3,5,7};const auto t=a::chinese_remainder_trace(r,m);verify_certificate(t,r,m);
  ASSERT_EQ(t.steps.size(),3u);EXPECT_EQ(t.steps[0].candidate_remainder,2);EXPECT_EQ(t.steps[1].candidate_remainder,8);EXPECT_EQ(t.steps[2].candidate_remainder,23);
  EXPECT_EQ(t.steps[1].bezout_first,2);EXPECT_EQ(t.steps[1].bezout_second,-1);EXPECT_EQ(t.steps[1].multiplier,2);
  EXPECT_EQ(t.steps[2].bezout_first,1);EXPECT_EQ(t.steps[2].bezout_second,-2);EXPECT_EQ(t.steps[2].difference,-6);EXPECT_EQ(t.steps[2].multiplier,1);
  EXPECT_EQ(t.work.euclidean_steps,8u);EXPECT_EQ(t.work.completed_merges,3u);
  verify_small({2,8},{6,12});verify_small({1,0},{2,4});
}
TEST(CrtTrace, FullRangeConsistentScaledSystemsPreserveKnownCandidateAndLcm) {
  std::mt19937_64 random(64);const std::array<std::int64_t,5> factors{1,2,3,5,7};
  for(int trial=0;trial<1000;++trial) {
    const std::size_t n=1+random()%5;std::vector<std::int64_t> r,m;std::int64_t base_period=1;
    std::vector<std::int64_t> selected;for(std::size_t i=0;i<n;++i){selected.push_back(factors[random()%factors.size()]);base_period=std::lcm(base_period,selected.back());}
    const auto scale=1+static_cast<std::int64_t>(random()%(maximum/base_period));
    auto candidate=static_cast<std::int64_t>(random()>>1);if(trial%2)candidate=-candidate;if(trial==0)candidate=minimum;
    for(const auto factor:selected){r.push_back(candidate);m.push_back(scale*factor);}
    const auto trace=a::chinese_remainder_trace(r,m);verify_certificate(trace,r,m);ASSERT_TRUE(trace.solution);ASSERT_TRUE(trace.solution->solvable);
    const auto period=scale*base_period;EXPECT_EQ(trace.solution->modulus,period);EXPECT_EQ(trace.solution->remainder,norm(candidate,period));
    EXPECT_TRUE(a::crt_residue_satisfies(trace.solution->remainder,r,m));
  }
}
TEST(CrtTrace, WidenedCorrectionProductCanExceedInt64WhileFinalPeriodFits) {
  const auto b=maximum/2;const std::vector<std::int64_t> r{0,b-1},m{2,b};
  const auto t=a::chinese_remainder_trace(r,m);verify_certificate(t,r,m);ASSERT_TRUE(t.solution);EXPECT_TRUE(t.solution->solvable);
  EXPECT_EQ(t.solution->modulus,maximum-1);EXPECT_EQ(t.solution->remainder,b-1);EXPECT_LT(t.steps[1].correction_product,-Wide(maximum));
  EXPECT_EQ(a::chinese_remainder(r,m).remainder,b-1);
}
TEST(CrtTrace, ExactInt64PeriodAndNormalizationEndpointsStayRepresentable) {
  const std::vector<std::int64_t> r{-1,-1},m{7,maximum};const auto t=a::chinese_remainder_trace(r,m);verify_certificate(t,r,m);
  ASSERT_TRUE(t.solution);EXPECT_EQ(t.solution->modulus,maximum);EXPECT_EQ(t.solution->remainder,maximum-1);
  EXPECT_EQ(a::chinese_remainder({minimum},{maximum}).remainder,maximum-1);
  EXPECT_EQ(a::crt_normalize(minimum,maximum),maximum-1);EXPECT_EQ(a::crt_normalize(maximum,maximum),0);EXPECT_EQ(a::crt_normalize(minimum,1),0);
}
TEST(CrtTrace, CompatiblePeriodOverflowHasWideDiagnosticAndNoFalseInt64Verdict) {
  const std::vector<std::int64_t> r{0,0},m{maximum,maximum-1};const auto t=a::chinese_remainder_trace(r,m);verify_certificate(t,r,m);
  EXPECT_EQ(t.status,a::CrtStatus::Overflow);EXPECT_FALSE(t.solution);EXPECT_EQ(t.prefix.modulus,maximum);EXPECT_EQ(t.work.completed_merges,1u);
  EXPECT_EQ(t.steps.back().candidate_modulus,Wide(maximum)*(maximum-1));EXPECT_EQ(t.steps.back().candidate_remainder,0);
  EXPECT_THROW(a::chinese_remainder(r,m),std::overflow_error);
  EXPECT_THROW(a::chinese_remainder_with_options(r,m),std::overflow_error);
  EXPECT_TRUE(a::crt_residue_satisfies(0,r,m));
}
TEST(CrtTrace, ContradictionPrecedesPeriodFormationAndUnvisitedSuffixHasNoVerdict) {
  const std::vector<std::int64_t> r{0,1},m{maximum-1,maximum-3};const auto bad=a::chinese_remainder_trace(r,m);verify_certificate(bad,r,m);
  EXPECT_EQ(bad.status,a::CrtStatus::Inconsistent);EXPECT_EQ(bad.steps.back().candidate_modulus,0);EXPECT_FALSE(a::chinese_remainder(r,m).solvable);
  const auto late=a::chinese_remainder_trace({0,0,1},{maximum,maximum-1,2});
  EXPECT_EQ(late.status,a::CrtStatus::Overflow);EXPECT_EQ(late.work.validated_moduli,3u);EXPECT_EQ(late.work.attempted_merges,2u);
  const auto early=a::chinese_remainder_trace({0,1,0},{maximum-1,2,maximum});EXPECT_EQ(early.status,a::CrtStatus::Inconsistent);
}
TEST(CrtTrace, LongEuclideanChainsRetainExactWideBezoutIdentity) {
  const std::vector<std::int64_t> r{1,-1},m{7540113804746346429LL,4660046610375530309LL};
  const auto t=a::chinese_remainder_trace(r,m);verify_certificate(t,r,m);EXPECT_EQ(t.status,a::CrtStatus::Overflow);
  EXPECT_GE(t.steps.back().euclidean_steps,89u);
}
TEST(CrtTrace, EmptyUnitDuplicateAndReorderedRepresentableSystemsHaveExactSemantics) {
  const auto empty=a::chinese_remainder_trace({},{});verify_certificate(empty,{},{});ASSERT_TRUE(empty.solution);EXPECT_EQ(empty.solution->modulus,1);EXPECT_EQ(empty.solution->remainder,0);
  verify_small({minimum,maximum,3},{1,1,1});verify_small({-1,5,-7},{6,6,6});
  std::vector<int> order{0,1,2};const std::vector<std::int64_t> r{2,3,2},m{3,5,7};
  do {std::vector<std::int64_t> rr,mm;for(const int i:order){rr.push_back(r[i]);mm.push_back(m[i]);}verify_small(rr,mm);EXPECT_EQ(a::chinese_remainder(rr,mm).remainder,23);}while(std::next_permutation(order.begin(),order.end()));
  a::CrtTrace unprocessed;EXPECT_EQ(unprocessed.status,a::CrtStatus::Unprocessed);EXPECT_FALSE(unprocessed.solution);
}
TEST(CrtTrace, ExactWholeCallCapsAndEarlyArithmeticStopHaveNoPartialFailureResult) {
  const std::vector<std::int64_t> r{2,3,2},m{3,5,7};a::CrtOptions exact;exact.max_merges=3;exact.max_euclidean_steps=8;
  EXPECT_EQ(a::chinese_remainder_with_options(r,m,exact).remainder,23);EXPECT_EQ(a::chinese_remainder_trace(r,m,exact).steps.size(),3u);
  auto low=exact;low.max_merges=2;
  EXPECT_THROW(a::chinese_remainder_with_options(r,m,low),std::length_error);
  EXPECT_THROW(a::chinese_remainder_trace(r,m,low),std::length_error);
  low=exact;low.max_euclidean_steps=7;
  EXPECT_THROW(a::chinese_remainder_trace(r,m,low),std::length_error);
  EXPECT_THROW(a::chinese_remainder_with_options(r,m,low),std::length_error);
  a::CrtOptions early;early.max_merges=2;EXPECT_FALSE(a::chinese_remainder_with_options({1,0,0},{2,4,3},early).solvable);
  EXPECT_EQ(r,(std::vector<std::int64_t>{2,3,2}));EXPECT_EQ(m,(std::vector<std::int64_t>{3,5,7}));
}
TEST(CrtTrace, InvalidInputTailRejectsBeforeInconsistencyCapsOrCandidateComparison) {
  for(const auto& m:std::vector<std::vector<std::int64_t>>{{2,4,0},{2,4,-1},{2,4,minimum}}) {
    EXPECT_THROW(a::chinese_remainder({1,0,0},m),std::invalid_argument);
    EXPECT_THROW(a::chinese_remainder_trace({1,0,0},m),std::invalid_argument);
    EXPECT_THROW(a::crt_residue_satisfies(0,{1,0,0},m),std::invalid_argument);
  }
  EXPECT_THROW(a::chinese_remainder_trace({1},{2,3}),std::invalid_argument);
  EXPECT_THROW(a::crt_residue_satisfies(0,{1},{2,3}),std::invalid_argument);
  EXPECT_THROW(a::crt_normalize(minimum,0),std::invalid_argument);
  EXPECT_THROW(a::crt_normalize(minimum,-1),std::invalid_argument);
}
TEST(CrtTrace, RepeatedUnitConstraintsUseConstantNumericStateWithoutRecursion) {
  const std::vector<std::int64_t> r(100000,minimum),m(100000,1);a::CrtOptions options;options.max_merges=options.max_euclidean_steps=100000;
  const auto s=a::chinese_remainder_with_options(r,m,options);EXPECT_TRUE(s.solvable);EXPECT_EQ(s.remainder,0);EXPECT_EQ(s.modulus,1);
}
TEST(CrtTrace, TraceCopiesOwnAllRecordsAndCallsShareNoMutableState) {
  const std::vector<std::int64_t> r{2,3,2},m{3,5,7};const auto original=a::chinese_remainder_trace(r,m);auto copy=original;
  copy.steps[0].before.modulus=99;copy.solution->remainder=99;
  EXPECT_EQ(original.steps[0].before.modulus,1);EXPECT_EQ(original.solution->remainder,23);EXPECT_EQ(a::chinese_remainder_trace(r,m).solution->remainder,23);
  EXPECT_TRUE(a::crt_residue_satisfies(minimum,{},{}));EXPECT_FALSE(a::crt_residue_satisfies(22,r,m));
}
