#include <datamunge/algorithms/chudnovsky.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <random>
namespace {
namespace a=datamunge::algorithms;
constexpr long double A=13591409,B=545140134,C=640320;
long double factorial(int n){long double out=1;for(int i=2;i<=n;++i)out*=i;return out;}
long double term(int k){const auto f=factorial(k);const auto magnitude=factorial(6*k)*(A+B*k)/(factorial(3*k)*f*f*f*std::pow(C,3*k));return k%2?-magnitude:magnitude;}
long double ratio(int k){const long double x=k;return -(6*x+1)*(2*x+1)*(6*x+5)*(A+B*(x+1))/((x+1)*(x+1)*(x+1)*(C*C*C/24)*(A+B*x));}
}
TEST(ChudnovskyTrace, FirstTwelveActualTermsAgreeWithIndependentFactorialsAndPrefixSums) {
  const auto t=a::chudnovsky_pi_trace(12);ASSERT_EQ(t.steps.size(),12u);long double sum=0;double previous=0;
  for(int k=0;k<12;++k) {
    const auto& s=t.steps[k];const auto expected=term(k);sum+=expected;
    EXPECT_EQ(s.index,k);EXPECT_NEAR(static_cast<long double>(s.term)/expected,1,3e-15);
    EXPECT_EQ(s.sum_before,previous);EXPECT_EQ(s.sum_after,s.sum_before+s.term);previous=s.sum_after;
    EXPECT_NEAR(s.sum_after/sum,1,3e-16);EXPECT_EQ(s.sum_changed,s.sum_after!=s.sum_before);
    const auto estimate=426880.0L*std::sqrt(10005.0L)/sum;EXPECT_NEAR(s.estimate,estimate,2e-15);
    EXPECT_EQ(s.estimate,a::chudnovsky_pi(k+1));
  }
  EXPECT_EQ(t.value,t.steps.back().estimate);EXPECT_EQ(t.terms_evaluated,12);EXPECT_EQ(t.ratio_evaluations,11);EXPECT_EQ(t.sum_additions,12);
}
TEST(ChudnovskyTrace, RatioUsesAllFactorialFactorsAndHandlesThousandFullRangeIndices) {
  EXPECT_NEAR(a::chudnovsky_term_ratio(0),-1.8790085379891469e-14,1e-29);
  std::mt19937_64 random(1989);
  for(int trial=0;trial<1000;++trial) {
    const auto k=static_cast<int>(random()%static_cast<unsigned>(std::numeric_limits<int>::max()));
    const auto actual=a::chudnovsky_term_ratio(k);EXPECT_LT(actual,0);EXPECT_TRUE(std::isfinite(actual));EXPECT_NEAR(actual/ratio(k),1,1e-15);
  }
  EXPECT_NEAR(a::chudnovsky_term_ratio(std::numeric_limits<int>::max())/ratio(std::numeric_limits<int>::max()),1,1e-15);
}
TEST(ChudnovskyTrace, LimitingRatioAndUniformPostInitialBoundMatchExactPolynomialIdentity) {
  constexpr long double rho=1728/(C*C*C);
  EXPECT_NEAR(a::chudnovsky_digits_per_term(),14.181647462725477,2e-14);
  for(int k=1;k<=10000;++k)EXPECT_LE(std::fabs(a::chudnovsky_term_ratio(k)),rho*(1+1e-15L));
  for(__int128 k=1;k<=1000;++k) {
    const __int128 av=13591409,bv=545140134,x=k-1;
    const auto difference=72*(k+1)*(k+1)*(k+1)*(av+bv*k)-(6*k+1)*(2*k+1)*(6*k+5)*(av+bv*(k+1));
    EXPECT_EQ(difference,66835011381+143166737776*x+94141694952*x*x+19625044824*x*x*x);EXPECT_GT(difference,0);
  }
}
TEST(ChudnovskyTrace, GeometricTruncationBoundDominatesIndependentFirstOmittedTerms) {
  const auto first=-term(1),lower=A-first,normalizer=426880.0L*std::sqrt(10005.0L);
  for(int n=1;n<=100;++n) {
    const auto observed=a::chudnovsky_log10_truncation_bound(n);
    const auto first_omitted=std::log10(normalizer)+std::log10(std::fabs(term(n)))-2*std::log10(lower);
    EXPECT_GE(observed+1e-11L,first_omitted);EXPECT_TRUE(std::isfinite(observed));
    if(n>1)EXPECT_NEAR(a::chudnovsky_log10_truncation_bound(n-1)-observed,a::chudnovsky_digits_per_term(),3e-13);
  }
  EXPECT_LT(a::chudnovsky_log10_truncation_bound(2),-27);EXPECT_GT(a::chudnovsky_log10_truncation_bound(1),-14);
  EXPECT_TRUE(std::isfinite(a::chudnovsky_log10_truncation_bound(std::numeric_limits<int>::max())));
}
TEST(ChudnovskyTrace, NonzeroTermsCanStopChangingTheSumWithoutStoppingRequestedEvaluation) {
  const auto t=a::chudnovsky_pi_trace(4);ASSERT_EQ(t.steps.size(),4u);EXPECT_EQ(t.first_unchanged_index,2);
  EXPECT_EQ(t.stop,a::ChudnovskyStop::RequestedTerms);EXPECT_TRUE(t.steps[0].sum_changed);EXPECT_TRUE(t.steps[1].sum_changed);
  EXPECT_FALSE(t.steps[2].sum_changed);EXPECT_NE(t.steps[2].term,0);EXPECT_NE(t.steps[3].term,0);
  EXPECT_EQ(t.steps[1].estimate,t.steps[2].estimate);EXPECT_EQ(t.steps[2].estimate,t.steps[3].estimate);
  const auto pi=std::numbers::pi_v<long double>;EXPECT_LT(std::fabs(t.steps[1].estimate-pi),std::fabs(t.steps[0].estimate-pi));
}
TEST(ChudnovskyTrace, HugeRequestsStopAtActualTermUnderflowWithoutGiantRecordReservation) {
  const auto t=a::chudnovsky_pi_trace(std::numeric_limits<int>::max());ASSERT_GT(t.steps.size(),4u);ASSERT_LT(t.steps.size(),64u);
  EXPECT_EQ(t.terms_requested,std::numeric_limits<int>::max());EXPECT_EQ(t.stop,a::ChudnovskyStop::TermUnderflow);
  EXPECT_EQ(t.steps.back().term,0);EXPECT_EQ(t.terms_evaluated,t.steps.size());EXPECT_EQ(t.ratio_evaluations,t.terms_evaluated-1);
  EXPECT_EQ(t.sum_additions,t.terms_evaluated);EXPECT_FALSE(t.steps.back().sum_changed);EXPECT_NE(t.steps[t.steps.size()-2].term,0);
  EXPECT_EQ(a::chudnovsky_pi_trace(t.terms_evaluated-1).stop,a::ChudnovskyStop::RequestedTerms);
  EXPECT_EQ(a::chudnovsky_pi_trace(t.terms_evaluated).stop,a::ChudnovskyStop::TermUnderflow);
  EXPECT_EQ(a::chudnovsky_pi(std::numeric_limits<int>::max()),t.value);
}
TEST(ChudnovskyTrace, ThousandsOfPositiveCountsRemainFinitePastOldFactorialOverflow) {
  const auto reference=std::numbers::pi_v<long double>;
  for(int n=1;n<=4096;++n) {
    const auto value=a::chudnovsky_pi(n);const auto t=a::chudnovsky_pi_trace(n);EXPECT_TRUE(std::isfinite(value));EXPECT_NEAR(value,reference,1e-12);
    EXPECT_EQ(value,t.value);EXPECT_LE(t.terms_evaluated,n);EXPECT_EQ(t.steps.size(),t.terms_evaluated);
    if(n>1)EXPECT_EQ(value,a::chudnovsky_pi(2));
  }
  EXPECT_TRUE(std::isfinite(a::chudnovsky_pi(29)));EXPECT_TRUE(std::isfinite(a::chudnovsky_pi(1000000)));
}
TEST(ChudnovskyTrace, NonpositiveCountsAndNegativeRatioIndicesRejectExplicitly) {
  for(const auto n:{0,-1,std::numeric_limits<int>::min()}) {
    EXPECT_THROW(a::chudnovsky_pi(n),std::invalid_argument);
    EXPECT_THROW(a::chudnovsky_pi_trace(n),std::invalid_argument);
    EXPECT_THROW(a::chudnovsky_log10_truncation_bound(n),std::invalid_argument);
  }
  EXPECT_THROW(a::chudnovsky_term_ratio(-1),std::invalid_argument);
  EXPECT_THROW(a::chudnovsky_term_ratio(std::numeric_limits<int>::min()),std::invalid_argument);
}
TEST(ChudnovskyTrace, RecordsAreOwnedAndCallsShareNoMutableState) {
  auto t=a::chudnovsky_pi_trace(4);const auto original=t;auto copy=t;t.steps[0].term=0;copy.steps.clear();
  EXPECT_EQ(original.steps[0].term,13591409);EXPECT_EQ(a::chudnovsky_pi_trace(4).steps[0].term,13591409);
  a::ChudnovskyTrace empty;EXPECT_EQ(empty.stop,a::ChudnovskyStop::Unprocessed);EXPECT_TRUE(empty.steps.empty());EXPECT_EQ(empty.first_unchanged_index,-1);
}
TEST(ChudnovskyTrace, PositivePrefixesAlternateUntilUnderflowWithoutArtificialLogFloor) {
  const auto t=a::chudnovsky_pi_trace(1000);
  for(std::size_t i=0;i<t.steps.size();++i) {
    const auto& s=t.steps[i];if(s.term!=0){EXPECT_EQ(std::signbit(s.term),bool(i%2));EXPECT_TRUE(std::isfinite(std::log10(std::fabs(s.term))));}
    if(i && s.term!=0)EXPECT_LT(std::fabs(s.term),std::fabs(t.steps[i-1].term));
  }
}
