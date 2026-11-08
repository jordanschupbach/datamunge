#include <datamunge/algorithms/cipolla.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <limits>
#include <random>
#include <set>
namespace {
namespace a=datamunge::algorithms;using U=std::uint64_t;
constexpr U maximum=std::numeric_limits<U>::max();
U add(U x,U y,U p){return x>=p-y?x-(p-y):x+y;}
U multiply(U x,U y,U p){U out=0;x%=p;while(y){if(y&1)out=add(out,x,p);y>>=1;if(y)x=add(x,x,p);}return out;}
U power(U x,U e,U p){U out=1;while(e){if(e&1)out=multiply(out,x,p);e>>=1;if(e)x=multiply(x,x,p);}return out;}
U difference(U x,U y,U p){return x>=y?x-y:p-(y-x);}
a::CipollaElement product(a::CipollaElement x,a::CipollaElement y,U w,U p) {
  return {add(multiply(x.x,y.x,p),multiply(multiply(x.y,y.y,p),w,p),p),add(multiply(x.x,y.y,p),multiply(x.y,y.x,p),p)};
}
void same(a::CipollaElement x,a::CipollaElement y){EXPECT_EQ(x.x,y.x);EXPECT_EQ(x.y,y.y);}
std::size_t scalar_products(U e){return std::bit_width(e)+std::popcount(e);}
bool prime(U p){if(p<2)return false;for(U d=2;d<=p/d;++d)if(p%d==0)return false;return true;}
void verify(U n,U p) {
  const auto t=a::cipolla_sqrt_trace(n,p);const auto ordinary=a::cipolla_sqrt_result(n,p);const auto legacy=a::cipolla_sqrt(n,p);const auto normalized=n%p;
  EXPECT_EQ(t.input_n,n);EXPECT_EQ(t.prime,p);EXPECT_EQ(t.normalized_n,normalized);EXPECT_EQ(t.work.primality_checks,1u);
  EXPECT_EQ(t.result.exists,ordinary.exists);EXPECT_EQ(t.result.root,ordinary.root);EXPECT_EQ(t.result.other_root,ordinary.other_root);EXPECT_EQ(t.result.legacy_root,legacy);
  if(t.result.exists) {
    EXPECT_GE(legacy,0);EXPECT_LT(t.result.root,p);EXPECT_LT(t.result.other_root,p);
    EXPECT_EQ(multiply(t.result.root,t.result.root,p),normalized);EXPECT_EQ(multiply(t.result.other_root,t.result.other_root,p),normalized);
    EXPECT_TRUE(a::cipolla_root_satisfies(n,p,t.result.root));EXPECT_TRUE(a::cipolla_root_satisfies(n,p,t.result.other_root));
    EXPECT_TRUE(U(legacy)==t.result.root || U(legacy)==t.result.other_root);EXPECT_EQ(multiply(U(legacy),U(legacy),p),normalized);
    EXPECT_EQ(t.result.other_root,t.result.root?p-t.result.root:0);
    if(t.result.root<=U(std::numeric_limits<long long>::max()))EXPECT_EQ(U(legacy),t.result.root);else EXPECT_EQ(U(legacy),t.result.other_root);
  } else {EXPECT_EQ(legacy,-1);EXPECT_EQ(t.result.root,0);EXPECT_EQ(t.result.other_root,0);}
  if(p==2 || !normalized) {
    EXPECT_EQ(t.path,p==2?a::CipollaPath::BinaryField:a::CipollaPath::Zero);EXPECT_TRUE(t.result.exists);EXPECT_EQ(t.residue_euler,0);EXPECT_EQ(t.work.residue_power_products,0u);
  } else {
    const auto euler=power(normalized,(p-1)/2,p);EXPECT_EQ(t.residue_euler,euler);EXPECT_EQ(t.work.residue_power_products,scalar_products((p-1)/2));
    if(euler!=1){EXPECT_EQ(t.path,a::CipollaPath::NonResidue);EXPECT_FALSE(t.result.exists);}else EXPECT_EQ(t.path,a::CipollaPath::Extension);
  }
  if(t.path!=a::CipollaPath::Extension) {
    EXPECT_TRUE(t.candidates.empty());EXPECT_TRUE(t.steps.empty());EXPECT_EQ(t.a,0);EXPECT_EQ(t.w,0);same(t.final_element,{0,0});
    EXPECT_EQ(t.work.candidate_tests,0u);EXPECT_EQ(t.work.candidate_squares,0u);EXPECT_EQ(t.work.candidate_power_products,0u);
    EXPECT_EQ(t.work.extension_result_products,0u);EXPECT_EQ(t.work.extension_base_squares,0u);EXPECT_EQ(t.work.extension_scalar_products,0u);EXPECT_EQ(t.work.verification_products,0u);return;
  }
  ASSERT_FALSE(t.candidates.empty());U next=1;
  for(std::size_t i=0;i<t.candidates.size();++i) {
    const auto& c=t.candidates[i];EXPECT_EQ(c.a,next);EXPECT_EQ(c.w,difference(multiply(c.a,c.a,p),normalized,p));EXPECT_EQ(c.euler_value,power(c.w,(p-1)/2,p));
    if(i+1==t.candidates.size()){EXPECT_EQ(c.euler_value,p-1);EXPECT_EQ(t.a,c.a);EXPECT_EQ(t.w,c.w);}else EXPECT_NE(c.euler_value,p-1);
    next=next==p-1?0:next+1;
  }
  EXPECT_EQ(t.work.candidate_tests,t.candidates.size());EXPECT_EQ(t.work.candidate_squares,t.candidates.size());EXPECT_EQ(t.work.candidate_power_products,t.candidates.size()*scalar_products((p-1)/2));
  U exponent=p/2+1;a::CipollaElement result{1,0},base{t.a,1};std::size_t result_products=0;
  for(const auto& step:t.steps) {
    EXPECT_EQ(step.remaining_exponent,exponent);same(step.result_before,result);same(step.base_before,base);EXPECT_EQ(step.multiplied,bool(exponent&1));
    if(exponent&1){result=product(result,base,t.w,p);++result_products;}
    base=product(base,base,t.w,p);same(step.result_after,result);same(step.base_after,base);exponent>>=1;
  }
  EXPECT_EQ(exponent,0);EXPECT_EQ(t.steps.size(),std::bit_width(p/2+1));EXPECT_EQ(t.work.extension_result_products,result_products);
  EXPECT_EQ(t.work.extension_base_squares,t.steps.size());EXPECT_EQ(t.work.extension_scalar_products,5*(result_products+t.steps.size()));EXPECT_EQ(t.work.verification_products,1u);
  same(t.final_element,result);EXPECT_EQ(result.y,0);EXPECT_EQ(result.x,t.result.root);
}
}
TEST(CipollaTrace, ExhaustiveSmallPrimeFieldsAgreeWithEnumeratedSquareSetsAndNormalizedAliases) {
  std::size_t fields=0,residues=0;
  for(U p=2;p<=257;++p)if(prime(p)) {
    std::vector<std::vector<U>> roots(p);for(U x=0;x<p;++x)roots[(x*x)%p].push_back(x);
    for(U n=0;n<p;++n){verify(n,p);const auto r=a::cipolla_sqrt_result(n+p,p);EXPECT_EQ(r.exists,!roots[n].empty());if(r.exists)EXPECT_NE(std::find(roots[n].begin(),roots[n].end(),r.root),roots[n].end());++residues;}
    ++fields;
  }
  EXPECT_EQ(fields,55u);EXPECT_GT(residues,6000u);
}
TEST(CipollaTrace, MissingZeroAuxiliaryModuloThreeIsReachedAfterBothNonzeroFailures) {
  verify(1,3);const auto t=a::cipolla_sqrt_trace(1,3);ASSERT_EQ(t.candidates.size(),3u);
  EXPECT_EQ(t.candidates[0].a,1);EXPECT_EQ(t.candidates[1].a,2);EXPECT_EQ(t.candidates[2].a,0);
  EXPECT_EQ(t.candidates[0].w,0);EXPECT_EQ(t.candidates[1].w,0);EXPECT_EQ(t.candidates[2].w,2);
  EXPECT_EQ(t.result.root,2);EXPECT_EQ(t.result.other_root,1);EXPECT_EQ(a::cipolla_sqrt(1,3),2);
}
TEST(CipollaTrace, WorkedFieldThirteenRetainsActualCandidatesBitsClosingSquareAndAllCounters) {
  verify(10,13);const auto t=a::cipolla_sqrt_trace(10,13);ASSERT_EQ(t.candidates.size(),2u);ASSERT_EQ(t.steps.size(),3u);
  EXPECT_EQ(t.a,2);EXPECT_EQ(t.w,7);same(t.steps[0].result_after,{2,1});same(t.steps[1].result_after,{11,6});same(t.steps[2].result_after,{6,0});same(t.steps[2].base_after,{12,6});
  EXPECT_EQ(t.work.residue_power_products,5u);EXPECT_EQ(t.work.candidate_squares,2u);EXPECT_EQ(t.work.candidate_power_products,10u);
  EXPECT_EQ(t.work.extension_result_products,3u);EXPECT_EQ(t.work.extension_base_squares,3u);EXPECT_EQ(t.work.extension_scalar_products,30u);EXPECT_EQ(t.work.verification_products,1u);
}
TEST(CipollaTrace, OriginalFourRootChoicesAndThousandNineCensusRemain) {
  const std::vector<U> n{2,10,100,500};const std::vector<long long> roots{439,162,999,587};
  for(std::size_t i=0;i<n.size();++i)EXPECT_EQ(a::cipolla_sqrt(n[i],1009),roots[i]);
  std::size_t nonzero_squares=0;for(U residue=0;residue<1009;++residue){verify(residue,1009);const auto t=a::cipolla_sqrt_result(residue,1009);if(residue && t.exists)++nonzero_squares;}
  EXPECT_EQ(nonzero_squares,504u);EXPECT_EQ(a::cipolla_sqrt(11,1009),-1);
}
TEST(CipollaTrace, ThousandKnownSquaresAcrossFullRangePrimesUseIndependentAddAndDoubleOracle) {
  const std::vector<U> primes{4294967291ULL,2305843009213693951ULL,9223372036854775783ULL,18446744073709551557ULL};
  std::mt19937_64 random(1903);std::size_t fallbacks=0;
  for(int trial=0;trial<1000;++trial) {
    const auto p=primes[trial%primes.size()],x=random()%p,n=multiply(x,x,p);verify(n,p);const auto r=a::cipolla_sqrt_result(n,p);ASSERT_TRUE(r.exists);
    if(r.root>U(std::numeric_limits<long long>::max())){++fallbacks;EXPECT_EQ(U(r.legacy_root),r.other_root);}
  }
  EXPECT_GT(fallbacks,0u);
}
TEST(CipollaTrace, UnsignedEndpointResiduesNormalizeAndEveryReturnedLegacyRootIsNonnegative) {
  for(const auto p:{2ULL,3ULL,13ULL,1009ULL,18446744073709551557ULL})verify(maximum,p);
  const U p=18446744073709551557ULL;for(const U x:std::vector<U>{0,1,2,p-1,p-2,p/2}){const auto n=multiply(x,x,p);verify(n,p);EXPECT_GE(a::cipolla_sqrt(n,p),0);}
}
TEST(CipollaTrace, PrimeValidationPrecedesZeroShortcutsAndAllUnusedCaps) {
  a::CipollaOptions options;options.max_candidates=1;options.max_extension_products=1;
  for(const U p:std::vector<U>{0,1,4,9,15,341,561,1105,1729,3215031751ULL,3825123056546413051ULL,maximum-1,maximum}) {
    EXPECT_THROW(a::cipolla_sqrt(0,p),std::invalid_argument);
    EXPECT_THROW(a::cipolla_sqrt_trace(0,p,options),std::invalid_argument);
    EXPECT_THROW(a::cipolla_sqrt_result(0,p,options),std::invalid_argument);
    EXPECT_THROW(a::cipolla_root_satisfies(0,p,0),std::invalid_argument);
  }
}
TEST(CipollaTrace, ZeroBinaryFieldAndNonresiduePathsSkipUnusedExtensionWork) {
  a::CipollaOptions cap;cap.max_candidates=1;cap.max_extension_products=1;
  for(const U n:std::vector<U>{0,1,maximum}){verify(n,2);EXPECT_TRUE(a::cipolla_sqrt_trace(n,2,cap).result.exists);}
  verify(0,13);EXPECT_TRUE(a::cipolla_sqrt_trace(0,13,cap).result.exists);
  const auto t=a::cipolla_sqrt_trace(11,1009,cap);EXPECT_EQ(t.path,a::CipollaPath::NonResidue);EXPECT_FALSE(t.result.exists);EXPECT_EQ(t.work.candidate_tests,0u);
  a::CipollaTrace unprocessed;EXPECT_EQ(unprocessed.path,a::CipollaPath::Unprocessed);EXPECT_FALSE(unprocessed.result.exists);
}
TEST(CipollaTrace, ExactAndOneBelowWholeCallCapsExposeNoPartialRootOrTrace) {
  a::CipollaOptions exact;exact.max_candidates=2;exact.max_extension_products=6;
  EXPECT_EQ(a::cipolla_sqrt_trace(10,13,exact).result.root,6);EXPECT_EQ(a::cipolla_sqrt_result(10,13,exact).root,6);
  auto low=exact;low.max_candidates=1;
  EXPECT_THROW(a::cipolla_sqrt_trace(10,13,low),std::length_error);
  EXPECT_THROW(a::cipolla_sqrt_result(10,13,low),std::length_error);
  low=exact;low.max_extension_products=5;
  EXPECT_THROW(a::cipolla_sqrt_trace(10,13,low),std::length_error);
  EXPECT_THROW(a::cipolla_sqrt_result(10,13,low),std::length_error);
  exact.max_candidates=3;exact.max_extension_products=3;EXPECT_EQ(a::cipolla_sqrt_result(1,3,exact).root,2);
  exact.max_candidates=2;
  EXPECT_THROW(a::cipolla_sqrt_trace(1,3,exact),std::length_error);
}
TEST(CipollaTrace, ClosingUnusedSquareStillConsumesItsActualProductBudget) {
  a::CipollaOptions options;options.max_extension_products=2;
  EXPECT_THROW(a::cipolla_sqrt_trace(1,3,options),std::length_error);
  options.max_extension_products=3;const auto t=a::cipolla_sqrt_trace(1,3,options);EXPECT_EQ(t.work.extension_result_products,1u);EXPECT_EQ(t.work.extension_base_squares,2u);
}
TEST(CipollaTrace, EveryCanonicalResidueHasTheCorrectRootMultiplicityForSmallOddPrimes) {
  for(const U p:{3ULL,5ULL,7ULL,13ULL,17ULL,97ULL}) {
    std::size_t squares=0;
    for(U n=0;n<p;++n){const auto r=a::cipolla_sqrt_result(n,p);if(!n){EXPECT_EQ(r.root,r.other_root);continue;}if(r.exists){++squares;EXPECT_NE(r.root,r.other_root);EXPECT_EQ(r.root+r.other_root,p);}}
    EXPECT_EQ(squares,(p-1)/2);
  }
}
TEST(CipollaTrace, ReturnedCandidateAndBitVectorsAreOwnedAcrossCallsAndCopies) {
  auto t=a::cipolla_sqrt_trace(10,13);const auto copy=t;t.candidates.clear();t.steps[0].result_after.x=0;
  EXPECT_EQ(copy.candidates.size(),2u);EXPECT_EQ(copy.steps[0].result_after.x,2);EXPECT_EQ(a::cipolla_sqrt_trace(10,13).steps[0].result_after.x,2);
}
TEST(CipollaTrace, RootEquationCheckerHandlesFullRangeRepresentativesWithoutSignedSquares) {
  EXPECT_TRUE(a::cipolla_root_satisfies(2,1009,439));EXPECT_FALSE(a::cipolla_root_satisfies(2,1009,438));EXPECT_TRUE(a::cipolla_root_satisfies(2+1009,1009,439+1009));
  const U p=18446744073709551557ULL;EXPECT_TRUE(a::cipolla_root_satisfies(1,p,p-1));EXPECT_TRUE(a::cipolla_root_satisfies(4,p,p-2));EXPECT_FALSE(a::cipolla_root_satisfies(5,p,p-2));
}
TEST(CipollaTrace, PowerLengthDependsOnPrimeWhileCandidateCountDependsOnTheResidue) {
  std::set<std::size_t> candidates;
  for(U n=1;n<1009;++n) {
    const auto t=a::cipolla_sqrt_trace(n,1009);if(!t.result.exists)continue;candidates.insert(t.work.candidate_tests);
    EXPECT_EQ(t.work.extension_result_products,std::popcount(U(505)));EXPECT_EQ(t.work.extension_base_squares,std::bit_width(U(505)));
  }
  EXPECT_GT(candidates.size(),1u);
}
