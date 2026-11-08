#include <datamunge/algorithms/polynomial_gf.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <random>
#include <set>
namespace {
namespace a=datamunge::algorithms;using P=a::GFPoly;using Wide=unsigned __int128;
std::int64_t norm(std::int64_t x,std::int64_t p) {const auto r=static_cast<__int128>(x)%p;return static_cast<std::int64_t>(r<0?r+p:r);}
P normalize(P f,std::int64_t p) {for(auto& c:f)c=norm(c,p);while(!f.empty()&&!f.back())f.pop_back();return f;}
std::int64_t power(std::int64_t x,std::uint64_t exponent,std::int64_t p) {
  Wide result=1,base=static_cast<Wide>(norm(x,p));
  while(exponent){if(exponent&1)result=result*base%p;base=base*base%p;exponent>>=1;}return static_cast<std::int64_t>(result);
}
std::int64_t horner(const P& f,std::int64_t x,std::int64_t p) {
  Wide value=0;for(std::size_t j=f.size();j-->0;)value=(value*norm(x,p)+norm(f[j],p))%p;return static_cast<std::int64_t>(value);
}
std::int64_t generator(std::int64_t p) {
  if(p==2)return 1;
  // Independent order oracle: walk every power, without factoring p-1.
  for(std::int64_t g=2;g<p;++g) {
    std::set<std::int64_t> visited;std::int64_t x=1;
    for(std::int64_t i=0;i<p-1;++i){visited.insert(x);x=static_cast<std::int64_t>(Wide(x)*g%p);}
    if(visited.size()==static_cast<std::size_t>(p-1))return g;
  }
  return 0;
}
std::size_t binary_products(std::uint64_t exponent) {
  if(!exponent)return 0;std::size_t ones=0,bits=0;while(exponent){ones+=exponent&1;++bits;exponent>>=1;}return ones+bits-1;
}
void verify(const P& raw,std::int64_t p) {
  const auto f=normalize(raw,p);const auto t=a::chien_search_trace(raw,p);const auto legacy=a::chien_search(raw,p);
  EXPECT_EQ(t.modulus,p);EXPECT_EQ(t.input,f);EXPECT_EQ(legacy,t.roots);EXPECT_EQ(a::chien_search_with_options(raw,p),t.roots);
  std::vector<std::int64_t> numeric_roots;
  if(!f.empty())for(std::int64_t x=0;x<p;++x)if(!horner(f,x,p))numeric_roots.push_back(x);
  auto sorted=t.roots;std::sort(sorted.begin(),sorted.end());EXPECT_EQ(sorted,numeric_roots);
  if(f.size()<3) {
    EXPECT_EQ(t.path,f.empty()?a::ChienPath::Zero:f.size()==1?a::ChienPath::Constant:a::ChienPath::Linear);
    EXPECT_FALSE(t.zero_checked);EXPECT_TRUE(t.steps.empty());EXPECT_TRUE(t.generator_tests.empty());EXPECT_TRUE(t.generator_factors.empty());
    EXPECT_TRUE(t.multipliers.empty());EXPECT_TRUE(t.final_terms.empty());EXPECT_EQ(t.final_point,0);EXPECT_EQ(t.generator,0);
    EXPECT_EQ(t.work.field_points,0u);EXPECT_EQ(t.work.generator_steps,0u);EXPECT_EQ(t.work.term_products,0u);
    EXPECT_EQ(t.work.linear_products,f.size()==2?binary_products(p-2)+1:0u);return;
  }
  EXPECT_EQ(t.path,a::ChienPath::Incremental);EXPECT_EQ(t.generator,generator(p));EXPECT_TRUE(t.zero_checked);EXPECT_EQ(t.zero_value,f[0]);
  EXPECT_EQ(t.work.field_points,static_cast<std::size_t>(p));ASSERT_EQ(t.steps.size(),static_cast<std::size_t>(p-1));
  EXPECT_EQ(t.work.term_products,f.size()*(p-1));EXPECT_EQ(t.work.sum_additions,f.size()*(p-1));EXPECT_EQ(t.work.point_products,static_cast<std::size_t>(p-1));
  EXPECT_EQ(t.work.linear_products,0u);EXPECT_EQ(t.final_point,1);EXPECT_EQ(t.final_terms,f);
  ASSERT_EQ(t.multipliers.size(),f.size());std::size_t setup=0;
  for(std::size_t j=0;j<f.size();++j){EXPECT_EQ(t.multipliers[j],power(t.generator,j,p));setup+=binary_products(j);}
  EXPECT_EQ(t.work.setup_products,setup);
  std::set<std::int64_t> visited;std::vector<std::int64_t> ordered_roots;if(!f[0])ordered_roots.push_back(0);
  for(std::size_t i=0;i<t.steps.size();++i) {
    const auto& s=t.steps[i];EXPECT_EQ(s.exponent,i);EXPECT_EQ(s.point,power(t.generator,i,p));EXPECT_TRUE(visited.insert(s.point).second);
    EXPECT_EQ(s.value,horner(f,s.point,p));EXPECT_EQ(s.root,s.value==0);ASSERT_EQ(s.terms.size(),f.size());
    Wide sum=0;
    for(std::size_t j=0;j<f.size();++j) {EXPECT_EQ(s.terms[j],static_cast<std::int64_t>(Wide(f[j])*power(s.point,j,p)%p));sum+=s.terms[j];}
    EXPECT_EQ(s.value,static_cast<std::int64_t>(sum%p));
    if(s.root)ordered_roots.push_back(s.point);
  }
  EXPECT_EQ(t.roots,ordered_roots);EXPECT_LE(t.roots.size(),f.size()-1);
  EXPECT_EQ(t.work.generator_tests,t.generator_tests.size());EXPECT_EQ(t.work.generator_steps,t.work.generator_divisors+t.work.generator_tests);
  std::size_t test=0,products=0;
  if(p>2)for(std::int64_t g=2;g<=t.generator;++g) {
    bool accepted=true;
    for(const auto q:t.generator_factors) {
      ASSERT_LT(test,t.generator_tests.size());const auto& check=t.generator_tests[test++];
      EXPECT_EQ(check.candidate,g);EXPECT_EQ(check.q,q);EXPECT_EQ((p-1)%q,0);
      const auto value=power(g,(p-1)/q,p);EXPECT_EQ(check.value,value);products+=binary_products((p-1)/q);
      if(value==1){accepted=false;break;}
    }
    EXPECT_EQ(accepted,g==t.generator);
  }
  EXPECT_EQ(test,t.generator_tests.size());EXPECT_EQ(t.work.generator_products,products);
}
}
TEST(ChienTrace, ExhaustiveSmallCoefficientVectorsIncludingTrailingZerosAndGF2) {
  std::size_t total=0;
  for(const std::int64_t p:{2,3,5,7})for(std::size_t length=0,choices=1;length<=5;++length,choices*=p)
    for(std::size_t code=0;code<choices;++code) {
      P f(length);auto digits=code;for(auto& c:f){c=digits%p;digits/=p;}verify(f,p);++total;
    }
  EXPECT_EQ(total,23941u);
}
TEST(ChienTrace, ThousandSignedRandomPolynomialsAgainstIndependentHornerAndPowerWalk) {
  std::mt19937_64 random(1964);const std::array<std::int64_t,7> primes{2,3,5,7,11,13,31};
  for(int trial=0;trial<1000;++trial) {
    const auto p=primes[random()%primes.size()];P f(random()%20);
    for(auto& c:f)c=static_cast<std::int64_t>(random()%200001)-100000;
    if(trial%5==0)f.insert(f.end(),3,0);verify(f,p);
  }
}
TEST(ChienTrace, SmallWorkedExampleHasUnsortedRootsAndActualRejectedGenerator) {
  const auto t=a::chien_search_trace({6,2,1},7);verify({6,2,1},7);
  EXPECT_EQ(t.generator,3);EXPECT_EQ(t.roots,(P{3,2}));EXPECT_EQ(t.multipliers,(P{1,3,2}));
  ASSERT_EQ(t.generator_tests.size(),3u);EXPECT_EQ(t.generator_tests[0].candidate,2);EXPECT_EQ(t.generator_tests[0].value,1);
  EXPECT_EQ(t.steps[0].terms,(P{6,2,1}));EXPECT_EQ(t.steps[1].terms,(P{6,6,2}));EXPECT_EQ(t.steps[2].terms,(P{6,4,4}));
  EXPECT_EQ(t.work.generator_divisors,1u);EXPECT_EQ(t.work.generator_steps,4u);EXPECT_EQ(t.work.generator_products,8u);
  EXPECT_EQ(t.work.term_products,18u);EXPECT_EQ(t.work.point_products,6u);EXPECT_EQ(t.work.setup_products,3u);EXPECT_EQ(t.work.sum_additions,18u);
}
TEST(ChienTrace, OriginalGF13ValuesDoNotCoverAllResiduesAndCountsIncludeClosingUpdates) {
  const P f{4,3,2,1},values{4,10,0,6,8,12,11,11,5,12,12,11,2};verify(f,13);
  const auto t=a::chien_search_trace(f,13);EXPECT_EQ(t.roots,(P{2}));EXPECT_EQ(t.generator,2);
  for(std::int64_t x=0;x<13;++x)EXPECT_EQ(horner(f,x,13),values[x]);
  const std::set<std::int64_t> distinct(values.begin(),values.end());EXPECT_EQ(distinct.size(),9u);
  EXPECT_EQ(t.work.field_points,13u);EXPECT_EQ(t.work.term_products,48u);EXPECT_EQ(t.work.point_products,12u);EXPECT_EQ(t.work.setup_products,6u);
  EXPECT_EQ(t.work.generator_divisors,1u);EXPECT_EQ(t.work.generator_steps,3u);EXPECT_EQ(t.work.generator_products,7u);
}
TEST(ChienTrace, ZeroConstantLinearAndZeroRootPathsRetainTheirSeparateContracts) {
  verify({},7);verify({0,7,-14,0},7);verify({5,0,0},7);verify({2,2,0},5);verify({0,4,0,1},5);
  const auto zero=a::chien_search_trace({},7);EXPECT_EQ(zero.path,a::ChienPath::Zero);EXPECT_TRUE(zero.roots.empty());
  const auto linear=a::chien_search_trace({0,1},7);EXPECT_EQ(linear.roots,(P{0}));EXPECT_FALSE(linear.zero_checked);EXPECT_TRUE(linear.steps.empty());
}
TEST(ChienTrace, NonzeroFormalPolynomialCanVanishOnEntireFieldAndRootsHaveNoMultiplicity) {
  for(const std::int64_t p:{2,3,5,7}) {
    P f(p+1,0);f[1]=-1;f[p]=1;verify(f,p);
    const auto t=a::chien_search_trace(f,p);EXPECT_FALSE(t.input.empty());EXPECT_EQ(t.path,a::ChienPath::Incremental);EXPECT_EQ(t.roots.size(),static_cast<std::size_t>(p));
    for(const auto& step:t.steps)EXPECT_EQ(step.value,0);
  }
  verify({4,-4,1},13);EXPECT_EQ(a::chien_search({4,-4,1},13),(P{2}));
}
TEST(ChienTrace, EveryNamedBudgetAcceptsEqualityAndRejectsOneBelowWithoutPartialOutput) {
  const P f{6,2,1};a::ChienOptions exact;exact.max_points=7;exact.max_term_products=18;exact.max_generator_steps=4;
  EXPECT_EQ(a::chien_search_with_options(f,7,exact),(P{3,2}));EXPECT_EQ(a::chien_search_trace(f,7,exact).roots,(P{3,2}));
  auto low=exact;low.max_points=6;
  EXPECT_THROW(a::chien_search_trace(f,7,low),std::length_error);
  EXPECT_THROW(a::chien_search_with_options(f,7,low),std::length_error);
  low=exact;low.max_term_products=17;
  EXPECT_THROW(a::chien_search_trace(f,7,low),std::length_error);
  EXPECT_THROW(a::chien_search_with_options(f,7,low),std::length_error);
  low=exact;low.max_generator_steps=3;
  EXPECT_THROW(a::chien_search_trace(f,7,low),std::length_error);
  EXPECT_THROW(a::chien_search_with_options(f,7,low),std::length_error);
  EXPECT_EQ(f,(P{6,2,1}));
}
TEST(ChienTrace, HugeSweepsRejectBeforeGeneratorWorkOrGiantTraceReservation) {
  const auto p=9223372036854775783LL;a::ChienOptions points;points.max_points=13;
  EXPECT_THROW(a::chien_search_trace({1,0,1},p,points),std::length_error);
  a::ChienOptions terms;terms.max_term_products=1;
  EXPECT_THROW(a::chien_search_trace({1,0,1},p,terms),std::length_error);
  EXPECT_THROW(a::chien_search({1,0,1},p),std::length_error);
  // The generator cap must reject before reserving nearly a billion snapshots.
  a::ChienOptions generator_limit;generator_limit.max_generator_steps=1;
  EXPECT_THROW(a::chien_search_trace({1,0,1},1000000007,generator_limit),std::length_error);
  EXPECT_THROW(a::chien_search_with_options({1,0,1},1000000007,generator_limit),std::length_error);
}
TEST(ChienTrace, ShortcutsDoNotConsumeSweepBudgetsEvenAtNearLimitPrime) {
  const auto p=9223372036854775783LL;a::ChienOptions options;options.max_points=options.max_term_products=options.max_generator_steps=1;
  const auto t=a::chien_search_trace({-1,1},p,options);
  EXPECT_EQ(t.path,a::ChienPath::Linear);EXPECT_EQ(t.roots,(P{1}));EXPECT_EQ(t.work.field_points,0u);EXPECT_EQ(t.work.generator_steps,0u);
  EXPECT_EQ(t.work.linear_products,binary_products(p-2)+1);
  EXPECT_TRUE(a::chien_search_with_options({},p,options).empty());EXPECT_TRUE(a::chien_search_with_options({3},p,options).empty());
}
TEST(ChienTrace, ThousandNearLimitSignedLinearInputsSatisfyTheirUniqueRootEquation) {
  const auto p=9223372036854775783LL;std::mt19937_64 random(64);
  for(int trial=0;trial<1000;++trial) {
    P f{static_cast<std::int64_t>(random()>>1),static_cast<std::int64_t>(1+random()%(p-1))};
    if(trial%2)f[0]=-f[0];if(trial%3)f[1]=-f[1];
    const auto t=a::chien_search_trace(f,p);ASSERT_EQ(t.roots.size(),1u);EXPECT_EQ(horner(f,t.roots[0],p),0);EXPECT_GE(t.roots[0],0);EXPECT_LT(t.roots[0],p);
  }
  const P endpoints{std::numeric_limits<std::int64_t>::min(),std::numeric_limits<std::int64_t>::max()};
  const auto t=a::chien_search_trace(endpoints,p);ASSERT_EQ(t.roots.size(),1u);EXPECT_EQ(horner(endpoints,t.roots[0],p),0);
}
TEST(ChienTrace, InvalidFieldsRejectBeforeZeroConstantLinearOrLimitShortcuts) {
  for(const std::int64_t p:{-7,0,1,4,9,15,256})for(const P f:{P{},P{1},P{0,1},P{1,0,1}}) {
    EXPECT_THROW(a::chien_search(f,p),std::invalid_argument);
    EXPECT_THROW(a::chien_search_with_options(f,p),std::invalid_argument);
    EXPECT_THROW(a::chien_search_trace(f,p),std::invalid_argument);
  }
}
TEST(ChienTrace, OwnedRecordsAndInputsRemainIndependentAcrossCopiesAndCalls) {
  P f{6,2,1};const auto t=a::chien_search_trace(f,7);auto copy=t;f[0]=0;
  copy.input[0]=0;copy.steps[0].terms[0]=0;copy.generator_tests[0].candidate=99;
  EXPECT_EQ(t.input,(P{6,2,1}));EXPECT_EQ(t.steps[0].terms,(P{6,2,1}));EXPECT_EQ(t.generator_tests[0].candidate,2);
  EXPECT_EQ(a::chien_search_trace({6,2,1},7).roots,(P{3,2}));
}
