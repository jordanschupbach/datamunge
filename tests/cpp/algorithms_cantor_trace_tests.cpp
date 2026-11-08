#include <gtest/gtest.h>
#include <datamunge/algorithms/polynomial_gf.hpp>
#include <algorithm>
#include <limits>
#include <random>
namespace {
namespace a=datamunge::algorithms;
using P=a::GFPoly;using U=unsigned __int128;
// Separate exact arithmetic oracle; no production polynomial helpers are used here.
std::int64_t norm(std::int64_t x,std::int64_t p) {auto r=static_cast<__int128>(x)%p;return static_cast<std::int64_t>(r<0?r+p:r);}
void trim(P& f) {while(!f.empty() && !f.back())f.pop_back();}
P normalize(P f,std::int64_t p) {for(auto& c:f)c=norm(c,p);trim(f);return f;}
std::int64_t scalar(std::int64_t x,std::uint64_t e,std::int64_t p) {
  std::int64_t r=1;while(e) {if(e&1)r=static_cast<std::int64_t>(static_cast<U>(r)*x%p);x=static_cast<std::int64_t>(static_cast<U>(x)*x%p);e>>=1;}return r;
}
P multiply(const P& x,const P& y,std::int64_t p) {
  if(x.empty() || y.empty())return {};P r(x.size()+y.size()-1);
  for(std::size_t i=0;i<x.size();++i)for(std::size_t j=0;j<y.size();++j)r[i+j]=static_cast<std::int64_t>((static_cast<U>(x[i])*y[j]+r[i+j])%p);
  trim(r);return r;
}
P remainder(P f,const P& divisor,std::int64_t p) {
  const auto inverse=scalar(divisor.back(),p-2,p);
  while(!f.empty() && f.size()>=divisor.size()) {
    const auto shift=f.size()-divisor.size(),coefficient=static_cast<std::uint64_t>(static_cast<U>(f.back())*inverse%p);
    for(std::size_t j=0;j<divisor.size();++j)f[j+shift]=norm(f[j+shift]-static_cast<std::int64_t>(static_cast<U>(coefficient)*divisor[j]%p),p);
    trim(f);
  }
  return f;
}
P monic(P f,std::int64_t p) {if(!f.empty()){const auto inverse=scalar(f.back(),p-2,p);for(auto& c:f)c=static_cast<std::int64_t>(static_cast<U>(c)*inverse%p);}return f;}
P gcd(P f,P g,std::int64_t p) {while(!g.empty()){auto r=remainder(f,g,p);f=std::move(g);g=std::move(r);}return monic(std::move(f),p);}
P sub(P f,const P& g,std::int64_t p) {f.resize(std::max(f.size(),g.size()));for(std::size_t i=0;i<g.size();++i)f[i]=norm(f[i]-g[i],p);trim(f);return f;}
P power(P base,U exponent,const P& modulus,std::int64_t p) {
  P r=remainder({1},modulus,p);base=remainder(base,modulus,p);
  while(exponent) {if(exponent&1)r=remainder(multiply(r,base,p),modulus,p);base=remainder(multiply(base,base,p),modulus,p);exponent>>=1;}return r;
}
P monic_from_code(std::size_t code,std::size_t degree,std::int64_t p) {
  P f(degree+1);for(std::size_t i=0;i<degree;++i){f[i]=code%p;code/=p;}f.back()=1;return f;
}
std::vector<P> trial_factorization(P f,std::int64_t p) {
  // Ascending-degree monic trial division; all smaller factors are already removed.
  std::vector<P> factors;
  for(std::size_t d=1,choices=p;f.size()>1 && d<=(f.size()-1)/2;++d,choices*=p)
    for(std::size_t code=0;code<choices;++code) {
      const auto divisor=monic_from_code(code,d,p);
      while(f.size()>=divisor.size() && remainder(f,divisor,p).empty()) {
        P quotient(f.size()-divisor.size()+1),remaining=f;
        while(!remaining.empty() && remaining.size()>=divisor.size()) {
          const auto shift=remaining.size()-divisor.size();const auto coefficient=remaining.back();quotient[shift]=coefficient;
          for(std::size_t j=0;j<divisor.size();++j)remaining[j+shift]=norm(remaining[j+shift]-static_cast<std::int64_t>(static_cast<U>(coefficient)*divisor[j]%p),p);
          trim(remaining);
        }
        trim(quotient);f=std::move(quotient);factors.push_back(divisor);
      }
    }
  if(f.size()>1)factors.push_back(f);std::sort(factors.begin(),factors.end());return factors;
}
std::size_t binary_products(std::uint64_t exponent) {
  if(!exponent)return 0;std::size_t bits=0,ones=0;while(exponent){++bits;ones+=exponent&1;exponent>>=1;}return ones+bits-1;
}
void verify_trace(const a::CantorZassenhausTrace& t) {
  const auto p=t.modulus;P product{1};for(const auto& factor:t.factors)product=multiply(product,factor,p);
  EXPECT_EQ(product,t.monic_input);EXPECT_EQ(multiply(product,{t.leading_coefficient},p),t.input);EXPECT_EQ(t.squarefree_gcd,P{1});
  P remaining=t.monic_input;U p_power=1;
  for(std::size_t i=0;i<t.distinct_degree.size();++i) {
    const auto& step=t.distinct_degree[i];EXPECT_EQ(step.degree,i+1);EXPECT_EQ(step.remaining_before,remaining);p_power*=p;
    EXPECT_EQ(step.frobenius,power({0,1},p_power,remaining,p));EXPECT_EQ(step.bucket,gcd(sub(step.frobenius,{0,1},p),remaining,p));
    EXPECT_EQ(multiply(step.bucket,step.remaining_after,p),remaining);remaining=step.remaining_after;
  }
  EXPECT_EQ(t.residual,remaining);EXPECT_EQ(t.work.distinct_degree_steps,t.distinct_degree.size());EXPECT_EQ(t.work.random_trials,t.equal_degree.size());
  std::size_t words=0,gcds=1+t.distinct_degree.size(),powers=t.distinct_degree.size(),products=powers*binary_products(p);
  for(std::size_t i=0;i<t.equal_degree.size();++i) {
    const auto& trial=t.equal_degree[i];EXPECT_EQ(trial.attempt,i);words+=trial.target.size()-1;
    if(trial.candidate.size()<2) {EXPECT_TRUE(trial.initial_gcd.empty());EXPECT_FALSE(trial.used_character);EXPECT_FALSE(trial.accepted);continue;}
    ++gcds;EXPECT_EQ(trial.initial_gcd,gcd(trial.candidate,trial.target,p));P expected=trial.initial_gcd;
    if(trial.used_character) {
      U q=1;for(std::size_t d=0;d<trial.degree;++d)q*=p;
      EXPECT_EQ(trial.character,power(trial.candidate,(q-1)/2,trial.target,p));
      expected=gcd(sub(trial.character,{1},p),trial.target,p);++gcds;powers+=trial.degree;
      products+=binary_products((p-1)/2)+(trial.degree-1)*(binary_products(p)+1);
    }
    EXPECT_EQ(trial.split_gcd,expected);const bool proper=expected.size()>1 && expected.size()<trial.target.size();EXPECT_EQ(trial.accepted,proper);
    if(proper)EXPECT_EQ(multiply(trial.split_gcd,trial.quotient,p),trial.target);
    else EXPECT_TRUE(trial.quotient.empty());
  }
  EXPECT_EQ(t.work.random_words,words);EXPECT_EQ(t.work.gcd_calls,gcds);EXPECT_EQ(t.work.modular_power_calls,powers);EXPECT_EQ(t.work.modular_products,products);
}
}
TEST(CantorTrace, OriginalGF5HasOneDdfStepAndResidualQuadratic) {
  const auto t=a::cantor_zassenhaus_trace({4,4,4,2,1},5);ASSERT_EQ(t.distinct_degree.size(),1u);
  EXPECT_EQ(t.distinct_degree[0].bucket,(P{2,2,1}));EXPECT_EQ(t.residual,(P{2,0,1}));EXPECT_EQ(t.factors.size(),3u);verify_trace(t);
  EXPECT_EQ(a::cantor_zassenhaus(t.input,5),t.factors);
}
TEST(CantorTrace, ExhaustiveMonicPolynomialsWithTrialDivisionOracle) {
  std::size_t total=0,squarefree=0,repeated=0;
  for(auto [p,maximum]:std::vector<std::pair<std::int64_t,std::size_t>>{{3,6},{5,5},{7,4}})
    for(std::size_t degree=0,choices=1;degree<=maximum;++degree,choices*=p)for(std::size_t code=0;code<choices;++code) {
      const auto f=monic_from_code(code,degree,p);const auto expected=trial_factorization(f,p);
      const bool is_squarefree=std::adjacent_find(expected.begin(),expected.end())==expected.end();
      EXPECT_EQ(a::poly_is_squarefree_gf(f,p),is_squarefree);EXPECT_EQ(a::poly_is_irreducible_gf(f,p),degree>0 && expected.size()==1 && expected[0]==f);
      if(is_squarefree) {
        a::CantorZassenhausOptions options;options.max_random_trials=1000;
        const auto trace=a::cantor_zassenhaus_trace(f,p,options);auto factors=trace.factors;std::sort(factors.begin(),factors.end());EXPECT_EQ(factors,expected);verify_trace(trace);++squarefree;
      } else {EXPECT_THROW(a::cantor_zassenhaus(f,p),std::invalid_argument);++repeated;}
      ++total;
    }
  EXPECT_EQ(total,7800u);EXPECT_EQ(squarefree+repeated,total);
  std::cout<<"Exhaustive monic inputs: "<<total<<", squarefree "<<squarefree<<", repeated "<<repeated<<"\n";
}
TEST(CantorTrace, SeededMixedDegreeFactorsAgainstIndependentOracle) {
  std::mt19937_64 rng(1981);
  for(int trial=0;trial<1000;++trial) {
    const std::int64_t p=std::vector<std::int64_t>{3,5,7}[rng()%3];const auto degree=1+rng()%6;
    P f(degree+1);for(auto& c:f)c=rng()%p;f.back()=1;
    const auto expected=trial_factorization(f,p);
    if(std::adjacent_find(expected.begin(),expected.end())!=expected.end())continue;
    a::CantorZassenhausOptions options;options.seed=rng();options.max_random_trials=1000;
    const auto trace=a::cantor_zassenhaus_trace(f,p,options);auto factors=trace.factors;std::sort(factors.begin(),factors.end());EXPECT_EQ(factors,expected);verify_trace(trace);
    EXPECT_EQ(a::cantor_zassenhaus_with_options(f,p,options),trace.factors);
  }
}
TEST(CantorTrace, NearInt64PrimeAndExponentBeyondUint64) {
  const std::int64_t p=9223372036854775783LL;
  EXPECT_EQ(a::poly_multiply_gf({p-1,p-1},{1,1},p),(P{p-1,p-2,p-1}));
  EXPECT_EQ(a::poly_normalize_gf({std::numeric_limits<std::int64_t>::min(),p-1,0},p),(P{p-25,p-1}));
  EXPECT_EQ(scalar(p-1,(p-1)/2,p),p-1);EXPECT_EQ(scalar(p-2,(p-1)/2,p),p-1);
  const P first{1,0,1},second{2,0,1};EXPECT_TRUE(a::poly_is_irreducible_gf(first,p));EXPECT_TRUE(a::poly_is_irreducible_gf(second,p));
  const auto f=multiply(first,second,p);a::CantorZassenhausOptions options;options.max_random_trials=100;
  const auto trace=a::cantor_zassenhaus_trace(f,p,options);ASSERT_EQ(trace.factors.size(),2u);verify_trace(trace);
  EXPECT_EQ(trace.factors[0].size(),3u);EXPECT_EQ(trace.factors[1].size(),3u);EXPECT_EQ(trace.distinct_degree.size(),2u);
  EXPECT_GT(static_cast<U>(p)*p,std::numeric_limits<std::uint64_t>::max());
}
TEST(CantorTrace, LargePrimeRandomArithmeticMatchesWidenedOracle) {
  const std::int64_t p=9223372036854775783LL;std::mt19937_64 rng(9);
  for(int trial=0;trial<300;++trial) {
    P left(1+rng()%8),right(1+rng()%5);for(auto& c:left)c=static_cast<std::int64_t>(rng()>>1);for(auto& c:right)c=static_cast<std::int64_t>(rng()>>1);
    const auto l=normalize(left,p),r=normalize(right,p);if(r.empty())continue;
    EXPECT_EQ(a::poly_multiply_gf(left,right,p),multiply(l,r,p));const auto qr=a::poly_divmod_gf(left,right,p);
    EXPECT_EQ(qr.remainder,remainder(l,r,p));P recon=multiply(qr.quotient,r,p);recon.resize(std::max(recon.size(),qr.remainder.size()));
    for(std::size_t i=0;i<qr.remainder.size();++i)recon[i]=static_cast<std::int64_t>((static_cast<U>(recon[i])+qr.remainder[i])%p);trim(recon);EXPECT_EQ(recon,l);
    const auto x=static_cast<std::int64_t>(rng()>>1)%p;std::int64_t value=0;for(std::size_t i=l.size();i-- >0;)value=static_cast<std::int64_t>((static_cast<U>(value)*x+l[i])%p);
    EXPECT_EQ(a::poly_evaluate_gf(left,x,p),value);
  }
}
TEST(CantorTrace, CompleteCallBudgetsMatchExactActualCounts) {
  P f{1};for(std::int64_t root=0;root<6;++root)f=multiply(f,{norm(-root,7),1},7);
  const auto full=a::cantor_zassenhaus_trace(f,7);ASSERT_GE(full.work.random_trials,5u);
  for(std::size_t cap=1;cap<full.work.random_trials;++cap) {a::CantorZassenhausOptions o;o.max_random_trials=cap;EXPECT_THROW(a::cantor_zassenhaus_trace(f,7,o),std::length_error);}
  for(std::size_t cap=1;cap<full.work.modular_products;++cap) {a::CantorZassenhausOptions o;o.max_modular_products=cap;EXPECT_THROW(a::cantor_zassenhaus_with_options(f,7,o),std::length_error);}
  a::CantorZassenhausOptions exact;exact.max_random_trials=full.work.random_trials;exact.max_modular_products=full.work.modular_products;
  EXPECT_EQ(a::cantor_zassenhaus_with_options(f,7,exact),full.factors);EXPECT_EQ(a::cantor_zassenhaus_trace(f,7,exact).factors,full.factors);
}
TEST(CantorTrace, InvalidFieldsDivisorsAndRepeatedFactorsThrow) {
  for(std::int64_t p:{-7,0,1,4,9,561}) {
    EXPECT_THROW(a::poly_normalize_gf({},p),std::invalid_argument);
    EXPECT_THROW(a::poly_divmod_gf({1},{1},p),std::invalid_argument);
    EXPECT_THROW(a::cantor_zassenhaus({1},p),std::invalid_argument);
    EXPECT_THROW(a::chien_search({},p),std::invalid_argument);
  }
  EXPECT_THROW(a::poly_divmod_gf({1},{},5),std::invalid_argument);
  EXPECT_THROW(a::poly_divmod_gf({1},{0,5},5),std::invalid_argument);
  EXPECT_THROW(a::cantor_zassenhaus({},5),std::invalid_argument);
  EXPECT_THROW(a::cantor_zassenhaus({5,0},5),std::invalid_argument);
  EXPECT_THROW(a::cantor_zassenhaus({1,2,1},5),std::invalid_argument);
  EXPECT_THROW(a::cantor_zassenhaus({1,0,0,1},3),std::invalid_argument);
  EXPECT_THROW(a::cantor_zassenhaus({1,1},2),std::invalid_argument);
}
TEST(CantorTrace, NormalizationUnitsZeroAndNonmonicReconstruction) {
  EXPECT_TRUE(a::cantor_zassenhaus({3,0},5).empty());const auto unit=a::cantor_zassenhaus_trace({3},5);EXPECT_EQ(unit.leading_coefficient,3);EXPECT_EQ(unit.monic_input,P{1});EXPECT_EQ(unit.residual,P{1});
  EXPECT_FALSE(a::poly_is_squarefree_gf({},5));EXPECT_TRUE(a::poly_is_squarefree_gf({3},5));EXPECT_FALSE(a::poly_is_irreducible_gf({3},5));
  const P raw{-2,0,2,0};const auto trace=a::cantor_zassenhaus_trace(raw,5);EXPECT_EQ(trace.input,(P{3,0,2}));EXPECT_EQ(trace.monic_input,(P{4,0,1}));verify_trace(trace);
  EXPECT_EQ(raw,(P{-2,0,2,0}));EXPECT_TRUE(a::poly_gcd_gf({}, {},5).empty());EXPECT_EQ(a::poly_gcd_gf({2},{4},5),P{1});
  EXPECT_EQ(a::poly_divmod_gf({1,2},{2},5).quotient,(P{3,1}));EXPECT_TRUE(a::poly_derivative_gf({1,0,0,1},3).empty());
  EXPECT_EQ(a::poly_string_gf({},5),"0");EXPECT_EQ(a::poly_string_gf({4,4,4,2,1},5),"x^4 + 2x^3 + 4x^2 + 4x + 4");
}
TEST(CantorTrace, GF2ArithmeticAndIrreducibilityRemainSupported) {
  EXPECT_EQ(a::poly_multiply_gf({1,1},{1,1},2),(P{1,0,1}));EXPECT_TRUE(a::poly_is_irreducible_gf({1,1,1},2));EXPECT_FALSE(a::poly_is_irreducible_gf({1,0,1},2));
  EXPECT_TRUE(a::poly_is_squarefree_gf({1,1,1},2));EXPECT_FALSE(a::poly_is_squarefree_gf({1,0,1},2));
}
TEST(CantorTrace, ChienKeepsZeroConventionAndCheckedOrder) {
  EXPECT_TRUE(a::chien_search({},5).empty());EXPECT_TRUE(a::chien_search({1},5).empty());
  EXPECT_EQ(a::chien_search({0,4,0,1},5),(std::vector<std::int64_t>{0,1,4}));
  EXPECT_EQ(a::chien_search({2,2},5),(std::vector<std::int64_t>{4}));EXPECT_EQ(a::chien_search({1,1},2),(std::vector<std::int64_t>{1}));
  EXPECT_EQ(a::chien_search({-1,1},9223372036854775783LL),(std::vector<std::int64_t>{1}));
}
TEST(CantorTrace, SeedReplayZeroMappingAndOwnedRecords) {
  const P f{4,4,4,2,1};a::CantorZassenhausOptions zero;zero.seed=0;auto fallback=zero;fallback.seed=0x1234567;
  const auto first=a::cantor_zassenhaus_trace(f,5,zero),second=a::cantor_zassenhaus_trace(f,5,fallback);EXPECT_EQ(first.factors,second.factors);ASSERT_EQ(first.equal_degree.size(),second.equal_degree.size());
  for(std::size_t i=0;i<first.equal_degree.size();++i)EXPECT_EQ(first.equal_degree[i].candidate,second.equal_degree[i].candidate);
  auto copy=first;copy.input.clear();copy.factors[0].clear();copy.distinct_degree[0].bucket.clear();EXPECT_FALSE(first.input.empty());EXPECT_FALSE(first.factors[0].empty());EXPECT_FALSE(first.distinct_degree[0].bucket.empty());
}
