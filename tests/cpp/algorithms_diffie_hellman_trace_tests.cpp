#include <datamunge/algorithms/diffie_hellman.hpp>
#include <datamunge/algorithms/miller_rabin.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
using namespace datamunge::algorithms;
namespace {
std::uint64_t naive(std::uint64_t g,std::uint64_t e,std::uint64_t m) {
 std::uint64_t r=1%m;for(std::uint64_t k=0;k<e;++k)r=r*(g%m)%m;return r;
}
std::uint64_t add(std::uint64_t a,std::uint64_t b,std::uint64_t m){return a>=m-b?a-(m-b):a+b;}
std::uint64_t multiply_by_additions(std::uint64_t a,std::uint64_t b,std::uint64_t m) {
 std::uint64_t r=0;a%=m;while(b){if(b&1)r=add(r,a,m);a=add(a,a,m);b>>=1;}return r;
}
std::uint64_t left_to_right(std::uint64_t a,std::uint64_t e,std::uint64_t m) {
 std::uint64_t r=1%m;for(int bit=63;bit>=0;--bit){r=multiply_by_additions(r,r,m);if((e>>bit)&1)r=multiply_by_additions(r,a,m);}return r;
}
bool small_prime(std::uint64_t n){if(n<2)return false;for(std::uint64_t d=2;d*d<=n;++d)if(n%d==0)return false;return true;}
}
TEST(DiffieHellmanTrace, OriginalBothExchanges) {
 const auto small=dh_exchange(5,6,15,23);EXPECT_EQ(small.alice_public,8);EXPECT_EQ(small.bob_public,19);EXPECT_EQ(small.alice_shared,2);EXPECT_EQ(small.bob_shared,2);
 const auto large=dh_exchange(7,123456,987654,2147483647);EXPECT_EQ(large.alice_public,744111946);EXPECT_EQ(large.bob_public,1598667707);EXPECT_EQ(large.alice_shared,490982923);EXPECT_EQ(large.bob_shared,490982923);
 EXPECT_EQ(dh_public_key(5,6,23),8);EXPECT_EQ(dh_shared_secret(19,6,23),2);EXPECT_EQ(small.generator,5);EXPECT_EQ(small.modulus,23);
}
TEST(DiffieHellmanTrace, ExhaustiveSmallArithmeticExchangesIncludingCompositeModuli) {
 std::size_t cases=0;
 for(std::uint64_t p=1;p<=40;++p)for(std::uint64_t g=0;g<=p+1;++g)for(std::uint64_t a=0;a<=8;++a)for(std::uint64_t b=0;b<=8;++b) {
  const auto t=dh_exchange(g,a,b,p);const auto A=naive(g,a,p),B=naive(g,b,p);
  EXPECT_EQ(t.alice_public,A);EXPECT_EQ(t.bob_public,B);EXPECT_EQ(t.alice_shared,naive(B,a,p));EXPECT_EQ(t.bob_shared,naive(A,b,p));EXPECT_EQ(t.alice_shared,t.bob_shared);++cases;
 }
 EXPECT_EQ(cases,72900);
}
TEST(DiffieHellmanTrace, ActualBitStatesAndInvariantFor103275SmallPowers) {
 std::size_t cases=0;
 for(std::uint64_t m=1;m<=25;++m)for(std::uint64_t g=0;g<=50;++g)for(std::uint64_t e=0;e<=80;++e) {
  const auto t=mod_pow_trace(g,e,m);const std::size_t width=m==1?0:std::bit_width(e),ones=m==1?0:std::popcount(e);
  EXPECT_EQ(t.result,naive(g,e,m));EXPECT_EQ(t.result,mod_pow(g,e,m));EXPECT_EQ(t.initial_base,g%m);EXPECT_EQ(t.steps.size(),width);
  EXPECT_EQ(t.work.bits,width);EXPECT_EQ(t.work.squarings,width);EXPECT_EQ(t.work.accumulator_products,ones);EXPECT_EQ(t.work.products,width+ones);
  std::uint64_t accumulator=1%m,running=g%m,remaining=e;
  for(std::size_t i=0;i<t.steps.size();++i) {
   const auto& s=t.steps[i];EXPECT_EQ(s.bit_index,i);EXPECT_EQ(s.remaining_before,remaining);EXPECT_EQ(s.running_before,running);EXPECT_EQ(s.accumulator_before,accumulator);EXPECT_EQ(s.multiply,(remaining&1)!=0);
   EXPECT_EQ((accumulator*naive(running,remaining,m))%m,t.result);
   if(remaining&1)accumulator=accumulator*running%m;
   running=running*running%m;remaining>>=1;
   EXPECT_EQ(s.accumulator_after,accumulator);EXPECT_EQ(s.squared_base,running);
  }
  ++cases;
 }
 EXPECT_EQ(cases,103275);
 const auto t=mod_pow_trace(5,6,23);ASSERT_EQ(t.steps.size(),3);EXPECT_FALSE(t.steps[0].multiply);EXPECT_EQ(t.steps[0].squared_base,2);EXPECT_EQ(t.steps[1].accumulator_after,2);EXPECT_EQ(t.steps[2].accumulator_after,8);EXPECT_EQ(t.steps[2].squared_base,16);
}
TEST(DiffieHellmanTrace, FullWidthArithmeticAgainstAdditionOnlyOracle) {
 std::mt19937_64 rng(19761976);
 for(int k=0;k<5000;++k){const auto a=rng(),b=rng(),m=rng()|1;EXPECT_EQ(mul_mod(a,b,m),multiply_by_additions(a,b,m));}
 for(int k=0;k<500;++k) {
  const auto a=rng(),e=rng(),m=rng()|1;const auto t=mod_pow_trace(a,e,m);
  EXPECT_EQ(t.result,left_to_right(a,e,m));EXPECT_EQ(t.result,mod_pow(a,e,m));
  for(const auto& s:t.steps){EXPECT_EQ(s.squared_base,multiply_by_additions(s.running_before,s.running_before,m));if(s.multiply)EXPECT_EQ(s.accumulator_after,multiply_by_additions(s.accumulator_before,s.running_before,m));}
 }
 const auto max=std::numeric_limits<std::uint64_t>::max();EXPECT_EQ(mul_mod(max-1,max-2,max),2);EXPECT_EQ(((max-1)*(max-2))%max,6);
 EXPECT_EQ(mul_mod(max,max,max),0);EXPECT_EQ(mul_mod(max,max,1),0);
}
TEST(DiffieHellmanTrace, AllSmallPrimeElementCyclesAndFullWidthOrderTwo) {
 for(std::uint64_t p=2;p<=127;++p)if(small_prime(p))for(std::uint64_t g=1;g<p;++g) {
  const auto t=dh_power_cycle(g,p);std::uint64_t value=1,order=0;
  do{EXPECT_EQ(t.values.at(order),value);value=value*g%p;++order;}while(value!=1);
  EXPECT_EQ(t.order,order);EXPECT_EQ(t.values.size(),order);EXPECT_EQ((p-1)%order,0);EXPECT_EQ(std::set<std::uint64_t>(t.values.begin(),t.values.end()).size(),order);
  for(std::uint64_t x=0;x<order;++x){EXPECT_EQ(t.values[x],naive(g,x,p));EXPECT_EQ(t.values[x],naive(g,x+order,p));}
 }
 const std::uint64_t p=18446744073709551557ULL;ASSERT_TRUE(is_probable_prime(p));const auto full=dh_power_cycle(p-1,p,2);
 EXPECT_EQ(full.order,2);EXPECT_EQ(full.values,(std::vector<std::uint64_t>{1,p-1}));EXPECT_EQ(dh_power_cycle(1,p,1).order,1);
}
TEST(DiffieHellmanTrace, CyclesExposePeriodAndTinyTableInversion) {
 EXPECT_EQ(dh_power_cycle(1,23).order,1);EXPECT_EQ(dh_power_cycle(22,23).order,2);EXPECT_EQ(dh_power_cycle(4,23).order,11);
 const auto c=dh_power_cycle(5,23);ASSERT_EQ(c.order,22);EXPECT_EQ(c.values[6],8);EXPECT_EQ(c.values[15],19);EXPECT_EQ(c.values.back(),14);
 EXPECT_EQ(std::find(c.values.begin(),c.values.end(),8)-c.values.begin(),6);EXPECT_EQ(dh_public_key(5,28,23),8);
 const auto bad=dh_exchange(22,6,15,23);EXPECT_EQ(bad.alice_public,1);EXPECT_EQ(bad.bob_public,22);EXPECT_EQ(bad.alice_shared,1);
 const auto identity=dh_exchange(1,6,15,23);EXPECT_EQ(identity.alice_shared,1);
 EXPECT_EQ(dh_shared_secret(0,6,23),0);EXPECT_EQ(dh_shared_secret(1,15,23),1);EXPECT_EQ(dh_shared_secret(22,6,23),1);EXPECT_EQ(dh_shared_secret(22,15,23),22);
}
TEST(DiffieHellmanTrace, UnauthenticatedTwoSessionSubstitution) {
 const auto left=dh_exchange(5,6,13,23),right=dh_exchange(5,7,15,23);
 EXPECT_EQ(left.alice_public,8);EXPECT_EQ(right.bob_public,19);EXPECT_EQ(left.bob_public,21);EXPECT_EQ(right.alice_public,17);
 EXPECT_EQ(left.alice_shared,18);EXPECT_EQ(left.bob_shared,18);EXPECT_EQ(right.alice_shared,15);EXPECT_EQ(right.bob_shared,15);EXPECT_NE(left.alice_shared,right.bob_shared);
 EXPECT_EQ(dh_shared_secret(left.alice_public,13,23),left.alice_shared);EXPECT_EQ(dh_shared_secret(right.bob_public,7,23),right.bob_shared);
}
TEST(DiffieHellmanTrace, ExactCapsValidationAndZeroWorkEdges) {
 const auto max=std::numeric_limits<std::uint64_t>::max();
 EXPECT_THROW(mul_mod(0,0,0),std::invalid_argument);
 EXPECT_THROW(dh_public_key(5,6,0),std::invalid_argument);
 EXPECT_THROW(dh_shared_secret(19,6,0),std::invalid_argument);
 EXPECT_THROW(dh_exchange(5,6,15,0),std::invalid_argument);
 EXPECT_THROW(mod_pow_trace(0,max,0,1),std::invalid_argument);
 EXPECT_THROW(mod_pow_trace(5,6,23,2),std::length_error);
 EXPECT_NO_THROW(mod_pow_trace(5,6,23,3));
 EXPECT_NO_THROW(mod_pow_trace(5,6,23,0));
 EXPECT_THROW(mod_pow_trace(5,max,23,63),std::length_error);
 EXPECT_EQ(mod_pow_trace(5,max,23,64).work.products,128);
 EXPECT_EQ(mod_pow_trace(5,max,1,1).work.products,0);EXPECT_EQ(mod_pow_trace(0,0,1).result,0);EXPECT_EQ(mod_pow_trace(0,0,23).result,1);
 for(auto p:{0ULL,1ULL,4ULL,15ULL,341ULL,561ULL,1105ULL,2047ULL,3825123056546413051ULL}) {EXPECT_THROW(dh_power_cycle(1,p,1),std::invalid_argument);}
 EXPECT_THROW(dh_power_cycle(0,23,1),std::invalid_argument);
 EXPECT_THROW(dh_power_cycle(23,23,1),std::invalid_argument);
 EXPECT_THROW(dh_power_cycle(5,23,21),std::length_error);
 EXPECT_NO_THROW(dh_power_cycle(5,23,22));
 EXPECT_NO_THROW(dh_power_cycle(5,23,0));
 EXPECT_EQ(dh_power_cycle(1,2,1).order,1);
 const auto one=dh_exchange(max,max,max,1);EXPECT_EQ(one.alice_public,0);EXPECT_EQ(one.bob_shared,0);
}
TEST(DiffieHellmanTrace, CountsAtEveryWidthAndOwnedRecords) {
 for(unsigned k=1;k<=64;++k) {
  const auto sparse=std::uint64_t{1}<<(k-1),dense=std::numeric_limits<std::uint64_t>::max()>>(64-k);
  const auto a=mod_pow_trace(7,sparse,23),b=mod_pow_trace(7,dense,23);EXPECT_EQ(a.work.squarings,k);EXPECT_EQ(b.work.squarings,k);EXPECT_EQ(a.work.products,k+1);EXPECT_EQ(b.work.products,2*k);
 }
 auto t=mod_pow_trace(5,6,23);auto copy=t;t.steps[1].accumulator_after=42;t.result=0;
 EXPECT_EQ(copy.steps[1].accumulator_after,2);EXPECT_EQ(copy.result,8);
 auto c=dh_power_cycle(5,23);auto d=c;c.values[0]=0;EXPECT_EQ(d.values[0],1);EXPECT_EQ(d.order,22);
}
