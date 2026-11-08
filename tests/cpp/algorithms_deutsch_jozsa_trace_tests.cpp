#include <datamunge/algorithms/deutsch_jozsa.hpp>
#include <datamunge/algorithms/fourier_sampling.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <limits>
#include <numeric>
#include <stdexcept>
using namespace datamunge::algorithms;

TEST(DeutschJozsaTrace, ExhaustiveTablesAgainstDirectCharacterSums) {
  std::size_t tables=0,promised=0;
  for(int n=0;n<=4;++n) {
    const std::size_t N=std::size_t{1}<<n;
    for(std::size_t table=0;table<(std::size_t{1}<<N);++table) {
      std::vector<int> f(N);std::size_t calls=0,ones=0;
      for(std::size_t x=0;x<N;++x)ones+=(f[x]=(table>>x)&1);
      const auto t=deutsch_jozsa_trace(n,[&](std::size_t x){EXPECT_EQ(x,calls++);return f[x];});
      EXPECT_EQ(calls,N);EXPECT_EQ(t.truth_table,f);EXPECT_EQ(t.ones,ones);
      const bool constant=ones==0||ones==N,balanced=N>1&&ones==N/2;
      EXPECT_EQ(t.promise,constant?DeutschJozsaPromise::constant:balanced?DeutschJozsaPromise::balanced:DeutschJozsaPromise::neither);
      EXPECT_DOUBLE_EQ(t.mean_phase,(double(N)-2*double(ones))/double(N));
      double norm=0;
      for(std::size_t s=0;s<N;++s) {
        int sum=0;
        for(std::size_t x=0;x<N;++x)sum+=((f[x]+std::popcount(s&x))&1)?-1:1;
        const double want=double(sum)/double(N);
        EXPECT_NEAR(t.final_state[s].real(),want,3e-14);EXPECT_DOUBLE_EQ(t.final_state[s].imag(),0);
        EXPECT_NEAR(t.before_oracle[s].real(),1/std::sqrt(double(N)),3e-14);
        EXPECT_EQ(t.after_oracle[s],f[s]?-t.before_oracle[s]:t.before_oracle[s]);
        norm+=std::norm(t.final_state[s]);
      }
      EXPECT_NEAR(norm,1,3e-14);EXPECT_NEAR(t.result.prob_all_zero,t.mean_phase*t.mean_phase,3e-14);
      EXPECT_EQ(t.result.constant,t.result.prob_all_zero>0.5);
      const auto old=deutsch_jozsa(n,[&](std::size_t x){return f[x];});
      EXPECT_EQ(old.constant,t.result.constant);EXPECT_DOUBLE_EQ(old.prob_all_zero,t.result.prob_all_zero);
      EXPECT_EQ(t.work.oracle_applications,1);EXPECT_EQ(t.work.callback_calls,N);
      EXPECT_EQ(t.work.hadamard_gates,2*n);EXPECT_EQ(t.work.butterflies,n*N);EXPECT_EQ(t.work.phase_flips,ones);
      if(constant||balanced) {
        ++promised;std::size_t classical_calls=0;
        const auto c=deutsch_jozsa_classical(n,[&](std::size_t x){EXPECT_EQ(x,classical_calls++);return f[x];});
        EXPECT_EQ(c.constant,constant);EXPECT_EQ(c.queries,classical_calls);
        EXPECT_LE(c.queries,N==1?1:N/2+1);EXPECT_EQ(t.result.constant,constant);
      }
      ++tables;
    }
  }
  EXPECT_EQ(tables,65814);EXPECT_EQ(promised,12958);
}
TEST(DeutschJozsaTrace, OriginalFiveOracles) {
  const std::vector<std::function<int(std::size_t)>> fs{
    [](std::size_t){return 0;},[](std::size_t){return 1;},
    [](std::size_t x){return std::popcount(x)&1;},
    [](std::size_t x){return (x>>3)&1;},[](std::size_t x){return x&1;}};
  for(std::size_t i=0;i<fs.size();++i) {
    const auto t=deutsch_jozsa_trace(4,fs[i]);EXPECT_EQ(t.result.constant,i<2);
    EXPECT_NEAR(t.result.prob_all_zero,i<2?1:0,1e-13);
    EXPECT_EQ(most_probable(t.final_state),i<2?0:i==2?15:i==3?8:1);
  }
}
TEST(DeutschJozsaTrace, AllAffineParitiesAndComplements) {
  for(int n=1;n<=8;++n)for(std::size_t mask=0;mask<(std::size_t{1}<<n);++mask) {
    const auto t=deutsch_jozsa_trace(n,[&](std::size_t x){return std::popcount(x&mask)&1;});
    const auto c=deutsch_jozsa_trace(n,[&](std::size_t x){return 1^(std::popcount(x&mask)&1);});
    EXPECT_EQ(t.promise,mask?DeutschJozsaPromise::balanced:DeutschJozsaPromise::constant);
    for(std::size_t x=0;x<t.final_state.size();++x) {
      EXPECT_NEAR(t.final_state[x].real(),x==mask?1:0,3e-14);
      EXPECT_EQ(c.final_state[x],-t.final_state[x]);
    }
    EXPECT_DOUBLE_EQ(t.result.prob_all_zero,c.result.prob_all_zero);
  }
}
TEST(DeutschJozsaTrace, NonlinearBalancedAndSharedFourierCircuit) {
  for(const std::vector<int>& f:std::vector<std::vector<int>>{{0,1,0,1,0,0,1,1},{0,0,0,1},{1},{0,1,1,0}}) {
    const int n=std::countr_zero(f.size());const auto t=deutsch_jozsa_trace(n,[&](std::size_t x){return f[x];});
    const auto psi=fourier_sampling_state(f);const auto spectrum=walsh_spectrum(f);
    EXPECT_EQ(psi,t.final_state);
    for(std::size_t x=0;x<f.size();++x)EXPECT_NEAR(psi[x].real(),spectrum[x],1e-14);
  }
  const auto t=deutsch_jozsa_trace(3,[](std::size_t x){return x<4?(x&1):((x>>1)&1);});
  EXPECT_EQ(t.promise,DeutschJozsaPromise::balanced);EXPECT_NEAR(t.result.prob_all_zero,0,1e-14);
  int support=0;for(const auto& a:t.final_state)if(std::norm(a)>1e-10)++support;
  EXPECT_EQ(support,4);
}
TEST(DeutschJozsaTrace, OffPromiseDoesNotCertifyEitherThresholdDecision) {
  const auto near=deutsch_jozsa_trace(4,[](std::size_t x){return x==15;});
  const auto and2=deutsch_jozsa_trace(2,[](std::size_t x){return x==3;});
  EXPECT_EQ(near.promise,DeutschJozsaPromise::neither);EXPECT_TRUE(near.result.constant);
  EXPECT_NEAR(near.result.prob_all_zero,49.0/64,1e-14);
  EXPECT_EQ(and2.promise,DeutschJozsaPromise::neither);EXPECT_FALSE(and2.result.constant);
  EXPECT_NEAR(and2.result.prob_all_zero,0.25,1e-14);
  // Ascending promised classical querying misses the exceptional last input.
  const auto c=deutsch_jozsa_classical(4,[](std::size_t x){return x==15;});
  EXPECT_TRUE(c.constant);EXPECT_EQ(c.queries,9);
}
TEST(DeutschJozsaTrace, SequentialClassicalBestAndWorstQueries) {
  for(int n=1;n<=16;++n) {
    const std::size_t N=std::size_t{1}<<n;
    const auto c=deutsch_jozsa_classical(n,[](std::size_t){return 0;});
    const auto b=deutsch_jozsa_classical(n,[&](std::size_t x){return x>=N/2;});
    const auto early=deutsch_jozsa_classical(n,[](std::size_t x){return x&1;});
    EXPECT_TRUE(c.constant);EXPECT_FALSE(b.constant);EXPECT_FALSE(early.constant);
    EXPECT_EQ(c.queries,N/2+1);EXPECT_EQ(b.queries,N/2+1);EXPECT_EQ(early.queries,2);
  }
  // Largest representable dimension is safe in the allocation-free early exit.
  const auto huge=deutsch_jozsa_classical(std::numeric_limits<std::size_t>::digits-1,[](std::size_t x){return x&1;});
  EXPECT_FALSE(huge.constant);EXPECT_EQ(huge.queries,2);
}
TEST(DeutschJozsaTrace, ValidationCapsAndSafeDimensions) {
  int calls=0;const std::function<int(std::size_t)> f=[&](std::size_t){++calls;return 0;};
  const std::function<int(std::size_t)> empty;
  for(int n:{-1,std::numeric_limits<int>::min(),std::numeric_limits<std::size_t>::digits,std::numeric_limits<int>::max()}) {
    EXPECT_THROW(deutsch_jozsa(n,f),std::invalid_argument);
    EXPECT_THROW(deutsch_jozsa_trace(n,f,1),std::invalid_argument);
    EXPECT_THROW(deutsch_jozsa_classical(n,f),std::invalid_argument);
  }
  EXPECT_THROW(deutsch_jozsa(0,empty),std::invalid_argument);
  EXPECT_THROW(deutsch_jozsa_trace(20,empty,1),std::invalid_argument);
  EXPECT_THROW(deutsch_jozsa_classical(0,empty),std::invalid_argument);
  EXPECT_THROW(deutsch_jozsa_trace(4,f,15),std::length_error);
  EXPECT_THROW(deutsch_jozsa_trace(21,f),std::length_error);
  EXPECT_THROW(deutsch_jozsa_trace(std::numeric_limits<std::size_t>::digits-1,f,0),std::length_error);
  EXPECT_THROW(deutsch_jozsa(std::numeric_limits<std::size_t>::digits-1,f),std::length_error);
  EXPECT_EQ(calls,0);
  EXPECT_NO_THROW(deutsch_jozsa_trace(4,f,16));
  EXPECT_NO_THROW(deutsch_jozsa_trace(4,f,0));
  EXPECT_EQ(calls,32);
}
TEST(DeutschJozsaTrace, ZeroQubitsAndLegacyIntegerLowBits) {
  for(int value:{-3,-2,-1,0,1,2,3}) {
    const auto t=deutsch_jozsa_trace(0,[&](std::size_t x){EXPECT_EQ(x,0);return value;},1);
    EXPECT_TRUE(t.result.constant);EXPECT_DOUBLE_EQ(t.result.prob_all_zero,1);
    EXPECT_EQ(t.promise,DeutschJozsaPromise::constant);EXPECT_EQ(t.ones,value&1);
    EXPECT_EQ(t.work.hadamard_gates,0);EXPECT_EQ(t.work.butterflies,0);
    EXPECT_EQ(t.work.callback_calls,1);EXPECT_EQ(t.work.oracle_applications,1);
    EXPECT_DOUBLE_EQ(t.final_state[0].real(),(value&1)?-1:1);
    const auto c=deutsch_jozsa_classical(0,[&](std::size_t){return value;});
    EXPECT_TRUE(c.constant);EXPECT_EQ(c.queries,1);
  }
  const auto t=deutsch_jozsa_trace(3,[](std::size_t x){return x&1?-3:2;});
  EXPECT_EQ(t.promise,DeutschJozsaPromise::balanced);EXPECT_FALSE(t.result.constant);
}
TEST(DeutschJozsaTrace, ActualSinglePassExceptionsAndOwnership) {
  std::size_t calls=0;
  auto t=deutsch_jozsa_trace(3,[&](std::size_t x){EXPECT_EQ(x,calls);return (++calls)%2;});
  EXPECT_EQ(calls,8);EXPECT_EQ(t.truth_table,(std::vector<int>{1,0,1,0,1,0,1,0}));
  auto copy=t;t.truth_table[0]=0;t.final_state[1]=42;t.after_oracle[0]=12;
  EXPECT_EQ(copy.truth_table[0],1);EXPECT_NE(copy.final_state[1],t.final_state[1]);
  EXPECT_NE(copy.after_oracle[0],t.after_oracle[0]);EXPECT_EQ(copy.work.callback_calls,8);
  for(int mode=0;mode<3;++mode) {
    calls=0;const std::function<int(std::size_t)> bad=[&](std::size_t x)->int{++calls;if(x==2)throw std::runtime_error("oracle");return 0;};
    if(mode==0) {EXPECT_THROW(deutsch_jozsa(3,bad),std::runtime_error);}
    if(mode==1) {EXPECT_THROW(deutsch_jozsa_trace(3,bad),std::runtime_error);}
    if(mode==2) {EXPECT_THROW(deutsch_jozsa_classical(3,bad),std::runtime_error);}
    EXPECT_EQ(calls,3);
  }
}
