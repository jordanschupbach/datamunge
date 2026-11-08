#include <gtest/gtest.h>
#include <datamunge/algorithms/cannon.hpp>
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>
namespace {
namespace a=datamunge::algorithms;
std::vector<double> integer_product(const std::vector<double>& A,const std::vector<double>& B,int n) {
  std::vector<double> result(n*n);
  for(int i=0;i<n;++i) for(int j=0;j<n;++j) {
    long long sum=0;for(int k=0;k<n;++k) sum+=static_cast<long long>(A[i*n+k])*static_cast<long long>(B[k*n+j]);
    result[i*n+j]=static_cast<double>(sum);
  }
  return result;
}
void check(const std::vector<double>& A,const std::vector<double>& B,int n,int q) {
  const auto trace=a::cannon_matmul_trace(A,B,n,q);const auto expected=integer_product(A,B,n);
  EXPECT_EQ(trace.output,expected);EXPECT_EQ(a::cannon_block_matmul(A,B,n,q),expected);
  EXPECT_EQ(trace.dimension,n);EXPECT_EQ(trace.mesh_dimension,q);ASSERT_EQ(trace.rounds.size(),static_cast<std::size_t>(q));
  const auto counts=a::cannon_work(n,q);
  EXPECT_EQ(trace.work.block_updates,counts.block_updates);EXPECT_EQ(trace.work.multiply_adds,counts.multiply_adds);
  EXPECT_EQ(trace.work.skew_copies,counts.skew_copies);EXPECT_EQ(trace.work.shift_copies,counts.shift_copies);
  if(!n) {EXPECT_TRUE(trace.final_a_sources.empty());return;}
  const int b=n/q;EXPECT_EQ(trace.block_dimension,b);
  std::vector<double> partial(n*n);
  std::vector<std::vector<bool>> seen(q*q,std::vector<bool>(q));
  // Independently enumerate closed-form source ids and original matrix block products.
  for(int step=0;step<q;++step) {
    const auto& round=trace.rounds[step];EXPECT_EQ(round.step,step);
    ASSERT_EQ(round.a_sources.size(),static_cast<std::size_t>(q*q));ASSERT_EQ(round.b_sources.size(),round.a_sources.size());
    for(int i=0;i<q;++i) for(int j=0;j<q;++j) {
      const int k=(i+j+step)%q,cell=i*q+j;
      EXPECT_EQ(round.a_sources[cell],i*q+k);EXPECT_EQ(round.b_sources[cell],k*q+j);
      EXPECT_FALSE(seen[cell][k]);seen[cell][k]=true;
      for(int r=0;r<b;++r) for(int c=0;c<b;++c) {
        long long contribution=0;
        for(int local=0;local<b;++local)
          contribution+=static_cast<long long>(A[(i*b+r)*n+k*b+local])*static_cast<long long>(B[(k*b+local)*n+j*b+c]);
        partial[(i*b+r)*n+j*b+c]+=contribution;
      }
    }
    EXPECT_EQ(round.partial,partial);
  }
  for(const auto& ks:seen) EXPECT_TRUE(std::all_of(ks.begin(),ks.end(),[](bool value){return value;}));
  EXPECT_EQ(trace.final_a_sources,trace.rounds[0].a_sources);EXPECT_EQ(trace.final_b_sources,trace.rounds[0].b_sources);
}
}
TEST(CannonTrace, OriginalTwoByTwoHandCalculation) {
  const auto trace=a::cannon_matmul_trace({1,2,3,4},{5,6,7,8},2,2);
  ASSERT_EQ(trace.rounds.size(),2u);
  EXPECT_EQ(trace.rounds[0].a_sources,(std::vector<std::size_t>{0,1,3,2}));
  EXPECT_EQ(trace.rounds[0].b_sources,(std::vector<std::size_t>{0,3,2,1}));
  EXPECT_EQ(trace.rounds[0].partial,(std::vector<double>{5,16,28,18}));
  EXPECT_EQ(trace.output,(std::vector<double>{19,22,43,50}));
  EXPECT_EQ(a::cannon_matmul({1,2,3,4},{5,6,7,8},2),trace.output);
  EXPECT_EQ(trace.work.block_updates,8u);EXPECT_EQ(trace.work.multiply_adds,8u);EXPECT_EQ(trace.work.skew_copies,8u);EXPECT_EQ(trace.work.shift_copies,16u);
}
TEST(CannonTrace, ExhaustiveTwoByTwoSignedIntegerProducts) {
  std::size_t checked=0;
  for(int left=0;left<81;++left) for(int right=0;right<81;++right) {
    std::vector<double> A(4),B(4);int x=left,y=right;
    for(int i=0;i<4;++i) {A[i]=x%3-1;B[i]=y%3-1;x/=3;y/=3;}
    check(A,B,2,1);check(A,B,2,2);++checked;
  }
  EXPECT_EQ(checked,6561u);
}
TEST(CannonTrace, RandomIntegerBlocksAndCompleteRoundPermutations) {
  std::mt19937 rng(1969);std::size_t checked=0;
  for(int n=1;n<=16;++n) for(int trial=0;trial<20;++trial) {
    std::vector<double> A(n*n),B(n*n);
    for(double& v:A) v=static_cast<int>(rng()%21)-10;
    for(double& v:B) v=static_cast<int>(rng()%21)-10;
    for(int q=1;q<=n;++q) if(n%q==0) {check(A,B,n,q);++checked;}
    EXPECT_EQ(a::cannon_matmul(A,B,n),integer_product(A,B,n));
  }
  EXPECT_EQ(checked,1000u);
}
TEST(CannonTrace, IdentityZerosAndAliasedInputs) {
  std::vector<double> A{1,2,3,4,5,6,7,8,9},I{1,0,0,0,1,0,0,0,1},zero(9);
  for(int q:{1,3}) {check(A,I,3,q);check(I,A,3,q);check(A,zero,3,q);check(zero,A,3,q);check(A,A,3,q);}
  EXPECT_EQ(a::cannon_matmul(A,A,3),integer_product(A,A,3));EXPECT_EQ(A,(std::vector<double>{1,2,3,4,5,6,7,8,9}));
}
TEST(CannonTrace, EveryFiniteScalarShapeAndEmptyProduct) {
  check({}, {},0,0);EXPECT_TRUE(a::cannon_matmul({}, {},0).empty());
  const auto t=a::cannon_matmul_trace({3},{7},1,1);EXPECT_EQ(t.output,(std::vector<double>{21}));
  EXPECT_EQ(t.work.block_updates,1u);EXPECT_EQ(t.work.skew_copies,2u);EXPECT_EQ(t.work.shift_copies,2u);
  EXPECT_EQ(t.rounds[0].a_sources,(std::vector<std::size_t>{0}));EXPECT_EQ(t.final_a_sources,t.rounds[0].a_sources);
}
TEST(CannonTrace, RejectsDimensionsAndExactInputLengthBeforeAccess) {
  EXPECT_THROW(a::cannon_matmul({}, {},-1),std::invalid_argument);
  EXPECT_THROW(a::cannon_matmul({1},{1},0),std::invalid_argument);
  EXPECT_THROW(a::cannon_matmul({1},{1},2),std::invalid_argument);
  EXPECT_THROW(a::cannon_matmul({1,2},{1},1),std::invalid_argument);
  EXPECT_THROW(a::cannon_matmul_trace({1},{1},1,0),std::invalid_argument);
  EXPECT_THROW(a::cannon_work(0,1),std::invalid_argument);
  EXPECT_THROW(a::cannon_work(4,-1),std::invalid_argument);
  EXPECT_THROW(a::cannon_work(4,3),std::invalid_argument);
  EXPECT_THROW(a::cannon_work(4,8),std::invalid_argument);
  EXPECT_THROW(a::cannon_work(std::numeric_limits<int>::max(),1),std::length_error);
  EXPECT_THROW(a::cannon_matmul({}, {},std::numeric_limits<int>::max()),std::length_error);
}
TEST(CannonTrace, WholeCallBudgetAndNoInputMutation) {
  const std::vector<double> A{1,2,3,4},B{5,6,7,8};
  EXPECT_THROW(a::cannon_block_matmul(A,B,2,2,7),std::length_error);
  EXPECT_THROW(a::cannon_matmul_trace(A,B,2,2,7),std::length_error);
  EXPECT_EQ(a::cannon_block_matmul(A,B,2,2,8),(std::vector<double>{19,22,43,50}));
  EXPECT_EQ(a::cannon_matmul_trace(A,B,2,2,8).work.block_updates,8u);
  EXPECT_EQ(a::cannon_matmul_trace(A,B,2,2,9).work.block_updates,8u);
  EXPECT_EQ(A,(std::vector<double>{1,2,3,4}));EXPECT_EQ(B,(std::vector<double>{5,6,7,8}));
}
TEST(CannonTrace, CountsFixedMatrixSizeWithDifferentMeshSides) {
  for(int q:{1,2,3,4,6,8,12,16,24,32,48,96}) {
    const auto w=a::cannon_work(96,q);EXPECT_EQ(w.multiply_adds,884736u);
    EXPECT_EQ(w.block_updates,static_cast<std::size_t>(q)*q*q);
    EXPECT_EQ(w.skew_copies,18432u);EXPECT_EQ(w.shift_copies,static_cast<std::size_t>(18432)*q);
  }
}
TEST(CannonTrace, FloatingPointOrderIsNotBitwiseInvariant) {
  // At cell (0,1), scalar Cannon visits k=1,2,0; ascending-k visits 0,1,2.
  const std::vector<double> A{1e16,-1e16,1,0,0,0,0,0,0},B{0,1,0,0,1,0,0,1,0};
  const auto scalar=a::cannon_matmul(A,B,3),single=a::cannon_block_matmul(A,B,3,1);
  EXPECT_EQ(scalar[1],0);EXPECT_EQ(single[1],1);
  const auto trace=a::cannon_matmul_trace(A,B,3,3);EXPECT_EQ(trace.output,scalar);
  EXPECT_EQ(trace.rounds[0].a_sources[1],1u);EXPECT_EQ(trace.rounds[1].a_sources[1],2u);EXPECT_EQ(trace.rounds[2].a_sources[1],0u);
}
TEST(CannonTrace, RandomFloatingProductsAgainstLongDoubleDotProducts) {
  std::mt19937 rng(11);std::uniform_real_distribution<double> draw(-5,5);
  for(int n=1;n<=20;++n) {
    std::vector<double> A(n*n),B(n*n);for(auto& v:A)v=draw(rng);for(auto& v:B)v=draw(rng);
    for(int q=1;q<=n;++q) if(n%q==0) {
      const auto got=a::cannon_block_matmul(A,B,n,q);
      for(int i=0;i<n;++i) for(int j=0;j<n;++j) {
        long double exact=0,scale=0;for(int k=0;k<n;++k) {const long double term=static_cast<long double>(A[i*n+k])*B[k*n+j];exact+=term;scale+=std::abs(term);}
        EXPECT_LE(std::abs(static_cast<long double>(got[i*n+j])-exact),4*n*std::numeric_limits<double>::epsilon()*scale+1e-14L);
      }
    }
  }
}
TEST(CannonTrace, NonfiniteArithmeticPropagatesAndSnapshotsOwnStorage) {
  EXPECT_TRUE(std::isnan(a::cannon_matmul({std::numeric_limits<double>::quiet_NaN()},{1},1)[0]));
  EXPECT_TRUE(std::isinf(a::cannon_matmul({std::numeric_limits<double>::infinity()},{1},1)[0]));
  EXPECT_TRUE(std::isinf(a::cannon_matmul({std::numeric_limits<double>::max()},{2},1)[0]));
  auto trace=a::cannon_matmul_trace({1,2,3,4},{5,6,7,8},2,2);auto copy=trace;
  copy.rounds[0].partial[0]=999;copy.final_a_sources[0]=999;copy.output[0]=999;
  EXPECT_EQ(trace.rounds[0].partial[0],5);EXPECT_EQ(trace.final_a_sources[0],0u);EXPECT_EQ(trace.output[0],19);
  trace.rounds[0].partial[0]=111;EXPECT_EQ(trace.rounds[1].partial[0],19);
}
