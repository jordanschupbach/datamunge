#include <gtest/gtest.h>
#include <datamunge/algorithms/dda_line.hpp>
#include <datamunge/algorithms/bresenham.hpp>
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
namespace {
int rational_oracle(int a,int b,std::uint64_t i,std::uint64_t n) {
  if(!n) return a;
  // Independent signed wide weighted-endpoint numerator, not production delta division.
  __int128 numerator=static_cast<__int128>(a)*(n-i)+static_cast<__int128>(b)*i;
  const int sign=numerator<0?-1:1;if(numerator<0)numerator=-numerator;
  const auto q=numerator/n,r=numerator%n;
  return static_cast<int>(sign*(q+(2*r>=n)));
}
void inspect(int x0,int y0,int x1,int y1) {
  const auto t=dda_line_trace(x0,y0,x1,y1);
  const auto reference=dda_line_exact(x0,y0,x1,y1);const auto ordinary=dda_line(x0,y0,x1,y1);
  ASSERT_EQ(t.points,ordinary);ASSERT_EQ(t.samples.size(),ordinary.size());ASSERT_EQ(reference.size(),ordinary.size());
  const auto n=t.steps;EXPECT_EQ(dda_point_count(x0,y0,x1,y1),n+1);EXPECT_EQ(t.coordinate_additions,2*n);EXPECT_EQ(t.roundings,2*(n+1));
  double x=x0,y=y0,maxerr=0,maxdrift=0;std::uint64_t differing=0;
  for(std::uint64_t i=0;i<=n;++i) {
    const auto& s=t.samples[i];const DDAPoint expected{rational_oracle(x0,x1,i,n),rational_oracle(y0,y1,i,n)};
    EXPECT_EQ(s.index,i);EXPECT_DOUBLE_EQ(s.x,x);EXPECT_DOUBLE_EQ(s.y,y);
    EXPECT_EQ(s.point,(DDAPoint{static_cast<int>(std::lround(x)),static_cast<int>(std::lround(y))}));
    EXPECT_EQ(s.exact_point,expected);EXPECT_EQ(reference[i],expected);EXPECT_EQ(dda_exact_sample(x0,y0,x1,y1,i),expected);
    // A direct long-double weighted numerator checks diagnostics on short/small-domain controls.
    const long double idealx=n?(static_cast<long double>(x0)*(n-i)+static_cast<long double>(x1)*i)/n:x0;
    const long double idealy=n?(static_cast<long double>(y0)*(n-i)+static_cast<long double>(y1)*i)/n:y0;
    EXPECT_NEAR(s.x_error,static_cast<long double>(s.point.first)-idealx,2e-10);
    EXPECT_NEAR(s.y_error,static_cast<long double>(s.point.second)-idealy,2e-10);
    EXPECT_NEAR(s.x_drift,static_cast<long double>(x)-idealx,2e-10);
    EXPECT_NEAR(s.y_drift,static_cast<long double>(y)-idealy,2e-10);
    EXPECT_LE(std::abs(static_cast<long double>(expected.first)-idealx),.5000000002L);
    EXPECT_LE(std::abs(static_cast<long double>(expected.second)-idealy),.5000000002L);
    maxerr=std::max({maxerr,std::abs(s.x_error),std::abs(s.y_error)});maxdrift=std::max({maxdrift,std::abs(s.x_drift),std::abs(s.y_drift)});differing+=s.point!=expected;
    if(i) {EXPECT_LE(std::abs(reference[i].first-reference[i-1].first),1);EXPECT_LE(std::abs(reference[i].second-reference[i-1].second),1);EXPECT_NE(reference[i],reference[i-1]);}
    x+=t.x_increment;y+=t.y_increment;
  }
  EXPECT_DOUBLE_EQ(t.max_coordinate_error,maxerr);EXPECT_DOUBLE_EQ(t.max_accumulator_drift,maxdrift);EXPECT_EQ(t.differing_samples,differing);
  EXPECT_EQ(reference.front(),(DDAPoint{x0,y0}));EXPECT_EQ(reference.back(),(DDAPoint{x1,y1}));
  auto reversed=dda_line_exact(x1,y1,x0,y0);std::reverse(reversed.begin(),reversed.end());EXPECT_EQ(reference,reversed);
}
}
TEST(DdaTrace, OriginalRasterAndSteepCount) {
  const std::vector<DDAPoint> expected{{0,0},{1,0},{2,1},{3,1},{4,2},{5,2},{6,3},{7,3},{8,3},{9,4},{10,4},{11,5},{12,5},{13,6},{14,6}};
  EXPECT_EQ(dda_line(0,0,14,6),expected);EXPECT_EQ(dda_line(0,0,3,10).size(),11);inspect(0,0,14,6);inspect(0,0,3,10);
}
TEST(DdaTrace, ExhaustiveSmallSignedEndpoints) {
  for(int x0=-5;x0<=5;++x0)for(int y0=-5;y0<=5;++y0)for(int x1=-5;x1<=5;++x1)for(int y1=-5;y1<=5;++y1)inspect(x0,y0,x1,y1);
}
TEST(DdaTrace, AccumulatedHalfwayDriftRemainsVisible) {
  const auto t=dda_line_trace(0,0,12,1);ASSERT_EQ(t.points.size(),13);
  EXPECT_LT(t.samples[6].y,.5);EXPECT_EQ(t.samples[6].point,(DDAPoint{6,0}));EXPECT_EQ(t.samples[6].exact_point,(DDAPoint{6,1}));
  EXPECT_LT(t.samples[6].y_drift,0);EXPECT_GT(t.differing_samples,0);EXPECT_NE(t.points,dda_line_exact(0,0,12,1));
  auto reverse=dda_line(10,7,0,0);std::reverse(reverse.begin(),reverse.end());EXPECT_NE(dda_line(0,0,10,7),reverse);
  inspect(0,0,12,1);inspect(12,1,0,0);
  const auto shifted=dda_line_trace(0,INT_MAX-1,10000,INT_MAX);
  EXPECT_GT(shifted.max_coordinate_error,.5);EXPECT_GT(shifted.differing_samples,0);
  EXPECT_EQ(shifted.points.back(),(DDAPoint{10000,INT_MAX}));
}
TEST(DdaTrace, SignedGlobalTiesAndBresenhamAreDistinct) {
  EXPECT_EQ(dda_exact_sample(0,0,2,-1,1),(DDAPoint{1,-1}));
  EXPECT_EQ(dda_exact_sample(0,1,2,0,1),(DDAPoint{1,1}));
  EXPECT_EQ(dda_exact_sample(0,-1,2,0,1),(DDAPoint{1,-1}));
  EXPECT_EQ(dda_line_exact(2,1,0,0)[1],(DDAPoint{1,1}));
  EXPECT_EQ(bresenham_line(2,1,0,0)[1],(DDAPoint{1,0}));
  // Signed ties across zero are not invariant under integer translation.
  EXPECT_NE(dda_exact_sample(0,-1,2,0,1).second+1,dda_exact_sample(0,0,2,1,1).second);
}
TEST(DdaTrace, SeededAccumulatorReplayAcrossOctants) {
  std::mt19937 rng(1973);
  for(int trial=0;trial<1000;++trial) {
    const int x=static_cast<int>(rng()%2000000001)-1000000000,y=static_cast<int>(rng()%2000000001)-1000000000;
    inspect(x,y,x+static_cast<int>(rng()%201)-100,y+static_cast<int>(rng()%201)-100);
  }
}
TEST(DdaTrace, ExactFullWidthSamplesWithoutAllocation) {
  std::mt19937_64 rng(1965);
  const std::array<int,5> extremes{INT_MIN,INT_MIN+1,0,INT_MAX-1,INT_MAX};
  for(int a:extremes)for(int b:extremes)for(int c:extremes)for(int d:extremes) {
    const auto n=dda_point_count(a,b,c,d)-1;
    for(auto i:{std::uint64_t(0),n/2,(n+1)/2,n})EXPECT_EQ(dda_exact_sample(a,b,c,d,i),(DDAPoint{rational_oracle(a,c,i,n),rational_oracle(b,d,i,n)}));
  }
  for(int trial=0;trial<25000;++trial) {
    const int a=static_cast<int>(static_cast<std::int64_t>(rng()%4294967296ULL)+INT_MIN),b=static_cast<int>(static_cast<std::int64_t>(rng()%4294967296ULL)+INT_MIN);
    const int c=static_cast<int>(static_cast<std::int64_t>(rng()%4294967296ULL)+INT_MIN),d=static_cast<int>(static_cast<std::int64_t>(rng()%4294967296ULL)+INT_MIN);
    const auto n=dda_point_count(a,b,c,d)-1,i=rng()%(n+1);
    EXPECT_EQ(dda_exact_sample(a,b,c,d,i),(DDAPoint{rational_oracle(a,c,i,n),rational_oracle(b,d,i,n)}));
  }
  EXPECT_EQ(dda_point_count(INT_MIN,INT_MAX,INT_MAX,INT_MIN),4294967296ULL);
}
TEST(DdaTrace, CompleteCapsAndIndexDomain) {
  EXPECT_NO_THROW(dda_line_trace(0,0,14,6,15));
  EXPECT_THROW(dda_line_trace(0,0,14,6,14),std::length_error);
  EXPECT_NO_THROW(dda_line_exact(0,0,14,6,15));
  EXPECT_THROW(dda_line_exact(0,0,14,6,14),std::length_error);
  EXPECT_NO_THROW(dda_line_trace(0,0,14,6,0));
  EXPECT_NO_THROW(dda_line_exact(0,0,14,6,0));
  EXPECT_THROW(dda_line_trace(INT_MIN,0,INT_MAX,1),std::length_error);
  EXPECT_THROW(dda_line_exact(INT_MIN,0,INT_MAX,1),std::length_error);
  EXPECT_THROW(dda_exact_sample(0,0,12,1,13),std::out_of_range);
  EXPECT_THROW(dda_exact_sample(1,2,1,2,1),std::out_of_range);
  EXPECT_THROW(dda_exact_sample(INT_MIN,0,INT_MAX,0,std::numeric_limits<std::uint64_t>::max()),std::out_of_range);
}
TEST(DdaTrace, CoincidenceExtremesOwnershipAndWorkLaw) {
  for(int a:{INT_MIN,0,INT_MAX}) {const auto t=dda_line_trace(a,a,a,a,1);EXPECT_EQ(t.points,(std::vector<DDAPoint>{{a,a}}));EXPECT_EQ(t.roundings,2);EXPECT_EQ(t.coordinate_additions,0);EXPECT_EQ(t.max_accumulator_drift,0);}
  inspect(INT_MIN,INT_MIN,INT_MIN+40,INT_MIN+17);inspect(INT_MAX,INT_MAX,INT_MAX-40,INT_MAX-17);
  auto t=dda_line_trace(0,0,4,2);auto copy=t;t.points[0]={9,9};EXPECT_EQ(copy.points[0],(DDAPoint{0,0}));
  for(int n:{8,16,32,64,128,256,1024}) {const auto t=dda_line_trace(0,0,n,n/2);EXPECT_EQ(t.points.size(),n+1);EXPECT_EQ(t.coordinate_additions,2*n);EXPECT_EQ(t.roundings,2*(n+1));EXPECT_EQ(t.differing_samples,0);EXPECT_EQ(t.max_accumulator_drift,0);EXPECT_EQ(t.max_coordinate_error,.5);}
}
TEST(DdaTrace, AccumulationBeyondIntIsCheckedBeforeConversion) {
  // At coordinates near INT_MAX, each 9/5000000 increment rounds to eight ULPs.
  // Its accumulated overshoot exceeds 0.5 near the end; no unsafe int conversion occurs.
  EXPECT_THROW(dda_line(0,INT_MAX-9,5000000,INT_MAX),std::overflow_error);
  EXPECT_EQ(dda_exact_sample(0,INT_MAX-9,5000000,INT_MAX,5000000),(DDAPoint{5000000,INT_MAX}));
}
