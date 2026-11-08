#include <gtest/gtest.h>
#include <datamunge/algorithms/cohen_sutherland.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace {
using namespace datamunge::algorithms;
// Test-only exact fractions: bounded coordinates [-20,20] keep every product
// below 2^31. No floating calculation participates in visibility decisions.
struct Rat {
  std::int64_t n=0,d=1;
  Rat(std::int64_t numerator=0,std::int64_t denominator=1):n(numerator),d(denominator) {
    if(d<0){n=-n;d=-d;}const auto common=std::gcd(n,d);n/=common;d/=common;
  }
  std::int64_t numerator()const{return n;}
  std::int64_t denominator()const{return d;}
  friend Rat operator+(Rat a,Rat b){return {a.n*b.d+b.n*a.d,a.d*b.d};}
  friend Rat operator*(Rat a,Rat b){return {a.n*b.n,a.d*b.d};}
  friend bool operator<(Rat a,Rat b){return a.n*b.d<b.n*a.d;}
  friend bool operator>(Rat a,Rat b){return b<a;}
};
struct Exact {bool visible=false;std::array<Rat,4> coordinates{};};
Exact oracle(const std::array<int,8>& a) {
  Rat low(0),high(1);
  for(int axis=0;axis<2;++axis) {
    const auto origin=a[axis],delta=a[axis+2]-origin;
    if(!delta) {if(origin<a[axis+4] || origin>a[axis+6])return {};continue;}
    auto enter=Rat(a[axis+4]-origin,delta),leave=Rat(a[axis+6]-origin,delta);
    if(enter>leave)std::swap(enter,leave);
    low=std::max(low,enter);high=std::min(high,leave);
    if(low>high)return {};
  }
  return {true,{Rat(a[0])+Rat(a[2]-a[0])*low,Rat(a[1])+Rat(a[3]-a[1])*low,
                Rat(a[0])+Rat(a[2]-a[0])*high,Rat(a[1])+Rat(a[3]-a[1])*high}};
}
auto trace(const std::array<double,8>& a) {
  return cohen_sutherland_clip_trace(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7]);
}
unsigned code(long double x,long double y,const std::array<double,8>& a) {
  return (x<a[4]?1U:0U)|(x>a[6]?2U:0U)|(y<a[5]?4U:0U)|(y>a[7]?8U:0U);
}
void check(const std::array<int,8>& integers) {
  std::array<double,8> a;std::copy(integers.begin(),integers.end(),a.begin());
  const auto expected=oracle(integers);const auto t=trace(a);
  ASSERT_EQ(t.result.visible,expected.visible) << ::testing::PrintToString(integers);
  const std::array<double,4> output{t.result.x0,t.result.y0,t.result.x1,t.result.y1};
  if(expected.visible) {
    for(std::size_t i=0;i<4;++i)
      ASSERT_NEAR(output[i],double(expected.coordinates[i].numerator())/expected.coordinates[i].denominator(),1e-12);
    ASSERT_EQ(code(output[0],output[1],a),0);ASSERT_EQ(code(output[2],output[3],a),0);
  } else ASSERT_EQ(output,(std::array<double,4>{0,0,0,0}));
  const auto plain=cohen_sutherland_clip(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7]);
  ASSERT_EQ(plain.visible,t.result.visible);ASSERT_EQ(plain.x0,t.result.x0);ASSERT_EQ(plain.y0,t.result.y0);
  ASSERT_EQ(plain.x1,t.result.x1);ASSERT_EQ(plain.y1,t.result.y1);
  ASSERT_LE(t.intersections,4);ASSERT_EQ(t.steps.size(),t.intersections+1);
  ASSERT_EQ(t.outcode_evaluations,2+t.intersections);ASSERT_EQ(t.or_tests,t.steps.size());
  ASSERT_EQ(t.and_tests,t.steps.size()-t.result.visible);
  std::array<long double,4> state{a[0],a[1],a[2],a[3]};unsigned processed=0;
  for(std::size_t i=0;i<t.steps.size();++i) {
    const auto& s=t.steps[i];
    ASSERT_EQ((std::array<long double,4>{s.x0,s.y0,s.x1,s.y1}),state);
    ASSERT_EQ(s.code0,code(s.x0,s.y0,a));ASSERT_EQ(s.code1,code(s.x1,s.y1,a));
    ASSERT_EQ((s.code0|s.code1)&processed,0);
    if(s.action==ClipAction::Accept) {ASSERT_EQ(s.code0|s.code1,0);ASSERT_EQ(i+1,t.steps.size());continue;}
    if(s.action==ClipAction::Reject) {ASSERT_NE(s.code0&s.code1,0);ASSERT_EQ(i+1,t.steps.size());continue;}
    ASSERT_NE(s.code0|s.code1,0);ASSERT_EQ(s.code0&s.code1,0);
    ASSERT_EQ(s.replaced_endpoint,s.code0?0:1);
    const unsigned outside=s.code0?s.code0:s.code1;
    const unsigned edge=outside&8?8:outside&4?4:outside&2?2:1;
    ASSERT_EQ(processed&edge,0);processed|=edge;
    ASSERT_TRUE(std::isfinite(s.intersection_x));ASSERT_TRUE(std::isfinite(s.intersection_y));
    ASSERT_GE(s.intersection_x,std::min(s.x0,s.x1));ASSERT_LE(s.intersection_x,std::max(s.x0,s.x1));
    ASSERT_GE(s.intersection_y,std::min(s.y0,s.y1));ASSERT_LE(s.intersection_y,std::max(s.y0,s.y1));
    if(edge==8){ASSERT_EQ(s.action,ClipAction::Top);ASSERT_EQ(s.intersection_y,a[7]);}
    if(edge==4){ASSERT_EQ(s.action,ClipAction::Bottom);ASSERT_EQ(s.intersection_y,a[5]);}
    if(edge==2){ASSERT_EQ(s.action,ClipAction::Right);ASSERT_EQ(s.intersection_x,a[6]);}
    if(edge==1){ASSERT_EQ(s.action,ClipAction::Left);ASSERT_EQ(s.intersection_x,a[4]);}
    ASSERT_EQ(s.new_code,code(s.intersection_x,s.intersection_y,a));ASSERT_EQ(s.new_code&processed,0);
    state[2*s.replaced_endpoint]=s.intersection_x;state[2*s.replaced_endpoint+1]=s.intersection_y;
  }
}
}

TEST(CohenTrace, OriginalFourSegmentsRetainExactWorkedResults) {
  check({-5,5,15,5,0,0,10,10});check({2,-3,8,13,0,0,10,10});
  check({3,3,7,6,0,0,10,10});check({-5,20,15,20,0,0,10,10});
  const auto c=cohen_sutherland_clip(2,-3,8,13,0,0,10,10);
  EXPECT_EQ(c.x0,3.125);EXPECT_EQ(c.y0,0);EXPECT_EQ(c.x1,6.875);EXPECT_EQ(c.y1,10);
}

TEST(CohenTrace, AllNineCodesIncludeClosedBoundaries) {
  const unsigned expected[3][3]={{5,4,6},{1,0,2},{9,8,10}};
  for(int y=0;y<3;++y)for(int x=0;x<3;++x)
    EXPECT_EQ(cohen_sutherland_outcode(2*x-1,2*y-1,0,0,2,2),expected[y][x]);
  for(double x:{0.,1.,2.})for(double y:{0.,1.,2.})EXPECT_EQ(cohen_sutherland_outcode(x,y,0,0,2,2),0);
}

TEST(CohenTrace, ZeroAndMaskCanStillMissAndAcquireADifferentOutsideBit) {
  const auto t=cohen_sutherland_clip_trace(-1,.5,.5,2,0,0,1,1);
  ASSERT_FALSE(t.result.visible);ASSERT_EQ(t.steps.size(),2);
  EXPECT_EQ(t.steps[0].code0,1);EXPECT_EQ(t.steps[0].code1,8);
  EXPECT_EQ(t.steps[0].action,ClipAction::Left);EXPECT_EQ(t.steps[0].new_code,8);
  EXPECT_EQ(t.steps[0].intersection_x,0);EXPECT_EQ(t.steps[0].intersection_y,1.5);
  EXPECT_EQ(t.steps[1].action,ClipAction::Reject);
}

TEST(CohenTrace, ExhaustiveGridSegmentsAndDegenerateRectanglesMatchExactRationals) {
  for(int xmin=-1;xmin<=1;++xmin)for(int xmax=xmin;xmax<=1;++xmax)
  for(int ymin=-1;ymin<=1;++ymin)for(int ymax=ymin;ymax<=1;++ymax)
  for(int x0=-3;x0<=3;++x0)for(int y0=-3;y0<=3;++y0)
  for(int x1=-3;x1<=3;++x1)for(int y1=-3;y1<=3;++y1) {
    check({x0,y0,x1,y1,xmin,ymin,xmax,ymax});if(::testing::Test::HasFatalFailure())return;
  }
}

TEST(CohenTrace, SeededIntegerProblemsMatchExactParametricIntervals) {
  std::mt19937 engine(90125);
  for(int trial=0;trial<3000;++trial) {
    std::array<int,8> a;
    for(auto& x:a)x=static_cast<int>(engine()%41)-20;
    if(a[4]>a[6])std::swap(a[4],a[6]);if(a[5]>a[7])std::swap(a[5],a[7]);
    check(a);if(::testing::Test::HasFatalFailure())return;
  }
}

TEST(CohenTrace, ReversalAndExactPowerOfTwoSimilaritiesPreserveVisibleGeometry) {
  const std::array<double,8> original{-4,-1,7,5,-2,-2,3,4};
  const auto a=trace(original).result;
  auto reversed=original;std::swap(reversed[0],reversed[2]);std::swap(reversed[1],reversed[3]);
  const auto b=trace(reversed).result;ASSERT_TRUE(a.visible);ASSERT_TRUE(b.visible);
  EXPECT_DOUBLE_EQ(a.x0,b.x1);EXPECT_DOUBLE_EQ(a.y0,b.y1);EXPECT_DOUBLE_EQ(a.x1,b.x0);EXPECT_DOUBLE_EQ(a.y1,b.y0);
  for(int power=-20;power<=20;++power) {
    auto moved=original;const double scale=std::ldexp(1.,power),shift=3*scale;
    for(auto& x:moved)x=x*scale+shift;
    const auto c=trace(moved).result;ASSERT_EQ(c.visible,a.visible);
    EXPECT_DOUBLE_EQ(c.x0,a.x0*scale+shift);EXPECT_DOUBLE_EQ(c.y0,a.y0*scale+shift);
    EXPECT_DOUBLE_EQ(c.x1,a.x1*scale+shift);EXPECT_DOUBLE_EQ(c.y1,a.y1*scale+shift);
  }
}

TEST(CohenTrace, ValidatesEveryCoordinateBeforeTrivialDecisions) {
  for(double bad:{std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()}) {
    for(std::size_t i=0;i<8;++i) {
      std::array<double,8> a{0,0,0,0,-1,-1,1,1};a[i]=bad;EXPECT_THROW(trace(a),std::invalid_argument);
      EXPECT_THROW(cohen_sutherland_clip(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7]),std::invalid_argument);
    }
    for(std::size_t i=0;i<6;++i) {
      std::array<double,6> a{0,0,-1,-1,1,1};a[i]=bad;
      EXPECT_THROW(cohen_sutherland_outcode(a[0],a[1],a[2],a[3],a[4],a[5]),std::invalid_argument);
    }
  }
  EXPECT_THROW(cohen_sutherland_clip(0,0,0,0,1,-1,-1,1),std::invalid_argument);
  EXPECT_THROW(cohen_sutherland_clip(0,0,0,0,-1,1,1,-1),std::invalid_argument);
}

TEST(CohenTrace, HugeDifferencesAndSubnormalCoordinatesStayFinite) {
  const auto maximum=std::numeric_limits<double>::max();
  if(std::numeric_limits<long double>::max_exponent>std::numeric_limits<double>::max_exponent) {
    const auto c=cohen_sutherland_clip(-maximum,1,maximum,1,-1,0,1,2);
    EXPECT_TRUE(c.visible);EXPECT_EQ(c.x0,-1);EXPECT_EQ(c.x1,1);EXPECT_EQ(c.y0,1);EXPECT_EQ(c.y1,1);
    const auto d=cohen_sutherland_clip(-maximum,-maximum,maximum,maximum,-1,-1,1,1);
    EXPECT_TRUE(d.visible);EXPECT_EQ(d.x0,-1);EXPECT_EQ(d.y0,-1);EXPECT_EQ(d.x1,1);EXPECT_EQ(d.y1,1);
    const auto anti=cohen_sutherland_clip(-maximum,maximum,maximum,-maximum,-1,-1,1,1);
    EXPECT_TRUE(anti.visible);EXPECT_EQ(anti.x0,-1);EXPECT_EQ(anti.y0,1);EXPECT_EQ(anti.x1,1);EXPECT_EQ(anti.y1,-1);
  } else EXPECT_THROW(cohen_sutherland_clip(-maximum,1,maximum,1,-1,0,1,2),std::overflow_error);
  const double tiny=std::numeric_limits<double>::denorm_min();
  const auto c=cohen_sutherland_clip(-tiny,0,tiny,0,0,0,tiny,tiny);
  EXPECT_TRUE(c.visible);EXPECT_EQ(c.x0,0);EXPECT_EQ(c.x1,tiny);
  const auto p=cohen_sutherland_clip(0,0,0,0,0,0,0,0);EXPECT_TRUE(p.visible);
}

TEST(CohenTrace, TerminalCountersAndOwnedRecordsAreActual) {
  auto inside=cohen_sutherland_clip_trace(0,0,1,1,0,0,1,1);
  ASSERT_EQ(inside.steps.size(),1);EXPECT_EQ(inside.intersections,0);EXPECT_EQ(inside.or_tests,1);EXPECT_EQ(inside.and_tests,0);
  const auto outside=cohen_sutherland_clip_trace(-2,0,-1,1,0,0,1,1);
  EXPECT_EQ(outside.intersections,0);EXPECT_EQ(outside.and_tests,1);
  auto crossed=cohen_sutherland_clip_trace(-1,.5,2,.5,0,0,1,1);
  ASSERT_EQ(crossed.steps.size(),3);crossed.steps[0].intersection_x=999;
  EXPECT_EQ(crossed.steps[1].x0,0);EXPECT_EQ(crossed.result.x0,0);EXPECT_EQ(crossed.result.x1,1);
}

TEST(CohenTrace, OneUlpCornerControlsUseNoHiddenTolerance) {
  const auto below=cohen_sutherland_clip(-1,0,1,std::nextafter(2.,0.),0,0,1,1);
  const auto touch=cohen_sutherland_clip(-1,0,1,2,0,0,1,1);
  const auto above=cohen_sutherland_clip(-1,0,1,std::nextafter(2.,3.),0,0,1,1);
  EXPECT_TRUE(below.visible);EXPECT_TRUE(touch.visible);EXPECT_FALSE(above.visible);
  EXPECT_EQ(touch.x0,0);EXPECT_EQ(touch.y0,1);EXPECT_EQ(touch.x1,0);EXPECT_EQ(touch.y1,1);
}

TEST(CohenTrace, ExtremeScaleThinSlabRetainsItsActualPointContact) {
  // Exact line: y=1+x/M. Its intersection with y=1, 0<=x<=1 is only (0,1).
  // Clipping to that supporting line before a rounded x cut keeps the point.
  const double magnitude=std::ldexp(1.,1000);
  const auto c=cohen_sutherland_clip(-magnitude,0,magnitude,2,0,1,1,1);
  EXPECT_TRUE(c.visible);EXPECT_EQ(c.x0,0);EXPECT_EQ(c.y0,1);EXPECT_EQ(c.x1,0);EXPECT_EQ(c.y1,1);
}
