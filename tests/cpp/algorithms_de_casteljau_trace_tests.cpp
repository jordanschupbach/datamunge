#include <gtest/gtest.h>
#include <datamunge/algorithms/interpolation.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
namespace {
void near(PlanarPoint a,PlanarPoint b,double tolerance=1e-11) {EXPECT_NEAR(a.x,b.x,tolerance);EXPECT_NEAR(a.y,b.y,tolerance);}
__int128 power(int a,std::size_t exponent) {__int128 out=1;while(exponent--)out*=a;return out;}
long double exact_component(const std::vector<PlanarPoint>& p,std::size_t start,std::size_t degree,int j,bool y) {
  __int128 sum=0,binomial=1;
  for(std::size_t i=0;i<=degree;++i) {
    const int coordinate=static_cast<int>(y?p[start+i].y:p[start+i].x);
    sum+=binomial*power(j,i)*power(8-j,degree-i)*coordinate;
    if(i<degree)binomial=binomial*(degree-i)/(i+1);
  }
  return static_cast<long double>(sum)/static_cast<long double>(power(8,degree));
}
long double bernstein(const std::vector<PlanarPoint>& p,long double t,bool y) {
  long double result=0,binomial=1;const auto n=p.size()-1;
  for(std::size_t i=0;i<=n;++i) {result+=binomial*std::pow(t,static_cast<int>(i))*std::pow(1-t,static_cast<int>(n-i))*(y?p[i].y:p[i].x);if(i<n)binomial=binomial*(n-i)/(i+1);}
  return result;
}
}
TEST(CasteljauTrace, OriginalCompleteTriangleAndEndpoints) {
  const std::vector<PlanarPoint> c{{0,0},{1,2},{3,3},{4,0}};const auto t=de_casteljau_trace(c,.5);
  const std::vector<std::vector<PlanarPoint>> expected{c,{{.5,1},{2,2.5},{3.5,1.5}},{{1.25,1.75},{2.75,2}},{{2,1.875}}};
  ASSERT_EQ(t.levels.size(),4);EXPECT_EQ(t.point_blends,6);EXPECT_EQ(t.shape.stored_points,10);near(t.point,{2,1.875},0);
  for(std::size_t r=0;r<4;++r)for(std::size_t i=0;i<expected[r].size();++i)near(t.levels[r][i],expected[r][i],0);
  near(de_casteljau(c,0),c.front(),0);near(de_casteljau(c,1),c.back(),0);
}
TEST(CasteljauTrace, ExactDyadicBernsteinChecksEveryActualRow) {
  std::mt19937 rng(1959);
  for(int trial=0;trial<300;++trial) {
    const auto n=rng()%13;std::vector<PlanarPoint> c(n+1);for(auto& p:c)p={double(int(rng()%257)-128),double(int(rng()%257)-128)};
    for(int j=0;j<=8;++j) {
      const auto t=de_casteljau_trace(c,j/8.);EXPECT_EQ(t.point_blends,n*(n+1)/2);ASSERT_EQ(t.levels.size(),n+1);
      for(std::size_t r=0;r<=n;++r) {
        ASSERT_EQ(t.levels[r].size(),n+1-r);
        for(std::size_t i=0;i<t.levels[r].size();++i) {EXPECT_DOUBLE_EQ(t.levels[r][i].x,static_cast<double>(exact_component(c,i,r,j,false)));EXPECT_DOUBLE_EQ(t.levels[r][i].y,static_cast<double>(exact_component(c,i,r,j,true)));}
      }
      near(t.point,de_casteljau(c,j/8.),0);
    }
  }
}
TEST(CasteljauTrace, SubdivisionIdentitiesAndActualDiagonals) {
  std::mt19937 rng(1985);
  for(int trial=0;trial<100;++trial) {
    std::vector<PlanarPoint> c(1+rng()%10);for(auto& p:c)p={double(int(rng()%101)-50),double(int(rng()%101)-50)};
    for(double s:{0.,.25,.5,.75,1.}) {
      const auto split=de_casteljau_subdivide(c,s);const auto trace=de_casteljau_trace(c,s);ASSERT_EQ(split.left.size(),c.size());ASSERT_EQ(split.right.size(),c.size());EXPECT_EQ(split.point_blends,trace.point_blends);
      for(std::size_t r=0;r<c.size();++r) {near(split.left[r],trace.levels[r].front(),0);near(split.right[c.size()-1-r],trace.levels[r].back(),0);}
      near(split.left.back(),split.right.front(),0);
      for(int j=0;j<=16;++j) {const double u=j/16.;near(de_casteljau(split.left,u),de_casteljau(c,s*u));near(de_casteljau(split.right,u),de_casteljau(c,s+(1-s)*u));}
    }
  }
}
TEST(CasteljauTrace, RandomBernsteinAffineAndHullControls) {
  std::mt19937 rng(2002);
  for(int trial=0;trial<1000;++trial) {
    std::vector<PlanarPoint> c(1+rng()%13),affine;
    for(auto& p:c) {p={double(int(rng()%201)-100)/7,double(int(rng()%201)-100)/7};affine.push_back({2*p.x-p.y+3,p.x+3*p.y-2});}
    const double t=(rng()%10001)/10000.;const auto p=de_casteljau(c,t),a=de_casteljau(affine,t);
    near(p,{double(bernstein(c,t,false)),double(bernstein(c,t,true))});near(a,{2*p.x-p.y+3,p.x+3*p.y-2});
    double xmin=c[0].x,xmax=xmin,ymin=c[0].y,ymax=ymin;for(const auto& q:c) {xmin=std::min(xmin,q.x);xmax=std::max(xmax,q.x);ymin=std::min(ymin,q.y);ymax=std::max(ymax,q.y);}
    EXPECT_GE(p.x,xmin-1e-12);EXPECT_LE(p.x,xmax+1e-12);EXPECT_GE(p.y,ymin-1e-12);EXPECT_LE(p.y,ymax+1e-12);
    const double outside=trial%2?-1.:2.;const auto e=de_casteljau(c,outside);near(e,{double(bernstein(c,outside,false)),double(bernstein(c,outside,true))},1e-6);
  }
}
TEST(CasteljauTrace, UniformGridEndpointsReversalAndDegenerateIntervals) {
  const std::vector<PlanarPoint> c{{0,0},{1,2},{3,3},{4,0}};const auto p=de_casteljau_sample(c);ASSERT_EQ(p.size(),101);near(p.front(),c.front(),0);near(p.back(),c.back(),0);
  for(std::size_t i=0;i<p.size();++i)near(p[i],de_casteljau(c,i/100.),0);
  const auto rev=de_casteljau_sample(c,100,1,0);for(std::size_t i=0;i<p.size();++i)near(p[i],rev[100-i]);
  for(const auto& q:de_casteljau_sample(c,8,.25,.25))near(q,de_casteljau(c,.25),0);
  const auto outside=de_casteljau_sample(c,4,-1,2);near(outside.front(),de_casteljau(c,-1),0);near(outside.back(),de_casteljau(c,2),0);
  const double tiny=std::numeric_limits<double>::denorm_min();const auto tinygrid=de_casteljau_sample({{0,0},{1,1}},4,tiny,2*tiny);near(tinygrid.front(),{tiny,tiny},0);near(tinygrid.back(),{2*tiny,2*tiny},0);
}
TEST(CasteljauTrace, EmptySingletonOwnershipAndShapeWork) {
  near(de_casteljau({},2),{},0);const auto e=de_casteljau_trace({},-1,1);EXPECT_TRUE(e.levels.empty());EXPECT_EQ(e.point_blends,0);
  for(const auto& p:de_casteljau_sample({},3))near(p,{},0);
  const auto single=de_casteljau_subdivide({{2,3}},.25,2);near(single.left[0],{2,3},0);near(single.right[0],{2,3},0);EXPECT_EQ(single.point_blends,0);
  std::vector<PlanarPoint> c{{0,0},{2,4},{4,0}};auto t=de_casteljau_trace(c,.25);const auto split=de_casteljau_subdivide(c,.25);c[0]={99,99};near(t.levels[0][0],{},0);near(split.left[0],{},0);
  for(std::size_t degree=0;degree<=64;++degree) {const auto m=degree+1;const auto a=de_casteljau_trace(std::vector<PlanarPoint>(m,{1,2}),.5);std::size_t count=0;for(const auto& row:a.levels)count+=row.size();EXPECT_EQ(count,m*(m+1)/2);EXPECT_EQ(a.point_blends,degree*m/2);EXPECT_EQ(a.shape.coordinate_blends,2*a.point_blends);near(a.point,{1,2},0);}
  const auto zero=de_casteljau_shape(0);EXPECT_EQ(zero.stored_points,0);EXPECT_EQ(de_casteljau_shape(1).stored_points,1);
}
TEST(CasteljauTrace, CompleteTriangleOutputAndWorkCaps) {
  const std::vector<PlanarPoint> c{{0,0},{1,2},{3,3},{4,0}};
  EXPECT_NO_THROW(de_casteljau_trace(c,.5,10));
  EXPECT_THROW(de_casteljau_trace(c,.5,9),std::length_error);
  EXPECT_NO_THROW(de_casteljau_subdivide(c,.5,8));
  EXPECT_THROW(de_casteljau_subdivide(c,.5,7),std::length_error);
  EXPECT_NO_THROW(de_casteljau_sample(c,4,0,1,5,30));
  EXPECT_THROW(de_casteljau_sample(c,4,0,1,4,30),std::length_error);
  EXPECT_THROW(de_casteljau_sample(c,4,0,1,5,29),std::length_error);
  EXPECT_NO_THROW(de_casteljau_trace(c,.5,0));
  EXPECT_NO_THROW(de_casteljau_subdivide(c,.5,0));
  EXPECT_NO_THROW(de_casteljau_sample(c,4,0,1,0,0));
  EXPECT_THROW(de_casteljau_shape(std::numeric_limits<std::size_t>::max()),std::length_error);
  EXPECT_THROW(de_casteljau_shape(std::numeric_limits<std::size_t>::max()/2),std::length_error);
  EXPECT_THROW(de_casteljau_sample(c,std::numeric_limits<std::size_t>::max(),0,1,0,0),std::length_error);
  EXPECT_THROW(de_casteljau_sample(c,std::numeric_limits<std::size_t>::max()/2,0,1,0,0),std::length_error);
}
TEST(CasteljauTrace, FiniteValidationPrecedesCapsAndOverflowIsExplicit) {
  const auto inf=std::numeric_limits<double>::infinity(),nan=std::numeric_limits<double>::quiet_NaN(),big=std::numeric_limits<double>::max();
  EXPECT_THROW(de_casteljau({},nan),std::invalid_argument);
  EXPECT_THROW(de_casteljau({{inf,0}},0),std::invalid_argument);
  EXPECT_THROW(de_casteljau_trace({{0,0},{nan,0}},.5,1),std::invalid_argument);
  EXPECT_THROW(de_casteljau_subdivide({},.5),std::invalid_argument);
  EXPECT_THROW(de_casteljau_subdivide({{1,2}},-1),std::invalid_argument);
  EXPECT_THROW(de_casteljau_subdivide({{1,2}},2),std::invalid_argument);
  EXPECT_THROW(de_casteljau_sample({{0,0},{1,inf}},100,0,1,1,1),std::invalid_argument);
  EXPECT_THROW(de_casteljau_sample({},0),std::invalid_argument);
  EXPECT_THROW(de_casteljau_sample({},1,0,inf),std::invalid_argument);
  EXPECT_THROW(de_casteljau({{big,0},{big,0}},2),std::overflow_error);
  EXPECT_THROW(de_casteljau_trace({{big,0},{big,0}},2),std::overflow_error);
  EXPECT_THROW(de_casteljau_sample({{big,0},{big,0}},1,0,2),std::overflow_error);
  near(de_casteljau({{big,0}},2),{big,0},0);near(de_casteljau({{big,0},{big,0}},.5),{big,0},0);
}
TEST(CasteljauTrace, ControlledSensitivityAndCancellation) {
  const std::vector<PlanarPoint> c{{0,0},{1,2},{3,3},{4,0}};auto perturbed=c;perturbed[1].y+=1;
  for(int j=0;j<=100;++j) {const double t=j/100.;const auto a=de_casteljau(c,t),b=de_casteljau(perturbed,t);EXPECT_NEAR(b.y-a.y,3*t*(1-t)*(1-t),1e-14);EXPECT_LE(std::abs(b.y-a.y),1+1e-14);}
  near(de_casteljau({{1e16,0},{-1e16,0}},.5),{},0);near(de_casteljau({{1e16,0},{-1e16+2,0}},.5),{1,0},0);
}
