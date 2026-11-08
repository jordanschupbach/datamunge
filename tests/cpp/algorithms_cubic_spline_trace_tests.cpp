#include <gtest/gtest.h>
#include <datamunge/algorithms/interpolation.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
namespace {
std::vector<double> grid(double lo,double hi,std::size_t count) {
  std::vector<double> x(count);for(std::size_t i=0;i<count;++i)x[i]=lo+(hi-lo)*i/(count-1);return x;
}
}
TEST(NaturalSpline, ThreeKnotExactSystemCoefficientsAndDerivatives) {
  const auto t=natural_cubic_spline_trace({0,1,2},{0,1,0},3);
  EXPECT_EQ(t.widths,(std::vector<double>{1,1}));EXPECT_EQ(t.secants,(std::vector<double>{1,-1}));
  EXPECT_EQ(t.lower,(std::vector<double>{0}));EXPECT_EQ(t.upper,(std::vector<double>{0}));
  EXPECT_EQ(t.diagonal,(std::vector<double>{4}));EXPECT_EQ(t.rhs,(std::vector<double>{-12}));
  EXPECT_EQ(t.pivots,t.diagonal);EXPECT_EQ(t.reduced_rhs,t.rhs);EXPECT_EQ(t.moments,(std::vector<double>{0,-3,0}));
  EXPECT_EQ(t.spline.a,(std::vector<double>{0,1}));EXPECT_EQ(t.spline.b,(std::vector<double>{1.5,0}));
  EXPECT_EQ(t.spline.c,(std::vector<double>{0,-1.5}));EXPECT_EQ(t.spline.d,(std::vector<double>{-.5,.5}));
  EXPECT_EQ(t.eliminations,0);EXPECT_EQ(t.back_substitutions,1);EXPECT_EQ(t.coefficient_intervals,2);
  const PreparedCubicSpline p(t.spline);EXPECT_EQ(p.evaluate(.5),.6875);EXPECT_EQ(p.evaluate(.5,1),1.125);
  EXPECT_EQ(p.evaluate(1,2),-3);EXPECT_EQ(p.evaluate(1,3),3);EXPECT_EQ(p.evaluate(std::nextafter(1.,0.),3),-3);
}
TEST(NaturalSpline, RandomNonuniformTraceReplayContinuityAndLinearity) {
  std::mt19937 rng(20261107);
  for(int trial=0;trial<1000;++trial) {
    const auto n=2+rng()%19;std::vector<double> x(n),y(n),affine(n);x[0]=-5;
    for(std::size_t i=0;i<n;++i) { if(i)x[i]=x[i-1]+1+rng()%4;y[i]=static_cast<int>(rng()%21)-10;affine[i]=2*y[i]+3; }
    const auto t=natural_cubic_spline_trace(x,y);const PreparedCubicSpline p(t.spline),q(natural_cubic_spline(x,affine));
    auto diagonal=t.diagonal,rhs=t.rhs;
    for(std::size_t i=1;i<diagonal.size();++i) {
      const auto w=t.lower[i]/diagonal[i-1];EXPECT_DOUBLE_EQ(w,t.multipliers[i]);
      diagonal[i]-=w*t.upper[i-1];rhs[i]-=w*rhs[i-1];
    }
    EXPECT_EQ(diagonal,t.pivots);EXPECT_EQ(rhs,t.reduced_rhs);
    EXPECT_EQ(t.eliminations,n>=3?n-3:0);EXPECT_EQ(t.back_substitutions,n-2);
    EXPECT_EQ(t.coefficient_intervals,n-1);EXPECT_EQ(t.moments.front(),0);EXPECT_EQ(t.moments.back(),0);
    for(std::size_t i=0;i<n;++i) EXPECT_NEAR(p.evaluate(x[i]),y[i],1e-11);
    for(std::size_t i=0;i+1<n;++i) {
      const auto h=x[i+1]-x[i];const auto& s=t.spline;
      if(i+2<n) {
        EXPECT_NEAR(s.b[i]+2*s.c[i]*h+3*s.d[i]*h*h,s.b[i+1],1e-11);
        EXPECT_NEAR(2*s.c[i]+6*s.d[i]*h,2*s.c[i+1],1e-11);
      }
      for(double fraction:{0.,.25,.5,.75,1.}) {
        const auto at=x[i]+fraction*h;EXPECT_NEAR(q.evaluate(at),2*p.evaluate(at)+3,1e-11);
      }
    }
    EXPECT_NEAR(p.evaluate(x.front(),2),0,1e-11);EXPECT_NEAR(p.evaluate(x.back(),2),0,1e-11);
  }
}
TEST(NaturalSpline, EmptySingletonTwoKnotsAndOwnedPreparation) {
  const PreparedCubicSpline empty(natural_cubic_spline({},{}));
  for(unsigned order=0;order<=3;++order)EXPECT_EQ(empty.evaluate(10,order),0);
  EXPECT_TRUE(empty.evaluate(std::vector<double>{}).empty());
  const auto t=natural_cubic_spline_trace({2},{7});EXPECT_EQ(t.moments,(std::vector<double>{0}));EXPECT_TRUE(t.widths.empty());
  const PreparedCubicSpline constant(t.spline);EXPECT_EQ(constant.evaluate(-100),7);EXPECT_EQ(constant.evaluate(100,1),0);
  auto record=natural_cubic_spline({0,2},{1,5});const PreparedCubicSpline line(record);record.a[0]=99;
  EXPECT_EQ(line.evaluate(3),7);EXPECT_EQ(line.evaluate(3,1),2);EXPECT_EQ(line.evaluate(3,2),0);
  EXPECT_EQ(line.coefficients().a[0],1);EXPECT_EQ(line.evaluate(std::vector<double>{3,1,0}),(std::vector<double>{7,3,1}));
}
TEST(NaturalSpline, OvershootPositivityAndNonlinearExtrapolationCounterexamples) {
  const PreparedCubicSpline p(natural_cubic_spline({0,1,2},{0,1,1}));
  EXPECT_EQ(p.evaluate(1.5),1.09375);EXPECT_EQ(p.evaluate(2,2),0);
  EXPECT_EQ(p.evaluate(2,1),-.25);EXPECT_EQ(p.evaluate(3),1);EXPECT_EQ(p.evaluate(3,2),1.5);
  const PreparedCubicSpline negative(natural_cubic_spline({0,1,2},{0,0,1}));EXPECT_EQ(negative.evaluate(.5),-.09375);
  const PreparedCubicSpline quadratic(natural_cubic_spline({0,1},{0,1}));EXPECT_EQ(quadratic.evaluate(.5),.5);EXPECT_NE(quadratic.evaluate(.5),.25);
}
TEST(NaturalSpline, FullFitValidationAndCompleteTraceCaps) {
  EXPECT_NO_THROW(natural_cubic_spline_trace({0,1,2},{0,1,0},3));
  EXPECT_THROW(natural_cubic_spline_trace({0,1,2},{0,1,0},2),std::length_error);
  EXPECT_NO_THROW(natural_cubic_spline_trace({0,1,2},{0,1,0},0));
  EXPECT_THROW(natural_cubic_spline_trace({0,1,1},{0,1,0},1),std::invalid_argument);
  EXPECT_THROW(natural_cubic_spline({0,1},{0}),std::invalid_argument);
  EXPECT_THROW(natural_cubic_spline({}, {1}),std::invalid_argument);
  EXPECT_THROW(natural_cubic_spline({1,0},{0,1}),std::invalid_argument);
  EXPECT_THROW(natural_cubic_spline({0,std::numeric_limits<double>::infinity()},{0,1}),std::invalid_argument);
  EXPECT_THROW(natural_cubic_spline({0},{std::numeric_limits<double>::quiet_NaN()}),std::invalid_argument);
}
TEST(NaturalSpline, PublicMutableRecordShapesAndFiniteQueriesValidate) {
  CubicSpline s; s.a={1};
  EXPECT_THROW(cubic_spline_eval(s,0),std::invalid_argument);
  EXPECT_THROW(PreparedCubicSpline{s},std::invalid_argument);
  s=natural_cubic_spline({0,1,2},{0,1,0});s.b.pop_back();
  EXPECT_THROW(cubic_spline_eval(s,1),std::invalid_argument);
  s=natural_cubic_spline({0,1,2},{0,1,0});s.c[1]=std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(PreparedCubicSpline{s},std::invalid_argument);
  const PreparedCubicSpline p(natural_cubic_spline({0,1},{0,1}));
  EXPECT_THROW(p.evaluate(std::numeric_limits<double>::infinity()),std::invalid_argument);
  EXPECT_THROW(p.evaluate(0,4),std::invalid_argument);
  EXPECT_THROW(p.evaluate(std::vector<double>{},4),std::invalid_argument);
  EXPECT_THROW(p.evaluate(std::vector<double>{0,std::numeric_limits<double>::quiet_NaN()}),std::invalid_argument);
  EXPECT_THROW(cubic_spline_eval({},std::numeric_limits<double>::quiet_NaN()),std::invalid_argument);
}
TEST(NaturalSpline, PreparedRecordsAreStructuralNotContinuityCertificates) {
  CubicSpline s{{0,1,2},{0,0},{0,0},{0,0},{1,2}};const PreparedCubicSpline p(s);
  EXPECT_EQ(p.evaluate(1,3),12);EXPECT_EQ(p.evaluate(std::nextafter(1.,0.),3),6);
  EXPECT_EQ(p.evaluate(2,3),12);EXPECT_EQ(p.evaluate(1),0);
}
TEST(NaturalSpline, OriginalRungeGridAndPolynomialComparison) {
  const auto f=[](double x){return 1/(1+25*x*x);};const auto x=grid(-1,1,11),query=grid(-1,1,2001);
  std::vector<double> y;for(const auto v:x)y.push_back(f(v));
  const auto s=natural_cubic_spline(x,y);const PreparedCubicSpline p(s);
  const auto spline=interpolation_error_scan([&](double v){return p.evaluate(v);},f,query);
  const auto polynomial=interpolation_error_scan([&](double v){return lagrange_interpolate(x,y,v);},f,query);
  EXPECT_NEAR(spline.max_abs_error,.0219738,1e-7);EXPECT_NEAR(polynomial.max_abs_error,1.91564,1e-5);
  EXPECT_GT(polynomial.max_abs_error/spline.max_abs_error,80);
  for(const auto v:x)EXPECT_NEAR(p.evaluate(v),f(v),1e-15);
  for(std::size_t i=0;i<query.size();++i)EXPECT_EQ(cubic_spline_eval(s,query[i]),spline.predictions[i]);
}
TEST(NaturalSpline, RefinementDistinguishesCompatibleAndIncompatibleEndpoints) {
  const auto query=grid(0,1,8193);double previous_sin=0,previous_quadratic=0;
  for(std::size_t intervals:{4,8,16,32,64,128}) {
    const auto x=grid(0,1,intervals+1);std::vector<double> ys,yq;
    for(auto v:x) {ys.push_back(std::sin(std::acos(-1.)*v));yq.push_back(v*v);}
    const PreparedCubicSpline ps(natural_cubic_spline(x,ys)),pq(natural_cubic_spline(x,yq));
    const auto es=interpolation_error_scan([&](double v){return ps.evaluate(v);},[](double v){return std::sin(std::acos(-1.)*v);},query);
    const auto eq=interpolation_error_scan([&](double v){return pq.evaluate(v);},[](double v){return v*v;},query);
    if(previous_sin && intervals>=16) { EXPECT_GT(previous_sin/es.max_abs_error,15.5);EXPECT_LT(previous_sin/es.max_abs_error,17); }
    if(previous_quadratic && intervals>=16) { EXPECT_GT(previous_quadratic/eq.max_abs_error,3.98);EXPECT_LT(previous_quadratic/eq.max_abs_error,4.02); }
    previous_sin=es.max_abs_error;previous_quadratic=eq.max_abs_error;
  }
}
TEST(NaturalSpline, ErrorCensusOrderTiesEmptyCapsAndOwnership) {
  std::vector<int> order;auto m=[&](double x){order.push_back(1);return x;};auto f=[&](double){order.push_back(2);return 0.;};
  auto e=interpolation_error_scan(m,f,{-2,0,2},3);
  EXPECT_EQ(order,(std::vector<int>{1,2,1,2,1,2}));EXPECT_EQ(e.worst_index,0);EXPECT_EQ(e.max_abs_error,2);
  EXPECT_NEAR(e.rms_error,std::sqrt(8./3),1e-15);e.predictions[0]=99;EXPECT_EQ(e.absolute_errors[0],2);
  const auto zero=interpolation_error_scan(m,m,{1,2});EXPECT_EQ(zero.worst_index,0);EXPECT_EQ(zero.max_abs_error,0);
  const auto empty=interpolation_error_scan(m,f,{});EXPECT_FALSE(empty.has_samples);EXPECT_EQ(empty.worst_index,0);EXPECT_EQ(empty.rms_error,0);
  order.clear();
  EXPECT_THROW(interpolation_error_scan(m,f,{1,2},1),std::length_error);EXPECT_TRUE(order.empty());
  EXPECT_THROW(interpolation_error_scan(m,f,{1,std::numeric_limits<double>::infinity()},1),std::invalid_argument);EXPECT_TRUE(order.empty());
  EXPECT_NO_THROW(interpolation_error_scan(m,f,{1,2},0));
  EXPECT_THROW(interpolation_error_scan({},f,{}),std::invalid_argument);
  EXPECT_THROW(interpolation_error_scan([](double){return std::numeric_limits<double>::quiet_NaN();},f,{1}),std::invalid_argument);
  EXPECT_THROW(interpolation_error_scan([](double)->double{throw std::runtime_error("callback");},f,{1}),std::runtime_error);
}
TEST(NaturalSpline, CheckedArithmeticAndScaledRms) {
  const double high=std::numeric_limits<double>::max();
  EXPECT_THROW(natural_cubic_spline({-high,high},{0,1}),std::overflow_error);
  EXPECT_THROW(natural_cubic_spline({0,1},{-high,high}),std::overflow_error);
  const PreparedCubicSpline p(natural_cubic_spline({0,1,2},{0,1,0}));
  EXPECT_THROW(p.evaluate(high),std::overflow_error);
  EXPECT_THROW(interpolation_error_scan([=](double){return high;},[=](double){return -high;},{0}),std::overflow_error);
  const auto error=interpolation_error_scan([=](double x){return high*x;},[](double){return 0.;},{.25,.5});
  EXPECT_TRUE(std::isfinite(error.rms_error));EXPECT_DOUBLE_EQ(error.max_abs_error,high/2);
  EXPECT_NEAR(error.rms_error/(high*std::sqrt(.15625)),1,3e-16);
}
