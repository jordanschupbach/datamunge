#include <datamunge/algorithms/bilinear_interpolation.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

using datamunge::algorithms::bilinear_interpolate;

TEST(Bilinear,HandCheckedUnevenCell) {
  EXPECT_DOUBLE_EQ(bilinear_interpolate({0,2},{0,4},{0,4,2,10},.5,3),4.25);
}

TEST(Bilinear,AllNodesAndNonuniformBilinearPolynomialIncludingExtrapolation) {
  const std::vector<double> xs{-2,0,3},ys{-3,-1,4,5};
  const auto f=[](double x,double y){return 1-2*x+.5*y+.3*x*y;};
  std::vector<double> z;
  for (double x:xs) for (double y:ys) z.push_back(f(x,y));
  for (double x:xs) for (double y:ys) EXPECT_NEAR(bilinear_interpolate(xs,ys,z,x,y),f(x,y),1e-13);
  for (double x:{-5.,-1.,.4,2.,4.}) for (double y:{-6.,-2.,0.,4.5,8.})
    EXPECT_NEAR(bilinear_interpolate(xs,ys,z,x,y),f(x,y),1e-12);
}

TEST(Bilinear,ExtrapolatesRatherThanClamping) {
  EXPECT_DOUBLE_EQ(bilinear_interpolate({0,1},{0,1},{0,0,1,1},2,.5),2);
  EXPECT_DOUBLE_EQ(bilinear_interpolate({0,1},{0,1},{0,1,0,1},.5,-1),-1);
}

TEST(Bilinear,QuadraticMidpointErrorEqualsHalfSquaredSpacing) {
  for (int m:{1,2,4,8,16}) {
    std::vector<double> x,z;
    for (int i=0;i<=m;++i) x.push_back(double(i)/m);
    for (double a:x) for (double b:x) z.push_back(a*a+b*b);
    for (int i=0;i<m;++i) for (int j=0;j<m;++j) {
      const double a=(i+.5)/m,b=(j+.5)/m;
      EXPECT_NEAR(bilinear_interpolate(x,x,z,a,b)-(a*a+b*b),.5/(m*m),1e-14);
    }
  }
}

TEST(Bilinear,RandomCellsMatchIndependentCornerWeightsAndConvexHull) {
  std::mt19937 rng(20261106);
  std::uniform_real_distribution<double> value(-100,100),position(0,1);
  for (int k=0;k<2000;++k) {
    const double tx=position(rng),ty=position(rng);
    const std::vector<double> z{value(rng),value(rng),value(rng),value(rng)};
    const double expected=(1-tx)*(1-ty)*z[0]+(1-tx)*ty*z[1]+tx*(1-ty)*z[2]+tx*ty*z[3];
    const double actual=bilinear_interpolate({-2,3},{-1,7},z,-2+5*tx,-1+8*ty);
    EXPECT_NEAR(actual,expected,1e-12);
    EXPECT_GE(actual,*std::min_element(z.begin(),z.end())-1e-12);
    EXPECT_LE(actual,*std::max_element(z.begin(),z.end())+1e-12);
  }
}

TEST(Bilinear,RejectsMalformedAxesAndGrid) {
  for (const auto& axis:std::vector<std::vector<double>>{{},{0},{0,0},{1,0},{0,2,1}}) {
    EXPECT_THROW(bilinear_interpolate(axis,{0,1},{0,1,2,3},0,0),std::invalid_argument);
    EXPECT_THROW(bilinear_interpolate({0,1},axis,{0,1,2,3},0,0),std::invalid_argument);
  }
  EXPECT_THROW(bilinear_interpolate({0,1},{0,1},{0,1,2},0,0),std::invalid_argument);
}

TEST(Bilinear,RejectsNonfiniteAxesSamplesAndQueries) {
  for (double bad:{std::numeric_limits<double>::infinity(),
                   -std::numeric_limits<double>::infinity(),
                   std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(bilinear_interpolate({0,bad},{0,1},{0,1,2,3},0,0),std::invalid_argument);
    EXPECT_THROW(bilinear_interpolate({0,1},{0,bad},{0,1,2,3},0,0),std::invalid_argument);
    EXPECT_THROW(bilinear_interpolate({0,1},{0,1},{0,1,2,bad},0,0),std::invalid_argument);
    EXPECT_THROW(bilinear_interpolate({0,1},{0,1},{0,1,2,3},bad,0),std::invalid_argument);
    EXPECT_THROW(bilinear_interpolate({0,1},{0,1},{0,1,2,3},0,bad),std::invalid_argument);
  }
}

TEST(Bilinear,StableBlendAvoidsOverflowAtOppositeLargeValues) {
  const double big=std::numeric_limits<double>::max();
  EXPECT_DOUBLE_EQ(bilinear_interpolate({0,1},{0,1},{-big,-big,big,big},.5,.5),0);
  EXPECT_THROW(bilinear_interpolate({0,1},{0,1},{0,0,big,big},2,.5),std::overflow_error);
}
