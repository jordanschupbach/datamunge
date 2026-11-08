#include <gtest/gtest.h>
#include <datamunge/algorithms/crank_nicolson.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;

TEST(CrankNicolsonTrace, OnePointExactZeroAndAlternatingSteps) {
  const auto annihilate=crank_nicolson_heat_trace([](double){return 1.;},1,1,1,.25,1,7);
  ASSERT_EQ(annihilate.states.size(),2u);EXPECT_EQ(annihilate.grid.r,.5);
  EXPECT_EQ(annihilate.pivots,(std::vector<double>{2}));
  EXPECT_EQ(annihilate.states[1].rhs,(std::vector<double>{0}));
  EXPECT_EQ(annihilate.states[1].profile,(std::vector<double>{0,0,0}));
  const auto t=crank_nicolson_heat_trace([](double){return 1.;},1,1,1,1,4);
  EXPECT_EQ(t.factorizations,1u);EXPECT_EQ(t.solves,4u);EXPECT_EQ(t.rhs_entries,4u);
  for(int i=0;i<=4;++i) {
    EXPECT_NEAR(t.states[i].profile[1],std::pow(-.6,i),1e-15);
    EXPECT_EQ(t.states[i].profile.front(),0);EXPECT_EQ(t.states[i].profile.back(),0);
    EXPECT_NEAR(t.states[i].rms,std::fabs(t.states[i].profile[1]),1e-15);
    EXPECT_LT(t.states[i].scaled_residual,1e-15);
  }
}
TEST(CrankNicolsonTrace, TwoPointHandSolveAndActualFactors) {
  int calls=0;
  const auto t=crank_nicolson_heat_trace([&](double){return ++calls==1?1.:0.;},1,1,2,1./9,1);
  ASSERT_EQ(calls,2);ASSERT_EQ(t.pivots.size(),2u);
  EXPECT_NEAR(t.grid.r,.5,1e-15);EXPECT_NEAR(t.pivots[0],2,1e-15);
  EXPECT_NEAR(t.multipliers[1],-.25,1e-15);EXPECT_NEAR(t.pivots[1],1.875,1e-15);
  EXPECT_NEAR(t.states[1].rhs[0],0,1e-15);EXPECT_NEAR(t.states[1].rhs[1],.5,1e-15);
  EXPECT_NEAR(t.states[1].profile[1],1./15,1e-15);EXPECT_NEAR(t.states[1].profile[2],4./15,1e-15);
}
TEST(CrankNicolsonTrace, RandomDenseCholeskyOracleAndActualRecurrence) {
  std::mt19937 rng(20261107);
  for(int trial=0;trial<500;++trial) {
    const int n=1+rng()%12,steps=1+rng()%6;const double dt=std::pow(10.,-5.+(rng()%700)/100.);
    std::vector<double> initial(n);for(auto& v:initial)v=(static_cast<int>(rng()%2001)-1000)/100.;
    int calls=0;const auto t=crank_nicolson_heat_trace([&](double){return initial[calls++];},1,1,n,dt,steps);
    ASSERT_EQ(calls,n);const long double r=t.grid.r,diag=1+2*r;
    std::vector<std::vector<long double>> lower(n,std::vector<long double>(n,0));
    for(int i=0;i<n;++i)for(int j=0;j<=i;++j) {
      long double v=i==j?diag:(i==j+1?-r:0);
      for(int k=0;k<j;++k)v-=lower[i][k]*lower[j][k];
      lower[i][j]=i==j?std::sqrt(v):v/lower[j][j];
    }
    std::vector<long double> previous(initial.begin(),initial.end());
    for(int s=1;s<=steps;++s) {
      std::vector<long double> rhs(n),y(n),x(n);
      for(int i=0;i<n;++i) {
        rhs[i]=(1-2*r)*previous[i]+(i?r*previous[i-1]:0)+(i+1<n?r*previous[i+1]:0);
        EXPECT_NEAR(t.states[s].rhs[i],static_cast<double>(rhs[i]),2e-9*std::max(1.L,std::fabs(rhs[i])));
      }
      for(int i=0;i<n;++i) { auto v=rhs[i];for(int j=0;j<i;++j)v-=lower[i][j]*y[j];y[i]=v/lower[i][i]; }
      for(int i=n;i-- >0;) { auto v=y[i];for(int j=i+1;j<n;++j)v-=lower[j][i]*x[j];x[i]=v/lower[i][i]; }
      for(int i=0;i<n;++i)EXPECT_NEAR(t.states[s].profile[i+1],static_cast<double>(x[i]),2e-10);
      EXPECT_LE(t.states[s].rms,t.states[s-1].rms+1e-10);
      EXPECT_LT(t.states[s].scaled_residual,2e-15);
      previous=x;
    }
  }
}
TEST(CrankNicolsonTrace, OriginalSineFixtureAndCorrectCentralIndex) {
  const double pi=std::acos(-1.);auto u0=[=](double x){return std::sin(pi*x);};
  for(const int steps:{0,50,100}) {
    const auto reference=heat_sine_mode(1,1,60,1,.001,steps);
    const auto t=crank_nicolson_heat_trace(u0,1,1,60,.001,steps);
    const auto u=crank_nicolson_heat(u0,1,1,60,.001,steps);
    EXPECT_EQ(u,t.states.back().profile);EXPECT_EQ(t.grid.final_time,steps*.001);
    EXPECT_LT(heat_profile_error(u,reference.discrete),2e-14);
    EXPECT_LT(heat_profile_error(u,reference.continuous),8e-5);
    EXPECT_DOUBLE_EQ(u[30],t.states.back().profile[30]);
    EXPECT_NEAR(u[30],u[31],1e-14);
  }
}
TEST(CrankNicolsonTrace, EveryModeMatchesIndependentSpectralControl) {
  for(const int n:{1,2,3,7,16})for(int mode=1;mode<=n;++mode)for(const double dt:{0.,.0001,.01,1.}) {
    const auto m=heat_sine_mode(2,.7,n,mode,dt,5);
    std::size_t sampled=1;
    const auto t=crank_nicolson_heat_trace([&](double){return m.initial[sampled++];},2,.7,n,dt,5);
    EXPECT_LT(heat_profile_error(t.states.back().profile,m.discrete),3e-14);
    EXPECT_GE(m.amplification,-1);EXPECT_LE(m.amplification,1);
    EXPECT_LE(m.eigenvalue,0);
  }
}
TEST(CrankNicolsonTrace, SeparateSecondOrderSpatialAndTemporalControls) {
  double previous=0;
  for(int n:{15,31,63,127}) {
    const auto m=heat_sine_mode(1,1,n,1,.001,100);
    const double error=heat_profile_error(m.semidiscrete,m.continuous);
    if(previous)EXPECT_NEAR(previous/error,4,0.025);previous=error;
  }
  previous=0;
  for(int steps:{10,20,40,80}) {
    const double dt=.1/steps;const auto m=heat_sine_mode(1,1,31,1,dt,steps);
    const auto u=crank_nicolson_heat([](double x){return std::sin(std::acos(-1.)*x);},1,1,31,dt,steps);
    const double error=heat_profile_error(u,m.semidiscrete);
    if(previous)EXPECT_NEAR(previous/error,4,0.012);previous=error;
  }
}
TEST(CrankNicolsonTrace, ZeroStepsZeroDiffusionAndZeroTime) {
  for(const auto parameters:std::vector<std::pair<double,double>>{{0,.1},{1,0}}) {
    std::vector<double> sampled;const auto t=crank_nicolson_heat_trace([&](double x){sampled.push_back(x);return x+1;},1,parameters.first,3,parameters.second,4);
    EXPECT_EQ(sampled,(std::vector<double>{.25,.5,.75}));
    for(const auto& state:t.states)EXPECT_EQ(state.profile,(std::vector<double>{0,1.25,1.5,1.75,0}));
  }
  const auto t=crank_nicolson_heat_trace([](double){return 2.;},1,1,2,.1,0,4);
  EXPECT_EQ(t.states.size(),1u);EXPECT_TRUE(t.states[0].rhs.empty());EXPECT_TRUE(t.pivots.empty());
  EXPECT_EQ(t.factorizations,0u);EXPECT_EQ(t.solves,0u);EXPECT_EQ(t.rhs_entries,0u);
}
TEST(CrankNicolsonTrace, CompleteCapBeforeCallbacksAndFullScalarChecks) {
  int calls=0;auto callback=[&](double){++calls;return 1.;};
  EXPECT_THROW(crank_nicolson_heat_trace(callback,1,1,1,1,2,10),std::length_error);EXPECT_EQ(calls,0);
  EXPECT_NO_THROW(crank_nicolson_heat_trace(callback,1,1,1,1,2,11));EXPECT_EQ(calls,1);
  EXPECT_NO_THROW(crank_nicolson_heat_trace(callback,1,1,1,1,2,0));
  for(const int n:{0,-1})EXPECT_THROW(crank_nicolson_heat(callback,1,1,n,.1,1),std::invalid_argument);
  EXPECT_THROW(crank_nicolson_heat_trace(callback,1,-1,1,.1,2,1),std::invalid_argument);
  EXPECT_THROW(crank_nicolson_heat(callback,1,1,1,-.1,1),std::invalid_argument);
  EXPECT_THROW(crank_nicolson_heat(callback,1,1,1,.1,-1),std::invalid_argument);
  EXPECT_THROW(crank_nicolson_heat({},1,1,1,.1,1),std::invalid_argument);
  for(const double v:{0.,-1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()})
    EXPECT_THROW(crank_nicolson_heat(callback,v,1,1,.1,1),std::invalid_argument);
  EXPECT_THROW(crank_nicolson_heat([](double){return std::numeric_limits<double>::quiet_NaN();},1,1,1,.1,0),std::invalid_argument);
  EXPECT_THROW(crank_nicolson_heat([](double)->double{throw std::runtime_error("callback");},1,1,1,.1,0),std::runtime_error);
}
TEST(CrankNicolsonTrace, CheckedDerivedValuesAndArithmetic) {
  const auto high=std::numeric_limits<double>::max(),low=std::numeric_limits<double>::denorm_min();
  auto unit=[](double){return 1.;};
  EXPECT_THROW(crank_nicolson_heat(unit,low,1,1,1,0),std::overflow_error);
  EXPECT_THROW(crank_nicolson_heat(unit,18*low,0,10,0,0),std::overflow_error);
  EXPECT_THROW(crank_nicolson_heat(unit,1,1,1,high,0),std::overflow_error);
  EXPECT_THROW(crank_nicolson_heat(unit,1,low,1,low,0),std::overflow_error);
  EXPECT_THROW(crank_nicolson_heat(unit,1,0,1,high,2),std::overflow_error);
  EXPECT_THROW(crank_nicolson_heat([=](double){return high;},1,1,1,1,1),std::overflow_error);
  EXPECT_THROW(heat_sine_mode(1,1,3,0,.1,1),std::invalid_argument);
  EXPECT_THROW(heat_sine_mode(1,1,3,4,.1,1),std::invalid_argument);
  EXPECT_THROW(heat_sine_mode(1,high,1,1,0,0),std::overflow_error);
}
TEST(CrankNicolsonTrace, ErrorShapesFiniteValidationAndOwnedStorage) {
  EXPECT_EQ(heat_profile_error({},{}),0);EXPECT_DOUBLE_EQ(heat_profile_error({1,2},{1,2.5}),.5);
  EXPECT_THROW(heat_profile_error({0},{}),std::invalid_argument);
  EXPECT_THROW(heat_profile_error({std::numeric_limits<double>::infinity()},{0}),std::invalid_argument);
  EXPECT_THROW(heat_profile_error({std::numeric_limits<double>::max()},{-std::numeric_limits<double>::max()}),std::overflow_error);
  auto t=crank_nicolson_heat_trace([](double){return 1.;},1,0,2,.1,2);
  t.states[0].profile[1]=77;t.states[1].rhs[0]=99;t.pivots[0]=22;
  EXPECT_EQ(t.states[2].profile[1],1);EXPECT_EQ(t.states[2].rhs[0],1);
  EXPECT_EQ(crank_nicolson_heat_trace([](double){return 1.;},1,0,2,.1,2).pivots[0],1);
}
TEST(CrankNicolsonTrace, ScaledRmsSupportsVeryLargeFiniteIdentityProfiles) {
  const auto high=std::numeric_limits<double>::max();int sampled=0;
  const auto t=crank_nicolson_heat_trace([&](double){return high/++sampled;},1,0,3,1,1);
  ASSERT_EQ(t.states.size(),2u);
  EXPECT_EQ(t.states[0].profile,t.states[1].profile);
  const long double reference=static_cast<long double>(high)*std::sqrt((1.L+.25L+1.L/9)/3);
  EXPECT_NEAR(t.states[1].rms/static_cast<double>(reference),1,3e-16);
  EXPECT_EQ(t.states[1].max_abs,high);EXPECT_EQ(t.states[1].scaled_residual,0);
}
