#include <datamunge/algorithms/bisection.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

using namespace datamunge::algorithms;

TEST(BisectionTrace,PreservesTheSignBracketAndExactDyadicShrink) {
  const auto t=bisection_trace([](double x){return x*x-2;},1,2,{1e-12,0,200});
  ASSERT_TRUE(t.result.converged);
  EXPECT_EQ(t.stop,BisectionStop::Interval);
  ASSERT_EQ(t.steps.size(),t.result.iterations);
  EXPECT_EQ(t.evaluations,t.steps.size()+2);
  EXPECT_LT(t.radius,1e-12);
  EXPECT_LE(std::abs(t.result.root-std::sqrt(2.)),t.radius+1e-15);
  for (std::size_t i=0;i<t.steps.size();++i) {
    const auto& step=t.steps[i];
    EXPECT_NE(std::signbit(step.f_lower),std::signbit(step.f_upper));
    EXPECT_DOUBLE_EQ(step.midpoint,std::midpoint(step.lower,step.upper));
    EXPECT_DOUBLE_EQ(step.radius,.5*(step.upper-step.lower));
    EXPECT_LE(step.lower,std::sqrt(2.));
    EXPECT_GE(step.upper,std::sqrt(2.));
    if (i) EXPECT_DOUBLE_EQ(step.upper-step.lower,.5*(t.steps[i-1].upper-t.steps[i-1].lower));
  }
}

TEST(BisectionTrace,LegacyAndRecordedCombinedToleranceUseTheSameLoop) {
  for (double tolerance:{1e-4,1e-8,1e-12}) {
    const auto f=[](double x){return x*x-2;};
    const auto plain=bisection(f,1,2,tolerance,200);
    const auto trace=bisection_trace(f,1,2,{tolerance,tolerance,200});
    EXPECT_EQ(plain.converged,trace.result.converged);
    EXPECT_DOUBLE_EQ(plain.root,trace.result.root);
    EXPECT_EQ(plain.iterations,trace.result.iterations);
    EXPECT_DOUBLE_EQ(plain.residual,trace.result.residual);
  }
}

TEST(BisectionTrace,SmallResidualDoesNotGuaranteePositionAccuracy) {
  const auto f=[](double x){return 1e-20*(x-.75);};
  const auto legacy=bisection_trace(f,0,2,{1e-10,1e-10,200});
  EXPECT_EQ(legacy.stop,BisectionStop::Residual);
  EXPECT_TRUE(legacy.result.converged);
  EXPECT_DOUBLE_EQ(legacy.result.root,1);
  EXPECT_DOUBLE_EQ(legacy.radius,1);
  const auto position=bisection_trace(f,0,2);
  EXPECT_EQ(position.stop,BisectionStop::ExactValue);
  EXPECT_DOUBLE_EQ(position.result.root,.75);
}

TEST(BisectionTrace,EndpointZerosAndReversedBrackets) {
  const auto first=bisection_trace([](double x){return x;},0,1);
  EXPECT_EQ(first.stop,BisectionStop::Endpoint);
  EXPECT_EQ(first.evaluations,1);
  EXPECT_TRUE(first.steps.empty());
  EXPECT_DOUBLE_EQ(first.radius,0);
  const auto second=bisection_trace([](double x){return x-1;},0,1);
  EXPECT_EQ(second.evaluations,2);
  EXPECT_DOUBLE_EQ(second.result.root,1);
  const auto reversed=bisection_trace([](double x){return x*x-2;},2,1);
  EXPECT_TRUE(reversed.result.converged);
  EXPECT_NEAR(reversed.result.root,std::sqrt(2.),1e-10);
}

TEST(BisectionTrace,SameSignsAreAnOrdinaryFailedBracket) {
  const auto t=bisection_trace([](double x){return x*x+1;},-1,1);
  EXPECT_FALSE(t.result.converged);
  EXPECT_EQ(t.stop,BisectionStop::NoBracket);
  EXPECT_TRUE(t.steps.empty());
  EXPECT_EQ(t.evaluations,2);
  EXPECT_TRUE(std::isinf(t.radius));
}

TEST(BisectionTrace,LargeEndpointsHaveAFiniteMidpoint) {
  const auto t=bisection_trace([](double x){return x-1.4e308;},1e308,1.6e308,{1e292,0,200});
  EXPECT_TRUE(t.result.converged);
  EXPECT_TRUE(std::isfinite(t.result.root));
  EXPECT_LE(std::abs(t.result.root-1.4e308),t.radius);
  EXPECT_TRUE(std::isfinite(t.steps.front().midpoint));
}

TEST(BisectionTrace,TinyCallbackValuesRetainTheirSigns) {
  const auto t=bisection_trace([](double x){return 1e-300*(x-.3);},0,1,{1e-12,0,200});
  EXPECT_TRUE(t.result.converged);
  EXPECT_NEAR(t.result.root,.3,1e-12);
  EXPECT_FALSE(bisection_trace([](double){return 1e-300;},0,1).result.converged);
}

TEST(BisectionTrace,ReportsBudgetExhaustionWithTheReturnedEstimatesBracket) {
  const auto t=bisection_trace([](double x){return x*x-2;},1,2,{1e-30,0,2});
  EXPECT_FALSE(t.result.converged);
  EXPECT_EQ(t.stop,BisectionStop::IterationLimit);
  EXPECT_EQ(t.result.iterations,2);
  EXPECT_EQ(t.evaluations,4);
  EXPECT_DOUBLE_EQ(t.result.root,1.25);
  EXPECT_DOUBLE_EQ(t.lower,1);
  EXPECT_DOUBLE_EQ(t.upper,1.5);
  EXPECT_DOUBLE_EQ(t.radius,.25);
}

TEST(BisectionTrace,ReportsFloatingPrecisionExhaustionWithoutRepeatedEndpoints) {
  const double a=1,b=std::nextafter(a,2.);
  const auto t=bisection_trace([a](double x){return x==a?-1.:1.;},a,b,
      {std::numeric_limits<double>::denorm_min(),0,200});
  EXPECT_FALSE(t.result.converged);
  EXPECT_EQ(t.stop,BisectionStop::Stagnation);
  EXPECT_EQ(t.steps.size(),1);
  EXPECT_DOUBLE_EQ(t.radius,b-a);
}

TEST(BisectionTrace,FiniteDiscontinuityCanPassTheIntervalTestWithoutARoot) {
  const auto t=bisection_trace([](double x){return x<.3?-1.:1.;},0,1,{1e-8,0,200});
  EXPECT_TRUE(t.result.converged);
  EXPECT_EQ(t.stop,BisectionStop::Interval);
  EXPECT_DOUBLE_EQ(t.result.residual,1);
}

TEST(BisectionTrace,RejectsInvalidOptionsAndNonfiniteCallbackValues) {
  const auto f=[](double x){return x;};
  EXPECT_THROW(bisection({},0,1),std::invalid_argument);
  EXPECT_THROW(bisection(f,0,1,0),std::invalid_argument);
  EXPECT_THROW(bisection(f,0,1,1e-10,0),std::invalid_argument);
  EXPECT_THROW(bisection_trace(f,0,1,{1e-10,-1,200}),std::invalid_argument);
  EXPECT_THROW(bisection_trace(f,0,1,{1e-10,0,-1}),std::invalid_argument);
  const double infinity=std::numeric_limits<double>::infinity(),nan=std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(bisection(f,infinity,1),std::invalid_argument);
  EXPECT_THROW(bisection(f,0,1,nan),std::invalid_argument);
  EXPECT_THROW(bisection_trace([nan](double){return nan;},0,1),std::domain_error);
  EXPECT_THROW(bisection_trace([](double x){return 1/x;},-1,1),std::domain_error);
  EXPECT_THROW(bisection_trace([](double)->double{throw std::runtime_error("callback");},0,1),std::runtime_error);
}

TEST(BisectionTrace,ReasonNamesAndUnknownValuesAreStable) {
  EXPECT_EQ(bisection_stop_name(BisectionStop::Interval),"interval");
  EXPECT_EQ(bisection_stop_name(BisectionStop::ExactValue),"zero callback");
  EXPECT_EQ(bisection_stop_name(static_cast<BisectionStop>(1000)),"unknown");
}
