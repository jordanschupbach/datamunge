#include <datamunge/algorithms/euler_method.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
using namespace datamunge::algorithms;
namespace {
void certificate(const EulerTrace& t,std::size_t dimension,std::size_t steps){const auto& o=t.outcome;ASSERT_EQ(o.solution.t.size(),steps+1);ASSERT_EQ(o.solution.y.size(),steps+1);ASSERT_EQ(t.derivatives.size(),steps);EXPECT_EQ(o.work.rhs_calls,steps);EXPECT_EQ(o.work.steps,steps);EXPECT_EQ(o.work.scalar_updates,steps*dimension);EXPECT_EQ(t.captured_entries,(2*steps+1)*(dimension+1));for(std::size_t n=0;n<steps;++n){ASSERT_EQ(t.derivatives[n].size(),dimension);ASSERT_EQ(o.solution.y[n].size(),dimension);for(std::size_t i=0;i<dimension;++i){const double increment=o.step_size*t.derivatives[n][i];EXPECT_EQ(o.solution.y[n+1][i],o.solution.y[n][i]+increment);}}}
}
TEST(EulerMethodTrace, ClosedDiscreteSolutionsAndNonautonomousQuadratureMatch){
 for(double rate:{-.5,-1.,-2.,.1})for(int n:{1,2,8,64,512}){const auto t=euler_method_trace([rate](double,const std::vector<double>& y){return std::vector<double>{rate*y[0]};},{1.5},2,5,n);certificate(t,1,n);const auto exact=1.5L*std::pow(1+static_cast<long double>(t.outcome.step_size)*rate,n);EXPECT_NEAR(t.outcome.solution.y.back()[0],double(exact),1e-11*(1+std::abs(double(exact))));}
 for(int n:{1,2,7,16,100}){const auto t=euler_method_trace([](double time,const std::vector<double>&){return std::vector<double>{time};},{3},.125,1.625,n);certificate(t,1,n);const long double h=t.outcome.step_size;const auto expected=3+h*(n*.125L+h*n*(n-1)/2);EXPECT_NEAR(t.outcome.solution.y.back()[0],double(expected),2e-14);EXPECT_EQ(t.outcome.solution.t.front(),.125);EXPECT_EQ(t.outcome.solution.t.back(),1.625);}
 const auto vector=euler_method_trace([](double,const std::vector<double>& y){return std::vector<double>{-y[0],-2*y[1],.5*y[2]};},{1,2,3},0,1,32);certificate(vector,3,32);const auto& v=vector.outcome.solution.y.back();EXPECT_NEAR(v[0],std::pow(31./32,32),1e-14);EXPECT_NEAR(v[1],2*std::pow(30./32,32),1e-14);EXPECT_NEAR(v[2],3*std::pow(65./64,32),1e-13);
 const auto oscillator=euler_method_trace([](double,const std::vector<double>& y){return std::vector<double>{y[1],-y[0]};},{1,0},0,100,10000);certificate(oscillator,2,10000);const auto expected=std::pow(std::complex<long double>{1,-static_cast<long double>(oscillator.outcome.step_size)},10000);const auto& last=oscillator.outcome.solution.y.back();EXPECT_NEAR(last[0],double(expected.real()),1e-12);EXPECT_NEAR(last[1],double(expected.imag()),1e-12);EXPECT_NEAR(last[0]*last[0]+last[1]*last[1],double(std::pow(1.L+.0001L,10000)),1e-11);
}
TEST(EulerMethodTrace, StatefulCallbacksAreCapturedOnceAndSharedWithOrdinaryWork){
 std::size_t calls=0;std::vector<double> times;std::vector<std::vector<double>> states,values;
 auto f=[&](double time,const std::vector<double>& y){++calls;times.push_back(time);states.push_back(y);std::vector<double> value{time+y[0]+calls};values.push_back(value);return value;};
 const auto trace=euler_method_trace(f,{1},0,1,8);certificate(trace,1,8);EXPECT_EQ(calls,8u);EXPECT_EQ(trace.derivatives,values);
 for(std::size_t i=0;i<calls;++i){EXPECT_EQ(times[i],trace.outcome.solution.t[i]);EXPECT_EQ(states[i],trace.outcome.solution.y[i]);}
 const auto first_times=times;const auto first_states=states;const auto first_values=values;calls=0;times.clear();states.clear();values.clear();const auto work=euler_method_work(f,{1},0,1,8);EXPECT_EQ(times,first_times);EXPECT_EQ(states,first_states);EXPECT_EQ(values,first_values);EXPECT_EQ(work.solution.y,trace.outcome.solution.y);EXPECT_EQ(work.solution.t,trace.outcome.solution.t);
 calls=0;times.clear();states.clear();values.clear();const auto ordinary=euler_method(f,{1},0,1,8);EXPECT_EQ(ordinary.y,work.solution.y);EXPECT_EQ(times,first_times);
}
TEST(EulerMethodTrace, FullKnownQuotasPrecedeCallbacksAndAllHistoryIsOwned){
 int calls=0;auto f=[&](double,const std::vector<double>& y){++calls;return std::vector<double>{-y[0]};};const auto t=euler_method_trace(f,{1},0,1,2,10);certificate(t,1,2);EXPECT_EQ(calls,2);
 for(std::size_t cap=1;cap<10;++cap){calls=0;EXPECT_THROW((void)euler_method_trace(f,{1},0,1,2,cap),std::length_error);EXPECT_EQ(calls,0);}
 EXPECT_EQ(euler_method_trace(f,{1},0,1,2,0).captured_entries,10u);EXPECT_EQ(euler_trace_size(2,2),15u);
 std::vector<double> initial{1};const auto owned=euler_method_trace(f,initial,0,1,2);initial[0]=99;auto copy=owned;copy.outcome.solution.t[0]=99;copy.outcome.solution.y[0][0]=99;copy.derivatives[0][0]=99;EXPECT_EQ(owned.outcome.solution.t[0],0);EXPECT_EQ(owned.outcome.solution.y[0][0],1);EXPECT_EQ(owned.derivatives[0][0],-1);
}
TEST(EulerMethodTrace, InvalidInputsDerivativesAndFiniteArithmeticFailExplicitly){
 const double nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity(),max=std::numeric_limits<double>::max();int calls=0;auto f=[&](double,const std::vector<double>& y){++calls;return std::vector<double>(y.size(),0);};
 for(int steps:{0,-1}){calls=0;EXPECT_THROW((void)euler_method_work(f,{1},0,1,steps),std::invalid_argument);EXPECT_EQ(calls,0);}
 EXPECT_THROW((void)euler_method_work({}, {1},0,1,1),std::invalid_argument);
 EXPECT_THROW((void)euler_method_work(f,{},0,1,1),std::invalid_argument);
 for(double bad:{nan,inf,-inf}){EXPECT_THROW((void)euler_method_work(f,{bad},0,1,1),std::invalid_argument);
  EXPECT_THROW((void)euler_method_work(f,{1},bad,1,1),std::invalid_argument);
  EXPECT_THROW((void)euler_method_trace(f,{1},0,bad,1),std::invalid_argument);
  EXPECT_THROW((void)euler_method_work([bad](double,const std::vector<double>&){return std::vector<double>{bad};},{1},0,1,1),std::invalid_argument);}
 for(std::size_t dim:{0,2}){EXPECT_THROW((void)euler_method_work([dim](double,const std::vector<double>&){return std::vector<double>(dim,0);},{1},0,1,1),std::invalid_argument);}
 EXPECT_THROW((void)euler_method_work([max](double,const std::vector<double>&){return std::vector<double>{max};},{1},0,2,1),std::overflow_error);
 EXPECT_THROW((void)euler_method_work([max](double,const std::vector<double>&){return std::vector<double>{max};},{max},0,1,1),std::overflow_error);
 EXPECT_THROW((void)euler_trace_size(std::numeric_limits<std::size_t>::max(),1),std::length_error);
 EXPECT_THROW((void)euler_trace_size(0,1),std::invalid_argument);
 EXPECT_THROW((void)euler_trace_size(1,0),std::invalid_argument);
 struct Marker:std::runtime_error{Marker():std::runtime_error("caller marker") {}};calls=0;
 EXPECT_THROW((void)euler_method_trace([&](double,const std::vector<double>&){if(++calls==2)throw Marker{};return std::vector<double>{0};},{1},0,1,3),Marker);
 EXPECT_EQ(calls,2);
 const auto wide=euler_method_work(f,{0},-max,max,4);EXPECT_TRUE(std::isfinite(wide.step_size));EXPECT_EQ(wide.solution.t.front(),-max);EXPECT_EQ(wide.solution.t.back(),max);for(double time:wide.solution.t)EXPECT_TRUE(std::isfinite(time));
}
TEST(EulerMethodTrace, RoundedTimeLimitsReverseAndZeroIntervalsAreDistinct){
 int calls=0;auto f=[&](double,const std::vector<double>&){++calls;return std::vector<double>{2};};const double tiny=std::numeric_limits<double>::denorm_min(),large=1e100,next=std::nextafter(large,std::numeric_limits<double>::infinity());
 EXPECT_THROW((void)euler_method_trace(f,{1},0,tiny,2),std::underflow_error);
 EXPECT_EQ(calls,0);
 EXPECT_THROW((void)euler_method_trace(f,{1},large,next,4),std::runtime_error);
 EXPECT_EQ(calls,0);
 const auto reverse=euler_method_trace(f,{1},2,-2,8);certificate(reverse,1,8);EXPECT_EQ(reverse.outcome.step_size,-.5);EXPECT_EQ(reverse.outcome.solution.y.back()[0],-7);for(std::size_t i=1;i<reverse.outcome.solution.t.size();++i)EXPECT_LT(reverse.outcome.solution.t[i],reverse.outcome.solution.t[i-1]);
 calls=0;const auto flat=euler_method_trace(f,{1},2,2,3);certificate(flat,1,3);EXPECT_EQ(calls,3);EXPECT_EQ(flat.outcome.step_size,0);for(std::size_t i=0;i<flat.outcome.solution.t.size();++i){EXPECT_EQ(flat.outcome.solution.t[i],2);EXPECT_EQ(flat.outcome.solution.y[i][0],1);}
}
TEST(EulerMethodTrace, OriginalDecayConvergenceAndStabilityBoundariesAreMeasured){
 const auto decay=[](double,const std::vector<double>& y){return std::vector<double>{-y[0]};};const auto original=euler_method_trace(decay,{1},0,4,8);certificate(original,1,8);EXPECT_EQ(original.outcome.solution.y.back()[0],1./256);
 double previous=0;for(int n=10;n<=2560;n*=2){const auto s=euler_method(decay,{1},0,1,n);const double error=std::exp(-1.)-s.y.back()[0];EXPECT_GT(error,0);if(previous){EXPECT_GT(previous/error,1.98);EXPECT_LT(previous/error,2.06);}previous=error;if(n==2560)EXPECT_NEAR(error*n,1/(2*std::exp(1.)),1e-4);}
 for(double q:{.5,1.,1.5,2.,2.5}){const auto t=euler_method_trace([q](double,const std::vector<double>& y){return std::vector<double>{-2*q*y[0]};},{1},0,4,8);certificate(t,1,8);EXPECT_EQ(t.outcome.solution.y.back()[0],std::pow(1-q,8));if(q==2)for(double y:t.outcome.solution.y.back())EXPECT_EQ(std::abs(y),1);}
}
