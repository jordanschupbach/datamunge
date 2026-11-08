#include <datamunge/algorithms/difference_map.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
using namespace datamunge::algorithms;
namespace {
const DifferenceProjection identity=[](const std::vector<double>& x){return x;};
const DifferenceProjection line_a=[](const std::vector<double>& x){return project_affine_hyperplane(x,{1,-1},0);};
const DifferenceProjection line_b=[](const std::vector<double>& x){return project_affine_hyperplane(x,{1,1},4);};
const DifferenceProjection finite_a=[](const std::vector<double>& x){return project_finite_set(x,{{0},{3}});};
const DifferenceProjection finite_b=[](const std::vector<double>& x){return project_finite_set(x,{{0},{2}});};
void near_vector(const std::vector<double>& a,const std::vector<double>& b,double tol=1e-12) {
 ASSERT_EQ(a.size(),b.size());for(std::size_t i=0;i<a.size();++i)EXPECT_NEAR(a[i],b[i],tol*(1+std::abs(b[i])));
}
}
TEST(DifferenceMapTrace, OriginalLinesAndActualStates) {
 const auto t=difference_map_trace(line_a,line_b,{7,-3},1,10000,1e-13);
 ASSERT_EQ(t.steps.size(),2);EXPECT_TRUE(t.result.converged);EXPECT_EQ(t.result.iterations,2);
 EXPECT_EQ(t.result.solution,(std::vector<double>{2,2}));EXPECT_EQ(t.final_iterate,t.result.solution);EXPECT_EQ(t.result.residual,0);
 const auto& s=t.steps[0];EXPECT_EQ(s.x_before,(std::vector<double>{7,-3}));EXPECT_EQ(s.projection_a,(std::vector<double>{2,2}));
 EXPECT_EQ(s.projection_b,s.x_before);EXPECT_EQ(s.f_a,s.x_before);EXPECT_EQ(s.f_b,s.x_before);
 EXPECT_EQ(s.estimate_a,(std::vector<double>{2,2}));EXPECT_EQ(s.estimate_b,s.x_before);EXPECT_EQ(s.x_after,t.final_iterate);
 EXPECT_NEAR(s.step_norm,std::sqrt(50.0),1e-14);EXPECT_DOUBLE_EQ(s.gap_norm,s.step_norm);
 EXPECT_EQ(t.work.projection_a_calls,4);EXPECT_EQ(t.work.projection_b_calls,4);EXPECT_EQ(t.work.coordinate_updates,4);EXPECT_EQ(t.work.history_entries,34);
 const auto r=difference_map(line_a,line_b,{7,-3},1,10000,1e-13);EXPECT_EQ(r.solution,t.result.solution);EXPECT_EQ(r.iterations,2);
 const auto axis_a=[](const std::vector<double>& x){return std::vector<double>{x[0],3};};
 const auto axis_b=[](const std::vector<double>& x){return std::vector<double>{2,x[1]};};
 EXPECT_EQ(difference_map(axis_a,axis_b,{-5,10}).solution,(std::vector<double>{2,3}));
}
TEST(DifferenceMapTrace, CorrectDecoderAtNonfeasibleFixedIterate) {
 const auto t=difference_map_trace(finite_a,finite_b,{2.25},.25);
 ASSERT_EQ(t.steps.size(),1);const auto& s=t.steps[0];EXPECT_TRUE(t.result.converged);EXPECT_DOUBLE_EQ(t.result.residual,0);
 EXPECT_EQ(s.projection_a,(std::vector<double>{3}));EXPECT_EQ(s.projection_b,(std::vector<double>{2}));
 EXPECT_EQ(s.f_a,(std::vector<double>{0}));EXPECT_EQ(s.f_b,(std::vector<double>{1}));
 EXPECT_EQ(s.estimate_a,(std::vector<double>{0}));EXPECT_EQ(s.estimate_b,(std::vector<double>{0}));
 EXPECT_EQ(t.final_iterate,(std::vector<double>{2.25}));EXPECT_EQ(t.result.solution,(std::vector<double>{0}));
 EXPECT_EQ(finite_b(t.final_iterate),(std::vector<double>{2}));EXPECT_NE(finite_a(finite_b(t.final_iterate)),finite_b(t.final_iterate));
 EXPECT_EQ(finite_a(finite_b({2.25})),(std::vector<double>{3}));EXPECT_EQ(finite_a(finite_b({3})),(std::vector<double>{3}));
 EXPECT_EQ(difference_map(finite_a,finite_b,{2.25},.25).solution,t.result.solution);
}
TEST(DifferenceMapTrace, IndependentTwoLineMatrixForAllRecordedIterations) {
 std::mt19937_64 rng(7717);std::size_t updates=0;
 for(double m:{.25,.5,1.,2.,4.})for(double beta:{-2.,-1.,-.5,.25,.5,1.,2.,4.})for(int trial=0;trial<25;++trial) {
  const DifferenceProjection a=[](const std::vector<double>& x){return project_affine_hyperplane(x,{0,1},0);};
  const DifferenceProjection b=[m](const std::vector<double>& x){return project_affine_hyperplane(x,{-m,1},0);};
  const std::vector<double> x0{double(int(rng()%41)-20),double(int(rng()%41)-20)};
  const auto t=difference_map_trace(a,b,x0,beta,12,0);ASSERT_EQ(t.steps.size(),12);EXPECT_FALSE(t.result.converged);
  long double u=x0[0],v=x0[1],c2=1/(1+static_cast<long double>(m)*m),s2=1-c2,cs=m*c2;
  for(const auto& s:t.steps) {
   const long double next_u=c2*u+beta*cs*v,next_v=-beta*cs*u+c2*v;
   near_vector(s.x_before,{double(u),double(v)},1e-10);near_vector(s.x_after,{double(next_u),double(next_v)},1e-10);
   near_vector(s.estimate_b,{double(c2*u+cs*v/beta),double(cs*u+s2*v/beta)},1e-10);
   EXPECT_NEAR(s.step_norm,double(std::hypot(next_u-u,next_v-v)),1e-10*(1+s.step_norm));
   EXPECT_NEAR(s.step_norm,std::abs(beta)*s.gap_norm,1e-10*(1+s.step_norm));u=next_u;v=next_v;++updates;
  }
  EXPECT_EQ(t.result.solution,t.steps.back().estimate_b);EXPECT_EQ(t.final_iterate,t.steps.back().x_after);
  EXPECT_EQ(t.work.projection_a_calls,24);EXPECT_EQ(t.work.projection_b_calls,24);EXPECT_EQ(t.work.coordinate_updates,24);EXPECT_EQ(t.work.history_entries,204);
 }
 EXPECT_EQ(updates,12000);
}
TEST(DifferenceMapTrace, ActualCallbackOrderMoveOnlyAndOwnership) {
 std::vector<std::pair<char,std::vector<double>>> calls;
 const DifferenceProjection a=[&](const std::vector<double>& x){calls.push_back({'A',x});return line_a(x);};
 const DifferenceProjection b=[&](const std::vector<double>& x){calls.push_back({'B',x});return line_b(x);};
 std::vector<double> x0{7,-3};auto t=difference_map_trace(a,b,x0);ASSERT_EQ(calls.size(),8);
 for(std::size_t k=0;k<t.steps.size();++k) {
  const auto& s=t.steps[k];EXPECT_EQ(calls[4*k],std::make_pair('A',s.x_before));EXPECT_EQ(calls[4*k+1],std::make_pair('B',s.x_before));
  EXPECT_EQ(calls[4*k+2],std::make_pair('A',s.f_b));EXPECT_EQ(calls[4*k+3],std::make_pair('B',s.f_a));
 }
 x0.clear();auto copy=t;t.steps[0].estimate_b[0]=42;t.initial.clear();t.result.solution[0]=99;
 EXPECT_EQ(copy.steps[0].estimate_b[0],7);EXPECT_EQ(copy.initial,(std::vector<double>{7,-3}));EXPECT_EQ(copy.result.solution[0],2);
 auto move_only=[p=std::make_unique<int>(0)](const std::vector<double>& x)mutable{++*p;return x;};
 EXPECT_EQ(difference_map(std::move(move_only),identity,{1}).solution,(std::vector<double>{1}));
}
TEST(DifferenceMapTrace, ExactOnlineHistoryCapsIncludingZeroDimension) {
 EXPECT_EQ(difference_map_history_cost(0),1);EXPECT_EQ(difference_map_history_cost(2),17);
 const auto max=std::numeric_limits<std::size_t>::max();EXPECT_EQ(difference_map_history_cost((max-1)/8),1+8*((max-1)/8));
 EXPECT_THROW(difference_map_history_cost((max-1)/8+1),std::length_error);
 EXPECT_NO_THROW(difference_map_trace(line_a,line_b,{7,-3},1,10000,1e-13,34));
 int calls=0;const DifferenceProjection a=[&](const std::vector<double>& x){++calls;return line_a(x);};
 const DifferenceProjection b=[&](const std::vector<double>& x){++calls;return line_b(x);};
 EXPECT_THROW(difference_map_trace(a,b,{7,-3},1,10000,1e-13,16),std::length_error);
 EXPECT_EQ(calls,0);
 EXPECT_THROW(difference_map_trace(a,b,{7,-3},1,10000,1e-13,33),std::length_error);
 EXPECT_EQ(calls,4);
 EXPECT_NO_THROW(difference_map_trace(line_a,line_b,{7,-3},1,10,0,0));
 const auto empty=difference_map_trace(identity,identity,{},1,3,0,3);EXPECT_EQ(empty.steps.size(),3);EXPECT_EQ(empty.work.history_entries,3);
 EXPECT_EQ(empty.work.coordinate_updates,0);EXPECT_EQ(empty.work.projection_a_calls,6);
 EXPECT_THROW(difference_map_trace(identity,identity,{},1,3,0,2),std::length_error);
 EXPECT_NO_THROW(difference_map_trace(identity,identity,{1},1,10000,1e-12,9));
 EXPECT_THROW(difference_map_trace(identity,identity,{1},1,10000,1e-12,8),std::length_error);
}
TEST(DifferenceMapTrace, ControlsCallbacksAndZeroBudget) {
 int calls=0;const DifferenceProjection id=[&](const std::vector<double>& x){++calls;return x;};const double nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();
 for(double beta:{0.,-0.,nan,inf}) {EXPECT_THROW(difference_map_trace(id,id,{1},beta),std::invalid_argument);}
 for(double tol:{-1.,nan,inf}) {EXPECT_THROW(difference_map_trace(id,id,{1},1,1,tol),std::invalid_argument);}
 EXPECT_THROW(difference_map_trace(id,id,{1},1,-1),std::invalid_argument);
 EXPECT_THROW(difference_map_trace(id,id,{nan}),std::invalid_argument);
 EXPECT_THROW(difference_map_trace(id,id,{inf}),std::invalid_argument);
 EXPECT_THROW(difference_map_trace(id,id,{1},std::numeric_limits<double>::denorm_min()),std::overflow_error);
 EXPECT_THROW(difference_map_trace({},id,{1}),std::invalid_argument);
 EXPECT_THROW(difference_map(DifferenceProjection{},id,{1}),std::invalid_argument);
 using Function=std::vector<double>(*)(const std::vector<double>&);const Function null=nullptr;
 EXPECT_THROW(difference_map(null,id,{1}),std::invalid_argument);
 EXPECT_EQ(calls,0);
 const auto zero=difference_map_trace(id,id,{2},1,0,0,1);EXPECT_EQ(calls,1);EXPECT_TRUE(zero.steps.empty());
 EXPECT_FALSE(zero.result.converged);EXPECT_EQ(zero.result.residual,0);EXPECT_EQ(zero.result.solution,(std::vector<double>{2}));
 EXPECT_EQ(zero.final_iterate,zero.initial);EXPECT_EQ(zero.work.projection_a_calls,0);EXPECT_EQ(zero.work.projection_b_calls,1);
 const auto fixed=difference_map_trace(identity,identity,{2},1,3,0);EXPECT_FALSE(fixed.result.converged);EXPECT_EQ(fixed.result.iterations,3);
}
TEST(DifferenceMapTrace, MalformedOutputsAndCallbackExceptionsStopImmediately) {
 for(int stage=0;stage<4;++stage)for(int kind=0;kind<3;++kind) {
  int calls=0;
  const DifferenceProjection projection=[&](const std::vector<double>& x){
   const bool bad=calls++==stage;if(!bad)return x;
   if(kind==0)return std::vector<double>{};if(kind==1)return std::vector<double>{1,2};
   return std::vector<double>{std::numeric_limits<double>::quiet_NaN()};};
  EXPECT_THROW(difference_map_trace(projection,projection,{1}),std::invalid_argument);EXPECT_EQ(calls,stage+1);
 }
 int calls=0;const DifferenceProjection throwing=[&](const std::vector<double>&)->std::vector<double>{++calls;throw std::runtime_error("oracle");};
 EXPECT_THROW(difference_map_trace(identity,throwing,{1}),std::runtime_error);EXPECT_EQ(calls,1);
 EXPECT_THROW(difference_map_trace(identity,[](const auto&){return std::vector<double>{};},{1},1,0),std::invalid_argument);
}
TEST(DifferenceMapTrace, HypotNormsAndCheckedArithmetic) {
 for(double scale:{1e200,1e-200}) {
  const auto a=[scale](const std::vector<double>&){return std::vector<double>{scale,scale};};
  const auto b=[](const std::vector<double>&){return std::vector<double>{0,0};};
  const auto t=difference_map_trace(a,b,{0,0},1,1,1e-250);
  EXPECT_TRUE(std::isfinite(t.result.residual));EXPECT_GT(t.result.residual,0);EXPECT_FALSE(t.result.converged);
  EXPECT_NEAR(t.result.residual/scale,std::sqrt(2.),1e-14);
 }
 const double max=std::numeric_limits<double>::max();
 const auto huge=[max](const auto&){return std::vector<double>{max};};
 const auto negative=[max](const auto&){return std::vector<double>{-max};};
 EXPECT_THROW(difference_map_trace(huge,negative,{0},1,1),std::overflow_error);
 EXPECT_THROW(difference_map_trace([](const auto&){return std::vector<double>{1e20};},identity,{0},1e-300,1),std::overflow_error);
}
TEST(DifferenceMapTrace, ProjectorPropertiesValidationAndNumericalTies) {
 std::mt19937_64 rng(717);
 for(int trial=0;trial<1000;++trial) {
  const std::size_t n=1+rng()%8;std::vector<double> x(n),normal(n);long double dot=0,squared=0;
  for(std::size_t i=0;i<n;++i){x[i]=int(rng()%41)-20;normal[i]=1+int(rng()%9);dot+=static_cast<long double>(x[i])*normal[i];squared+=static_cast<long double>(normal[i])*normal[i];}
  const double offset=int(rng()%21)-10;const auto projected=project_affine_hyperplane(x,normal,offset);long double residual=-offset;
  for(std::size_t i=0;i<n;++i){EXPECT_NEAR(projected[i],double(x[i]-(dot-offset)*normal[i]/squared),1e-12);residual+=static_cast<long double>(normal[i])*projected[i];}
  EXPECT_NEAR(double(residual),0,1e-11);near_vector(project_affine_hyperplane(projected,normal,offset),projected);
  std::vector<std::vector<double>> points(1+rng()%8,std::vector<double>(n));std::size_t best=0;long long best_dist=std::numeric_limits<long long>::max();
  for(std::size_t k=0;k<points.size();++k){long long dist=0;for(std::size_t i=0;i<n;++i){points[k][i]=int(rng()%41)-20;const long long delta=static_cast<long long>(x[i]-points[k][i]);dist+=delta*delta;}if(dist<best_dist){best_dist=dist;best=k;}}
  EXPECT_EQ(project_finite_set(x,points),points[best]);
 }
 EXPECT_EQ(project_finite_set({1},{{0},{2}}),(std::vector<double>{0}));EXPECT_EQ(project_finite_set({1},{{2},{0}}),(std::vector<double>{2}));
 EXPECT_TRUE(project_finite_set({},{{},{}}).empty());EXPECT_EQ(project_affine_hyperplane({1e300,1e300},{1e300,1e300},0),(std::vector<double>{0,0}));
 EXPECT_EQ(project_affine_hyperplane({1},{std::numeric_limits<double>::denorm_min()},0),(std::vector<double>{0}));
 EXPECT_THROW(project_affine_hyperplane({},{},0),std::invalid_argument);
 EXPECT_THROW(project_affine_hyperplane({1},{0},0),std::invalid_argument);
 EXPECT_THROW(project_affine_hyperplane({1},{1,2},0),std::invalid_argument);
 EXPECT_THROW(project_affine_hyperplane({1},{1},std::numeric_limits<double>::infinity()),std::invalid_argument);
 EXPECT_THROW(project_affine_hyperplane({1},{std::numeric_limits<double>::quiet_NaN()},0),std::invalid_argument);
 EXPECT_THROW(project_affine_hyperplane({1},{std::numeric_limits<double>::denorm_min()},1),std::overflow_error);
 EXPECT_THROW(project_finite_set({1},{}),std::invalid_argument);
 EXPECT_THROW(project_finite_set({1},{{1},{}}),std::invalid_argument);
 EXPECT_THROW(project_finite_set({1},{{1},{std::numeric_limits<double>::infinity()}}),std::invalid_argument);
}
TEST(DifferenceMapTrace, SmallBetaStopAndConvexDivergenceAreNotFeasibility) {
 const auto a=[](const auto&){return std::vector<double>{0};};const auto b=[](const auto&){return std::vector<double>{1};};
 const auto t=difference_map_trace(a,b,{0},1e-6,10,1e-5);ASSERT_EQ(t.steps.size(),1);EXPECT_TRUE(t.result.converged);
 EXPECT_DOUBLE_EQ(t.steps[0].gap_norm,1);EXPECT_EQ(t.result.solution,(std::vector<double>{1}));EXPECT_EQ(a(t.result.solution),(std::vector<double>{0}));
 const auto drift=difference_map_trace(a,b,{0},1,5,1e-12);EXPECT_FALSE(drift.result.converged);EXPECT_DOUBLE_EQ(drift.result.residual,1);EXPECT_EQ(drift.final_iterate,(std::vector<double>{-5}));
 const auto pa=[](const std::vector<double>& x){return project_affine_hyperplane(x,{0,1},0);};
 const auto pb=[](const std::vector<double>& x){return project_affine_hyperplane(x,{-1,1},0);};
 const auto stable=difference_map_trace(pa,pb,{3,4},1,12,0),unstable=difference_map_trace(pa,pb,{3,4},4,12,0);
 EXPECT_LT(std::hypot(stable.final_iterate[0],stable.final_iterate[1]),.1);
 EXPECT_GT(std::hypot(unstable.final_iterate[0],unstable.final_iterate[1]),10000);
}
