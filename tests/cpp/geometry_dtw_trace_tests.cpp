#include <gtest/gtest.h>
#include <datamunge/geometry/dtw.hpp>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
using namespace datamunge::geometry;
namespace {
// Independent exhaustive enumeration of legal alignment paths, with no DP cache.
double brute(const std::vector<double>&a,const std::vector<double>&b) {
 double best=std::numeric_limits<double>::infinity();
 std::function<void(std::size_t,std::size_t,double)> walk=[&](auto i,auto j,double cost){
  cost+=std::abs(a[i]-b[j]);
  if(i+1==a.size()&&j+1==b.size()){best=std::min(best,cost);return;}
  if(i+1<a.size())walk(i+1,j,cost);
  if(j+1<b.size())walk(i,j+1,cost);
  if(i+1<a.size()&&j+1<b.size())walk(i+1,j+1,cost);
 };
 walk(0,0,0);return best;
}
void check(const DTWTrace&t) {
 const auto n=t.a.size(),m=t.b.size();
 ASSERT_EQ(t.cumulative.size(),n*m);ASSERT_EQ(t.predecessors.size(),n*m);
 EXPECT_EQ(t.captured_entries,n+m+2*n*m);
 EXPECT_EQ(t.work.cells,n*m);EXPECT_EQ(t.work.predecessor_reads,3*n*m);
 EXPECT_EQ(t.work.traceback_steps,t.result.path_a.size());
 EXPECT_EQ(t.result.distance,dtw_path_cost(t.a,t.b,t.result.path_a,t.result.path_b));
 EXPECT_GE(t.result.path_a.size(),std::max(n,m));EXPECT_LE(t.result.path_a.size(),n+m-1);
 for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<m;++j){
  std::vector<double>a(t.a.begin(),t.a.begin()+i+1),b(t.b.begin(),t.b.begin()+j+1);
  EXPECT_EQ(t.cumulative[i*m+j],brute(a,b));
  const auto p=t.predecessors[i*m+j];
  if(i==0&&j==0)EXPECT_EQ(p,DTWPredecessor::origin);
  else {
   const auto inf=std::numeric_limits<double>::infinity();
   const auto diagonal=i&&j?t.cumulative[(i-1)*m+j-1]:inf;
   const auto up=i?t.cumulative[(i-1)*m+j]:inf,left=j?t.cumulative[i*m+j-1]:inf;
   EXPECT_EQ(p,diagonal<=up&&diagonal<=left?DTWPredecessor::diagonal:up<=left?DTWPredecessor::advance_a:DTWPredecessor::advance_b);
  }
 }
}
}
TEST(DTWTrace, AllShortTernarySequencesMatchExhaustivePathEnumerationAndActualCells) {
 std::vector<std::vector<double>> inputs;
 for(int n=1;n<=4;++n){int total=1;for(int i=0;i<n;++i)total*=3;
  for(int mask=0;mask<total;++mask){int k=mask;std::vector<double>x(n);for(auto&v:x){v=k%3-1;k/=3;}inputs.push_back(x);}}
 ASSERT_EQ(inputs.size(),120);
 for(const auto&a:inputs)for(const auto&b:inputs){const auto t=dynamic_time_warping_trace(a,b);check(t);EXPECT_EQ(t.result.distance,brute(a,b));
  const auto ordinary=dynamic_time_warping_work(a,b);EXPECT_EQ(ordinary.result.distance,t.result.distance);EXPECT_EQ(ordinary.result.path_a,t.result.path_a);EXPECT_EQ(ordinary.result.path_b,t.result.path_b);}
}
TEST(DTWTrace, OriginalTinyTieOrderNonmetricControlsAndSeededInvariants) {
 const auto r=dynamic_time_warping({1,2,3},{1,1,2,3});
 EXPECT_EQ(r.path_a,(std::vector<int>{0,0,1,2}));EXPECT_EQ(r.path_b,(std::vector<int>{0,1,2,3}));EXPECT_EQ(r.distance,0);
 EXPECT_EQ(dynamic_time_warping({0},{0,0}).distance,0);
 EXPECT_EQ(dynamic_time_warping({0,0,0,0},{1,1,1,1}).distance,4);
 EXPECT_EQ(dynamic_time_warping({0,0,0,0},{0,1}).distance,1);
 EXPECT_EQ(dynamic_time_warping({0,1},{1,1,1,1}).distance,1);
 std::mt19937 rng(73);
 for(int trial=0;trial<500;++trial){std::vector<double>a(1+rng()%12),b(1+rng()%12);for(auto&v:a)v=int(rng()%9)-4;for(auto&v:b)v=int(rng()%9)-4;
  const auto x=dynamic_time_warping(a,b);EXPECT_EQ(x.distance,dynamic_time_warping(b,a).distance);EXPECT_EQ(x.distance,dtw_path_cost(a,b,x.path_a,x.path_b));
  if(a.size()==b.size()){double rigid=0;for(std::size_t i=0;i<a.size();++i)rigid+=std::abs(a[i]-b[i]);EXPECT_LE(x.distance,rigid);}
  for(auto&v:a)v=3*v+7;for(auto&v:b)v=3*v+7;EXPECT_EQ(dynamic_time_warping(a,b).distance,3*x.distance);
 }
}
TEST(DTWTrace, CompleteInputValidationCaptureLimitsAndSafeGeometry) {
 EXPECT_EQ(dtw_trace_entries(3,4),31);EXPECT_EQ(dtw_trace_entries(1,1),4);
 EXPECT_THROW((void)dtw_trace_entries(0,1),std::invalid_argument);
 EXPECT_THROW((void)dtw_trace_entries(std::numeric_limits<std::size_t>::max(),2),std::length_error);
 EXPECT_THROW((void)dtw_trace_entries(static_cast<std::size_t>(std::numeric_limits<int>::max())+1,1),std::length_error);
 EXPECT_NO_THROW((void)dynamic_time_warping_trace({1},{2},4));
 EXPECT_NO_THROW((void)dynamic_time_warping_trace({1},{2},0));
 EXPECT_THROW((void)dynamic_time_warping_trace({1},{2},3),std::length_error);
 EXPECT_THROW((void)dynamic_time_warping({}, {1}),std::invalid_argument);
 for(double v:{std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()}){
  EXPECT_THROW((void)dynamic_time_warping({1,v},{1,2}),std::invalid_argument);
  EXPECT_THROW((void)dynamic_time_warping_trace({1},{v},1),std::invalid_argument);
 }
 const double big=std::numeric_limits<double>::max();
 EXPECT_THROW((void)dynamic_time_warping({big},{-big}),std::overflow_error);
 const auto harmless=dynamic_time_warping_trace({big,-big},{big,-big});
 EXPECT_EQ(harmless.result.distance,0);EXPECT_TRUE(std::isinf(harmless.cumulative[1]));
 EXPECT_EQ(harmless.predecessors[1],DTWPredecessor::unrepresentable);
 EXPECT_EQ(dtw_path_cost(harmless.a,harmless.b,harmless.result.path_a,harmless.result.path_b),0);
}
TEST(DTWTrace, WitnessValidationAndNativeDrawingAreExplicitlyFeasibilityChecks) {
 EXPECT_THROW((void)dtw_path_cost({0,1},{0,1},{0},{0,1}),std::invalid_argument);
 EXPECT_THROW((void)dtw_path_cost({0},{0},{-1},{0}),std::invalid_argument);
 EXPECT_THROW((void)dtw_path_cost({0,1},{0,1},{0,0,1},{0,0,1}),std::invalid_argument);
 EXPECT_THROW((void)dtw_path_cost({0,1,2},{0,1,2},{0,2},{0,2}),std::invalid_argument);
 EXPECT_THROW((void)dtw_path_cost({0,1},{0,1},{1},{1}),std::invalid_argument);
 const double big=std::numeric_limits<double>::max();
 EXPECT_THROW((void)dtw_path_cost({big},{-big},{0,0},{0,0}),std::invalid_argument);
 EXPECT_THROW((void)dtw_path_cost({big},{-big},{0},{0}),std::overflow_error);
 const auto r=dynamic_time_warping({1,2},{1,2});
 EXPECT_NO_THROW((void)plot_dtw_alignment({1,2},{1,2},r,1.6,2));
 EXPECT_NO_THROW((void)plot_dtw_alignment({1,2},{1,2},r,0,0));
 EXPECT_THROW((void)plot_dtw_alignment({1,2},{1,2},r,1.6,1),std::length_error);
 auto bad=r;bad.distance=1;EXPECT_THROW((void)plot_dtw_alignment({1,2},{1,2},bad),std::invalid_argument);
 EXPECT_THROW((void)plot_dtw_alignment({1,2},{1,2},r,-1),std::invalid_argument);
 const auto one=dynamic_time_warping({big},{big});
 EXPECT_THROW((void)plot_dtw_alignment({big},{big},one,big),std::overflow_error);
 EXPECT_NO_THROW((void)plot_dtw_alignment({0},{0},dynamic_time_warping({0},{0})));
}
TEST(DTWTrace, CapturesOwnInputsAndLongOneDimensionalTracebackUsesNoRecursion) {
 std::vector<double>a{0,1},b{0,1};auto t=dynamic_time_warping_trace(a,b);auto copy=t;a[0]=99;b[1]=99;t.cumulative[0]=99;
 EXPECT_EQ(copy.a[0],0);EXPECT_EQ(copy.b[1],1);EXPECT_EQ(copy.cumulative[0],0);
 std::vector<double> long_a(50001,2);const auto r=dynamic_time_warping_work(long_a,{2});
 EXPECT_EQ(r.result.distance,0);EXPECT_EQ(r.result.path_a.size(),50001);EXPECT_EQ(r.work.cells,50001);EXPECT_EQ(r.work.predecessor_reads,150003);
 EXPECT_EQ(r.result.path_a.back(),50000);EXPECT_EQ(r.result.path_b.back(),0);
}
