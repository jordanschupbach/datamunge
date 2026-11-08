#include <datamunge/algorithms/clarke_wright.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>
#include <random>
#include <tuple>
namespace {
namespace a=datamunge::algorithms;
using P=std::pair<double,double>;using Points=std::vector<P>;using R=long double;
R metric(P a,P b){return std::hypot(R(a.first)-b.first,R(a.second)-b.second);}
void verify(const Points& p,const std::vector<double>& demand,double cap,std::size_t depot=0) {
  const auto trace=a::clarke_wright_trace(p,demand,cap,depot);const auto ordinary=a::clarke_wright_savings(p,demand,cap,depot);
  EXPECT_EQ(trace.result.routes,ordinary.routes);EXPECT_EQ(trace.result.total_distance,ordinary.total_distance);
  const auto m=p.size()-1,pairs=m*(m?m-1:0)/2;
  std::vector<std::deque<std::size_t>> routes;std::vector<R> loads;
  R star=0;for(std::size_t i=0;i<p.size();++i)if(i!=depot){routes.push_back({i});loads.push_back(demand[i]);star+=2*metric(p[depot],p[i]);}
  std::vector<std::tuple<R,std::size_t,std::size_t>> sorted;
  for(std::size_t i=0;i<p.size();++i)if(i!=depot)for(std::size_t j=i+1;j<p.size();++j)if(j!=depot)sorted.push_back({metric(p[depot],p[i])+metric(p[depot],p[j])-metric(p[i],p[j]),i,j});
  std::sort(sorted.begin(),sorted.end(),[](const auto& x,const auto& y){if(std::get<0>(x)!=std::get<0>(y))return std::get<0>(x)>std::get<0>(y);return std::tie(std::get<1>(x),std::get<2>(x))<std::tie(std::get<1>(y),std::get<2>(y));});
  ASSERT_EQ(trace.steps.size(),pairs);R savings=0;std::size_t merges=0,reversals=0,appends=0;
  for(std::size_t k=0;k<sorted.size();++k) {
    const auto [value,i,j]=sorted[k];const auto& s=trace.steps[k];EXPECT_EQ(s.i,i);EXPECT_EQ(s.j,j);EXPECT_EQ(s.saving,value);
    std::optional<std::size_t> left,right;
    for(std::size_t slot=0;slot<m;++slot)for(const auto vertex:routes[slot]){if(vertex==i && (vertex==routes[slot].front() || vertex==routes[slot].back()))left=slot;if(vertex==j && (vertex==routes[slot].front() || vertex==routes[slot].back()))right=slot;}
    EXPECT_EQ(s.route_i,left);EXPECT_EQ(s.route_j,right);EXPECT_EQ(s.load_i,left?loads[*left]:0);EXPECT_EQ(s.load_j,right?loads[*right]:0);
    if(!left || !right)EXPECT_EQ(s.decision,a::ClarkeWrightDecision::MissingEndpoint);
    else if(left==right)EXPECT_EQ(s.decision,a::ClarkeWrightDecision::SameRoute);
    else if(loads[*left]+loads[*right]>cap)EXPECT_EQ(s.decision,a::ClarkeWrightDecision::CapacityExceeded);
    else {
      EXPECT_EQ(s.decision,a::ClarkeWrightDecision::Merged);auto& x=routes[*left];auto& y=routes[*right];
      const bool flip_x=x.front()==i,flip_y=y.back()==j;EXPECT_EQ(s.reversed_i,flip_x);EXPECT_EQ(s.reversed_j,flip_y);reversals+=flip_x+flip_y;appends+=y.size();
      std::deque<std::size_t> joined;
      if(flip_x)for(auto v=x.rbegin();v!=x.rend();++v)joined.push_back(*v);else for(auto v:x)joined.push_back(v);
      if(flip_y)for(auto v=y.rbegin();v!=y.rend();++v)joined.push_back(*v);else for(auto v:y)joined.push_back(v);
      EXPECT_EQ(s.merged_customers,(std::vector<std::size_t>(joined.begin(),joined.end())));x=joined;y.clear();loads[*left]+=loads[*right];++merges;savings+=value;
    }
    if(s.decision!=a::ClarkeWrightDecision::Merged){EXPECT_TRUE(s.merged_customers.empty());EXPECT_FALSE(s.reversed_i);EXPECT_FALSE(s.reversed_j);}
  }
  std::size_t output=0,edge_count=0;R distance=0;std::vector<std::size_t> visits(p.size());
  for(std::size_t slot=0;slot<m;++slot)if(!routes[slot].empty()) {
    ASSERT_LT(output,trace.result.routes.size());const auto& route=trace.result.routes[output];ASSERT_EQ(route.size(),routes[slot].size()+2);EXPECT_EQ(route.front(),depot);EXPECT_EQ(route.back(),depot);
    R load=0;for(std::size_t i=1;i+1<route.size();++i){EXPECT_EQ(route[i],routes[slot][i-1]);++visits[route[i]];load+=demand[route[i]];}
    EXPECT_LE(load,cap);EXPECT_EQ(trace.route_loads[output],load);
    for(std::size_t i=1;i<route.size();++i){distance+=metric(p[route[i-1]],p[route[i]]);++edge_count;}++output;
  }
  EXPECT_EQ(output,trace.result.routes.size());for(std::size_t v=0;v<p.size();++v)EXPECT_EQ(visits[v],v==depot?0u:1u);
  EXPECT_EQ(trace.work.validated_points,p.size());EXPECT_EQ(trace.work.savings_pairs,pairs);EXPECT_EQ(trace.work.decisions,pairs);EXPECT_EQ(trace.work.endpoint_slot_visits,pairs*m);EXPECT_EQ(trace.work.merges,merges);EXPECT_EQ(trace.work.route_reversals,reversals);EXPECT_EQ(trace.work.appended_customers,appends);EXPECT_EQ(trace.work.output_edges,edge_count);EXPECT_EQ(trace.work.distance_evaluations,m+pairs+edge_count);
  EXPECT_NEAR(trace.initial_star_distance,star,1e-11);EXPECT_NEAR(trace.accepted_savings,savings,1e-11);EXPECT_NEAR(trace.measured_distance,distance,1e-11);EXPECT_NEAR(trace.accounting_residual,0,1e-10);EXPECT_NEAR(trace.result.total_distance,distance,1e-10);
  const auto evaluated=a::evaluate_vehicle_routes(p,demand,cap,trace.result.routes,depot);
  EXPECT_TRUE(evaluated.feasible);EXPECT_EQ(evaluated.route_loads,trace.route_loads);EXPECT_NEAR(evaluated.total_distance,distance,1e-11);
}
// Independent exact oracle: enumerate every customer permutation and every cut mask.
R optimum(const Points& p,const std::vector<double>& d,double cap,std::size_t depot) {
  std::vector<std::size_t> order;for(std::size_t i=0;i<p.size();++i)if(i!=depot)order.push_back(i);if(order.empty())return 0;
  R best=std::numeric_limits<R>::infinity();
  do {for(std::size_t cuts=0;cuts<(std::size_t{1}<<(order.size()-1));++cuts){R cost=0,load=0;auto last=depot;bool feasible=true;for(std::size_t i=0;i<order.size();++i){load+=d[order[i]];if(load>cap){feasible=false;break;}cost+=metric(p[last],p[order[i]]);last=order[i];if(i+1==order.size() || cuts&(std::size_t{1}<<i)){cost+=metric(p[last],p[depot]);last=depot;load=0;}}if(feasible)best=std::min(best,cost);}}while(std::next_permutation(order.begin(),order.end()));return best;
}
Points line(std::size_t n){Points p;for(std::size_t i=0;i<n;++i)p.push_back({double(i),0});return p;}
}
TEST(ClarkeWrightTrace, OriginalThreePointRouteAndSavingIdentityRemain) {
  const auto t=a::clarke_wright_trace(line(3),{0,1,1},2);EXPECT_EQ(t.result.routes,(std::vector<std::vector<std::size_t>>{{0,1,2,0}}));EXPECT_EQ(t.initial_star_distance,6);EXPECT_EQ(t.accepted_savings,2);EXPECT_EQ(t.result.total_distance,4);verify(line(3),{0,1,1},2);
}
TEST(ClarkeWrightTrace, EndpointSameRouteAndCapacityReasonsAreActualAndDistinct) {
  auto t=a::clarke_wright_trace(line(5),{0,1,1,1,1},4);bool missing=false,same=false;for(const auto& s:t.steps){missing|=s.decision==a::ClarkeWrightDecision::MissingEndpoint;same|=s.decision==a::ClarkeWrightDecision::SameRoute;}EXPECT_TRUE(missing);EXPECT_TRUE(same);verify(line(5),{0,1,1,1,1},4);
  t=a::clarke_wright_trace({{0,0},{1,0},{2,0},{10,0}},{0,1,1,1},2);EXPECT_EQ(t.steps[0].i,2u);EXPECT_EQ(t.steps[0].j,3u);EXPECT_EQ(t.steps[1].decision,a::ClarkeWrightDecision::CapacityExceeded);EXPECT_EQ(t.result.routes.size(),2u);verify({{0,0},{1,0},{2,0},{10,0}},{0,1,1,1},2);
}
TEST(ClarkeWrightTrace, ExactTiesUseLexicographicOriginalIdsAndZeroSavingStillMerges) {
  const Points p{{0,0},{1,0},{-1,0},{0,1},{0,-1}};const auto t=a::clarke_wright_trace(p,{0,1,1,1,1},4);verify(p,{0,1,1,1,1},4);
  const auto zero=a::clarke_wright_trace({{0,0},{1,0},{-1,0}},{0,1,1},2);ASSERT_EQ(zero.steps.size(),1u);EXPECT_EQ(zero.steps[0].saving,0);EXPECT_EQ(zero.work.merges,1u);EXPECT_EQ(zero.result.total_distance,4);
}
TEST(ClarkeWrightTrace, ExhaustiveLineDemandsCapacitiesAndEveryDepotPreserveFeasibilityAndStates) {
  std::size_t cases=0;
  for(std::size_t n=1;n<=6;++n)for(std::size_t depot=0;depot<n;++depot)for(std::size_t mask=0;mask<(std::size_t{1}<<n);++mask)for(int cap=2;cap<=4;++cap){std::vector<double> d(n);for(std::size_t i=0;i<n;++i)d[i]=1+((mask>>i)&1);verify(line(n),d,cap,depot);++cases;}
  EXPECT_EQ(cases,1926u);
}
TEST(ClarkeWrightTrace, ThousandRandomEuclideanInputsCheckEveryActualDecisionAndOutputEdge) {
  std::mt19937_64 random(1964);
  for(int trial=0;trial<1000;++trial){const auto n=1+random()%10;Points p(n);std::vector<double> d(n);for(std::size_t i=0;i<n;++i){p[i]={double(int(random()%41)-20),double(int(random()%41)-20)};d[i]=random()%5;}const auto depot=random()%n;const double cap=5+random()%8;verify(p,d,cap,depot);}
}
TEST(ClarkeWrightTrace, SmallProblemsCompareWithExhaustivePermutationAndPartitionOptima) {
  std::mt19937_64 random(568);int strict=0;
  for(int trial=0;trial<100;++trial){const auto n=5+trial%2;Points p(n);std::vector<double> d(n,1);d[0]=0;for(auto& point:p)point={double(int(random()%15)-7),double(int(random()%15)-7)};const double cap=2+trial%3;const auto r=a::clarke_wright_savings(p,d,cap);const auto exact=optimum(p,d,cap,0);EXPECT_GE(R(r.total_distance)+1e-10L,exact);strict+=r.total_distance>exact+1e-8L;}
  EXPECT_GT(strict,0);
}
TEST(ClarkeWrightTrace, FullCapacityBoundaryUsesWidenedLoadsWithoutAnUndocumentedTolerance) {
  EXPECT_EQ(a::clarke_wright_savings(line(3),{0,0.5,0.5},1).routes.size(),1u);
  EXPECT_EQ(a::clarke_wright_savings(line(3),{0,0.5,0.5},std::nextafter(1.0,0.0)).routes.size(),2u);
  const auto t=a::clarke_wright_trace(line(3),{0,0.1,0.2},0.3);EXPECT_EQ(t.result.routes.size(),2u);EXPECT_EQ(t.steps[0].decision,a::ClarkeWrightDecision::CapacityExceeded);
}
TEST(ClarkeWrightTrace, DepotOnlyZeroDemandsAndNonzeroDepotIdHaveExplicitSemantics) {
  verify({{3,4}},{100},1);verify(line(4),{0,0,0,0},1,2);
  const auto t=a::clarke_wright_trace({{3,4}},{100},1);EXPECT_TRUE(t.result.routes.empty());EXPECT_TRUE(t.steps.empty());EXPECT_EQ(t.work.distance_evaluations,0u);EXPECT_EQ(t.initial_star_distance,0);
}
TEST(ClarkeWrightTrace, ShapeCapacityCoordinateAndDemandValidationRejectInvalidFullInputs) {
  const double nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();
  EXPECT_THROW(a::clarke_wright_savings({}, {},1),std::invalid_argument);
  EXPECT_THROW(a::clarke_wright_savings(line(2),{0},1),std::invalid_argument);
  EXPECT_THROW(a::clarke_wright_savings(line(2),{0,1},1,2),std::invalid_argument);
  for(const double cap:{0.0,-1.0,nan,inf}){EXPECT_THROW(a::clarke_wright_savings(line(2),{0,1},cap),std::invalid_argument);}
  for(const double bad:{-1.0,nan,inf}){EXPECT_THROW(a::clarke_wright_savings(line(2),{0,bad},1),std::invalid_argument);}
  EXPECT_THROW(a::clarke_wright_savings({{nan,0},{1,0}},{0,1},1),std::invalid_argument);
  EXPECT_THROW(a::clarke_wright_savings({{0,0},{1,inf}},{0,1},1),std::invalid_argument);
  EXPECT_THROW(a::clarke_wright_savings(line(2),{-1,1},1),std::invalid_argument);
  EXPECT_THROW(a::clarke_wright_savings(line(2),{0,2},1),std::invalid_argument);
}
TEST(ClarkeWrightTrace, AllValidationPrecedesCapsAndExactPairEndpointRecordCapsAreSeparate) {
  a::ClarkeWrightOptions o;o.max_savings_pairs=6;o.max_endpoint_visits=24;o.max_decisions=6;
  EXPECT_EQ(a::clarke_wright_trace(line(5),{0,1,1,1,1},4,0,o).steps.size(),6u);
  o.max_savings_pairs=5;
  EXPECT_THROW(a::clarke_wright_trace(line(5),{0,1,1,1,1},4,0,o),std::length_error);
  o.max_savings_pairs=6;o.max_endpoint_visits=23;
  EXPECT_THROW(a::clarke_wright_savings_with_options(line(5),{0,1,1,1,1},4,0,o),std::length_error);
  o.max_endpoint_visits=24;o.max_decisions=5;
  EXPECT_THROW(a::clarke_wright_trace(line(5),{0,1,1,1,1},4,0,o),std::length_error);
  EXPECT_EQ(a::clarke_wright_savings_with_options(line(5),{0,1,1,1,1},4,0,o).routes.size(),1u);
  o.max_savings_pairs=1;
  EXPECT_THROW(a::clarke_wright_trace(line(5),{0,1,1,1,-1},4,0,o),std::invalid_argument);
}
TEST(ClarkeWrightTrace, HugePairForecastRejectsBeforeQuadraticCandidatesOrCubicLookup) {
  const auto p=line(100000);std::vector<double> d(p.size(),1);a::ClarkeWrightOptions o;o.max_savings_pairs=1;
  EXPECT_THROW(a::clarke_wright_trace(p,d,2,0,o),std::length_error);
}
TEST(ClarkeWrightTrace, WideIntermediateStarCanExceedDoubleWhileMergedResultFits) {
  const double x=std::numeric_limits<double>::max()*0.4;
  const auto t=a::clarke_wright_trace({{0,0},{x,0},{x,0}},{0,1,1},2);EXPECT_GT(t.initial_star_distance,std::numeric_limits<double>::max());EXPECT_TRUE(std::isfinite(t.result.total_distance));EXPECT_NEAR(t.result.total_distance/x,2,1e-15);EXPECT_EQ(t.work.merges,1u);
}
TEST(ClarkeWrightTrace, WideCoordinateDifferencesNeverLeakInfiniteFinalDoubleDistance) {
  const double x=std::numeric_limits<double>::max();
  EXPECT_THROW(a::clarke_wright_savings({{-x,0},{x,0}},{0,1},1),std::overflow_error);
  EXPECT_THROW(a::clarke_wright_trace({{-x,0},{x,0}},{0,1},1),std::overflow_error);
}
TEST(ClarkeWrightTrace, OwnedTraceRecordsDoNotAliasOtherStepsInputsOrFinalRoutes) {
  auto p=line(5);std::vector<double> d{0,1,1,1,1};auto t=a::clarke_wright_trace(p,d,4);const auto final=t.result.routes;const auto later=t.steps[1].merged_customers;
  t.steps[0].merged_customers[0]=99;p[1]={999,999};d[1]=999;EXPECT_EQ(t.result.routes,final);EXPECT_EQ(t.steps[1].merged_customers,later);EXPECT_EQ(t.route_loads[0],4);
}
TEST(ClarkeWrightTrace, FixedEuclideanCounterexampleHasAnIndependentlyEnumeratedShorterFeasibleSolution) {
  const Points p{{0,0},{2,-3},{-2,5},{0,-2},{4,1},{5,-4}};const std::vector<double> d{0,1,1,1,1,1};
  const auto h=a::clarke_wright_savings(p,d,3);const auto alternative=a::evaluate_vehicle_routes(p,d,3,{{0,2,4,0},{0,3,1,5,0}});const auto exact=optimum(p,d,3,0);
  EXPECT_TRUE(alternative.feasible);EXPECT_NEAR(alternative.total_distance,30.52084285878116L,1e-12);EXPECT_NEAR(alternative.total_distance,exact,1e-12);EXPECT_NEAR(h.total_distance,30.655228771257836,1e-12);EXPECT_GT(h.total_distance,alternative.total_distance+0.1L);verify(p,d,3);
}
TEST(ClarkeWrightTrace, IndependentEvaluatorMeasuresEveryReturnedRouteAndIdentifiesStructuralDefects) {
  const auto p=line(4);const std::vector<double> d{0,1,1,1};const auto r=a::clarke_wright_savings(p,d,2);const auto good=a::evaluate_vehicle_routes(p,d,2,r.routes);EXPECT_TRUE(good.feasible);EXPECT_NEAR(good.total_distance,r.total_distance,1e-12);
  const auto bad=a::evaluate_vehicle_routes(p,d,2,{{1,1},{0,2,0,2,0},{}});EXPECT_FALSE(bad.feasible);EXPECT_EQ(bad.malformed_routes,2u);EXPECT_EQ(bad.interior_depot_visits,1u);EXPECT_EQ(bad.missing_customers,1u);EXPECT_EQ(bad.repeated_visits,2u);EXPECT_EQ(bad.overloaded_routes,0u);EXPECT_EQ(bad.route_loads,(std::vector<R>{2,2,0}));
  const auto overloaded=a::evaluate_vehicle_routes(p,d,2,{{0,1,2,3,0}});EXPECT_EQ(overloaded.overloaded_routes,1u);EXPECT_FALSE(overloaded.feasible);
  EXPECT_TRUE(a::evaluate_vehicle_routes({{0,0}},{0},1,{}).feasible);
}
TEST(ClarkeWrightTrace, EvaluatorRejectsTrailingBadVerticesAndKeepsWideFiniteDistances) {
  EXPECT_THROW(a::evaluate_vehicle_routes(line(3),{0,1,1},2,{{0,1,0},{0,2,3,0}}),std::invalid_argument);
  const double x=std::numeric_limits<double>::max();const auto m=a::evaluate_vehicle_routes({{-x,0},{x,0}},{0,1},1,{{0,1,0}});EXPECT_TRUE(m.feasible);EXPECT_GT(m.total_distance,x);EXPECT_TRUE(std::isfinite(m.total_distance));
}
