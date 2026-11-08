#include <datamunge/algorithms/circle_packing.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <random>
#include <set>
namespace {
namespace a=datamunge::algorithms;using Edge=std::pair<std::size_t,std::size_t>;using Edges=std::vector<Edge>;
std::set<Edge> canonical(const Edges& edges){std::set<Edge> out;for(auto [u,v]:edges)if(u!=v){if(u>v)std::swap(u,v);out.insert({u,v});}return out;}
double separation(a::Point2D x,a::Point2D y){const auto dx=x.x-y.x,dy=x.y-y.y;return std::sqrt(dx*dx+dy*dy);}
std::vector<std::vector<a::Point2D>> legacy_states(std::size_t n,const Edges& edges,std::size_t iterations,double step) {
  const auto adjacency=canonical(edges);std::vector<a::Point2D> p(n);
  for(std::size_t i=0;i<n;++i)p[i]={3*std::cos(2*std::numbers::pi*i/n),3*std::sin(2*std::numbers::pi*i/n)};
  std::vector<std::vector<a::Point2D>> states{p};if(n<2)return states;
  for(std::size_t t=0;t<iterations;++t) {
    std::vector<a::Point2D> delta(n);
    for(std::size_t u=0;u<n;++u)for(std::size_t v=u+1;v<n;++v){const auto dx=p[v].x-p[u].x,dy=p[v].y-p[u].y,d=std::max(1e-9,separation(p[u],p[v]));const bool edge=adjacency.contains({u,v});const double target=edge?2:2.15;const double force=(d-target)*(edge || d<target?step:0);delta[u].x+=dx/d*force;delta[u].y+=dy/d*force;delta[v].x-=dx/d*force;delta[v].y-=dy/d*force;}
    for(std::size_t v=0;v<n;++v){p[v].x+=delta[v].x;p[v].y+=delta[v].y;}states.push_back(p);
  }return states;
}
void verify(std::size_t n,const Edges& edges,a::CirclePackingOptions options) {
  const auto trace=a::circle_packing_trace(n,edges,options);const auto ordinary=a::circle_packing_layout_with_options(n,edges,options);const auto old=legacy_states(n,edges,options.iterations,options.step_size);const auto adjacency=canonical(edges);
  const auto sweeps=n<2?0:options.iterations,pairs=n*(n?n-1:0)/2;
  ASSERT_EQ(trace.circles.size(),n);ASSERT_EQ(ordinary.size(),n);EXPECT_EQ(trace.work.completed_iterations,sweeps);EXPECT_EQ(trace.work.pair_visits,pairs*sweeps);EXPECT_EQ(trace.work.position_updates,n*sweeps);
  EXPECT_EQ(trace.work.validated_edges,edges.size());EXPECT_EQ(trace.work.unique_edges,adjacency.size());std::size_t loops=0;for(const auto [u,v]:edges)loops+=u==v;
  EXPECT_EQ(trace.work.ignored_self_loops,loops);EXPECT_EQ(trace.work.collapsed_duplicates,edges.size()-loops-adjacency.size());
  const auto frames=1+sweeps/options.snapshot_stride+(sweeps%options.snapshot_stride!=0);ASSERT_EQ(trace.frames.size(),frames);EXPECT_EQ(trace.work.frame_pair_visits,pairs*frames);EXPECT_EQ(trace.work.output_pair_visits,pairs);
  std::size_t previous=0;
  for(std::size_t f=0;f<trace.frames.size();++f) {
    const auto& frame=trace.frames[f];ASSERT_LT(frame.iteration,old.size());ASSERT_EQ(frame.circles.size(),n);if(f)EXPECT_GT(frame.iteration,previous);else EXPECT_EQ(frame.iteration,0u);previous=frame.iteration;
    EXPECT_TRUE(frame.iteration==0 || frame.iteration==sweeps || frame.iteration%options.snapshot_stride==0);
    long double edge_energy=0,nonedge_energy=0;
    for(std::size_t v=0;v<n;++v){EXPECT_NEAR(frame.circles[v].center.x,old[frame.iteration][v].x,1e-11);EXPECT_NEAR(frame.circles[v].center.y,old[frame.iteration][v].y,1e-11);EXPECT_EQ(frame.circles[v].radius,1);}
    for(std::size_t u=0;u<n;++u)for(std::size_t v=u+1;v<n;++v){const auto d=separation(old[frame.iteration][u],old[frame.iteration][v]);if(adjacency.contains({u,v}))edge_energy+=(d-2)*(d-2);else if(d<2.15)nonedge_energy+=(2.15-d)*(2.15-d);}
    EXPECT_NEAR(frame.edge_stress,edge_energy,1e-9);EXPECT_NEAR(frame.nonedge_penalty,nonedge_energy,1e-9);EXPECT_NEAR(frame.energy,(edge_energy+nonedge_energy)/2,1e-9);
  }
  EXPECT_EQ(trace.frames.back().iteration,sweeps);
  for(std::size_t v=0;v<n;++v){EXPECT_EQ(trace.circles[v].center.x,ordinary[v].center.x);EXPECT_EQ(trace.circles[v].center.y,ordinary[v].center.y);EXPECT_EQ(ordinary[v].radius,0.45);const auto raw=trace.frames.back().circles[v].center;EXPECT_NEAR(trace.circles[v].center.x,(raw.x-trace.normalization_center.x)/trace.normalization_scale,1e-15);EXPECT_NEAR(trace.circles[v].center.y,(raw.y-trace.normalization_center.y)/trace.normalization_scale,1e-15);}
  const auto metrics=a::circle_packing_metrics(trace.circles,edges);EXPECT_EQ(trace.output_metrics.edge_contacts,metrics.edge_contacts);EXPECT_EQ(trace.output_metrics.overlapping_pairs,metrics.overlapping_pairs);EXPECT_EQ(trace.output_metrics.min_gap,metrics.min_gap);
}
Edges complete(std::size_t n){Edges e;for(std::size_t u=0;u<n;++u)for(std::size_t v=u+1;v<n;++v)e.emplace_back(u,v);return e;}
}
TEST(CirclePackingTrace, OriginalTriangleCoordinatesRemainButTangencyClaimIsMeasuredFalse) {
  const auto e=complete(3);const auto circles=a::circle_packing_layout(3,e);const auto m=a::circle_packing_metrics(circles,e);
  ASSERT_EQ(circles.size(),3u);EXPECT_NEAR(circles[0].center.x,1,1e-12);EXPECT_NEAR(circles[1].center.x,-0.5,1e-12);EXPECT_NEAR(circles[1].center.y,std::sqrt(3.0)/2,1e-12);
  EXPECT_EQ(m.overlapping_pairs,0u);EXPECT_EQ(m.edge_contacts,0u);EXPECT_NEAR(m.min_gap,std::sqrt(3.0)-0.9,1e-12);EXPECT_GT(m.edge_stress,2);
}
TEST(CirclePackingTrace, TwoCircleDistanceAndEnergyFollowIndependentClosedForm) {
  a::CirclePackingOptions options;options.iterations=20;options.snapshot_stride=1;const auto t=a::circle_packing_trace(2,{{0,1}},options);
  for(const auto& f:t.frames){const auto expected=2+4*std::pow(0.84,double(f.iteration));EXPECT_NEAR(separation(f.circles[0].center,f.circles[1].center),expected,1e-12);EXPECT_NEAR(f.energy,8*std::pow(0.84,2.0*f.iteration),1e-12);if(f.iteration)EXPECT_NEAR(f.max_displacement,0.32*std::pow(0.84,double(f.iteration-1)),1e-12);}
  EXPECT_NEAR(t.output_metrics.min_gap,1.1,1e-12);
}
TEST(CirclePackingTrace, TriangleRadialModeHasItsOwnClosedFormAndRawContacts) {
  a::CirclePackingOptions options;options.iterations=30;options.snapshot_stride=1;const auto t=a::circle_packing_trace(3,complete(3),options);const auto fixed=2/std::sqrt(3.0);
  for(const auto& f:t.frames){const auto radius=fixed+(3-fixed)*std::pow(0.76,double(f.iteration));EXPECT_NEAR(std::hypot(f.circles[0].center.x,f.circles[0].center.y),radius,1e-12);EXPECT_NEAR(f.energy,1.5*std::pow(std::sqrt(3.0)*radius-2,2),1e-11);}
  const auto settled=a::circle_packing_trace(3,complete(3));EXPECT_EQ(a::circle_packing_metrics(settled.frames.back().circles,complete(3)).edge_contacts,3u);EXPECT_EQ(settled.work.completed_iterations,1000u);
}
TEST(CirclePackingTrace, EveryGraphThroughFiveVerticesMatchesIndependentDoubleJacobiStates) {
  std::size_t graphs=0;a::CirclePackingOptions options;options.iterations=3;options.snapshot_stride=1;
  for(std::size_t n=0;n<=5;++n){const auto all=complete(n);for(std::size_t mask=0;mask<(std::size_t{1}<<all.size());++mask){Edges edges;for(std::size_t bit=0;bit<all.size();++bit)if(mask&(std::size_t{1}<<bit))edges.push_back(all[bit]);verify(n,edges,options);++graphs;}}
  EXPECT_EQ(graphs,1100u);
}
TEST(CirclePackingTrace, ThousandRandomMultigraphsStepsAndStridesRetainActualStatesAndCounters) {
  std::mt19937_64 random(45);const double steps[]{0,0.01,0.08,0.12};
  for(int trial=0;trial<1000;++trial){const auto n=random()%9;Edges edges;if(n)for(std::size_t j=0;j<random()%25;++j)edges.emplace_back(random()%n,random()%n);a::CirclePackingOptions options;options.iterations=random()%9;options.snapshot_stride=1+random()%5;options.step_size=steps[random()%4];verify(n,edges,options);}
}
TEST(CirclePackingTrace, DefaultCycleOfEightReturnsOverlappingDisksDespiteFiniteRelaxation) {
  Edges edges;for(std::size_t v=0;v<8;++v)edges.emplace_back(v,(v+1)%8);const auto t=a::circle_packing_trace(8,edges);
  EXPECT_EQ(t.output_metrics.overlapping_pairs,8u);EXPECT_EQ(t.output_metrics.edge_contacts,0u);EXPECT_NEAR(t.output_metrics.min_gap,2*std::sin(std::numbers::pi/8)-0.9,1e-10);
}
TEST(CirclePackingTrace, UnequalPositiveRadiiGiveKFourContactsThatLegacyEqualRadiiMiss) {
  const auto radius=2/std::sqrt(3.0);const std::vector<a::Circle2D> disks{{{radius,0},1},{{-radius/2,1},1},{{-radius/2,-1},1},{{0,0},radius-1}};
  const auto good=a::circle_packing_metrics(disks,complete(4));EXPECT_EQ(good.edge_contacts,6u);EXPECT_EQ(good.overlapping_pairs,0u);EXPECT_EQ(good.nonedge_contacts,0u);EXPECT_EQ(good.zero_radii,0u);EXPECT_LT(good.edge_stress,1e-27);
  const auto bad=a::circle_packing_metrics(a::circle_packing_layout(4,complete(4)),complete(4));EXPECT_EQ(bad.edge_contacts,0u);EXPECT_EQ(bad.overlapping_pairs,0u);
}
TEST(CirclePackingTrace, ToleranceCountsDoNotClipOverlapPenaltyOrHideZeroRadii) {
  const std::vector<a::Circle2D> c{{{0,0},1},{{2,0},1},{{0,0},0}};const Edges edges{{0,1},{1,0},{0,0}};const auto m=a::circle_packing_metrics(c,edges);
  EXPECT_EQ(m.unique_edges,1u);EXPECT_EQ(m.pair_visits,3u);EXPECT_EQ(m.edge_contacts,1u);EXPECT_EQ(m.overlapping_pairs,1u);EXPECT_EQ(m.min_gap,-1);EXPECT_EQ(m.overlap_penalty,1);EXPECT_EQ(m.zero_radii,1u);
  const auto empty_graph=a::circle_packing_metrics(c,{});EXPECT_EQ(empty_graph.nonedge_contacts,1u);
  const auto tolerant=a::circle_packing_metrics(c,edges,1.1);EXPECT_EQ(tolerant.overlapping_pairs,0u);EXPECT_EQ(tolerant.nonedge_contacts,2u);EXPECT_EQ(tolerant.overlap_penalty,1);
}
TEST(CirclePackingTrace, WholeInputAndOptionsValidationPrecedesPositiveWorkCaps) {
  a::CirclePackingOptions cap;cap.max_pair_visits=1;
  EXPECT_THROW(a::circle_packing_trace(3,{{0,1},{3,0}},cap),std::invalid_argument);
  EXPECT_THROW(a::circle_packing_layout(0,{{0,0}}),std::invalid_argument);
  cap.snapshot_stride=0;
  EXPECT_THROW(a::circle_packing_trace(0,{},cap),std::invalid_argument);
  cap.snapshot_stride=1;cap.step_size=-1;
  EXPECT_THROW(a::circle_packing_layout_with_options(0,{},cap),std::invalid_argument);
  cap.step_size=std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(a::circle_packing_trace(0,{},cap),std::invalid_argument);
  EXPECT_THROW(a::circle_packing_metrics({{{0,0},-1}},{}),std::invalid_argument);
  EXPECT_THROW(a::circle_packing_metrics({{{std::numeric_limits<double>::infinity(),0},1}},{}),std::invalid_argument);
  EXPECT_THROW(a::circle_packing_metrics({}, {},-1),std::invalid_argument);
  EXPECT_THROW(a::circle_packing_metrics({}, {},std::numeric_limits<double>::quiet_NaN()),std::invalid_argument);
}
TEST(CirclePackingTrace, ExactWorkAndSnapshotPreflightCapsAreSeparateAndExposeNoPartialResult) {
  a::CirclePackingOptions exact;exact.iterations=4;exact.snapshot_stride=2;exact.max_pair_visits=12;exact.max_snapshots=3;
  const auto t=a::circle_packing_trace(3,complete(3),exact);EXPECT_EQ(t.work.pair_visits,12u);EXPECT_EQ(t.frames.size(),3u);EXPECT_EQ(t.work.frame_pair_visits,9u);
  auto low=exact;low.max_pair_visits=11;
  EXPECT_THROW(a::circle_packing_trace(3,complete(3),low),std::length_error);
  EXPECT_THROW(a::circle_packing_layout_with_options(3,complete(3),low),std::length_error);
  low=exact;low.max_snapshots=2;
  EXPECT_THROW(a::circle_packing_trace(3,complete(3),low),std::length_error);
  EXPECT_EQ(a::circle_packing_layout_with_options(3,complete(3),low).size(),3u);
  low=exact;low.iterations=std::numeric_limits<std::size_t>::max();low.max_snapshots=1;
  EXPECT_EQ(a::circle_packing_trace(0,{},low).frames.size(),1u);const auto one=a::circle_packing_trace(1,{},low);EXPECT_EQ(one.circles[0].center.x,0);EXPECT_EQ(one.circles[0].radius,0.45);EXPECT_EQ(one.work.completed_iterations,0u);
}
TEST(CirclePackingTrace, HugeLogicalWorkRejectsBeforeDenseGraphOrTraceAllocation) {
  a::CirclePackingOptions cap;cap.iterations=1;cap.max_pair_visits=1;
  EXPECT_THROW(a::circle_packing_trace(1000000000,{},cap),std::length_error);
  cap.iterations=std::numeric_limits<std::size_t>::max();cap.max_pair_visits=0;
  EXPECT_THROW(a::circle_packing_layout_with_options(2,{},cap),std::length_error);
  EXPECT_THROW(a::circle_packing_layout(std::numeric_limits<std::size_t>::max(),{},0),std::length_error);
}
TEST(CirclePackingTrace, NonfiniteUpdatesThrowAndWideDiagnosticDistancesRemainMeasurable) {
  a::CirclePackingOptions o;o.iterations=1;o.step_size=std::numeric_limits<double>::max();
  EXPECT_THROW(a::circle_packing_trace(2,{{0,1}},o),std::overflow_error);
  EXPECT_THROW(a::circle_packing_layout_with_options(2,{{0,1}},o),std::overflow_error);
  const auto m=std::numeric_limits<double>::max();const auto wide=a::circle_packing_metrics({{{-m,0},0},{{m,0},0}},{});EXPECT_TRUE(std::isfinite(wide.min_gap));EXPECT_GT(wide.min_gap,m);EXPECT_EQ(wide.zero_radii,2u);
}
TEST(CirclePackingTrace, InitialStrideAndFinalFramesAreOwnedAndFinalIsNotDuplicated) {
  a::CirclePackingOptions o;o.iterations=5;o.snapshot_stride=2;auto t=a::circle_packing_trace(3,complete(3),o);ASSERT_EQ(t.frames.size(),4u);
  EXPECT_EQ(t.frames[0].iteration,0u);EXPECT_EQ(t.frames[1].iteration,2u);EXPECT_EQ(t.frames[2].iteration,4u);EXPECT_EQ(t.frames[3].iteration,5u);
  const auto copy=t;t.frames[0].circles[0].center.x=99;t.circles.clear();EXPECT_EQ(copy.frames[0].circles[0].center.x,3);EXPECT_EQ(copy.circles.size(),3u);EXPECT_EQ(copy.frames[0].circles[0].radius,1);
}
TEST(CirclePackingTrace, ZeroStepStillExecutesRequestedWorkAndDuplicateInputsCollapse) {
  a::CirclePackingOptions o;o.iterations=4;o.snapshot_stride=1;o.step_size=0;verify(3,{{0,1},{1,0},{0,0},{1,2},{2,1},{2,2},{2,0}},o);
  const auto t=a::circle_packing_trace(3,complete(3),o);EXPECT_EQ(t.work.completed_iterations,4u);EXPECT_EQ(t.work.pair_visits,12u);for(const auto& f:t.frames){EXPECT_EQ(f.max_displacement,0);EXPECT_EQ(f.energy,t.frames[0].energy);}
  o.iterations=0;o.max_pair_visits=1;const auto no_sweeps=a::circle_packing_trace(4,complete(4),o);EXPECT_EQ(no_sweeps.work.pair_visits,0u);EXPECT_EQ(no_sweeps.work.frame_pair_visits,6u);
}
