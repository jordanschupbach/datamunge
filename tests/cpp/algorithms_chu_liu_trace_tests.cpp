#include <datamunge/algorithms/chu_liu_edmonds.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <random>
#include <set>
namespace {
namespace a=datamunge::algorithms;
using Edges=std::vector<a::WeightedEdge>;
constexpr auto none=a::kNoBranchingEdge;
std::pair<bool,long double> brute(std::size_t n,const Edges& edges,std::size_t root) {
  std::vector<std::vector<std::size_t>> incoming(n);
  for(std::size_t i=0;i<edges.size();++i){const auto [u,v,w]=edges[i];if(u!=v && v!=root)incoming[v].push_back(i);}
  std::vector<std::size_t> parent(n,root);long double best=std::numeric_limits<long double>::infinity();
  std::function<void(std::size_t,long double)> visit=[&](std::size_t v,long double cost) {
    if(v==n) {
      for(std::size_t x=0;x<n;++x){auto u=x;std::size_t steps=0;while(u!=root && steps++<n)u=parent[u];if(u!=root)return;}
      best=std::min(best,cost);return;
    }
    if(v==root){visit(v+1,cost);return;}
    for(const auto id:incoming[v]){parent[v]=std::get<0>(edges[id]);visit(v+1,cost+std::get<2>(edges[id]));}
  };
  visit(0,0);return {std::isfinite(best),std::isfinite(best)?best:0};
}
void verify(std::size_t n,const Edges& edges,std::size_t root,bool oracle=true) {
  const auto trace=a::chu_liu_edmonds_trace(n,edges,root);const auto& tree=trace.tree;
  const auto ordinary=a::chu_liu_edmonds(n,edges,root);const auto recovered=a::chu_liu_edmonds_tree(n,edges,root);
  EXPECT_EQ(tree.result.exists,ordinary.exists);EXPECT_EQ(tree.result.total_weight,ordinary.total_weight);EXPECT_EQ(tree.parent,recovered.parent);EXPECT_EQ(tree.incoming_edges,recovered.incoming_edges);
  if(oracle){const auto want=brute(n,edges,root);EXPECT_EQ(tree.result.exists,want.first);if(want.first)EXPECT_EQ(tree.result.total_weight,static_cast<double>(want.second));}
  if(tree.result.exists) {
    ASSERT_EQ(tree.parent.size(),n);ASSERT_EQ(tree.incoming_edges.size(),n);EXPECT_EQ(tree.parent[root],root);EXPECT_EQ(tree.incoming_edges[root],none);
    long double sum=0;
    for(std::size_t v=0;v<n;++v)if(v!=root) {
      const auto id=tree.incoming_edges[v];ASSERT_LT(id,edges.size());EXPECT_EQ(std::get<1>(edges[id]),v);EXPECT_EQ(tree.parent[v],std::get<0>(edges[id]));sum+=std::get<2>(edges[id]);
      auto u=v;std::size_t hops=0;while(u!=root && hops++<n){ASSERT_LT(u,n);u=tree.parent[u];}EXPECT_EQ(u,root);
    }EXPECT_EQ(tree.result.total_weight,static_cast<double>(sum));
  } else {EXPECT_TRUE(tree.parent.empty());EXPECT_TRUE(tree.incoming_edges.empty());EXPECT_EQ(tree.result.total_weight,0);}
  ASSERT_FALSE(trace.phases.empty());EXPECT_LE(trace.phases.size(),n);EXPECT_EQ(trace.work.validated_edges,edges.size());EXPECT_EQ(trace.work.phases,trace.phases.size());
  std::size_t cycles=0,predecessors=0,reweight=0,saved=0,replacements=0;long double accumulated=0;
  for(std::size_t p=0;p<trace.phases.size();++p) {
    const auto& phase=trace.phases[p];const auto count=phase.vertex_count;ASSERT_EQ(phase.edges.size(),edges.size());ASSERT_EQ(phase.original_to_vertex.size(),n);
    ASSERT_EQ(phase.incoming_edges.size(),count);ASSERT_EQ(phase.incoming_weights.size(),count);
    std::vector<std::size_t> selected(count,none);std::vector<long double> weights(count,std::numeric_limits<long double>::infinity());weights[phase.root]=0;
    for(std::size_t id=0;id<edges.size();++id) {
      const auto& e=phase.edges[id];EXPECT_EQ(e.original_index,id);ASSERT_LT(e.from,count);ASSERT_LT(e.to,count);
      if(e.from!=e.to && e.to!=phase.root && e.weight<weights[e.to]){weights[e.to]=e.weight;selected[e.to]=id;}
      if(p==0){EXPECT_EQ(e.from,std::get<0>(edges[id]));EXPECT_EQ(e.to,std::get<1>(edges[id]));EXPECT_EQ(e.weight,std::get<2>(edges[id]));}
    }
    EXPECT_EQ(phase.incoming_edges,selected);EXPECT_EQ(phase.incoming_weights,weights);
    std::vector<std::size_t> missing;for(std::size_t v=0;v<count;++v)if(v!=phase.root && selected[v]==none)missing.push_back(v);
    EXPECT_EQ(phase.missing_vertices,missing);
    if(!missing.empty()){EXPECT_FALSE(tree.result.exists);EXPECT_EQ(p+1,trace.phases.size());EXPECT_EQ(phase.added_cost,0);EXPECT_TRUE(phase.component.empty());EXPECT_TRUE(phase.cycles.empty());EXPECT_EQ(phase.cumulative_cost,accumulated);continue;}
    predecessors+=count-1;long double addition=0;for(const auto w:weights)addition+=w;EXPECT_EQ(phase.added_cost,addition);accumulated+=addition;EXPECT_EQ(phase.cumulative_cost,accumulated);
    if(phase.cycles.empty()){EXPECT_EQ(p+1,trace.phases.size());EXPECT_TRUE(tree.result.exists);EXPECT_TRUE(phase.component.empty());continue;}
    cycles+=phase.cycles.size();reweight+=edges.size();saved+=count+n;
    ASSERT_EQ(phase.component.size(),count);ASSERT_LT(p+1,trace.phases.size());const auto& next=trace.phases[p+1];
    EXPECT_EQ(next.root,phase.component[phase.root]);replacements+=next.vertex_count-1;
    for(std::size_t v=0;v<n;++v)EXPECT_EQ(next.original_to_vertex[v],phase.component[phase.original_to_vertex[v]]);
    std::set<std::size_t> cyclic;
    for(std::size_t c=0;c<phase.cycles.size();++c) {
      const auto& group=phase.cycles[c];ASSERT_GE(group.size(),2u);
      for(std::size_t i=0;i<group.size();++i){const auto v=group[i];ASSERT_LT(v,count);EXPECT_TRUE(cyclic.insert(v).second);EXPECT_EQ(phase.component[v],c);EXPECT_NE(v,phase.root);EXPECT_EQ(phase.edges[selected[v]].from,group[(i+1)%group.size()]);}
    }
    for(std::size_t id=0;id<edges.size();++id) {
      const auto& before=phase.edges[id];const auto& after=next.edges[id];EXPECT_EQ(after.from,phase.component[before.from]);EXPECT_EQ(after.to,phase.component[before.to]);
      EXPECT_EQ(after.weight,after.from!=after.to?before.weight-weights[before.to]:before.weight);
    }
  }
  EXPECT_EQ(trace.reduced_total,accumulated);EXPECT_EQ(trace.work.incoming_edge_visits,trace.phases.size()*edges.size());EXPECT_EQ(trace.work.reweight_edge_visits,reweight);
  EXPECT_EQ(trace.work.cycles_contracted,cycles);EXPECT_EQ(trace.work.predecessor_steps,predecessors);EXPECT_EQ(trace.work.saved_vertex_entries,saved);
  EXPECT_EQ(trace.work.expansion_replacements,tree.result.exists?replacements:0);
}
}
TEST(ChuLiuTrace, OriginalCycleRecoversActualInputEdgesAndReducedEntryCosts) {
  const Edges e{{0,1,5},{0,2,6},{1,2,1},{2,1,1}};verify(3,e,0);const auto t=a::chu_liu_edmonds_trace(3,e,0);
  EXPECT_EQ(t.tree.parent,(std::vector<std::size_t>{0,0,1}));EXPECT_EQ(t.tree.incoming_edges,(std::vector<std::size_t>{none,0,2}));
  ASSERT_EQ(t.phases.size(),2u);EXPECT_EQ(t.phases[0].added_cost,2);EXPECT_EQ(t.phases[1].added_cost,4);EXPECT_EQ(t.phases[1].edges[0].weight,4);EXPECT_EQ(t.phases[1].edges[1].weight,5);
}
TEST(ChuLiuTrace, ExhaustiveThreeVertexSignedTernaryGraphsAndAllRootsMatchBruteForce) {
  std::size_t cases=0;for(int code=0;code<729;++code){Edges edges;int digits=code;for(std::size_t u=0;u<3;++u)for(std::size_t v=0;v<3;++v)if(u!=v){const auto digit=digits%3;digits/=3;if(digit)edges.emplace_back(u,v,digit==1?-2:3);}for(std::size_t r=0;r<3;++r){verify(3,edges,r);++cases;}}
  EXPECT_EQ(cases,2187u);
}
TEST(ChuLiuTrace, ExhaustiveFourVertexPresenceGraphsAndAllRootsMatchBruteForce) {
  std::size_t cases=0;for(unsigned mask=0;mask<4096;++mask){Edges edges;unsigned bit=0;for(std::size_t u=0;u<4;++u)for(std::size_t v=0;v<4;++v)if(u!=v){if(mask&(1u<<bit))edges.emplace_back(u,v,static_cast<int>((7*u+3*v)%9)-4);++bit;}for(std::size_t r=0;r<4;++r){verify(4,edges,r);++cases;}}
  EXPECT_EQ(cases,16384u);
}
TEST(ChuLiuTrace, ThousandRandomSignedMultigraphsMatchEnumeratedIncomingChoices) {
  std::mt19937_64 random(1965);for(int trial=0;trial<1000;++trial){const std::size_t n=1+random()%7;Edges edges;for(std::size_t v=0;v<n;++v)for(unsigned j=0;j<random()%4;++j)edges.emplace_back(random()%n,v,static_cast<int>(random()%31)-15);verify(n,edges,random()%n);}
}
TEST(ChuLiuTrace, NestedAndSimultaneousCyclesExpandAcrossEveryActualPartition) {
  const Edges nested{{0,1,5},{0,3,10},{1,2,0},{2,1,0},{2,3,1},{3,1,1}};verify(4,nested,0);const auto n=a::chu_liu_edmonds_trace(4,nested,0);
  ASSERT_EQ(n.phases.size(),3u);EXPECT_EQ(n.tree.result.total_weight,6);EXPECT_EQ(n.tree.parent,(std::vector<std::size_t>{0,0,1,2}));EXPECT_EQ(n.work.cycles_contracted,2u);
  const Edges simultaneous{{0,1,5},{0,3,7},{1,2,0},{2,1,0},{3,4,0},{4,3,0},{2,3,1}};verify(5,simultaneous,0);const auto s=a::chu_liu_edmonds_trace(5,simultaneous,0);
  EXPECT_EQ(s.phases[0].cycles.size(),2u);EXPECT_EQ(s.tree.result.total_weight,6);EXPECT_EQ(s.tree.parent,(std::vector<std::size_t>{0,0,1,2,3}));
}
TEST(ChuLiuTrace, InitialMissingAndUnreachableClosedCycleReturnNoPartialTree) {
  verify(3,{{0,1,1}},0);verify(3,{{1,2,1},{2,1,1}},0);verify(3,{{1,0,-9},{1,2,1},{2,1,1}},0);
  const auto t=a::chu_liu_edmonds_trace(3,{{1,2,1},{2,1,1}},0);ASSERT_EQ(t.phases.size(),2u);EXPECT_EQ(t.phases.back().missing_vertices.size(),1u);EXPECT_EQ(t.reduced_total,2);EXPECT_FALSE(t.tree.result.exists);
}
TEST(ChuLiuTrace, ParallelEdgesAndIgnoredRootArcsOrLoopsValidateAndRetainTieIdentity) {
  const Edges edges{{1,1,-100},{1,0,-100},{0,1,-3},{0,1,-3}};verify(2,edges,0);const auto t=a::chu_liu_edmonds_tree(2,edges,0);EXPECT_EQ(t.incoming_edges[1],2u);EXPECT_EQ(t.result.total_weight,-3);
  verify(1,{{0,0,-100}},0);
}
TEST(ChuLiuTrace, WidenedReducedCostsAndFinalCancellationAvoidDoubleIntermediateOverflow) {
  const auto m=std::numeric_limits<double>::max();const Edges e{{0,1,m},{0,2,m},{1,2,-m},{2,1,-m}};verify(3,e,0);const auto t=a::chu_liu_edmonds_trace(3,e,0);
  ASSERT_EQ(t.phases.size(),2u);EXPECT_GT(t.phases[1].edges[0].weight,m);EXPECT_EQ(t.tree.result.total_weight,0);
  EXPECT_EQ(a::chu_liu_edmonds(4,{{0,1,m},{0,2,m},{0,3,-m}},0).total_weight,m);
  EXPECT_THROW(a::chu_liu_edmonds(3,{{0,1,m},{0,2,m}},0),std::overflow_error);
  EXPECT_THROW(a::chu_liu_edmonds_tree(3,{{0,1,-m},{0,2,-m}},0),std::overflow_error);
}
TEST(ChuLiuTrace, FullValidationPrecedesCapsAndIgnoredInvalidArcsStillReject) {
  const auto nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();a::BranchingOptions cap;cap.max_edge_visits=1;
  EXPECT_THROW(a::chu_liu_edmonds_trace(2,{{0,1,1},{1,0,nan}},0,cap),std::invalid_argument);
  EXPECT_THROW(a::chu_liu_edmonds(1,{{0,0,inf}},0),std::invalid_argument);
  EXPECT_THROW(a::chu_liu_edmonds(2,{{0,1,-inf}},0),std::invalid_argument);
  EXPECT_THROW(a::chu_liu_edmonds(2,{{0,2,1}},0),std::invalid_argument);
  EXPECT_THROW(a::chu_liu_edmonds(0,{},0),std::invalid_argument);
  EXPECT_THROW(a::chu_liu_edmonds(2,{},2),std::invalid_argument);
}
TEST(ChuLiuTrace, ExactAndOneBelowPhaseAndWholeCallVisitCapsHaveNoPartialResult) {
  const Edges e{{0,1,5},{0,2,6},{1,2,1},{2,1,1}};a::BranchingOptions exact;exact.max_phases=2;exact.max_edge_visits=12;
  EXPECT_EQ(a::chu_liu_edmonds_trace(3,e,0,exact).tree.result.total_weight,6);EXPECT_EQ(a::chu_liu_edmonds_tree(3,e,0,exact).result.total_weight,6);
  auto low=exact;low.max_phases=1;
  EXPECT_THROW(a::chu_liu_edmonds_trace(3,e,0,low),std::length_error);
  low=exact;low.max_edge_visits=11;
  EXPECT_THROW(a::chu_liu_edmonds_tree(3,e,0,low),std::length_error);
  EXPECT_THROW(a::chu_liu_edmonds_trace(3,e,0,low),std::length_error);
  EXPECT_EQ(a::chu_liu_edmonds_tree(1,{},0,exact).result.total_weight,0);
}
TEST(ChuLiuTrace, RecordsAndOriginalChoicesAreOwnedAcrossCopiesAndCalls) {
  auto edges=a::rooted_cycle_edges(3);auto t=a::chu_liu_edmonds_trace(4,edges,0);const auto copy=t;t.phases[0].edges.clear();t.tree.parent.clear();edges.clear();
  EXPECT_EQ(copy.tree.result.total_weight,7);EXPECT_EQ(copy.tree.parent.size(),4u);EXPECT_EQ(copy.phases[0].edges.size(),6u);
  a::DirectedBranchingResult default_result;EXPECT_FALSE(default_result.exists);EXPECT_EQ(default_result.total_weight,0);
}
TEST(ChuLiuTrace, NativeCycleFamilyHasTwoRoundsAndSixVisitsPerCycleVertex) {
  for(const auto k:{2u,3u,8u,64u,1024u}) {
    const auto edges=a::rooted_cycle_edges(k);const auto t=a::chu_liu_edmonds_trace(k+1,edges,0);EXPECT_EQ(edges.size(),2*k);EXPECT_EQ(t.tree.result.total_weight,k+4);
    EXPECT_EQ(t.work.phases,2u);EXPECT_EQ(t.work.incoming_edge_visits,4*k);EXPECT_EQ(t.work.reweight_edge_visits,2*k);EXPECT_EQ(t.work.predecessor_steps,k+1);
  }
  EXPECT_THROW(a::rooted_cycle_edges(1),std::invalid_argument);
  EXPECT_THROW(a::rooted_cycle_edges(std::numeric_limits<std::size_t>::max()),std::length_error);
  EXPECT_THROW(a::rooted_cycle_edges(2,std::numeric_limits<double>::infinity()),std::invalid_argument);
}
TEST(ChuLiuTrace, HundredThousandVertexChainUsesIterativeLinearParentWalk) {
  constexpr std::size_t n=100000;Edges edges;edges.reserve(n-1);for(std::size_t v=1;v<n;++v)edges.emplace_back(v-1,v,1);
  const auto t=a::chu_liu_edmonds_trace(n,edges,0);EXPECT_EQ(t.tree.result.total_weight,n-1);EXPECT_EQ(t.work.phases,1u);EXPECT_EQ(t.work.predecessor_steps,n-1);EXPECT_EQ(t.work.saved_vertex_entries,0u);EXPECT_EQ(t.tree.parent.back(),n-2);
}
TEST(ChuLiuTrace, SizeTVertexDomainIsNotSilentlyNarrowedToInt) {
  const auto n=std::numeric_limits<std::size_t>::max();
  EXPECT_THROW(a::chu_liu_edmonds(n,{},n-1),std::length_error);
}
