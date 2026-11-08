#include <datamunge/algorithms/boruvka.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <set>

using namespace datamunge::algorithms;
namespace {
// Independent dense Prim forest, with a fresh visited array for every original component.
double dense_prim(std::size_t n,const std::vector<WeightedEdge>& edges) {
  const double inf=std::numeric_limits<double>::infinity();
  std::vector<std::vector<double>> weights(n,std::vector<double>(n,inf));
  for (auto [u,v,w]:edges) if (u!=v) weights[u][v]=weights[v][u]=std::min(weights[u][v],w);
  std::vector<bool> visited(n);std::vector<double> distance(n,inf);double result=0;
  for (std::size_t first=0;first<n;++first) if (!visited[first]) {
    distance[first]=0;
    for (;;) {
      auto chosen=n;for (std::size_t v=0;v<n;++v)
        if (!visited[v] && std::isfinite(distance[v]) && (chosen==n || distance[v]<distance[chosen])) chosen=v;
      if (chosen==n) break;
      visited[chosen]=true;result+=distance[chosen];
      for (std::size_t v=0;v<n;++v) if (!visited[v]) distance[v]=std::min(distance[v],weights[chosen][v]);
    }
  }
  return result;
}
// Reachability labels are intentionally independent of the production union-find.
std::vector<std::size_t> labels(std::size_t n,const std::vector<WeightedEdge>& accepted) {
  std::vector<std::vector<std::size_t>> adjacent(n);
  for (auto [u,v,w]:accepted) {adjacent[u].push_back(v);adjacent[v].push_back(u);}
  std::vector<std::size_t> result(n,n);
  for (std::size_t first=0;first<n;++first) if (result[first]==n) {
    std::vector<std::size_t> pending{first};result[first]=first;
    for (std::size_t i=0;i<pending.size();++i) for (auto v:adjacent[pending[i]]) if (result[v]==n) {
      result[v]=first;pending.push_back(v);
    }
  }
  return result;
}
void check_trace(std::size_t n,const std::vector<WeightedEdge>& edges,const BoruvkaTrace& trace) {
  const auto ordinary=boruvka(n,edges);
  ASSERT_EQ(trace.forest.edges,ordinary.edges);ASSERT_EQ(trace.forest.total_weight,ordinary.total_weight);
  ASSERT_EQ(trace.forest.is_connected,ordinary.is_connected);
  std::vector<WeightedEdge> accepted;
  for (const auto& phase:trace.phases) {
    const auto before=labels(n,accepted);ASSERT_EQ(phase.component_labels,before);
    const std::set<std::size_t> components(before.begin(),before.end());
    ASSERT_EQ(phase.components_before,components.size());ASSERT_EQ(phase.edges_scanned,edges.size());
    std::size_t expected_choices=0;
    for (auto component:components) {
      std::size_t best=edges.size();
      for (std::size_t i=0;i<edges.size();++i) {
        const auto [u,v,w]=edges[i];
        if ((before[u]==component)==(before[v]==component)) continue;
        if (best==edges.size() || w<std::get<2>(edges[best])) best=i;
      }
      if (best==edges.size()) continue;
      ++expected_choices;
      ASSERT_EQ(std::count_if(phase.choices.begin(),phase.choices.end(),[&](auto choice){
        return choice.component_label==component && choice.edge_index==best;}),1);
    }
    ASSERT_EQ(phase.choices.size(),expected_choices);
    for (auto index:phase.accepted_edge_indices) {
      ASSERT_LT(index,edges.size());const auto [u,v,w]=edges[index];
      const auto current=labels(n,accepted);ASSERT_NE(current[u],current[v]);
      ASSERT_TRUE(std::any_of(phase.choices.begin(),phase.choices.end(),[&](auto c){return c.edge_index==index;}));
      accepted.push_back(edges[index]);
    }
    ASSERT_EQ(phase.components_before-phase.accepted_edge_indices.size(),phase.components_after);
    if (trace.forest.is_connected) ASSERT_LE(phase.components_after,phase.components_before/2);
  }
  ASSERT_EQ(accepted,trace.forest.edges);
  ASSERT_EQ(trace.edge_scans,trace.phases.size()*edges.size());
  ASSERT_EQ(trace.component_count,n-accepted.size());
}
}
TEST(BoruvkaTrace, OriginalExampleRecordsRealChoicesAndMerges) {
  const std::vector<WeightedEdge> edges{{0,1,1},{1,2,2},{2,3,1},{0,3,9}};
  const auto trace=boruvka_trace(4,edges);check_trace(4,edges,trace);
  ASSERT_EQ(trace.phases.size(),2);EXPECT_EQ(trace.forest.total_weight,4);
  EXPECT_EQ(trace.phases[0].accepted_edge_indices,(std::vector<std::size_t>{0,2}));
  EXPECT_EQ(trace.phases[1].accepted_edge_indices,(std::vector<std::size_t>{1}));
  EXPECT_EQ(trace.phases[1].component_labels,(std::vector<std::size_t>{0,0,2,2}));
  EXPECT_EQ(trace.phases[0].choices.size(),4);EXPECT_EQ(trace.edge_scans,8);
}
TEST(BoruvkaTrace, All33868SimpleGraphsThroughSixVerticesMatchIndependentPrim) {
  std::size_t checked=0;
  for (std::size_t n=0;n<=6;++n) {
    std::vector<WeightedEdge> complete;
    for (std::size_t u=0;u<n;++u) for (std::size_t v=u+1;v<n;++v)
      complete.emplace_back(u,v,int((u*11+v*7)%9)-4);
    for (std::size_t mask=0;mask<(std::size_t{1}<<complete.size());++mask) {
      std::vector<WeightedEdge> edges;
      for (std::size_t i=0;i<complete.size();++i) if (mask&(std::size_t{1}<<i)) edges.push_back(complete[i]);
      const auto trace=boruvka_trace(n,edges);
      ASSERT_DOUBLE_EQ(trace.forest.total_weight,dense_prim(n,edges));
      check_trace(n,edges,trace);++checked;
    }
  }
  EXPECT_EQ(checked,33868);
}
TEST(BoruvkaTrace, ParallelLoopsTiesAndPermutedInputMatchIndependentPrim) {
  std::vector<WeightedEdge> edges{{0,0,-100},{0,1,4},{1,0,-2},{0,1,-2},{1,2,0},
      {2,0,0},{2,3,5},{3,4,5},{4,2,5},{4,4,-100}};
  std::mt19937_64 engine(31);
  for (int i=0;i<1000;++i) {
    std::shuffle(edges.begin(),edges.end(),engine);
    const auto trace=boruvka_trace(6,edges);check_trace(6,edges,trace);
    EXPECT_EQ(trace.forest.total_weight,dense_prim(6,edges));EXPECT_FALSE(trace.forest.is_connected);
  }
}
TEST(BoruvkaTrace, EmptySingletonAndDisconnectedTerminalScanAreExplicit) {
  const auto empty=boruvka_trace(0,{});EXPECT_TRUE(empty.forest.is_connected);EXPECT_TRUE(empty.phases.empty());
  const auto single=boruvka_trace(1,{{0,0,-7}});EXPECT_TRUE(single.forest.is_connected);EXPECT_TRUE(single.phases.empty());
  const auto isolated=boruvka_trace(5,{});ASSERT_EQ(isolated.phases.size(),1);EXPECT_EQ(isolated.component_count,5);
  EXPECT_EQ(isolated.phases[0].components_before,isolated.phases[0].components_after);
  const auto forest=boruvka_trace(6,{{0,1,-3},{2,3,2}});check_trace(6,{{0,1,-3},{2,3,2}},forest);
  ASSERT_EQ(forest.phases.size(),2);EXPECT_TRUE(forest.phases.back().choices.empty());EXPECT_EQ(forest.component_count,4);
}
TEST(BoruvkaTrace, BalancedPathNeedsLogarithmicallyManyProductivePhases) {
  for (unsigned exponent=1;exponent<=10;++exponent) {
    const std::size_t n=std::size_t{1}<<exponent;const auto edges=dyadic_path_edges(n);
    const auto trace=boruvka_trace(n,edges);
    ASSERT_EQ(trace.phases.size(),exponent);EXPECT_EQ(trace.edge_scans,exponent*(n-1));
    std::size_t count=n;
    for (const auto& phase:trace.phases) {EXPECT_EQ(phase.components_before,count);count/=2;EXPECT_EQ(phase.components_after,count);}
    EXPECT_EQ(trace.forest.edges.size(),n-1);EXPECT_EQ(trace.forest.total_weight,dense_prim(n,edges));
  }
}
TEST(BoruvkaTrace, DyadicGeneratorHasSpecifiedWeightsAndHandlesNonpowers) {
  EXPECT_TRUE(dyadic_path_edges(0).empty());EXPECT_TRUE(dyadic_path_edges(1).empty());
  EXPECT_EQ(dyadic_path_edges(8),(std::vector<WeightedEdge>{
      {0,1,1},{1,2,2},{2,3,1},{3,4,3},{4,5,1},{5,6,2},{6,7,1}}));
  const auto edges=dyadic_path_edges(7);const auto trace=boruvka_trace(7,edges);check_trace(7,edges,trace);
  EXPECT_EQ(trace.forest.total_weight,10);
  EXPECT_THROW(dyadic_path_edges(std::numeric_limits<std::size_t>::max()),std::length_error);
}
TEST(BoruvkaTrace, InvalidEdgesAndNonfiniteWeightsRejectedBeforeTraversal) {
  const double inf=std::numeric_limits<double>::infinity(),missing=std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(boruvka(0,{{0,0,1}}),std::invalid_argument);
  EXPECT_THROW(boruvka_trace(2,{{0,2,1}}),std::invalid_argument);
  EXPECT_THROW(boruvka(2,{{0,1,inf}}),std::invalid_argument);
  EXPECT_THROW(boruvka_trace(2,{{0,1,missing}}),std::invalid_argument);
}
TEST(BoruvkaTrace, UnrepresentableForestWeightIsReported) {
  const double maximum=std::numeric_limits<double>::max();
  EXPECT_EQ(boruvka(2,{{0,1,maximum}}).total_weight,maximum);
  EXPECT_THROW(boruvka(3,{{0,1,maximum},{1,2,maximum}}),std::overflow_error);
  EXPECT_THROW(boruvka_trace(3,{{0,1,-maximum},{1,2,-maximum}}),std::overflow_error);
}
