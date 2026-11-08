#include <gtest/gtest.h>
#include <datamunge/algorithms/dag_longest_path.hpp>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
namespace {
void verify_integer_dag(std::size_t n,const std::vector<WeightedEdge>& edges,std::size_t source) {
  const auto t=dag_longest_paths_trace(n,edges,source),copy=t;
  const auto legacy=dag_longest_paths(n,edges,source);
  EXPECT_EQ(t.result.distance,legacy.distance);
  EXPECT_EQ(t.result.predecessor,legacy.predecessor);
  // Independent enumeration of all source paths, with exact small-integer sums.
  const long long absent=std::numeric_limits<long long>::min();std::vector<long long> best(n,absent);
  std::function<void(std::size_t,long long)> visit=[&](std::size_t u,long long value) {
    best[u]=std::max(best[u],value);for(const auto& e:edges) if(std::get<0>(e)==u) visit(std::get<1>(e),value+static_cast<long long>(std::get<2>(e)));
  };visit(source,0);
  ASSERT_EQ(t.topological_order.size(),n);std::vector<std::size_t> position(n,n);
  for(std::size_t k=0;k<n;++k) {ASSERT_LT(t.topological_order[k],n);
  EXPECT_EQ(position[t.topological_order[k]],n);position[t.topological_order[k]]=k;}
  for(const auto& e:edges) EXPECT_LT(position[std::get<0>(e)],position[std::get<1>(e)]);
  std::vector<double> distance(n,-std::numeric_limits<double>::infinity());distance[source]=0;
  std::vector<std::size_t> parent(n,n),parent_edge(n,edges.size());std::size_t event=0,updates=0,reached=0;
  for(auto u:t.topological_order) if(std::isfinite(distance[u])) {
    ++reached;for(std::size_t i=0;i<edges.size();++i) if(std::get<0>(edges[i])==u) {
      ASSERT_LT(event,t.relaxations.size());const auto& e=t.relaxations[event++];const auto v=std::get<1>(edges[i]);const auto candidate=distance[u]+std::get<2>(edges[i]);
      EXPECT_EQ(e.edge_index,i);
  EXPECT_EQ(e.from_distance,distance[u]);
  EXPECT_EQ(e.previous_distance,distance[v]);
  EXPECT_EQ(e.candidate,candidate);
  EXPECT_EQ(e.improved,candidate>distance[v]);
      if(candidate>distance[v]) {distance[v]=candidate;parent[v]=u;parent_edge[v]=i;++updates;}
    }
  }
  EXPECT_EQ(event,t.relaxations.size());
  EXPECT_EQ(distance,t.result.distance);
  EXPECT_EQ(parent,t.result.predecessor);
  EXPECT_EQ(parent_edge,t.predecessor_edge);
  EXPECT_EQ(t.work.topological_pops,n);
  EXPECT_EQ(t.work.topological_edge_visits,edges.size());
  EXPECT_EQ(t.work.reachable_vertices,reached);
  EXPECT_EQ(t.work.relaxation_attempts,event);
  EXPECT_EQ(t.work.improvements,updates);
  for(std::size_t v=0;v<n;++v) {
    const auto p=dag_longest_path_to(t,v);
    if(best[v]==absent) {EXPECT_FALSE(p.found);
  EXPECT_EQ(t.result.distance[v],-std::numeric_limits<double>::infinity());
  EXPECT_EQ(t.result.predecessor[v],n);
  EXPECT_EQ(t.predecessor_edge[v],edges.size());
  EXPECT_TRUE(p.vertices.empty());
  EXPECT_TRUE(p.edge_indices.empty());}
    else {
      EXPECT_DOUBLE_EQ(t.result.distance[v],static_cast<double>(best[v]));
  ASSERT_TRUE(p.found);
  ASSERT_EQ(p.vertices.size(),p.edge_indices.size()+1);
  EXPECT_EQ(p.vertices.front(),source);
  EXPECT_EQ(p.vertices.back(),v);long long total=0;
      for(std::size_t i=0;i<p.edge_indices.size();++i) {ASSERT_LT(p.edge_indices[i],edges.size());const auto& e=edges[p.edge_indices[i]];
  EXPECT_EQ(std::get<0>(e),p.vertices[i]);
  EXPECT_EQ(std::get<1>(e),p.vertices[i+1]);total+=static_cast<long long>(std::get<2>(e));}
      EXPECT_EQ(total,best[v]);
  EXPECT_EQ(p.weight,t.result.distance[v]);
    }
  }
  EXPECT_EQ(t.edges,copy.edges);
}
}
TEST(DagLongestTrace, OriginalDiamondAndExactEdgeWitness) {
  const auto t=dag_longest_paths_trace(4,{{0,1,2},{0,2,1},{1,3,3},{2,3,8}},0);const auto p=dag_longest_path_to(t,3);
  EXPECT_EQ(t.result.distance,(std::vector<double>{0,2,1,9}));
  EXPECT_EQ(p.vertices,(std::vector<std::size_t>{0,2,3}));
  EXPECT_EQ(p.edge_indices,(std::vector<std::size_t>{1,3}));
  EXPECT_EQ(t.topological_order,(std::vector<std::size_t>{0,1,2,3}));
  ASSERT_EQ(t.relaxations.size(),4);
  EXPECT_EQ(t.relaxations.back().previous_distance,5);
  EXPECT_EQ(t.relaxations.back().candidate,9);
  EXPECT_EQ(t.work.improvements,4);
  verify_integer_dag(4,t.edges,0);
}
TEST(DagLongestTrace, EveryOrderedDagThroughSixVerticesAndEverySourceMatchesAllPaths) {
  std::size_t cases=0;
  for(std::size_t n=1;n<=6;++n) for(std::size_t mask=0;mask<(std::size_t{1}<<(n*(n-1)/2));++mask) {
    std::vector<WeightedEdge> edges;std::size_t bit=0;
    for(std::size_t u=0;u<n;++u) for(std::size_t v=u+1;v<n;++v,++bit) if((mask>>bit)&1) edges.emplace_back(u,v,static_cast<double>(static_cast<int>((u*13+v*7+mask%7)%11)-5));
    for(std::size_t source=0;source<n;++source) {verify_integer_dag(n,edges,source);++cases;}
  }
  EXPECT_EQ(cases,202013);
}
TEST(DagLongestTrace, SeededRelabelledParallelGraphsPreserveExactLengthsAndActualTies) {
  std::mt19937 rng(651);
  for(int trial=0;trial<1000;++trial) {
    const std::size_t n=2+rng()%8;std::vector<std::size_t> label(n);std::iota(label.begin(),label.end(),0);std::shuffle(label.begin(),label.end(),rng);std::vector<WeightedEdge> edges;
    for(std::size_t u=0;u<n;++u) for(std::size_t v=u+1;v<n;++v) if(rng()%3==0) {
      edges.emplace_back(label[u],label[v],static_cast<int>(rng()%21)-10);if(rng()%3==0) edges.emplace_back(label[u],label[v],static_cast<int>(rng()%21)-10);
    }
    std::shuffle(edges.begin(),edges.end(),rng);verify_integer_dag(n,edges,rng()%n);
  }
  const auto t=dag_longest_paths_trace(2,{{0,1,1},{0,1,3},{0,1,3}},0);
  EXPECT_EQ(t.predecessor_edge[1],1);
  EXPECT_EQ(t.work.relaxation_attempts,3);
  EXPECT_EQ(t.work.improvements,2);
  EXPECT_FALSE(t.relaxations.back().improved);
  EXPECT_EQ(dag_longest_path_to(t,1).edge_indices,(std::vector<std::size_t>{1}));
}
TEST(DagLongestTrace, NegativeUnreachableSourceAndTieConventions) {
  const auto t=dag_longest_paths_trace(6,{{0,1,-2},{0,2,-1},{1,3,5},{2,3,4},{4,5,9}},0);
  EXPECT_EQ(t.result.distance[3],3);
  EXPECT_EQ(t.result.predecessor[3],1);
  EXPECT_EQ(t.work.reachable_vertices,4);
  EXPECT_EQ(t.work.relaxation_attempts,4);
  EXPECT_EQ(t.work.topological_edge_visits,5);
  EXPECT_TRUE(dag_longest_path_to(t,1).found);
  EXPECT_EQ(dag_longest_path_to(t,1).weight,-2);
  EXPECT_FALSE(dag_longest_path_to(t,5).found);
  const auto one=dag_longest_paths_trace(1,{},0);
  EXPECT_EQ(one.result.distance,(std::vector<double>{0}));
  EXPECT_EQ(dag_longest_path_to(one,0).vertices,(std::vector<std::size_t>{0}));
  EXPECT_TRUE(one.relaxations.empty());
  verify_integer_dag(6,t.edges,0);
  EXPECT_THROW(dag_longest_paths(0,{},0),std::invalid_argument);
}
TEST(DagLongestTrace, FullDomainValidationGlobalCyclesAndCompleteCaps) {
  EXPECT_THROW(dag_longest_paths(2,{},2),std::invalid_argument);
  EXPECT_THROW(dag_longest_paths(2,{{0,2,1}},0),std::invalid_argument);
  for(double bad:{std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()}) {
    EXPECT_THROW(dag_longest_paths(3,{{1,2,bad}},0),std::invalid_argument);
  }
  EXPECT_THROW(dag_longest_paths(1,{{0,0,0}},0),std::invalid_argument);
  EXPECT_THROW(dag_longest_paths(4,{{2,3,-1},{3,2,-1}},0),std::invalid_argument);
  const auto huge=std::numeric_limits<double>::max();
  EXPECT_THROW(dag_longest_paths(5,{{0,1,huge},{1,2,huge},{3,4,0},{4,3,0}},0),std::invalid_argument);
  EXPECT_THROW(dag_longest_paths_trace(3,{{0,1,1}},0,2,0),std::length_error);
  EXPECT_THROW(dag_longest_paths_trace(3,{{0,1,1},{1,2,1}},0,0,1),std::length_error);
  EXPECT_THROW(dag_longest_paths_trace(3,{{0,1,1},{1,3,1}},0,1,1),std::invalid_argument);
  EXPECT_NO_THROW(dag_longest_paths_trace(3,{{0,1,1},{1,2,1}},0,3,2));
  EXPECT_NO_THROW(dag_longest_paths_trace(3,{{0,1,1},{1,2,1}},0,0,0));
}
TEST(DagLongestTrace, ReachableOverflowRejectsAndRoundedTotalsCanTie) {
  const auto max=std::numeric_limits<double>::max();
  EXPECT_THROW(dag_longest_paths(3,{{0,1,max},{1,2,max}},0),std::overflow_error);
  EXPECT_THROW(dag_longest_paths_trace(3,{{0,1,-max},{1,2,-max},{0,2,0}},0),std::overflow_error);
  EXPECT_NO_THROW(dag_longest_paths(4,{{1,2,max},{2,3,max}},0));
  const double large=std::ldexp(1.0,53);const auto t=dag_longest_paths_trace(4,{{0,1,large},{0,2,large},{1,3,0},{2,3,1}},0);
  EXPECT_EQ(t.result.distance[3],large);
  EXPECT_EQ(t.result.predecessor[3],1);
  EXPECT_EQ(dag_longest_path_to(t,3).vertices,(std::vector<std::size_t>{0,1,3}));
  EXPECT_FALSE(t.relaxations.back().improved);
  const double smaller=std::ldexp(1.0,52);const auto exact=dag_longest_paths(4,{{0,1,smaller},{0,2,smaller},{1,3,0},{2,3,1}},0);
  EXPECT_EQ(exact.distance[3],smaller+1);
  EXPECT_EQ(exact.predecessor[3],2);
}
TEST(DagLongestTrace, ForwardBandGeneratorAllSmallShapesAndCountGuards) {
  for(std::size_t n=0;n<40;++n) for(std::size_t width=0;width<45;++width) {
    const auto edges=dag_forward_band_edges(n,width,-2);std::vector<WeightedEdge> expected;
    for(std::size_t u=0;u<n;++u) for(std::size_t v=u+1;v<n;++v) if(v-u<=width) expected.emplace_back(u,v,-2);
    EXPECT_EQ(edges,expected);
  }
  EXPECT_TRUE(dag_forward_band_edges(std::numeric_limits<std::size_t>::max(),0).empty());
  EXPECT_THROW(dag_forward_band_edges(std::numeric_limits<std::size_t>::max(),std::numeric_limits<std::size_t>::max()),std::length_error);
  EXPECT_THROW(dag_forward_band_edges(4,2,1,4),std::length_error);
  EXPECT_EQ(dag_forward_band_edges(4,2,1,5).size(),5);
  EXPECT_EQ(dag_forward_band_edges(4,2,1,0).size(),5);
  EXPECT_EQ(dag_forward_band_edges(3,std::numeric_limits<std::size_t>::max()).size(),3);
  EXPECT_THROW(dag_forward_band_edges(0,0,std::numeric_limits<double>::infinity()),std::invalid_argument);
}
TEST(DagLongestTrace, OwnedWitnessesAndMalformedLocalRecordsRejectSafely) {
  std::vector<WeightedEdge> edges{{0,1,0},{1,2,0}};const auto t=dag_longest_paths_trace(4,edges,0);edges.clear();
  EXPECT_EQ(dag_longest_path_to(t,2).vertices,(std::vector<std::size_t>{0,1,2}));
  auto good=t;good.topological_order.clear();good.relaxations.clear();
  EXPECT_EQ(dag_longest_path_to(good,2).weight,0);
  EXPECT_THROW(dag_longest_path_to(t,4),std::out_of_range);
  auto bad=t;bad.result.predecessor.pop_back();
  EXPECT_THROW(dag_longest_path_to(bad,2),std::invalid_argument);
  bad=t;bad.predecessor_edge[0]=0;
  EXPECT_THROW(dag_longest_path_to(bad,2),std::invalid_argument);
  bad=t;bad.result.distance[2]=std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(dag_longest_path_to(bad,2),std::invalid_argument);
  bad=t;bad.predecessor_edge[2]=2;
  EXPECT_THROW(dag_longest_path_to(bad,2),std::invalid_argument);
  bad=t;bad.edges[1]={0,2,0};
  EXPECT_THROW(dag_longest_path_to(bad,2),std::invalid_argument);
  bad=t;bad.result.distance[2]=1;
  EXPECT_THROW(dag_longest_path_to(bad,2),std::invalid_argument);
  bad=t;bad.result.predecessor[3]=0;
  EXPECT_THROW(dag_longest_path_to(bad,3),std::invalid_argument);
  bad=t;bad.edges.push_back({2,1,0});bad.predecessor_edge[0]=3;bad.result.predecessor[1]=2;bad.predecessor_edge[1]=2;
  EXPECT_THROW(dag_longest_path_to(bad,2),std::invalid_argument);
}
TEST(DagLongestTrace, ActualLinearPhasesForChainsDenseDagsAndIsolatedVertices) {
  for(std::size_t n:{1,2,8,32,128}) for(std::size_t width:{std::size_t{1},n}) {
    const auto edges=dag_forward_band_edges(n,width);const auto t=dag_longest_paths_trace(n,edges,0);
    EXPECT_EQ(t.result.distance.back(),n-1);
  EXPECT_EQ(t.work.topological_pops,n);
  EXPECT_EQ(t.work.topological_edge_visits,edges.size());
  EXPECT_EQ(t.work.relaxation_attempts,edges.size());
  EXPECT_EQ(t.work.improvements,edges.size());
  }
  const auto t=dag_longest_paths_trace(100000,{},99999);
  EXPECT_EQ(t.work.topological_pops,100000);
  EXPECT_EQ(t.work.reachable_vertices,1);
  EXPECT_EQ(t.work.relaxation_attempts,0);
  EXPECT_EQ(t.topological_order.front(),0);
  EXPECT_EQ(t.topological_order.back(),99999);
}
