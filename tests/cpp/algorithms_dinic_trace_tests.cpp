#include <gtest/gtest.h>
#include <datamunge/algorithms/advanced_graph.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
using namespace datamunge::algorithms;
namespace {
const std::vector<CapacityEdge> classic{{0,1,16},{0,2,13},{1,2,10},{2,1,4},{1,3,12},{3,2,9},{2,4,14},{4,3,7},{3,5,20},{4,5,4}};
const std::vector<CapacityEdge> reroute{{0,1,1},{0,2,1},{1,3,1},{1,4,1},{2,3,1},{3,5,1},{4,5,1}};
double cut_oracle(std::size_t n,const std::vector<CapacityEdge>& edges,std::size_t s,std::size_t t){
 double best=std::numeric_limits<double>::infinity();
 for(std::size_t mask=0;mask<(std::size_t{1}<<n);++mask)if((mask>>s&1) && !(mask>>t&1)){
  double value=0;for(const auto& [u,v,c]:edges)if((mask>>u&1) && !(mask>>v&1))value+=c;best=std::min(best,value);
 }return best;
}
void replay(const DinicTrace& t){
 const auto& g=t.network;const auto n=g.vertex_count;std::vector<double> residual(2*g.edges.size(),0);std::vector<std::vector<std::size_t>> adjacency(n);
 for(std::size_t e=0;e<g.edges.size();++e){residual[2*e]=std::get<2>(g.edges[e]);adjacency[std::get<0>(g.edges[e])].push_back(2*e);adjacency[std::get<1>(g.edges[e])].push_back(2*e+1);}
 const auto endpoint=[&](std::size_t ai){const auto& [u,v,c]=g.edges[ai/2];(void)c;return ai%2?std::pair{v,u}:std::pair{u,v};};
 double flow=0;std::size_t consumed=0,total_bfs=0,total_dfs=0,total_advance=0,total_updates=0,previous_level=0;
 for(std::size_t phase=0;phase<t.phases.size();++phase){const auto& p=t.phases[phase];std::vector<std::size_t> level(n,n),queue{g.source};level[g.source]=0;std::size_t checks=0;
  for(std::size_t h=0;h<queue.size();++h)for(auto ai:adjacency[queue[h]]){++checks;const auto [u,v]=endpoint(ai);(void)u;if(residual[ai]>0 && level[v]==n){level[v]=level[queue[h]]+1;queue.push_back(v);}}
  EXPECT_EQ(p.levels,level);EXPECT_EQ(p.sink_level,level[g.sink]);EXPECT_EQ(p.bfs_arc_inspections,checks);total_bfs+=checks;
  ASSERT_EQ(p.augmentation_begin,consumed);EXPECT_DOUBLE_EQ(p.flow_before,flow);
  if(level[g.sink]!=n){EXPECT_GT(p.sink_level,previous_level);previous_level=p.sink_level;}else EXPECT_EQ(phase+1,t.phases.size());
  std::size_t path_updates=0;
  for(std::size_t j=0;j<p.augmentation_count;++j){const auto& a=t.augmentations[consumed++];EXPECT_EQ(a.phase,phase);EXPECT_DOUBLE_EQ(a.flow_before,flow);ASSERT_FALSE(a.path.empty());
   std::size_t vertex=g.source;double bottleneck=std::numeric_limits<double>::infinity();
   for(const auto& arc:a.path){const auto ai=2*arc.input_edge+arc.reverse;ASSERT_LT(ai,residual.size());const auto [u,v]=endpoint(ai);EXPECT_EQ(u,vertex);EXPECT_EQ(arc.from,u);EXPECT_EQ(arc.to,v);EXPECT_EQ(level[v],level[u]+1);vertex=v;bottleneck=std::min(bottleneck,residual[ai]);}
   EXPECT_EQ(vertex,g.sink);EXPECT_DOUBLE_EQ(a.amount,bottleneck);EXPECT_GT(a.amount,0);
   for(const auto& arc:a.path){const auto ai=2*arc.input_edge+arc.reverse;EXPECT_DOUBLE_EQ(arc.residual_before,residual[ai]);EXPECT_DOUBLE_EQ(arc.paired_before,residual[ai^1]);residual[ai]-=a.amount;residual[ai^1]+=a.amount;EXPECT_DOUBLE_EQ(arc.residual_after,residual[ai]);EXPECT_DOUBLE_EQ(arc.paired_after,residual[ai^1]);}
   flow+=a.amount;EXPECT_DOUBLE_EQ(a.flow_after,flow);path_updates+=a.path.size();
  }
  EXPECT_DOUBLE_EQ(p.flow_after,flow);EXPECT_LE(p.current_arc_advances,2*g.edges.size());EXPECT_EQ(p.dfs_arc_inspections,p.current_arc_advances+path_updates);
  if(level[g.sink]!=n){std::vector<bool> reached(n,false);queue={g.source};reached[g.source]=true;
   for(std::size_t h=0;h<queue.size();++h)for(auto ai:adjacency[queue[h]]){const auto [u,v]=endpoint(ai);if(residual[ai]>0 && level[v]==level[u]+1 && !reached[v]){reached[v]=true;queue.push_back(v);}}
   EXPECT_FALSE(reached[g.sink]);
  }
  total_dfs+=p.dfs_arc_inspections;total_advance+=p.current_arc_advances;total_updates+=path_updates;
 }
 EXPECT_EQ(consumed,t.augmentations.size());EXPECT_DOUBLE_EQ(flow,t.result.max_flow);EXPECT_EQ(total_bfs,t.work.bfs_arc_inspections);EXPECT_EQ(total_dfs,t.work.dfs_arc_inspections);EXPECT_EQ(total_advance,t.work.current_arc_advances);EXPECT_EQ(total_updates,t.work.path_arc_updates);EXPECT_EQ(t.work.augmentations,consumed);EXPECT_EQ(t.work.bfs_calls,t.phases.size());EXPECT_EQ(t.work.blocking_phases+1,t.work.bfs_calls);
 for(std::size_t e=0;e<g.edges.size();++e)EXPECT_DOUBLE_EQ(t.result.edge_flow[e],residual[2*e+1]);
 const auto certificate=max_flow_certificate(n,g.edges,g.source,g.sink,t.result);EXPECT_TRUE(certificate.feasible);EXPECT_TRUE(certificate.optimal_within_tolerance);
}
}
TEST(DinicTrace, OriginalSingleEdgeAndPositiveScalesRetainFlowInsteadOfThresholding){
 for(double scale:{1e-200,1e-13,1.,1e200}){const double capacity=7.5*scale;const auto t=dinic_max_flow_trace(2,{{0,1,capacity}},0,1,8);
  EXPECT_DOUBLE_EQ(t.result.max_flow,capacity);EXPECT_EQ(t.result.min_cut_source_side,(std::vector<std::size_t>{0}));EXPECT_EQ(t.work.bfs_calls,2u);EXPECT_EQ(t.work.augmentations,1u);EXPECT_EQ(t.work.path_arc_updates,1u);replay(t);}
 EXPECT_THROW((void)dinic_max_flow_trace(2,{{0,1,7.5}},0,1,7),std::length_error);
 EXPECT_NO_THROW(dinic_max_flow_trace(2,{{0,1,7.5}},0,1,0));
}
TEST(DinicTrace, ClassicNetworkHasTwentyThreeFlowAndIndependentCutCertificate){
 const auto t=dinic_max_flow_trace(6,classic,0,5);EXPECT_DOUBLE_EQ(t.result.max_flow,23);
 EXPECT_EQ(t.result.min_cut_source_side,(std::vector<std::size_t>{0,1,2,4}));EXPECT_EQ(dinic_max_flow(6,classic,0,5).edge_flow,t.result.edge_flow);replay(t);
 const auto c=max_flow_certificate(6,classic,0,5,t.result);EXPECT_DOUBLE_EQ(c.cut_capacity,23);EXPECT_EQ(c.crossing_edges,(std::vector<std::size_t>{4,7,9}));
 EXPECT_DOUBLE_EQ(edmonds_karp(6,classic,0,5).max_flow,23);EXPECT_DOUBLE_EQ(push_relabel_max_flow(6,classic,0,5).max_flow,23);
}
TEST(DinicTrace, SecondPhaseUsesReverseResidualToCancelAnEarlierMatching){
 const auto t=dinic_max_flow_trace(6,reroute,0,5);ASSERT_EQ(t.phases.size(),3u);EXPECT_EQ(t.phases[0].sink_level,3u);EXPECT_EQ(t.phases[1].sink_level,5u);EXPECT_EQ(t.phases[2].sink_level,6u);EXPECT_DOUBLE_EQ(t.result.max_flow,2);
 ASSERT_EQ(t.augmentations.size(),2u);const auto& a=t.augmentations[1];ASSERT_EQ(a.path.size(),5u);EXPECT_EQ(a.path[2].input_edge,2u);EXPECT_TRUE(a.path[2].reverse);EXPECT_DOUBLE_EQ(a.path[2].residual_before,1);EXPECT_DOUBLE_EQ(a.path[2].residual_after,0);EXPECT_DOUBLE_EQ(t.result.edge_flow[2],0);replay(t);
}
TEST(DinicTrace, ExhaustiveSmallNetworksMatchIndependentEnumeratedMinimumCuts){
 for(std::size_t n:{3u,4u}){const unsigned radix=n==3?3:2,total=n==3?729:4096;
  for(unsigned code=0;code<total;++code){unsigned digits=code;std::vector<CapacityEdge> edges;
   for(std::size_t u=0;u<n;++u)for(std::size_t v=0;v<n;++v)if(u!=v){const double c=digits%radix;digits/=radix;edges.emplace_back(u,v,c);}
   for(std::size_t s=0;s<n;++s)for(std::size_t t=0;t<n;++t)if(s!=t){const auto trace=dinic_max_flow_trace(n,edges,s,t);
    EXPECT_DOUBLE_EQ(trace.result.max_flow,cut_oracle(n,edges,s,t));const auto plain=dinic_max_flow(n,edges,s,t);EXPECT_EQ(plain.edge_flow,trace.result.edge_flow);EXPECT_EQ(plain.min_cut_source_side,trace.result.min_cut_source_side);replay(trace);}
  }
 }
}
TEST(DinicTrace, SeparatePairsHandleLoopsParallelAntiparallelAndOwnedInputs){
 std::vector<CapacityEdge> edges{{0,0,99},{0,1,2},{0,1,3},{1,0,7},{1,1,88},{1,2,4},{0,2,1},{2,2,50}};
 const auto t=dinic_max_flow_trace(3,edges,0,2);EXPECT_DOUBLE_EQ(t.result.max_flow,5);EXPECT_DOUBLE_EQ(t.result.edge_flow[0],0);EXPECT_DOUBLE_EQ(t.result.edge_flow[4],0);EXPECT_DOUBLE_EQ(t.result.edge_flow[7],0);replay(t);
 edges.clear();EXPECT_EQ(t.network.edges.size(),8u);auto copy=t;copy.augmentations[0].path[0].residual_before=999;EXPECT_NE(t.augmentations[0].path[0].residual_before,999);
}
TEST(DinicTrace, EqualAndStaircaseFamiliesMatchPhaseClosedFormsAndLongPathsUseHeapStack){
 for(std::size_t count:{0u,1u,2u,8u,32u}){
  const auto g=dinic_disjoint_paths(count,4);const auto t=dinic_max_flow_trace(g.vertex_count,g.edges,0,1);EXPECT_DOUBLE_EQ(t.result.max_flow,count);EXPECT_EQ(t.work.blocking_phases,count?1u:0u);EXPECT_EQ(t.work.augmentations,count);replay(t);
  const auto h=dinic_staircase_paths(count);const auto u=dinic_max_flow_trace(h.vertex_count,h.edges,0,1);EXPECT_EQ(h.edges.size(),count*(count+1)/2);EXPECT_EQ(h.vertex_count,2+count*(count-1)/2);EXPECT_EQ(u.work.blocking_phases,count);EXPECT_DOUBLE_EQ(u.result.max_flow,count);replay(u);
 }
 const auto g=dinic_disjoint_paths(1,30000);const auto t=dinic_max_flow_trace(g.vertex_count,g.edges,0,1);EXPECT_DOUBLE_EQ(t.result.max_flow,1);EXPECT_EQ(t.work.max_path_edges,30000u);EXPECT_EQ(t.work.path_arc_updates,30000u);EXPECT_EQ(t.work.bfs_arc_inspections,60001u);EXPECT_EQ(t.work.dfs_arc_inspections,60000u);
}
TEST(DinicTrace, ValidationCapacityBudgetsOverflowAndLostProgressFailExplicitly){
 const double inf=std::numeric_limits<double>::infinity();
 for(double bad:{-1.,inf,-inf,std::numeric_limits<double>::quiet_NaN()}){
  EXPECT_THROW((void)dinic_max_flow(4,{{2,3,bad}},0,1),std::invalid_argument);
 }
 EXPECT_THROW((void)dinic_max_flow(2,{},0,0),std::invalid_argument);
 EXPECT_THROW((void)dinic_max_flow(0,{},0,1),std::invalid_argument);
 EXPECT_THROW((void)dinic_max_flow(2,{{0,2,1}},0,1),std::invalid_argument);
 EXPECT_THROW((void)dinic_max_flow(std::numeric_limits<std::size_t>::max(),{{2,3,-1}},0,1),std::invalid_argument);
 EXPECT_THROW((void)dinic_max_flow(2,{{0,1,1e308},{0,1,1e308}},0,1),std::overflow_error);
 EXPECT_THROW((void)dinic_max_flow(3,{{0,1,1e100},{1,2,1}},0,2),std::runtime_error);
 EXPECT_DOUBLE_EQ(dinic_max_flow(2,{{0,1,std::numeric_limits<double>::max()}},0,1).max_flow,std::numeric_limits<double>::max());
 EXPECT_DOUBLE_EQ(dinic_max_flow(2,{{0,1,std::numeric_limits<double>::denorm_min()}},0,1).max_flow,std::numeric_limits<double>::denorm_min());
 EXPECT_THROW((void)dinic_disjoint_paths(1,0),std::invalid_argument);
 EXPECT_THROW((void)dinic_disjoint_paths(0,2,inf),std::invalid_argument);
 EXPECT_THROW((void)dinic_disjoint_paths(4,3,1,11),std::length_error);
 EXPECT_NO_THROW(dinic_disjoint_paths(4,3,1,12));
 EXPECT_NO_THROW(dinic_disjoint_paths(4,3,1,0));
 EXPECT_THROW((void)dinic_staircase_paths(4,9),std::length_error);
 EXPECT_NO_THROW(dinic_staircase_paths(4,10));
 EXPECT_THROW((void)dinic_staircase_paths(std::numeric_limits<std::size_t>::max(),0),std::length_error);
 EXPECT_THROW((void)dinic_disjoint_paths(std::numeric_limits<std::size_t>::max(),3,1,0),std::length_error);
}
TEST(FlowCertificate, DistinguishesFeasibilityOptimalityAndMalformedRecords){
 auto c=max_flow_certificate(2,{{0,1,10}},0,1,MaxFlowResult{7,{7},{0}});EXPECT_TRUE(c.feasible);EXPECT_FALSE(c.optimal_within_tolerance);EXPECT_DOUBLE_EQ(c.duality_gap,3);
 c=max_flow_certificate(2,{{0,1,10}},0,1,MaxFlowResult{11,{11},{0}});EXPECT_FALSE(c.feasible);EXPECT_DOUBLE_EQ(c.max_capacity_violation,1);EXPECT_DOUBLE_EQ(c.duality_gap,-1);
 c=max_flow_certificate(3,{{0,1,3},{1,2,3}},0,2,MaxFlowResult{3,{3,2},{0}});EXPECT_FALSE(c.feasible);EXPECT_DOUBLE_EQ(c.max_conservation_error,1);EXPECT_DOUBLE_EQ(c.value_balance_error,1);
 c=max_flow_certificate(2,{{0,1,10}},0,1,MaxFlowResult{10,{10},{0,1}});EXPECT_TRUE(c.feasible);EXPECT_FALSE(c.cut_separates);EXPECT_FALSE(c.optimal_within_tolerance);
 c=max_flow_certificate(2,{{0,1,10}},0,1,MaxFlowResult{9.5,{9.5},{0}},.5);EXPECT_TRUE(c.optimal_within_tolerance);EXPECT_DOUBLE_EQ(c.duality_gap,.5);
 EXPECT_THROW((void)max_flow_certificate(2,{{0,1,10}},0,1,MaxFlowResult{10,{},{0}}),std::invalid_argument);
 EXPECT_THROW((void)max_flow_certificate(2,{{0,1,10}},0,1,MaxFlowResult{10,{10},{0,0}}),std::invalid_argument);
 EXPECT_THROW((void)max_flow_certificate(2,{{0,1,10}},0,1,MaxFlowResult{10,{10},{2}}),std::invalid_argument);
 EXPECT_THROW((void)max_flow_certificate(2,{{0,1,10}},0,1,MaxFlowResult{10,{std::numeric_limits<double>::infinity()},{0}}),std::invalid_argument);
 EXPECT_THROW((void)max_flow_certificate(2,{{0,1,10}},0,1,MaxFlowResult{10,{10},{0}},-1),std::invalid_argument);
 EXPECT_THROW((void)max_flow_certificate(2,{{0,1,1e308},{0,1,1e308}},0,1,MaxFlowResult{0,{0,0},{0}}),std::overflow_error);
}
