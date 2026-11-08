#include <gtest/gtest.h>
#include <datamunge/algorithms/dijkstra.hpp>
#include <datamunge/algorithms/dag_longest_path.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
using namespace datamunge::algorithms;
namespace {
using Edge=std::tuple<std::size_t,std::size_t,double>;
const std::vector<Edge> roads{{0,1,4},{0,2,2},{2,1,1},{1,3,5},{2,3,8},{2,4,10},{1,4,6},{3,5,3},{4,5,4},{3,4,2}};
const double inf=std::numeric_limits<double>::infinity();
void certificate(const DijkstraTrace& t,const std::vector<Edge>& edges){
 std::vector<double> live(t.result.distance.size(),inf);live[t.source]=0;
 std::vector<std::size_t> parent(live.size(),kNoPredecessor),selected(live.size(),kNoPredecessor),order;
 std::multiset<std::pair<double,std::size_t>> heap{{0,t.source}};
 std::size_t checks=0,updates=0,stale=0,peak=1;
 for(const auto& pop:t.pops){ASSERT_FALSE(heap.empty());EXPECT_EQ(heap.size(),pop.heap_before);
  EXPECT_EQ(*heap.begin(),std::make_pair(pop.key,pop.vertex));heap.erase(heap.begin());
  EXPECT_DOUBLE_EQ(pop.live_distance,live[pop.vertex]);EXPECT_EQ(pop.stale,pop.key>live[pop.vertex]);
  if(pop.stale){++stale;EXPECT_EQ(pop.relaxation_count,0u);}
  else {order.push_back(pop.vertex);std::vector<std::size_t> outgoing;
   for(std::size_t e=0;e<edges.size();++e)if(std::get<0>(edges[e])==pop.vertex)outgoing.push_back(e);
   ASSERT_EQ(outgoing.size(),pop.relaxation_count);ASSERT_EQ(pop.relaxation_begin,checks);
   for(std::size_t k=0;k<outgoing.size();++k){const auto& event=t.relaxations[checks++];const auto e=outgoing[k];
    const auto [u,v,w]=edges[e];EXPECT_EQ(event.edge_index,e);EXPECT_EQ(event.from,u);EXPECT_EQ(event.to,v);EXPECT_DOUBLE_EQ(event.weight,w);
    EXPECT_DOUBLE_EQ(event.from_distance,live[u]);EXPECT_DOUBLE_EQ(event.distance_before,live[v]);EXPECT_EQ(event.predecessor_before,parent[v]);
    EXPECT_DOUBLE_EQ(event.candidate,live[u]+w);EXPECT_EQ(event.improved,event.candidate<live[v]);
    if(event.improved){++updates;live[v]=event.candidate;parent[v]=u;selected[v]=e;heap.emplace(event.candidate,v);peak=std::max(peak,heap.size());}
    EXPECT_EQ(event.heap_after,heap.size());
   }
  }EXPECT_EQ(pop.heap_after,heap.size());
 }
 EXPECT_TRUE(heap.empty());EXPECT_EQ(live,t.result.distance);EXPECT_EQ(parent,t.result.predecessor);EXPECT_EQ(selected,t.predecessor_edge);
 EXPECT_EQ(order,t.settled_order);EXPECT_EQ(checks,t.work.edges_examined);EXPECT_EQ(updates,t.work.successful_relaxations);
 EXPECT_EQ(stale,t.work.stale_pops);EXPECT_EQ(peak,t.work.max_heap_size);EXPECT_EQ(order.size(),t.work.settled_vertices);
 EXPECT_EQ(updates+1,t.work.heap_pushes);EXPECT_EQ(t.work.heap_pushes,t.work.heap_pops);EXPECT_EQ(t.work.heap_pops,t.pops.size());
 EXPECT_EQ(t.relaxations.size(),checks);EXPECT_EQ(t.work.heap_pops,t.work.stale_pops+t.work.settled_vertices);
}
}
TEST(DijkstraTrace, TinyGraphRecordsRealStaleRemovalAndStrictImprovement){
 const std::vector<Edge> edges{{0,1,4},{0,2,1},{2,1,1}};const auto t=dijkstra_trace(3,edges,0,7);
 EXPECT_EQ(t.result.distance,(std::vector<double>{0,2,1}));EXPECT_EQ(t.settled_order,(std::vector<std::size_t>{0,2,1}));
 ASSERT_EQ(t.pops.size(),4u);EXPECT_EQ(t.pops[3].vertex,1u);EXPECT_DOUBLE_EQ(t.pops[3].key,4);EXPECT_DOUBLE_EQ(t.pops[3].live_distance,2);EXPECT_TRUE(t.pops[3].stale);
 EXPECT_EQ(t.predecessor_edge,(std::vector<std::size_t>{kNoPredecessor,2,1}));EXPECT_EQ(reconstruct_path(t.result,1),(std::vector<std::size_t>{0,2,1}));certificate(t,edges);
}
TEST(DijkstraTrace, OriginalSixCitiesRetainDistancesParentsRoutesAndLogicalCounts){
 const auto t=dijkstra_trace(6,roads,0,19);EXPECT_EQ(t.result.distance,(std::vector<double>{0,3,2,8,9,11}));
 EXPECT_EQ(reconstruct_path(t.result,5),(std::vector<std::size_t>{0,2,1,3,5}));EXPECT_EQ(t.settled_order,(std::vector<std::size_t>{0,2,1,3,4,5}));
 EXPECT_EQ(t.work.edges_examined,10u);EXPECT_EQ(t.work.successful_relaxations,8u);EXPECT_EQ(t.work.heap_pops,9u);EXPECT_EQ(t.work.stale_pops,3u);EXPECT_EQ(t.work.max_heap_size,5u);
 EXPECT_EQ(dijkstra(6,roads,0).predecessor,t.result.predecessor);certificate(t,roads);
}
TEST(DijkstraTrace, AllThreeVertexIntegerDigraphsMatchIndependentFloydWarshall){
 for(unsigned mask=0;mask<4096;++mask){unsigned digits=mask;std::vector<Edge> edges;double distance[3][3];
  for(std::size_t i=0;i<3;++i)for(std::size_t j=0;j<3;++j)distance[i][j]=i==j?0:inf;
  for(std::size_t u=0;u<3;++u)for(std::size_t v=0;v<3;++v)if(u!=v){const unsigned choice=digits%4;digits/=4;if(choice){const double w=choice-1;edges.emplace_back(u,v,w);distance[u][v]=w;}}
  for(std::size_t k=0;k<3;++k)for(std::size_t i=0;i<3;++i)for(std::size_t j=0;j<3;++j)distance[i][j]=std::min(distance[i][j],distance[i][k]+distance[k][j]);
  for(std::size_t source=0;source<3;++source){const auto t=dijkstra_trace(3,edges,source);const auto plain=dijkstra(3,edges,source);
   for(std::size_t v=0;v<3;++v)EXPECT_DOUBLE_EQ(t.result.distance[v],distance[source][v]);
   EXPECT_EQ(plain.distance,t.result.distance);EXPECT_EQ(plain.predecessor,t.result.predecessor);certificate(t,edges);
  }
 }
}
TEST(DijkstraTrace, ZeroCyclesLoopsDuplicatesAndTieOrderingDoNotReexpand){
 const std::vector<Edge> edges{{0,2,1},{0,1,1},{1,3,0},{2,3,0},{3,2,0},{3,3,0},{0,1,1},{1,0,0}};
 const auto t=dijkstra_trace(5,edges,0);EXPECT_EQ(t.settled_order,(std::vector<std::size_t>{0,1,2,3}));
 EXPECT_EQ(t.result.predecessor[3],1u);EXPECT_EQ(t.predecessor_edge[1],1u);EXPECT_EQ(t.work.successful_relaxations,3u);
 EXPECT_EQ(t.work.stale_pops,0u);EXPECT_EQ(t.result.distance[4],inf);EXPECT_TRUE(reconstruct_path(t.result,4).empty());certificate(t,edges);
}
TEST(DijkstraTrace, ParallelArcsExposeHeapGrowthAndExactBudgetBoundaries){
 for(std::size_t n:{0u,1u,2u,3u,16u,257u,4096u}){
  const auto edges=dijkstra_parallel_workload(n);const auto t=dijkstra_trace(2,edges,0,2*n+1);certificate(t,edges);
  EXPECT_EQ(t.work.heap_pushes,n+1);EXPECT_EQ(t.work.max_heap_size,std::max(std::size_t{1},n));
  EXPECT_EQ(t.work.stale_pops,n?n-1:0);EXPECT_EQ(t.work.settled_vertices,n?2u:1u);
  EXPECT_DOUBLE_EQ(t.result.distance[1],n?1:inf);
  if(n){EXPECT_EQ(t.predecessor_edge[1],n-1);
   EXPECT_THROW((void)dijkstra_trace(2,edges,0,2*n),std::length_error);
  }
 }
 EXPECT_NO_THROW(dijkstra_trace(6,roads,0,0));
 EXPECT_THROW((void)dijkstra_trace(6,roads,0,18),std::length_error);
 EXPECT_THROW((void)dijkstra_parallel_workload(4,3),std::length_error);
 EXPECT_NO_THROW(dijkstra_parallel_workload(4,4));
 EXPECT_NO_THROW(dijkstra_parallel_workload(4,0));
 EXPECT_THROW((void)dijkstra_parallel_workload(std::numeric_limits<std::size_t>::max(),0),std::length_error);
}
TEST(DijkstraTrace, ForwardBandsMatchClosedFormDistancesAndWork){
 for(std::size_t n:{1u,2u,5u,32u,257u})for(std::size_t width:{1u,2u,8u,300u}){
  const auto edges=dag_forward_band_edges(n,width);const auto t=dijkstra_trace(n,edges,0);const auto w=std::min(n-1,width);
  EXPECT_EQ(t.work.edges_examined,w*(n-w)+w*(w-1)/2);EXPECT_EQ(t.work.successful_relaxations,n-1);
  EXPECT_EQ(t.work.heap_pushes,n);EXPECT_EQ(t.work.stale_pops,0u);EXPECT_EQ(t.work.max_heap_size,std::max(w,std::size_t{1}));
  if(w)for(std::size_t v=0;v<n;++v)EXPECT_DOUBLE_EQ(t.result.distance[v],(v+w-1)/w);
 }
}
TEST(DijkstraTrace, ValidatesAllArcsBeforeAllocationAndRejectsNonfiniteExaminedSums){
 for(double bad:{-1.,inf,-inf,std::numeric_limits<double>::quiet_NaN()}){
  const std::vector<Edge> edges{{1,2,bad}};
  EXPECT_THROW((void)dijkstra_trace(3,edges,0),std::invalid_argument);
  EXPECT_THROW((void)dijkstra(3,edges,0),std::invalid_argument);
 }
 EXPECT_THROW((void)dijkstra(std::numeric_limits<std::size_t>::max(),{{1,2,-1}},0),std::invalid_argument);
 EXPECT_THROW((void)dijkstra_trace(0,{},0),std::invalid_argument);
 EXPECT_THROW((void)dijkstra_trace(2,{{0,2,0}},0),std::invalid_argument);
 EXPECT_THROW((void)dijkstra_trace(2,{{2,0,0}},0),std::invalid_argument);
 EXPECT_THROW((void)dijkstra_trace(3,{{0,1,1e308},{1,2,1e308}},0,3),std::length_error);
 EXPECT_THROW((void)dijkstra_trace(3,{{0,1,1e308},{1,2,1e308}},0,4),std::overflow_error);
 EXPECT_THROW((void)dijkstra(3,{{0,1,1e308},{0,2,1},{1,2,1e308}},0),std::overflow_error);
 const double max=std::numeric_limits<double>::max(),tiny=std::numeric_limits<double>::denorm_min();
 EXPECT_DOUBLE_EQ(dijkstra(2,{{0,1,max}},0).distance[1],max);
 EXPECT_DOUBLE_EQ(dijkstra(3,{{0,1,tiny},{1,2,tiny}},0).distance[2],2*tiny);
 EXPECT_DOUBLE_EQ(dijkstra(2,{{0,1,-0.}},0).distance[1],0);
}
TEST(DijkstraTrace, PathRecoveryRejectsMalformedChainsWithoutUncheckedIndexingOrLoops){
 EXPECT_THROW((void)reconstruct_path(ShortestPathResult{{0,1},{kNoPredecessor}},1),std::invalid_argument);
 EXPECT_THROW((void)reconstruct_path(ShortestPathResult{{0,1},{kNoPredecessor,9}},1),std::invalid_argument);
 EXPECT_THROW((void)reconstruct_path(ShortestPathResult{{0,1},{kNoPredecessor,1}},1),std::invalid_argument);
 EXPECT_THROW((void)reconstruct_path(ShortestPathResult{{0,1,2},{kNoPredecessor,2,1}},2),std::invalid_argument);
 EXPECT_THROW((void)reconstruct_path(ShortestPathResult{{0,-1},{kNoPredecessor,0}},1),std::invalid_argument);
 EXPECT_THROW((void)reconstruct_path(ShortestPathResult{{inf,1},{kNoPredecessor,0}},1),std::invalid_argument);
 EXPECT_THROW((void)reconstruct_path(ShortestPathResult{{std::numeric_limits<double>::quiet_NaN()},{kNoPredecessor}},0),std::invalid_argument);
 EXPECT_TRUE(reconstruct_path(ShortestPathResult{{0,inf},{}},1).empty());
 EXPECT_TRUE(reconstruct_path(ShortestPathResult{{0,1},{}},9).empty());
 EXPECT_EQ(reconstruct_path(ShortestPathResult{{0,7},{kNoPredecessor,kNoPredecessor}},1),(std::vector<std::size_t>{1}));
}
TEST(DijkstraTrace, TraceRecordsOwnOperandsAndDoNotRetainCallerViews){
 auto input=roads;const auto t=dijkstra_trace(6,input,0);input.clear();
 EXPECT_DOUBLE_EQ(t.relaxations[0].weight,4);EXPECT_EQ(t.predecessor_edge[5],7u);
 auto copy=t;copy.pops[0].key=99;copy.result.distance[0]=99;EXPECT_DOUBLE_EQ(t.pops[0].key,0);EXPECT_DOUBLE_EQ(t.result.distance[0],0);
}
