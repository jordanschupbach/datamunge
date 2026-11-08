#include <datamunge/algorithms/breadth_first_search.hpp>
#include <datamunge/algorithms/backtracking.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
using namespace datamunge::algorithms;
namespace {
using Edge=std::pair<std::size_t,std::size_t>;
std::vector<std::vector<std::size_t>> floyd(std::size_t n,const std::vector<Edge>& edges,bool directed) {
  std::vector<std::vector<std::size_t>> distance(n,std::vector<std::size_t>(n,n));
  for (std::size_t i=0;i<n;++i) distance[i][i]=0;
  for (auto [u,v]:edges) {distance[u][v]=std::min<std::size_t>(distance[u][v],1);if (!directed) distance[v][u]=std::min<std::size_t>(distance[v][u],1);}
  for (std::size_t k=0;k<n;++k) for (std::size_t i=0;i<n;++i) for (std::size_t j=0;j<n;++j)
    distance[i][j]=std::min(distance[i][j],distance[i][k]+distance[k][j]);
  return distance;
}
void check_trace(std::size_t n,const std::vector<Edge>& edges,std::size_t start,std::size_t goal,bool directed,
                 const BreadthFirstTrace& trace,const std::vector<std::size_t>& shortest) {
  const auto ordinary=breadth_first_search(n,edges,start,goal,directed);
  ASSERT_EQ(trace.result.path,ordinary.path);ASSERT_EQ(trace.result.order,ordinary.order);ASSERT_EQ(trace.result.found,ordinary.found);
  ASSERT_EQ(trace.result.found,shortest[goal]<n);ASSERT_EQ(trace.distance.size(),n);ASSERT_EQ(trace.predecessor.size(),n);
  ASSERT_EQ(trace.directed,directed);ASSERT_EQ(trace.adjacency_entries,edges.size()*(directed?1:2));
  ASSERT_TRUE(trace.goal_requested);
  std::vector<std::vector<std::size_t>> adjacent(n);
  for (auto [u,v]:edges) {adjacent[u].push_back(v);if (!directed) adjacent[v].push_back(u);}
  std::vector<std::size_t> pending{start};std::vector<bool> marked(n),popped(n);marked[start]=true;
  std::size_t inspected=0,peak=1,previous_depth=0;
  ASSERT_EQ(trace.steps.size(),trace.result.order.size());
  for (std::size_t i=0;i<trace.steps.size();++i) {
    const auto& step=trace.steps[i];ASSERT_FALSE(pending.empty());ASSERT_EQ(step.vertex,pending.front());
    ASSERT_EQ(step.vertex,trace.result.order[i]);ASSERT_FALSE(popped[step.vertex]);popped[step.vertex]=true;
    ASSERT_EQ(step.queue_size_before,pending.size());
    if (trace.queues_recorded) ASSERT_EQ(step.queue_before,pending);else ASSERT_TRUE(step.queue_before.empty());
    pending.erase(pending.begin());ASSERT_EQ(step.depth,shortest[step.vertex]);ASSERT_GE(step.depth,previous_depth);previous_depth=step.depth;
    ASSERT_EQ(step.reached_goal,step.vertex==goal);
    std::vector<std::size_t> discovered;
    if (step.reached_goal) {ASSERT_EQ(i+1,trace.steps.size());ASSERT_EQ(step.edges_examined,0);}
    else {
      ASSERT_EQ(step.edges_examined,adjacent[step.vertex].size());
      for (auto v:adjacent[step.vertex]) if (!marked[v]) {
        marked[v]=true;discovered.push_back(v);pending.push_back(v);ASSERT_EQ(trace.predecessor[v],step.vertex);
      }
    }
    ASSERT_EQ(step.discovered,discovered);ASSERT_EQ(step.queue_size_after,pending.size());
    inspected+=step.edges_examined;peak=std::max(peak,pending.size());
  }
  if (!trace.result.found) ASSERT_TRUE(pending.empty());
  ASSERT_EQ(trace.edges_examined,inspected);ASSERT_EQ(trace.max_queue_size,peak);
  ASSERT_EQ(trace.vertices_enqueued,std::count(marked.begin(),marked.end(),true));
  std::vector<std::size_t> layers;
  for (std::size_t i=0;i<n;++i) {
    if (marked[i]) {
      ASSERT_EQ(trace.distance[i],shortest[i]);const auto depth=trace.distance[i];
      if (layers.size()<=depth) layers.resize(depth+1);++layers[depth];
      if (i==start) ASSERT_EQ(trace.predecessor[i],n);
      else {ASSERT_LT(trace.predecessor[i],n);ASSERT_EQ(trace.distance[trace.predecessor[i]]+1,depth);}
    } else {ASSERT_EQ(trace.distance[i],n);ASSERT_EQ(trace.predecessor[i],n);}
    if (!trace.result.found) ASSERT_EQ(trace.distance[i],shortest[i]);
  }
  ASSERT_EQ(trace.layer_counts,layers);
  if (trace.result.found) {
    ASSERT_EQ(trace.result.path.front(),start);ASSERT_EQ(trace.result.path.back(),goal);ASSERT_EQ(trace.result.path.size(),shortest[goal]+1);
    for (std::size_t i=1;i<trace.result.path.size();++i) {
      const auto u=trace.result.path[i-1],v=trace.result.path[i];
      ASSERT_NE(std::find(adjacent[u].begin(),adjacent[u].end(),v),adjacent[u].end());
    }
  } else ASSERT_TRUE(trace.result.path.empty());
}
}
TEST(BreadthFirstTrace, OriginalThreeVertexExampleIsRetained) {
  const std::vector<Edge> edges{{0,1},{1,2}};const auto trace=breadth_first_search_trace(3,edges,0,2);
  check_trace(3,edges,0,2,true,trace,floyd(3,edges,true)[0]);
  EXPECT_EQ(trace.result.path,(std::vector<std::size_t>{0,1,2}));EXPECT_EQ(trace.edges_examined,2);
  EXPECT_EQ(trace.vertices_enqueued,3);EXPECT_EQ(trace.max_queue_size,1);
}
TEST(BreadthFirstTrace, ActualQueueAndFirstDiscoveryTieArePreserved) {
  const std::vector<Edge> edges{{0,1},{0,2},{1,3},{2,4},{4,5},{3,5}};
  const auto trace=breadth_first_search_trace(6,edges,0,5);check_trace(6,edges,0,5,true,trace,floyd(6,edges,true)[0]);
  EXPECT_EQ(trace.result.path,(std::vector<std::size_t>{0,1,3,5}));EXPECT_EQ(trace.predecessor[5],3);
  EXPECT_EQ(trace.layer_counts,(std::vector<std::size_t>{1,2,2,1}));EXPECT_EQ(trace.edges_examined,6);EXPECT_EQ(trace.max_queue_size,2);
  const std::vector<std::vector<std::size_t>> queues{{0},{1,2},{2,3},{3,4},{4,5},{5}};
  ASSERT_EQ(trace.steps.size(),queues.size());for (std::size_t i=0;i<queues.size();++i) EXPECT_EQ(trace.steps[i].queue_before,queues[i]);
}
TEST(BreadthFirstTrace, All66129DirectedGraphTerminalPairsThroughFourVerticesMatchFloyd) {
  std::size_t checked=0,graphs=0,full_traversals=0;
  for (std::size_t n=1;n<=4;++n) {
    std::vector<Edge> possible;for (std::size_t u=0;u<n;++u) for (std::size_t v=0;v<n;++v) if (u!=v) possible.emplace_back(u,v);
    for (std::size_t mask=0;mask<(std::size_t{1}<<possible.size());++mask) {
      std::vector<Edge> edges;for (std::size_t i=0;i<possible.size();++i) if (mask&(std::size_t{1}<<i)) edges.push_back(possible[i]);
      const auto shortest=floyd(n,edges,true);++graphs;
      for (std::size_t start=0;start<n;++start) {
        const auto full=breadth_first_traversal(n,edges,start,true,(mask+start)%2==0);
        ASSERT_EQ(full.distance,shortest[start]);ASSERT_FALSE(full.goal_requested);ASSERT_FALSE(full.result.found);
        ASSERT_TRUE(full.result.path.empty());
        ASSERT_EQ(full.result.order.size(),std::count_if(shortest[start].begin(),shortest[start].end(),[&](auto d){return d<n;}));
        for (std::size_t v=0;v<n;++v) if (v!=start && full.distance[v]<n) {
          ASSERT_LT(full.predecessor[v],n);ASSERT_EQ(full.distance[full.predecessor[v]]+1,full.distance[v]);
          ASSERT_NE(std::find(edges.begin(),edges.end(),Edge{full.predecessor[v],v}),edges.end());
        }
        ++full_traversals;
        for (std::size_t goal=0;goal<n;++goal) {
          const auto trace=breadth_first_search_trace(n,edges,start,goal,true,(mask+start+goal)%2==0);
          check_trace(n,edges,start,goal,true,trace,shortest[start]);if (::testing::Test::HasFatalFailure()) return;++checked;
        }
      }
    }
  }
  EXPECT_EQ(checked,66129);EXPECT_EQ(graphs,4165);EXPECT_EQ(full_traversals,16585);
}
TEST(BreadthFirstTrace, ShuffledMultigraphsLoopsAndUndirectedEntriesMatchFloyd) {
  std::mt19937_64 rng(472);
  for (int trial=0;trial<1000;++trial) {
    const std::size_t n=1+rng()%8;std::vector<Edge> edges;
    for (std::size_t i=0,count=rng()%40;i<count;++i) edges.emplace_back(rng()%n,rng()%n);
    std::shuffle(edges.begin(),edges.end(),rng);
    for (bool directed:{true,false}) {
      const auto shortest=floyd(n,edges,directed);const auto start=rng()%n,goal=rng()%n;
      check_trace(n,edges,start,goal,directed,breadth_first_search_trace(n,edges,start,goal,directed),shortest[start]);
      if (::testing::Test::HasFatalFailure()) return;
    }
  }
}
TEST(BreadthFirstTrace, GoalPopCanLeaveDiscoveredDeeperVerticesPending) {
  const std::vector<Edge> edges{{0,1},{0,2},{1,3},{2,4}};
  const auto trace=breadth_first_search_trace(5,edges,0,2);check_trace(5,edges,0,2,true,trace,floyd(5,edges,true)[0]);
  EXPECT_EQ(trace.result.order,(std::vector<std::size_t>{0,1,2}));EXPECT_EQ(trace.distance,(std::vector<std::size_t>{0,1,1,2,5}));
  EXPECT_EQ(trace.layer_counts,(std::vector<std::size_t>{1,2,1}));EXPECT_EQ(trace.steps.back().queue_size_after,1);
  EXPECT_TRUE(trace.steps.back().reached_goal);EXPECT_EQ(trace.steps.back().edges_examined,0);
}
TEST(BreadthFirstTrace, StartEqualsGoalStillValidatesEveryEdge) {
  const auto trace=breadth_first_search_trace(3,{{0,1},{1,2}},1,1);
  EXPECT_EQ(trace.result.path,(std::vector<std::size_t>{1}));EXPECT_EQ(trace.result.order,trace.result.path);
  EXPECT_EQ(trace.distance,(std::vector<std::size_t>{3,0,3}));EXPECT_EQ(trace.edges_examined,0);EXPECT_EQ(trace.vertices_enqueued,1);
  EXPECT_THROW(breadth_first_search(3,{{0,3}},1,1),std::invalid_argument);
}
TEST(BreadthFirstTrace, DisconnectedAndReverseDirectedCasesAreExplicit) {
  EXPECT_FALSE(breadth_first_search(3,{{0,1},{1,2}},2,0).found);
  EXPECT_EQ(breadth_first_search(3,{{0,1},{1,2}},2,0,false).path,(std::vector<std::size_t>{2,1,0}));
  const std::vector<Edge> edges{{0,0},{0,1},{0,1},{1,1},{1,2},{2,0}};
  const auto trace=breadth_first_search_trace(4,edges,0,3,false);check_trace(4,edges,0,3,false,trace,floyd(4,edges,false)[0]);
  EXPECT_EQ(trace.edges_examined,12);EXPECT_EQ(trace.vertices_enqueued,3);EXPECT_FALSE(trace.result.found);
}
TEST(BreadthFirstTrace, LengthOnlyCaptureKeepsWideSearchLinearInStoredEvents) {
  const std::size_t vertices=20000;std::vector<Edge> edges;
  for (std::size_t i=1;i<vertices;++i) edges.emplace_back(0,i);
  const auto trace=breadth_first_search_trace(vertices+1,edges,0,vertices,true,false);
  EXPECT_EQ(trace.vertices_enqueued,vertices);EXPECT_EQ(trace.steps.size(),vertices);EXPECT_EQ(trace.max_queue_size,vertices-1);
  EXPECT_EQ(trace.edges_examined,vertices-1);EXPECT_EQ(trace.layer_counts,(std::vector<std::size_t>{1,vertices-1}));
  EXPECT_FALSE(trace.queues_recorded);for (const auto& step:trace.steps) EXPECT_TRUE(step.queue_before.empty());
  EXPECT_EQ(trace.steps[1].queue_size_before,vertices-1);EXPECT_EQ(trace.steps.back().queue_size_before,1);
}
TEST(BreadthFirstTrace, GridDistancesAndCompleteIncidenceCountsMatchClosedForms) {
  for (const auto [rows,cols]:std::vector<std::pair<std::size_t,std::size_t>>{{1,1},{1,8},{8,1},{5,7},{12,10}}) {
    const auto vertices=rows*cols;const auto edges=grid_graph_edges(rows,cols);
    const auto trace=breadth_first_search_trace(vertices+1,edges,0,vertices,false,false);
    EXPECT_EQ(edges.size(),rows*(cols-1)+cols*(rows-1));EXPECT_EQ(trace.edges_examined,2*edges.size());EXPECT_EQ(trace.vertices_enqueued,vertices);
    EXPECT_FALSE(trace.result.found);EXPECT_EQ(trace.layer_counts.size(),rows+cols-1);
    for (std::size_t r=0;r<rows;++r) for (std::size_t c=0;c<cols;++c) EXPECT_EQ(trace.distance[r*cols+c],r+c);
    EXPECT_EQ(trace.distance.back(),vertices+1);
  }
}
TEST(BreadthFirstTrace, CompleteTraversalNeedsNoArtificialGoalAndExpandsEveryReachableVertex) {
  const std::vector<Edge> edges{{0,1},{0,2},{1,3},{2,4},{4,4}};
  const auto full=breadth_first_traversal(6,edges,0);const auto shortest=floyd(6,edges,true)[0];
  EXPECT_EQ(full.distance,shortest);EXPECT_FALSE(full.goal_requested);EXPECT_FALSE(full.queues_recorded);
  EXPECT_FALSE(full.result.found);EXPECT_TRUE(full.result.path.empty());EXPECT_EQ(full.result.order,(std::vector<std::size_t>{0,1,2,3,4}));
  EXPECT_EQ(full.edges_examined,edges.size());EXPECT_EQ(full.vertices_enqueued,5);EXPECT_EQ(full.predecessor[5],6);
  for (const auto& step:full.steps) {EXPECT_FALSE(step.reached_goal);EXPECT_TRUE(step.queue_before.empty());}
  const auto queues=breadth_first_traversal(6,edges,0,true,true);
  ASSERT_EQ(queues.steps.size(),full.steps.size());EXPECT_EQ(queues.distance,full.distance);
  EXPECT_TRUE(queues.queues_recorded);EXPECT_EQ(queues.steps[1].queue_before,(std::vector<std::size_t>{1,2}));
  EXPECT_EQ(queues.steps.back().queue_size_after,0);
  const auto grid=grid_graph_edges(8,10);const auto lattice=breadth_first_traversal(80,grid,0,false);
  EXPECT_EQ(lattice.vertices_enqueued,80);EXPECT_EQ(lattice.edges_examined,2*grid.size());
  for (std::size_t r=0;r<8;++r) for (std::size_t c=0;c<10;++c) EXPECT_EQ(lattice.distance[r*10+c],r+c);
  EXPECT_THROW(breadth_first_traversal(0,{},0),std::invalid_argument);
  EXPECT_THROW(breadth_first_traversal(2,{{2,0}},0),std::invalid_argument);
}
TEST(BreadthFirstTrace, InvalidTerminalsAndEdgesRejectedBeforeAllocation) {
  EXPECT_THROW(breadth_first_search(0,{},0,0),std::invalid_argument);
  EXPECT_THROW(breadth_first_search_trace(2,{},2,1),std::invalid_argument);
  EXPECT_THROW(breadth_first_search(2,{{0,2}},0,1),std::invalid_argument);
  const auto large=std::numeric_limits<std::size_t>::max();
  EXPECT_THROW(breadth_first_search(large,{{large,0}},0,0),std::invalid_argument);
}
