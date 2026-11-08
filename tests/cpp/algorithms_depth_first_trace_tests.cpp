#include <gtest/gtest.h>
#include <datamunge/algorithms/search_more.hpp>
#include <datamunge/algorithms/breadth_first_search.hpp>
#include <algorithm>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
namespace {
using Edges=std::vector<std::pair<std::size_t,std::size_t>>;
const Edges branching{{0,1},{1,3},{3,0},{0,2},{2,4},{4,5},{5,6},{0,6}};
struct Oracle {
  std::vector<std::size_t> pre,post,parent,parent_edge,depth,discovery,finish,path;
  std::vector<DepthFirstState> state;
  std::size_t scanned=0,white=0,gray=0,black=0,peak=0;
  bool found=false;
};
Oracle recursive(std::size_t n,const Edges& e,std::size_t start,std::size_t goal,bool complete) {
  Oracle o;o.parent.assign(n,n);o.parent_edge.assign(n,e.size());o.depth.assign(n,n);o.discovery.assign(n,2*n);o.finish.assign(n,2*n);o.state.assign(n,DepthFirstState::unseen);
  std::size_t time=0;std::vector<std::size_t> path;
  std::function<bool(std::size_t)> visit=[&](std::size_t u) {
    o.state[u]=DepthFirstState::active;o.discovery[u]=time++;o.pre.push_back(u);path.push_back(u);o.peak=std::max(o.peak,path.size());o.depth[u]=path.size()-1;
    if(!complete && u==goal) {o.path=path;o.found=true;return true;}
    // Scan the original edge sequence rather than using production adjacency frames.
    for(std::size_t i=0;i<e.size();++i)if(e[i].first==u) {
      const auto v=e[i].second;++o.scanned;
      if(o.state[v]==DepthFirstState::unseen) {++o.white;o.parent[v]=u;o.parent_edge[v]=i;if(visit(v))return true;}
      else if(o.state[v]==DepthFirstState::active)++o.gray;else ++o.black;
    }
    o.state[u]=DepthFirstState::finished;o.finish[u]=time++;o.post.push_back(u);path.pop_back();return false;
  };visit(start);return o;
}
void compare(std::size_t n,const Edges& e,std::size_t s,std::size_t g,bool complete) {
  const auto o=recursive(n,e,s,g,complete);const auto t=complete?depth_first_traversal(n,e,s):depth_first_search_trace(n,e,s,g);
  EXPECT_EQ(t.result.path,o.path);EXPECT_EQ(t.result.found,o.found);EXPECT_EQ(t.result.expanded,o.pre.size());EXPECT_EQ(t.preorder,o.pre);EXPECT_EQ(t.postorder,o.post);
  EXPECT_EQ(t.parent,o.parent);EXPECT_EQ(t.parent_edge,o.parent_edge);EXPECT_EQ(t.depth,o.depth);EXPECT_EQ(t.discovered_at,o.discovery);EXPECT_EQ(t.finished_at,o.finish);EXPECT_EQ(t.state,o.state);
  EXPECT_EQ(t.work.discovered,o.pre.size());EXPECT_EQ(t.work.finished,o.post.size());EXPECT_EQ(t.work.edges_examined,o.scanned);EXPECT_EQ(t.work.unseen_edges,o.white);EXPECT_EQ(t.work.active_edges,o.gray);EXPECT_EQ(t.work.finished_edges,o.black);EXPECT_EQ(t.work.max_stack_size,o.peak);
  EXPECT_EQ(t.events.size(),o.pre.size()+o.post.size()+o.scanned+o.found);
  std::vector<std::size_t> stack;std::vector<DepthFirstState> replayed(n,DepthFirstState::unseen);std::size_t entries=0,exits=0,scans=0,goals=0;
  for(const auto& x:t.events) {
    if(x.kind==DepthFirstEventKind::discover) {EXPECT_EQ(replayed[x.vertex],DepthFirstState::unseen);replayed[x.vertex]=DepthFirstState::active;stack.push_back(x.vertex);++entries;EXPECT_EQ(x.edge_index,o.parent_edge[x.vertex]);EXPECT_EQ(x.depth,stack.size()-1);}
    else if(x.kind==DepthFirstEventKind::finish) {EXPECT_EQ(replayed[x.vertex],DepthFirstState::active);replayed[x.vertex]=DepthFirstState::finished;ASSERT_FALSE(stack.empty());EXPECT_EQ(stack.back(),x.vertex);EXPECT_EQ(x.depth,stack.size()-1);stack.pop_back();++exits;}
    else if(x.kind==DepthFirstEventKind::examine) {ASSERT_FALSE(stack.empty());EXPECT_EQ(stack.back(),x.vertex);ASSERT_LT(x.edge_index,e.size());EXPECT_EQ(e[x.edge_index],(std::pair{x.vertex,x.neighbor}));EXPECT_EQ(x.neighbor_state,replayed[x.neighbor]);EXPECT_EQ(x.depth,stack.size()-1);++scans;}
    else {EXPECT_EQ(x.kind,DepthFirstEventKind::goal);ASSERT_FALSE(stack.empty());EXPECT_EQ(stack.back(),g);++goals;}
    EXPECT_EQ(x.stack_size,stack.size());if(x.kind!=DepthFirstEventKind::examine)EXPECT_EQ(x.neighbor,n);
  }
  EXPECT_EQ(replayed,t.state);EXPECT_EQ(entries,o.pre.size());EXPECT_EQ(exits,o.post.size());EXPECT_EQ(scans,o.scanned);EXPECT_EQ(goals,o.found);if(o.found)EXPECT_EQ(stack,o.path);else EXPECT_TRUE(stack.empty());
  if(!complete) {const auto r=depth_first_search(n,e,s,g);EXPECT_EQ(r.path,o.path);EXPECT_EQ(r.found,o.found);EXPECT_EQ(r.expanded,o.pre.size());}
}
}
TEST(DepthFirstTrace, OriginalTwoVertexExampleAndActualEvents) {
  compare(2,{{0,1}},0,1,false);const auto t=depth_first_search_trace(2,{{0,1}},0,1);EXPECT_EQ(t.result.path,(std::vector<std::size_t>{0,1}));EXPECT_EQ(t.result.expanded,2);EXPECT_EQ(t.events.size(),4);EXPECT_EQ(t.work.finished,0);EXPECT_EQ(t.finished_at,(std::vector<std::size_t>{4,4}));
}
TEST(DepthFirstTrace, ExhaustiveGraphsAgainstRecursiveOracleAndTransitiveClosure) {
  for(std::size_t n=1;n<=4;++n) {
    Edges possible;for(std::size_t u=0;u<n;++u)for(std::size_t v=0;v<n;++v)if(n<4 || u!=v)possible.push_back({u,v});
    for(unsigned mask=0;mask<(1u<<possible.size());++mask) {
      Edges e;std::vector<std::vector<bool>> reach(n,std::vector<bool>(n,false));for(std::size_t i=0;i<n;++i)reach[i][i]=true;
      for(std::size_t i=0;i<possible.size();++i)if(mask&(1u<<i)) {e.push_back(possible[i]);reach[e.back().first][e.back().second]=true;}
      for(std::size_t k=0;k<n;++k)for(std::size_t i=0;i<n;++i)for(std::size_t j=0;j<n;++j)reach[i][j]=reach[i][j]||(reach[i][k]&&reach[k][j]);
      for(std::size_t s=0;s<n;++s) {
        compare(n,e,s,n,true);
        for(std::size_t g=0;g<n;++g) {compare(n,e,s,g,false);EXPECT_EQ(depth_first_search(n,e,s,g).found,reach[s][g]);}
      }
    }
  }
}
TEST(DepthFirstTrace, BacktrackingEarlyStopAndAdjacencyOrderAreNotShortest) {
  const auto t=depth_first_search_trace(7,branching,0,6);EXPECT_EQ(t.result.path,(std::vector<std::size_t>{0,2,4,5,6}));EXPECT_EQ(t.preorder,(std::vector<std::size_t>{0,1,3,2,4,5,6}));EXPECT_EQ(t.postorder,(std::vector<std::size_t>{3,1}));EXPECT_EQ(t.work.edges_examined,7);EXPECT_EQ(t.work.active_edges,1);EXPECT_EQ(t.work.max_stack_size,5);EXPECT_EQ(t.events.size(),17);
  EXPECT_EQ(breadth_first_search(7,branching,0,6).path,(std::vector<std::size_t>{0,6}));
  auto reordered=branching;std::rotate(reordered.begin(),reordered.end()-1,reordered.end());const auto early=depth_first_search_trace(7,reordered,0,6);EXPECT_EQ(early.result.path,(std::vector<std::size_t>{0,6}));EXPECT_EQ(early.result.expanded,2);EXPECT_EQ(early.work.edges_examined,1);
  compare(7,branching,0,6,false);compare(7,reordered,0,6,false);compare(8,branching,0,7,false);
}
TEST(DepthFirstTrace, CompleteIntervalsDuplicateLoopsAndUnseenComponent) {
  auto e=branching;e.push_back({0,1});e.push_back({2,2});e.push_back({7,7});const auto t=depth_first_traversal(8,e,0);compare(8,e,0,8,true);
  EXPECT_FALSE(t.goal_requested);EXPECT_EQ(t.goal,8);EXPECT_FALSE(t.result.found);EXPECT_TRUE(t.result.path.empty());EXPECT_EQ(t.work.discovered,7);EXPECT_EQ(t.work.finished,7);EXPECT_EQ(t.work.edges_examined,10);EXPECT_EQ(t.work.active_edges,2);EXPECT_EQ(t.work.finished_edges,2);EXPECT_EQ(t.events.size(),24);EXPECT_EQ(t.state[7],DepthFirstState::unseen);EXPECT_EQ(t.discovered_at[7],16);
  for(auto u:t.preorder) {EXPECT_LT(t.discovered_at[u],t.finished_at[u]);if(t.parent[u]!=8) {EXPECT_LT(t.discovered_at[t.parent[u]],t.discovered_at[u]);EXPECT_LT(t.finished_at[u],t.finished_at[t.parent[u]]);}}
  const auto tree=rooted_tree_edges(64,2);const auto dag=depth_first_traversal(64,tree,0);auto order=dag.postorder;std::reverse(order.begin(),order.end());std::vector<std::size_t> rank(64);for(std::size_t i=0;i<64;++i)rank[order[i]]=i;for(auto [a,b]:tree)EXPECT_LT(rank[a],rank[b]);EXPECT_EQ(dag.work.active_edges,0);
}
TEST(DepthFirstTrace, SeededOrderedMultigraphs) {
  std::mt19937 rng(1972);for(int trial=0;trial<1000;++trial) {const std::size_t n=1+rng()%8;Edges e;for(unsigned i=0,m=rng()%41;i<m;++i)e.push_back({rng()%n,rng()%n});const auto s=rng()%n,g=rng()%n;compare(n,e,s,g,false);compare(n,e,s,n,true);std::reverse(e.begin(),e.end());compare(n,e,s,g,false);}
}
TEST(DepthFirstTrace, DeepGraphsUseExplicitFrames) {
  const std::size_t n=200000;const auto e=rooted_tree_edges(n,1);const auto r=depth_first_search(n,e,0,n-1);EXPECT_TRUE(r.found);EXPECT_EQ(r.path.size(),n);EXPECT_EQ(r.expanded,n);EXPECT_EQ(r.path.back(),n-1);
  const auto full=depth_first_traversal(20000,rooted_tree_edges(20000,1),0);EXPECT_EQ(full.work.max_stack_size,20000);EXPECT_EQ(full.work.finished,20000);EXPECT_EQ(full.events.size(),59999);EXPECT_EQ(full.postorder.front(),19999);EXPECT_EQ(full.postorder.back(),0);
}
TEST(DepthFirstTrace, CompleteCapsValidationAndTimestampArithmetic) {
  EXPECT_NO_THROW(depth_first_search_trace(7,branching,0,6,{7,8,17}));
  EXPECT_THROW(depth_first_search_trace(7,branching,0,6,{6,8,17}),std::length_error);
  EXPECT_THROW(depth_first_search_trace(7,branching,0,6,{7,7,17}),std::length_error);
  EXPECT_THROW(depth_first_search_trace(7,branching,0,6,{7,8,16}),std::length_error);
  EXPECT_NO_THROW(depth_first_search_trace(7,branching,0,6,{0,0,0}));
  EXPECT_NO_THROW(depth_first_search_trace(7,branching,0,0,{7,8,2}));
  EXPECT_THROW(depth_first_search_trace(7,branching,0,0,{7,8,1}),std::length_error);
  EXPECT_NO_THROW(depth_first_traversal(7,branching,0,{7,8,22}));
  EXPECT_THROW(depth_first_traversal(7,branching,0,{7,8,21}),std::length_error);
  EXPECT_THROW(depth_first_search(0,{},0,0),std::invalid_argument);
  EXPECT_THROW(depth_first_search(1,{{1,0}},0,0),std::invalid_argument);
  EXPECT_THROW(depth_first_search_trace(7,{{0,7}},0,0,{1,1,1}),std::invalid_argument);
  EXPECT_THROW(depth_first_traversal(7,{{7,0}},0,{1,1,1}),std::invalid_argument);
  EXPECT_THROW(depth_first_search_trace(1,{},0,1),std::invalid_argument);
  EXPECT_THROW(depth_first_traversal(std::numeric_limits<std::size_t>::max(),{},0,{0,0,0}),std::length_error);
}
TEST(DepthFirstTrace, RootedTreeShapesAndPreallocationGuards) {
  EXPECT_TRUE(rooted_tree_edges(0).empty());EXPECT_TRUE(rooted_tree_edges(1).empty());
  EXPECT_EQ(rooted_tree_edges(4,2),(Edges{{0,1},{0,2},{1,3}}));EXPECT_EQ(rooted_tree_edges(4,1),(Edges{{0,1},{1,2},{2,3}}));
  for(std::size_t n:{8,16,32,64,128,256,512})for(std::size_t b:{1,2,1024}) {const auto t=depth_first_traversal(n,rooted_tree_edges(n,b),0);EXPECT_EQ(t.work.discovered,n);EXPECT_EQ(t.work.finished,n);EXPECT_EQ(t.work.edges_examined,n-1);EXPECT_EQ(t.events.size(),3*n-1);if(b==1)EXPECT_EQ(t.work.max_stack_size,n);if(b==1024)EXPECT_EQ(t.work.max_stack_size,2);}
  EXPECT_THROW(rooted_tree_edges(0,0),std::invalid_argument);
  EXPECT_THROW(rooted_tree_edges(4,2,2),std::length_error);
  EXPECT_NO_THROW(rooted_tree_edges(4,2,3));
  EXPECT_THROW(rooted_tree_edges(std::numeric_limits<std::size_t>::max()),std::length_error);
}
TEST(DepthFirstTrace, OwnedRecordsAndSingleVertexStates) {
  Edges e{{0,1},{0,1},{1,1}};const auto t=depth_first_traversal(2,e,0);e.clear();EXPECT_EQ(t.edges.size(),3);EXPECT_EQ(t.work.unseen_edges,1);EXPECT_EQ(t.work.active_edges,1);EXPECT_EQ(t.work.finished_edges,1);EXPECT_EQ(t.result.expanded,2);
  const auto own=t;auto edited=t;edited.parent[1]=99;EXPECT_EQ(own.parent[1],0);compare(1,{{0,0},{0,0}},0,0,false);compare(1,{{0,0},{0,0}},0,1,true);
}
