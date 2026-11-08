#include <datamunge/algorithms/blossom.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
using namespace datamunge::algorithms;
namespace {
using Edges=std::vector<std::pair<std::size_t,std::size_t>>;
std::vector<std::uint32_t> adjacency(std::size_t n,const Edges& edges){
  std::vector<std::uint32_t> graph(n,0);
  for(auto [u,v]:edges)if(u!=v){graph[u]|=1U<<v;graph[v]|=1U<<u;}
  return graph;
}
int subset_oracle(const std::vector<std::uint32_t>& graph){
  std::vector<int> memo(1U<<graph.size(),-1);memo[0]=0;
  std::function<int(std::uint32_t)> solve=[&](std::uint32_t mask){
    if(memo[mask]>=0)return memo[mask];
    const auto v=std::countr_zero(mask);const auto rest=mask&~(1U<<v);
    int best=solve(rest);
    for(auto candidates=graph[v]&rest;candidates;candidates&=candidates-1){
      const auto u=std::countr_zero(candidates);best=std::max(best,1+solve(rest&~(1U<<u)));
    }
    return memo[mask]=best;
  };
  return solve((1U<<graph.size())-1);
}
void valid_matching(const GeneralMatchingResult& m,const std::vector<std::uint32_t>& graph){
  ASSERT_EQ(m.mate.size(),graph.size());std::size_t matched=0;
  for(std::size_t v=0;v<graph.size();++v){
    const auto u=m.mate[v];if(u==kGeneralMatchingUnmatched)continue;
    ASSERT_LT(u,graph.size());EXPECT_NE(v,u);EXPECT_EQ(m.mate[u],v);
    EXPECT_TRUE(graph[v]&(1U<<u));++matched;
  }
  EXPECT_EQ(m.size,matched/2);EXPECT_EQ(matched%2,0);
}
void valid_trace(const GeneralMatchingTrace& trace,const std::vector<std::uint32_t>& graph){
  valid_matching(trace.result,graph);
  std::vector<std::size_t> previous(graph.size(),kGeneralMatchingUnmatched);std::size_t count=0;
  for(const auto& a:trace.augmentations){
    ASSERT_LT(a.search_index,trace.searches);ASSERT_GE(a.path.size(),2);ASSERT_EQ(a.path.size()%2,0);
    auto sorted=a.path;std::sort(sorted.begin(),sorted.end());
    EXPECT_EQ(std::adjacent_find(sorted.begin(),sorted.end()),sorted.end());
    ASSERT_LT(a.path.front(),graph.size());ASSERT_LT(a.path.back(),graph.size());
    EXPECT_EQ(previous[a.path.front()],kGeneralMatchingUnmatched);
    EXPECT_EQ(previous[a.path.back()],kGeneralMatchingUnmatched);
    for(std::size_t j=0;j+1<a.path.size();++j){
      const auto u=a.path[j],v=a.path[j+1];ASSERT_LT(u,graph.size());ASSERT_LT(v,graph.size());
      EXPECT_TRUE(graph[u]&(1U<<v));EXPECT_EQ(previous[u]==v,j%2==1);
    }
    auto expected=previous;
    for(std::size_t j=0;j<a.path.size();j+=2){expected[a.path[j]]=a.path[j+1];expected[a.path[j+1]]=a.path[j];}
    EXPECT_EQ(expected,a.matching.mate);EXPECT_EQ(a.matching.size,++count);
    valid_matching(a.matching,graph);previous=a.matching.mate;
  }
  EXPECT_EQ(previous,trace.result.mate);EXPECT_EQ(count,trace.result.size);
  for(const auto& b:trace.contractions){
    ASSERT_LT(b.search_index,trace.searches);ASSERT_LT(b.root,graph.size());ASSERT_LT(b.base,graph.size());
    ASSERT_GE(b.members.size(),3);EXPECT_EQ(b.members.size()%2,1);EXPECT_TRUE(std::is_sorted(b.members.begin(),b.members.end()));
    EXPECT_EQ(std::adjacent_find(b.members.begin(),b.members.end()),b.members.end());
    EXPECT_TRUE(std::binary_search(b.members.begin(),b.members.end(),b.base));
    EXPECT_TRUE(std::binary_search(b.members.begin(),b.members.end(),b.first));
    EXPECT_TRUE(std::binary_search(b.members.begin(),b.members.end(),b.second));
    EXPECT_TRUE(graph[b.first]&(1U<<b.second));
    std::vector<std::size_t> before(graph.size(),kGeneralMatchingUnmatched);
    for(const auto& a:trace.augmentations)if(a.search_index<b.search_index)before=a.matching.mate;
    EXPECT_EQ(before[b.root],kGeneralMatchingUnmatched);
    for(auto v:b.members){ASSERT_LT(v,graph.size());if(v!=b.base)
      EXPECT_TRUE(std::binary_search(b.members.begin(),b.members.end(),before[v]));}
    EXPECT_FALSE(std::binary_search(b.members.begin(),b.members.end(),before[b.base]));
  }
  EXPECT_LE(trace.augmentations.size(),trace.searches);EXPECT_LE(trace.searches,graph.size());
}
}
TEST(BlossomTrace, FourVertexOddCycleActuallyChangesTheLiftedRoute){
  const Edges edges{{0,1},{1,2},{2,0},{1,3}};
  const auto t=blossom_maximum_matching_trace(4,edges);valid_trace(t,adjacency(4,edges));
  ASSERT_EQ(t.contractions.size(),1);EXPECT_EQ(t.contractions[0].base,2);
  EXPECT_EQ(t.contractions[0].members,(std::vector<std::size_t>{0,1,2}));
  ASSERT_EQ(t.augmentations.size(),2);EXPECT_EQ(t.augmentations[0].path,(std::vector<std::size_t>{0,1}));
  EXPECT_EQ(t.augmentations[1].path,(std::vector<std::size_t>{2,0,1,3}));EXPECT_EQ(t.result.size,2);
}
TEST(BlossomTrace, EverySimpleGraphThroughSixVerticesMatchesSubsetOracle){
  std::size_t cases=0;
  for(std::size_t n=0;n<=6;++n){
    Edges possible;for(std::size_t u=0;u<n;++u)for(std::size_t v=u+1;v<n;++v)possible.emplace_back(u,v);
    for(std::uint32_t mask=0;mask<(1U<<possible.size());++mask){
      Edges edges;for(std::size_t i=0;i<possible.size();++i)if((mask>>i)&1)edges.push_back(possible[i]);
      const auto graph=adjacency(n,edges);
      const auto t=blossom_maximum_matching_trace(n,edges);
      valid_trace(t,graph);EXPECT_EQ(t.result.size,subset_oracle(graph));
      const auto plain=blossom_maximum_matching(n,edges);EXPECT_EQ(plain.mate,t.result.mate);EXPECT_EQ(plain.size,t.result.size);
      EXPECT_LE(t.edge_scans,2*edges.size()*t.searches);++cases;
    }
  }
  EXPECT_EQ(cases,33868);
}
TEST(BlossomTrace, NestedOddSetsAndRandomEdgeOrdersRemainValidAndOptimal){
  const Edges nested{{0,1},{0,3},{1,6},{1,8},{2,3},{2,6},{2,7},{3,8},{4,7},{5,6},{5,7},{5,8}};
  const auto t=blossom_maximum_matching_trace(9,nested);valid_trace(t,adjacency(9,nested));
  EXPECT_EQ(t.result.size,4);bool has_nested=false;
  for(std::size_t j=0;j<t.contractions.size();++j)for(std::size_t i=0;i<j;++i){
    const auto& a=t.contractions[i];const auto& b=t.contractions[j];
    has_nested|=a.search_index==b.search_index && a.members.size()<b.members.size() &&
        std::includes(b.members.begin(),b.members.end(),a.members.begin(),a.members.end());
  }
  EXPECT_TRUE(has_nested);
  std::mt19937_64 rng(1965);
  for(int sample=0;sample<1000;++sample){
    const std::size_t n=1+rng()%12;Edges edges;
    for(std::size_t u=0;u<n;++u)for(std::size_t v=u+1;v<n;++v)if(rng()%3==0)edges.emplace_back(u,v);
    const auto graph=adjacency(n,edges);const auto optimum=subset_oracle(graph);
    std::shuffle(edges.begin(),edges.end(),rng);const auto trace=blossom_maximum_matching_trace(n,edges);
    valid_trace(trace,graph);EXPECT_EQ(trace.result.size,optimum);
  }
}
TEST(BlossomTrace, EmptyGraphsLoopsDuplicatesAndUnmatchedSentinelAreExplicit){
  EXPECT_TRUE(blossom_maximum_matching_trace(0,{}).result.mate.empty());
  const auto loops=blossom_maximum_matching_trace(3,{{0,0},{2,2}});
  EXPECT_EQ(loops.searches,3);EXPECT_EQ(loops.edge_scans,0);EXPECT_EQ(loops.result.size,0);
  EXPECT_EQ(loops.result.mate,(std::vector<std::size_t>(3,kGeneralMatchingUnmatched)));
  const Edges repeated{{0,1},{0,1},{1,0},{1,2},{2,0},{1,3},{1,3}};
  const auto trace=blossom_maximum_matching_trace(5,repeated);valid_trace(trace,adjacency(5,repeated));
  EXPECT_EQ(trace.result.size,2);EXPECT_EQ(trace.result.mate[4],kGeneralMatchingUnmatched);
  EXPECT_THROW(blossom_maximum_matching(0,{{0,0}}),std::invalid_argument);
  EXPECT_THROW(blossom_maximum_matching_trace(3,{{0,3}}),std::invalid_argument);
  EXPECT_THROW(blossom_maximum_matching(3,{{0,kGeneralMatchingUnmatched}}),std::invalid_argument);
  const auto too_many=static_cast<std::size_t>(std::numeric_limits<int>::max())+1;
  EXPECT_THROW(blossom_maximum_matching(too_many,{}),std::length_error);
  EXPECT_THROW(blossom_maximum_matching_trace(too_many,{}),std::length_error);
}
