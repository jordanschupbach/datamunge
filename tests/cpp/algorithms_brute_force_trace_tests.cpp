#include <datamunge/algorithms/brute_force_search.hpp>
#include <datamunge/algorithms/search_more.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <limits>
#include <random>
#include <set>
namespace bf=datamunge::algorithms;
namespace {
using Edges=std::vector<std::pair<std::size_t,std::size_t>>;
using Adjacency=std::vector<std::vector<std::size_t>>;
Adjacency make_adjacency(std::size_t n,const Edges& edges) {
    Adjacency next(n);for (auto [u,v]:edges) next[u].push_back(v);return next;
}
// Independent all-pairs distances followed by greedy first-adjacency shortest-path recovery.
std::vector<std::size_t> floyd_path(std::size_t n,const Edges& edges,std::size_t start,std::size_t goal) {
    std::vector<std::vector<std::size_t>> d(n,std::vector<std::size_t>(n,n));
    for (std::size_t u=0;u<n;++u) d[u][u]=0;
    for (auto [u,v]:edges) if (u!=v) d[u][v]=1;
    for (std::size_t k=0;k<n;++k) for (std::size_t u=0;u<n;++u) for (std::size_t v=0;v<n;++v)
        d[u][v]=std::min(d[u][v],d[u][k]+d[k][v]);
    if (d[start][goal]==n) return {};
    const auto next=make_adjacency(n,edges);
    std::vector<std::size_t> path{start};
    while (path.back()!=goal) {
        const auto u=path.back();bool advanced=false;
        for (auto v:next[u]) if (d[v][goal]+1==d[u][goal]) {path.push_back(v);advanced=true;break;}
        if (!advanced) return {};
    }
    return path;
}
void verify_trace(const bf::BruteForceTrace& trace,const Adjacency& next,std::size_t goal) {
    ASSERT_EQ(trace.steps.size(),trace.result.expanded);
    std::size_t incumbent=0,edges=0,cycles=0,goals=0,improvements=0,prunes=0,max_depth=0;
    std::vector<std::size_t> best;
    for (const auto& row:trace.steps) {
        ASSERT_LT(row.id,trace.steps.size());EXPECT_EQ(&row,&trace.steps[row.id]);
        ASSERT_TRUE(trace.paths_recorded);ASSERT_FALSE(row.path.empty());
        EXPECT_EQ(row.path.size(),row.depth+1);EXPECT_EQ(row.path.back(),row.vertex);
        const std::set<std::size_t> members(row.path.begin(),row.path.end());EXPECT_EQ(members.size(),row.path.size());
        if (row.id) {
            ASSERT_LT(row.parent,row.id);const auto& parent=trace.steps[row.parent];
            auto expected=parent.path;expected.push_back(row.vertex);EXPECT_EQ(row.path,expected);
            EXPECT_FALSE(parent.goal);EXPECT_FALSE(parent.pruned);
            EXPECT_NE(std::find(next[parent.vertex].begin(),next[parent.vertex].end(),row.vertex),next[parent.vertex].end());
        } else EXPECT_EQ(row.depth,0u);
        EXPECT_EQ(row.incumbent_vertices_before,incumbent);
        const bool improved=row.vertex==goal && (!incumbent || row.path.size()<incumbent);
        EXPECT_EQ(row.improved,improved);EXPECT_EQ(row.goal,row.vertex==goal);
        if (improved) {incumbent=row.path.size();best=row.path;++improvements;}
        EXPECT_EQ(row.incumbent_vertices_after,incumbent);
        const bool pruned=!row.goal && incumbent && row.path.size()>=incumbent;
        EXPECT_EQ(row.pruned,pruned);prunes+=pruned;goals+=row.goal;
        const auto examined=row.goal || pruned?0:next[row.vertex].size();
        std::size_t rejected=0;
        if (!row.goal && !pruned) for (auto v:next[row.vertex]) rejected+=members.contains(v);
        EXPECT_EQ(row.edges_examined,examined);EXPECT_EQ(row.cycle_rejections,rejected);
        edges+=examined;cycles+=rejected;max_depth=std::max(max_depth,row.depth);
    }
    EXPECT_EQ(trace.result.path,best);EXPECT_EQ(trace.result.found,incumbent!=0);
    EXPECT_EQ(trace.edges_examined,edges);EXPECT_EQ(trace.cycle_rejections,cycles);
    EXPECT_EQ(trace.goal_visits,goals);EXPECT_EQ(trace.improvements,improvements);
    EXPECT_EQ(trace.bound_prunes,prunes);EXPECT_EQ(trace.max_depth,max_depth);
    EXPECT_EQ(edges-cycles+1,trace.result.expanded);
}
const Edges worked{{0,1},{1,2},{2,0},{2,5},{0,3},{3,5},{0,4},{4,1},{4,5}};
}
TEST(BruteForceTrace, ExhaustiveDirectedGraphsThroughFourVerticesAgreeWithFloydAndFirstShortestTie) {
    std::size_t graphs=0,searches=0;
    for (std::size_t n=1;n<=4;++n) {
        const auto possible=bf::complete_directed_graph_edges(n);
        for (std::size_t mask=0;mask<(std::size_t{1}<<possible.size());++mask) {
            Edges edges;for (std::size_t i=0;i<possible.size();++i) if (mask&(std::size_t{1}<<i)) edges.push_back(possible[i]);
            const auto next=make_adjacency(n,edges);++graphs;
            for (std::size_t start=0;start<n;++start) for (std::size_t goal=0;goal<n;++goal) {
                const auto expected=floyd_path(n,edges,start,goal);
                const auto trace=bf::brute_force_search_trace(n,edges,start,goal);
                const auto ordinary=bf::brute_force_search(n,edges,start,goal);
                EXPECT_EQ(trace.result.path,expected);EXPECT_EQ(ordinary.path,expected);
                EXPECT_EQ(ordinary.found,!expected.empty());EXPECT_EQ(ordinary.expanded,trace.result.expanded);
                verify_trace(trace,next,goal);++searches;
            }
        }
    }
    EXPECT_EQ(graphs,4165u);EXPECT_EQ(searches,66129u);
}
TEST(BruteForceTrace, WorkedCasePreservesVisitOrderAndHistoricalLocalIncumbents) {
    const auto trace=bf::brute_force_search_trace(6,worked,0,5);
    const std::vector<std::vector<std::size_t>> paths{{0},{0,1},{0,1,2},{0,1,2,5},{0,3},{0,3,5},{0,4},{0,4,1},{0,4,5}};
    const std::vector<std::size_t> before{0,0,0,0,4,4,3,3,3},after{0,0,0,4,4,3,3,3,3};
    ASSERT_EQ(trace.steps.size(),9u);
    for (std::size_t i=0;i<paths.size();++i) {
        EXPECT_EQ(trace.steps[i].path,paths[i]);EXPECT_EQ(trace.steps[i].incumbent_vertices_before,before[i]);
        EXPECT_EQ(trace.steps[i].incumbent_vertices_after,after[i]);
    }
    EXPECT_EQ(trace.result.path,(std::vector<std::size_t>{0,3,5}));EXPECT_EQ(trace.result.expanded,9u);
    EXPECT_EQ(trace.edges_examined,9u);EXPECT_EQ(trace.cycle_rejections,1u);
    EXPECT_EQ(trace.goal_visits,3u);EXPECT_EQ(trace.improvements,2u);EXPECT_EQ(trace.bound_prunes,1u);
    EXPECT_TRUE(trace.steps[7].pruned);EXPECT_EQ(trace.steps[7].edges_examined,0u);
    EXPECT_TRUE(trace.steps.back().goal);EXPECT_FALSE(trace.steps.back().improved);
    verify_trace(trace,make_adjacency(6,worked),5);
    bf::BasicPathResult (*original_signature)(std::size_t,const Edges&,std::size_t,std::size_t)=bf::brute_force_search;
    EXPECT_EQ(original_signature(6,worked,0,5).expanded,9u);
}
TEST(BruteForceTrace, OriginalChainDirectedFailureSingletonAndFirstEqualShortestAlternative) {
    const auto chain=bf::brute_force_search(3,{{0,1},{1,2}},0,2);
    EXPECT_EQ(chain.path,(std::vector<std::size_t>{0,1,2}));EXPECT_EQ(chain.expanded,3u);
    EXPECT_FALSE(bf::brute_force_search(3,{{0,1},{1,2}},2,0).found);
    const auto singleton=bf::brute_force_search_trace(3,{{0,1},{1,2}},1,1);
    EXPECT_EQ(singleton.result.path,(std::vector<std::size_t>{1}));EXPECT_EQ(singleton.result.expanded,1u);
    EXPECT_EQ(singleton.edges_examined,0u);EXPECT_EQ(singleton.improvements,1u);
    const Edges tied{{0,4},{4,5},{0,3},{3,5}};
    EXPECT_EQ(bf::brute_force_search(6,tied,0,5).path,(std::vector<std::size_t>{0,4,5}));
}
TEST(BruteForceTrace, RandomOrderedMultigraphsMatchIndependentShortestPathsWithoutMutation) {
    std::mt19937_64 rng(83);
    for (std::size_t trial=0;trial<1000;++trial) {
        const std::size_t n=1+rng()%7;Edges edges;
        for (std::size_t i=0;i<3*n;++i) edges.emplace_back(rng()%n,rng()%n);
        std::shuffle(edges.begin(),edges.end(),rng);const auto saved=edges;
        const auto start=rng()%n,goal=rng()%n;
        const auto trace=bf::brute_force_search_trace(n,edges,start,goal);
        EXPECT_EQ(trace.result.path,floyd_path(n,edges,start,goal));
        EXPECT_EQ(trace.result.path,bf::brute_force_search(n,edges,start,goal).path);
        EXPECT_EQ(edges,saved);verify_trace(trace,make_adjacency(n,edges),goal);
    }
}
TEST(BruteForceTrace, CompleteReachableComponentWithIsolatedGoalHasExactFactorialWork) {
    std::size_t expected=1;
    for (std::size_t k=1;k<=9;++k) {
        if (k>1) expected=1+(k-1)*expected;
        const auto edges=bf::complete_directed_graph_edges(k);
        const auto trace=bf::brute_force_search_trace(k+1,edges,0,k,{0,false});
        EXPECT_EQ(trace.result.expanded,expected);EXPECT_EQ(trace.edges_examined,(k-1)*expected);
        EXPECT_EQ(trace.cycle_rejections,trace.edges_examined-expected+1);
        EXPECT_EQ(trace.max_depth,k-1);EXPECT_FALSE(trace.result.found);EXPECT_TRUE(trace.result.path.empty());
        EXPECT_EQ(trace.goal_visits,0u);EXPECT_EQ(trace.improvements,0u);EXPECT_EQ(trace.bound_prunes,0u);
        for (const auto& row:trace.steps) EXPECT_TRUE(row.path.empty());
        EXPECT_EQ(bf::brute_force_search(k+1,edges,0,k).expanded,expected);
    }
    EXPECT_EQ(expected,109601u);
}
TEST(BruteForceTrace, DeepChainUsesAnExplicitStackAndCanOmitQuadraticSnapshots) {
    const std::size_t n=50000;Edges edges;edges.reserve(n-1);
    for (std::size_t i=1;i<n;++i) edges.emplace_back(i-1,i);
    const auto trace=bf::brute_force_search_trace(n,edges,0,n-1,{n,false});
    ASSERT_EQ(trace.result.path.size(),n);EXPECT_EQ(trace.result.expanded,n);
    EXPECT_EQ(trace.max_depth,n-1);EXPECT_EQ(trace.edges_examined,n-1);
    EXPECT_FALSE(trace.paths_recorded);EXPECT_EQ(trace.steps.size(),n);
    for (std::size_t i=0;i<n;++i) {EXPECT_EQ(trace.result.path[i],i);EXPECT_TRUE(trace.steps[i].path.empty());}
    EXPECT_EQ(bf::brute_force_search(n,edges,0,n-1).path,trace.result.path);
}
TEST(BruteForceTrace, OmittingSnapshotsPreservesAllOtherActualFieldsAndCounters) {
    const auto full=bf::brute_force_search_trace(6,worked,0,5);
    const auto lean=bf::brute_force_search_trace(6,worked,0,5,{0,false});
    EXPECT_EQ(full.result.path,lean.result.path);EXPECT_EQ(full.result.expanded,lean.result.expanded);
    EXPECT_EQ(full.edges_examined,lean.edges_examined);EXPECT_EQ(full.cycle_rejections,lean.cycle_rejections);
    ASSERT_EQ(full.steps.size(),lean.steps.size());
    for (std::size_t i=0;i<full.steps.size();++i) {
        const auto& f=full.steps[i];const auto& l=lean.steps[i];EXPECT_TRUE(l.path.empty());
        EXPECT_EQ(f.id,l.id);EXPECT_EQ(f.parent,l.parent);EXPECT_EQ(f.vertex,l.vertex);EXPECT_EQ(f.depth,l.depth);
        EXPECT_EQ(f.incumbent_vertices_before,l.incumbent_vertices_before);EXPECT_EQ(f.incumbent_vertices_after,l.incumbent_vertices_after);
        EXPECT_EQ(f.edges_examined,l.edges_examined);EXPECT_EQ(f.cycle_rejections,l.cycle_rejections);
        EXPECT_EQ(f.goal,l.goal);EXPECT_EQ(f.pruned,l.pruned);EXPECT_EQ(f.improved,l.improved);
    }
}
TEST(BruteForceTrace, PositiveVisitLimitsNeverReturnAnIncompleteOptimum) {
    const auto saved=worked;
    for (std::size_t budget=1;budget<9;++budget) {
        EXPECT_THROW(bf::brute_force_search_trace(6,worked,0,5,{budget,true}),std::runtime_error);
    }
    EXPECT_EQ(worked,saved);
    EXPECT_EQ(bf::brute_force_search_trace(6,worked,0,5,{9,true}).result.path,(std::vector<std::size_t>{0,3,5}));
    EXPECT_EQ(bf::brute_force_search_trace(1,{},0,0,{1,true}).result.expanded,1u);
}
TEST(BruteForceTrace, InvalidTerminalsAndEdgesFailBeforeAdjacencyAllocationEvenForTrivialGoal) {
    EXPECT_THROW(bf::brute_force_search(0,{},0,0),std::invalid_argument);
    EXPECT_THROW(bf::brute_force_search_trace(3,{},3,0),std::invalid_argument);
    EXPECT_THROW(bf::brute_force_search(3,{},0,3),std::invalid_argument);
    EXPECT_THROW(bf::brute_force_search(3,{{0,3}},0,0),std::invalid_argument);
    EXPECT_THROW(bf::brute_force_search_trace(3,{{3,0}},0,0),std::invalid_argument);
    const auto huge=std::numeric_limits<std::size_t>::max();
    EXPECT_THROW(bf::brute_force_search(huge,{{huge,0}},0,0),std::invalid_argument);
    EXPECT_THROW(bf::brute_force_search_trace(huge,{},huge,0),std::invalid_argument);
}
TEST(BruteForceTrace, CompleteDirectedGeneratorIsOrderedUniqueAndChecksCountsBeforeAllocation) {
    EXPECT_TRUE(bf::complete_directed_graph_edges(0).empty());EXPECT_TRUE(bf::complete_directed_graph_edges(1).empty());
    EXPECT_EQ(bf::complete_directed_graph_edges(2),(Edges{{0,1},{1,0}}));
    for (std::size_t n:{2,3,7}) {
        const auto edges=bf::complete_directed_graph_edges(n);EXPECT_EQ(edges.size(),n*(n-1));
        EXPECT_TRUE(std::is_sorted(edges.begin(),edges.end()));EXPECT_EQ(std::adjacent_find(edges.begin(),edges.end()),edges.end());
        for (auto [u,v]:edges) {EXPECT_LT(u,n);EXPECT_LT(v,n);EXPECT_NE(u,v);}
    }
    EXPECT_THROW(bf::complete_directed_graph_edges(std::numeric_limits<std::size_t>::max()),std::overflow_error);
    const auto capacity_case=std::size_t{1}<<(std::numeric_limits<std::size_t>::digits/2);
    EXPECT_THROW(bf::complete_directed_graph_edges(capacity_case),std::length_error);
}
