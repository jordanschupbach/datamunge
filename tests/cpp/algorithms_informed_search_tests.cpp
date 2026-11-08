#include <datamunge/algorithms/tour_search.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

using namespace datamunge::algorithms;

TEST(InformedSearch,OriginalMultipleGoalExample) {
  const auto t = b_star_search_trace(4,{{0,1,1},{1,2,5},{0,3,2}},0,{2,3},{2,5,0,0});
  EXPECT_TRUE(t.result.found);
  EXPECT_DOUBLE_EQ(t.result.cost,2);
  EXPECT_EQ(t.result.path,(std::vector<std::size_t>{0,3}));
  ASSERT_EQ(t.expansions.size(),2);
  EXPECT_EQ(t.expansions.back().vertex,3);
  EXPECT_EQ(t.expansions.back().queued,1);
  EXPECT_EQ(t.result.expanded,t.expansions.size());
}

TEST(InformedSearch,AdmissibleInconsistentBoundReopens) {
  const auto t = b_star_search_trace(5,{{0,1,3},{0,2,1},{2,1,1},{1,3,3}},
                                   0,{3,4},{0,0,4,0,0});
  EXPECT_DOUBLE_EQ(t.result.cost,5);
  EXPECT_EQ(t.result.path,(std::vector<std::size_t>{0,2,1,3}));
  ASSERT_EQ(t.expansions.size(),5);
  EXPECT_EQ(t.expansions[1].vertex,1);
  EXPECT_DOUBLE_EQ(t.expansions[1].path_cost,3);
  EXPECT_EQ(t.expansions[3].vertex,1);
  EXPECT_DOUBLE_EQ(t.expansions[3].path_cost,2);
}

TEST(InformedSearch,InadmissibleBoundCanReturnMoreExpensiveGoal) {
  const auto r = b_star_search(5,{{0,1,3},{0,2,1},{2,1,1},{1,3,3}},
                               0,{3,4},{0,0,10,0,0});
  EXPECT_DOUBLE_EQ(r.cost,6);
}

TEST(InformedSearch,StartGoalDuplicatesAndDisconnected) {
  const auto r = b_star_search(1,{},0,{0,0},{0});
  EXPECT_TRUE(r.found);
  EXPECT_EQ(r.path,(std::vector<std::size_t>{0}));
  EXPECT_EQ(r.expanded,1);
  EXPECT_DOUBLE_EQ(r.cost,0);
  const auto no = b_star_search_trace(2,{},0,{1},{0,0});
  EXPECT_FALSE(no.result.found);
  EXPECT_TRUE(no.result.path.empty());
  EXPECT_DOUBLE_EQ(no.result.cost,0);
  EXPECT_EQ(no.expansions.size(),1);
}

TEST(InformedSearch,ZeroCyclesParallelEdgesAndStableTies) {
  const auto t = b_star_search_trace(4,{{0,0,0},{0,1,9},{0,1,1},{0,2,1},
      {1,2,0},{2,1,0},{1,3,1},{2,3,1}},0,{3},{0,0,0,0});
  EXPECT_EQ(t.result.path,(std::vector<std::size_t>{0,1,3}));
  EXPECT_DOUBLE_EQ(t.result.cost,2);
  EXPECT_EQ(t.result.expanded,4);
}

TEST(InformedSearch,StaleQueueEntriesAreNotExpansions) {
  const auto t = b_star_search_trace(4,{{0,1,10},{0,2,1},{2,1,1}},0,{3},{0,0,0,0});
  EXPECT_FALSE(t.result.found);
  EXPECT_EQ(t.result.expanded,3);
  EXPECT_EQ(t.expansions.size(),3);
}

TEST(InformedSearch,RejectsInvalidShapesVerticesAndGoals) {
  EXPECT_THROW(b_star_search(0,{},0,{0},{}),std::invalid_argument);
  EXPECT_THROW(b_star_search(1,{},0,{}, {0}),std::invalid_argument);
  EXPECT_THROW(b_star_search(1,{},0,{1},{0}),std::invalid_argument);
  EXPECT_THROW(b_star_search(1,{},1,{0},{0}),std::invalid_argument);
  EXPECT_THROW(b_star_search(2,{},0,{1},{0}),std::invalid_argument);
  EXPECT_THROW(b_star_search(2,{{0,2,1}},0,{1},{0,0}),std::invalid_argument);
  EXPECT_THROW(a_star_search(2,{},0,2,{0,0}),std::invalid_argument);
}

TEST(InformedSearch,RejectsNonfiniteWeightsAndHeuristicsEvenAtStartGoal) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (double bad : {nan,inf,-inf,-1.})
    EXPECT_THROW(b_star_search(2,{{0,1,bad}},0,{0},{0,0}),std::invalid_argument);
  for (double bad : {nan,inf,-inf})
    EXPECT_THROW(b_star_search(2,{},0,{0},{0,bad}),std::invalid_argument);
}

TEST(InformedSearch,RejectsAccumulatedCostAndPriorityOverflow) {
  const double huge = std::numeric_limits<double>::max();
  EXPECT_THROW(b_star_search(3,{{0,1,huge},{1,2,huge}},0,{2},{0,0,0}),std::overflow_error);
  EXPECT_THROW(b_star_search(2,{{0,1,huge}},0,{1},{0,huge}),std::overflow_error);
}

TEST(InformedSearch,RandomGraphsMatchIndependentFloydWarshallOracle) {
  std::mt19937 rng(20261106);
  const double inf = std::numeric_limits<double>::infinity();
  for (int trial=0; trial<600; ++trial) {
    const std::size_t n=2+rng()%7, start=rng()%n;
    std::vector<std::tuple<std::size_t,std::size_t,double>> edges;
    std::vector<std::vector<double>> d(n,std::vector<double>(n,inf));
    for (std::size_t i=0;i<n;++i) {
      d[i][i]=0;
      for (std::size_t j=0;j<n;++j) if (rng()%3==0) {
        double w=rng()%10;
        edges.emplace_back(i,j,w);
        d[i][j]=std::min(d[i][j],w);
      }
    }
    for (std::size_t k=0;k<n;++k)
      for (std::size_t i=0;i<n;++i)
        for (std::size_t j=0;j<n;++j) d[i][j]=std::min(d[i][j],d[i][k]+d[k][j]);
    std::vector<std::size_t> goals{rng()%n};
    if (rng()%2) goals.push_back(rng()%n);
    std::vector<double> h(n);
    for (std::size_t i=0;i<n;++i) {
      double nearest=inf;
      for (auto goal:goals) nearest=std::min(nearest,d[i][goal]);
      h[i]=std::isfinite(nearest)?std::floor(nearest*(rng()%101)/100.):double(rng()%30);
    }
    const auto t=b_star_search_trace(n,edges,start,goals,h);
    const auto plain=b_star_search(n,edges,start,goals,h);
    double expected=inf;
    for (auto goal:goals) expected=std::min(expected,d[start][goal]);
    ASSERT_EQ(t.result.found,std::isfinite(expected)) << trial;
    EXPECT_EQ(t.result.path,plain.path);
    EXPECT_EQ(t.result.expanded,t.expansions.size());
    EXPECT_EQ(t.result.expanded,plain.expanded);
    if (!t.result.found) continue;
    EXPECT_DOUBLE_EQ(t.result.cost,expected) << trial;
    ASSERT_EQ(t.result.path.front(),start);
    ASSERT_NE(std::find(goals.begin(),goals.end(),t.result.path.back()),goals.end());
    double cost=0;
    for (std::size_t k=1;k<t.result.path.size();++k) {
      double w=inf;
      for (auto [u,v,c]:edges)
        if (u==t.result.path[k-1] && v==t.result.path[k]) w=std::min(w,c);
      ASSERT_TRUE(std::isfinite(w));
      cost+=w;
    }
    EXPECT_DOUBLE_EQ(cost,expected);
    const auto single=a_star_search(n,edges,start,goals.front(),std::vector<double>(n,0));
    EXPECT_EQ(single.found,std::isfinite(d[start][goals.front()]));
    if (single.found) EXPECT_DOUBLE_EQ(single.cost,d[start][goals.front()]);
  }
}
