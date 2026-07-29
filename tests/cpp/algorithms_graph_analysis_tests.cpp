#include <gtest/gtest.h>

#include <datamunge/algorithms/graph_layout.hpp>
#include <datamunge/algorithms/network_analysis.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

using namespace datamunge::algorithms;

TEST(GraphLayouts, CirclePackingSeparatesTriangleDisks) {
  const auto circles = circle_packing_layout(3, {{0,1},{1,2},{2,0}});
  ASSERT_EQ(circles.size(), 3U);
  for (std::size_t i=0;i<3;++i) for(std::size_t j=i+1;j<3;++j) {
    const double distance=std::hypot(circles[i].center.x-circles[j].center.x,
                                     circles[i].center.y-circles[j].center.y);
    EXPECT_GE(distance+1e-9,circles[i].radius+circles[j].radius);
  }
}

TEST(GraphLayouts, ForceDirectedIsDeterministicAndPullsEdgesTogether) {
  const auto a=force_directed_layout(4,{{0,1},{1,2}},300,7);
  const auto b=force_directed_layout(4,{{0,1},{1,2}},300,7);
  ASSERT_EQ(a.size(),4U);
  for(std::size_t i=0;i<4;++i){EXPECT_DOUBLE_EQ(a[i].x,b[i].x);EXPECT_DOUBLE_EQ(a[i].y,b[i].y);}
  EXPECT_LT(std::hypot(a[0].x-a[1].x,a[0].y-a[1].y),
            std::hypot(a[0].x-a[3].x,a[0].y-a[3].y));
}

TEST(GraphLayouts, SpectralLayoutSeparatesPathEnds) {
  const auto p=spectral_layout(4,{{0,1},{1,2},{2,3}});
  ASSERT_EQ(p.size(),4U);
  EXPECT_GT(std::hypot(p[0].x-p[3].x,p[0].y-p[3].y),0.5);
  for(const auto& point:p){EXPECT_TRUE(std::isfinite(point.x));EXPECT_TRUE(std::isfinite(point.y));}
}

TEST(NetworkAnalysis, PageRankIsProbabilityAndRewardsPopularSink) {
  const auto rank=page_rank(3,{{0,2},{1,2},{2,2}});
  EXPECT_NEAR(std::accumulate(rank.begin(),rank.end(),0.0),1.0,1e-12);
  EXPECT_GT(rank[2],rank[0]); EXPECT_NEAR(rank[0],rank[1],1e-12);
}

TEST(NetworkAnalysis, HITSDistinguishesHubAndAuthority) {
  const auto result=hits(4,{{0,2},{0,3},{1,2}});
  EXPECT_GT(result.hubs[0],result.hubs[1]);
  EXPECT_GT(result.authorities[2],result.authorities[3]);
  EXPECT_GT(result.iterations,0U);
}

TEST(NetworkAnalysis, GirvanNewmanCutsBridgeBetweenTriangles) {
  const auto groups=girvan_newman(6,{{0,1},{1,2},{2,0},{2,3},{3,4},{4,5},{5,3}},2);
  ASSERT_EQ(groups.size(),2U);
  std::set<std::set<std::size_t>> actual;
  for(const auto& group:groups) actual.emplace(group.begin(),group.end());
  EXPECT_EQ(actual,(std::set<std::set<std::size_t>>{{0,1,2},{3,4,5}}));
}

TEST(GraphAnalysis, RejectsInvalidInputs) {
  EXPECT_THROW((void)page_rank(2,{{0,2}}),std::invalid_argument);
  EXPECT_THROW((void)circle_packing_layout(2,{{2,0}}),std::invalid_argument);
  EXPECT_THROW((void)girvan_newman(2,{},3),std::invalid_argument);
}
