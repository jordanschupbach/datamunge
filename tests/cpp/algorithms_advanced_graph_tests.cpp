#include <gtest/gtest.h>

#include <datamunge/algorithms/advanced_graph.hpp>
#include <datamunge/algorithms/network_analysis.hpp>

#include <algorithm>
#include <numeric>

using namespace datamunge::algorithms;

namespace {
std::vector<CapacityEdge> clrs() {
  return {{0,1,16},{0,2,13},{1,2,10},{2,1,4},{1,3,12},
          {3,2,9},{2,4,14},{4,3,7},{3,5,20},{4,5,4}};
}
}

TEST(AdvancedGraph, TrustRankPersonalizesTeleportation) {
  const auto rank=trust_rank(3,{{0,1},{1,2},{2,0}},{0});
  EXPECT_NEAR(std::accumulate(rank.begin(),rank.end(),0.0),1.0,1e-10);
  EXPECT_GT(rank[0],rank[1]);
}

TEST(AdvancedGraph, DinicSolvesClassicNetwork) {
  const auto r=dinic_max_flow(6,clrs(),0,5);
  EXPECT_NEAR(r.max_flow,23.0,1e-9);
  EXPECT_EQ(r.edge_flow.size(),clrs().size());
}

TEST(AdvancedGraph, PushRelabelSolvesClassicNetwork) {
  const auto r=push_relabel_max_flow(6,clrs(),0,5);
  EXPECT_NEAR(r.max_flow,23.0,1e-9);
  EXPECT_EQ(r.edge_flow.size(),clrs().size());
}

TEST(AdvancedGraph, KargerFindsBridgeCut) {
  const auto r=karger_min_cut(6,{{0,1},{1,2},{2,0},{2,3},{3,4},{4,5},{5,3}},300,7);
  EXPECT_EQ(r.cut_size,1U);
  EXPECT_FALSE(r.side_a.empty());EXPECT_FALSE(r.side_b.empty());
}

TEST(AdvancedGraph, ChuLiuEdmondsContractsCycle) {
  // Cheapest incoming edges 2->1 and 1->2 form a cycle; contraction chooses root->1.
  const auto r=chu_liu_edmonds(3,{{0,1,5},{0,2,6},{1,2,1},{2,1,1}},0);
  EXPECT_TRUE(r.exists);
  EXPECT_DOUBLE_EQ(r.total_weight,6.0);
}

TEST(AdvancedGraph, EuclideanMstOfUnitSquareHasWeightThree) {
  const auto r=euclidean_minimum_spanning_tree({{0,0},{1,0},{1,1},{0,1}});
  EXPECT_TRUE(r.is_connected);EXPECT_EQ(r.edges.size(),3U);EXPECT_NEAR(r.total_weight,3.0,1e-12);
}

TEST(AdvancedGraph, RejectsMalformedInputs) {
  EXPECT_THROW((void)trust_rank(2,{{0,1}},{2}),std::invalid_argument);
  EXPECT_THROW((void)dinic_max_flow(2,{{0,2,1}},0,1),std::invalid_argument);
  EXPECT_FALSE(chu_liu_edmonds(3,{{0,1,1}},0).exists);
}
