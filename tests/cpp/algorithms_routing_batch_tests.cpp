#include <gtest/gtest.h>
#include <datamunge/algorithms/routing_batch.hpp>
#include <cmath>
#include <set>
using namespace datamunge::algorithms;

TEST(RoutingBatch,DagLongestPath){auto r=dag_longest_paths(5,{{0,1,2},{0,2,1},{1,3,3},{2,3,8},{3,4,-1}},0);EXPECT_DOUBLE_EQ(r.distance[4],8);}
TEST(RoutingBatch,DagLongestRejectsCycle){EXPECT_THROW((void)dag_longest_paths(2,{{0,1,1},{1,0,1}},0),std::invalid_argument);}
TEST(RoutingBatch,BoruvkaFindsKnownMst){auto r=boruvka(4,{{0,1,1},{1,2,2},{2,3,1},{0,3,9},{0,2,5}});EXPECT_TRUE(r.is_connected);EXPECT_DOUBLE_EQ(r.total_weight,4);EXPECT_EQ(r.edges.size(),3U);}
TEST(RoutingBatch,ReverseDeleteFindsKnownMst){auto r=reverse_delete(4,{{0,1,1},{1,2,2},{2,3,1},{0,3,9},{0,2,5}});EXPECT_TRUE(r.is_connected);EXPECT_DOUBLE_EQ(r.total_weight,4);EXPECT_EQ(r.edges.size(),3U);}
TEST(RoutingBatch,SwitchAlternatesSharedGroups){auto route=nonblocking_switch_routes(2,2,{{0,0},{0,1},{1,1},{1,0}});ASSERT_EQ(route.size(),4U);EXPECT_NE(route[0],route[1]);EXPECT_NE(route[1],route[2]);EXPECT_NE(route[2],route[3]);EXPECT_NE(route[3],route[0]);}
TEST(RoutingBatch,JohnsonKnownDistances){auto r=johnson_all_pairs_shortest_paths(4,{{0,1,1},{1,2,-2},{0,2,4},{2,3,2}});EXPECT_FALSE(r.has_negative_cycle);EXPECT_DOUBLE_EQ(r.distance[0][3],1);EXPECT_TRUE(std::isinf(r.distance[3][0]));}
TEST(RoutingBatch,JohnsonDetectsNegativeCycle){const auto r=johnson_all_pairs_shortest_paths(2,{{0,1,-1},{1,0,-1}});EXPECT_TRUE(r.has_negative_cycle);}
TEST(RoutingBatch,TransitiveClosureFindsIndirectReachability){auto r=transitive_closure(4,{{0,1},{1,2},{2,3}});EXPECT_TRUE(r[0][3]);EXPECT_FALSE(r[3][0]);EXPECT_TRUE(r[2][2]);}
