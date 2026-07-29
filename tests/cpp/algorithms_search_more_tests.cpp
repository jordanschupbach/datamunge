#include <gtest/gtest.h>
#include <datamunge/algorithms/search_more.hpp>
using namespace datamunge::algorithms;
TEST(SearchMore,BruteForceFindsShortestSimplePath){auto r=brute_force_search(4,{{0,1},{1,3},{0,2},{2,1}},0,3);EXPECT_TRUE(r.found);EXPECT_EQ(r.path,(std::vector<std::size_t>{0,1,3}));}
TEST(SearchMore,DepthFirstFindsPath){EXPECT_TRUE(depth_first_search(3,{{0,1},{1,2}},0,2).found);}
TEST(SearchMore,IddfsFindsShallowest){auto r=iterative_deepening_dfs(4,{{0,1},{1,2},{2,3},{0,3}},0,3,3);EXPECT_EQ(r.path,(std::vector<std::size_t>{0,3}));}
TEST(SearchMore,DStarReplansAroundObstacle){auto r=d_star_replan({{0,0,0},{0,1,0},{0,0,0}},{0,0},{2,2});EXPECT_TRUE(r.found);EXPECT_DOUBLE_EQ(r.cost,4);}
TEST(SearchMore,JumpPointOpenGrid){auto r=jump_point_search(std::vector<std::vector<bool>>(5,std::vector<bool>(5)),{0,0},{4,4});EXPECT_TRUE(r.found);EXPECT_DOUBLE_EQ(r.cost,8);}
TEST(SearchMore,GpsBuildsPlan){std::vector<PlanningAction>a{{"get-key",0,1,0},{"open",1,2,0}};auto r=general_problem_solver(0,2,a);EXPECT_TRUE(r.found);EXPECT_EQ(r.actions.size(),2U);}
