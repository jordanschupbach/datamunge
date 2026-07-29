#include <gtest/gtest.h>
#include <datamunge/algorithms/tour_search.hpp>
#include <set>
using namespace datamunge::algorithms;
static auto square(){return std::vector<std::vector<double>>{{0,1,1.41421356237,1},{1,0,1,1.41421356237},{1.41421356237,1,0,1},{1,1.41421356237,1,0}};}
TEST(TourSearch,NearestNeighborSquare){auto r=nearest_neighbor_tsp(square());EXPECT_NEAR(r.cost,4,1e-9);EXPECT_EQ(r.tour.front(),r.tour.back());}
TEST(TourSearch,ChristofidesSquare){auto r=christofides_tsp(square());EXPECT_NEAR(r.cost,4,1e-9);EXPECT_EQ(r.tour.size(),5U);}
TEST(TourSearch,ClarkeWrightMergesCompatibleCustomers){auto r=clarke_wright_savings({{0,0},{1,0},{2,0},{10,0}},{0,1,1,1},2);EXPECT_EQ(r.routes.size(),2U);for(auto&x:r.routes){EXPECT_EQ(x.front(),0U);EXPECT_EQ(x.back(),0U);}}
TEST(TourSearch,WarnsdorffToursEightByEight){auto p=warnsdorff_knight_tour(8,8);ASSERT_EQ(p.size(),64U);std::set<std::pair<int,int>>s(p.begin(),p.end());EXPECT_EQ(s.size(),64U);}
TEST(TourSearch,AStarFindsCheapestPath){auto r=a_star_search(4,{{0,1,1},{1,3,2},{0,2,2},{2,3,5}},0,3,{2,2,5,0});EXPECT_TRUE(r.found);EXPECT_DOUBLE_EQ(r.cost,3);EXPECT_EQ(r.path,(std::vector<std::size_t>{0,1,3}));}
TEST(TourSearch,BStarChoosesCheapestGoal){auto r=b_star_search(5,{{0,1,1},{1,3,5},{0,2,2},{2,4,1}},0,{3,4},{3,5,1,0,0});EXPECT_TRUE(r.found);EXPECT_EQ(r.path.back(),4U);EXPECT_DOUBLE_EQ(r.cost,3);}
