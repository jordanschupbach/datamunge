#include <gtest/gtest.h>
#include <datamunge/algorithms/search_batch.hpp>
using namespace datamunge::algorithms;
static auto edges(){return std::vector<std::pair<std::size_t,std::size_t>>{{0,1},{0,2},{1,3},{2,4},{4,5},{3,5}};}
TEST(SearchBatch,BreadthFirstShortest){auto r=breadth_first_search(6,edges(),0,5);EXPECT_TRUE(r.found);EXPECT_EQ(r.path.size(),4U);}
TEST(SearchBatch,BacktrackingFindsPath){auto r=backtracking_search(6,edges(),0,5);EXPECT_TRUE(r.found);EXPECT_EQ(r.path.front(),0U);EXPECT_EQ(r.path.back(),5U);}
TEST(SearchBatch,BestFirstUsesHeuristic){auto r=best_first_search(6,edges(),0,5,{3,2,1,1,1,0});EXPECT_TRUE(r.found);}
TEST(SearchBatch,BeamFindsPath){auto r=beam_search(6,edges(),0,5,{3,2,1,1,1,0},2);EXPECT_TRUE(r.found);}
TEST(SearchBatch,BeamStackRecoversPrunedBranch){auto r=beam_stack_search(5,{{0,1},{0,2},{1,3},{2,4}},0,4,{2,0,3,0,0},1);EXPECT_TRUE(r.found);EXPECT_EQ(r.path,(std::vector<std::size_t>{0,2,4}));}
TEST(SearchBatch,BidirectionalFindsShortest){auto r=bidirectional_search(6,edges(),0,5);EXPECT_TRUE(r.found);EXPECT_EQ(r.path.size(),4U);}
