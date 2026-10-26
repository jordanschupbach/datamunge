#include <gtest/gtest.h>
#include <datamunge/dstruct/dstruct.hpp>
using namespace datamunge::dstruct;
TEST(DirectedAcyclicGraph, RejectsCycleAndTopologicallySorts) { DirectedAcyclicGraph<int> graph; graph.add_edge(1, 2); graph.add_edge(2, 3); EXPECT_THROW(graph.add_edge(3, 1), std::invalid_argument); EXPECT_EQ(graph.topological_order(), (std::vector<int>{1, 2, 3})); }
