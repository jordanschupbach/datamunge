#include <gtest/gtest.h>

#include <datamunge/dstruct/dstruct.hpp>

#include <string>
#include <vector>

using datamunge::dstruct::UndirectedGraph;

TEST(UndirectedGraph, SupportsVertexAndEdgeMutation) {
  UndirectedGraph<int> graph;

  EXPECT_TRUE(graph.add_vertex(1));
  EXPECT_FALSE(graph.add_vertex(1));
  EXPECT_TRUE(graph.add_edge(1, 2));
  EXPECT_TRUE(graph.add_edge(2, 3));
  EXPECT_FALSE(graph.add_edge(1, 2));
  EXPECT_FALSE(graph.add_edge(2, 1)); // same edge, order shouldn't matter

  EXPECT_EQ(graph.vertex_count(), 3u);
  EXPECT_EQ(graph.edge_count(), 2u);
  EXPECT_TRUE(graph.has_edge(1, 2));
  EXPECT_TRUE(graph.has_edge(2, 1));
  EXPECT_FALSE(graph.has_edge(3, 1));
}

TEST(UndirectedGraph, EdgesAreSymmetricInNeighborsAndDegree) {
  UndirectedGraph<int> graph;
  graph.add_edge(1, 2);
  graph.add_edge(1, 3);

  EXPECT_EQ(graph.degree(1), 2u);
  EXPECT_EQ(graph.neighbors(1), (std::vector<int>{2, 3}));
  EXPECT_EQ(graph.neighbors(2), (std::vector<int>{1}));
  EXPECT_EQ(graph.neighbors(3), (std::vector<int>{1}));
}

TEST(UndirectedGraph, RemoveEdgeIsSymmetric) {
  UndirectedGraph<int> graph;
  graph.add_edge(1, 2);

  EXPECT_TRUE(graph.remove_edge(2, 1));
  EXPECT_FALSE(graph.has_edge(1, 2));
  EXPECT_FALSE(graph.has_edge(2, 1));
  EXPECT_EQ(graph.edge_count(), 0u);
}

TEST(UndirectedGraph, RemovingVertexRemovesIncidentEdgesIncludingSelfLoop) {
  UndirectedGraph<int> graph;
  graph.add_edge(1, 2);
  graph.add_edge(2, 3);
  graph.add_edge(2, 2); // self-loop

  EXPECT_TRUE(graph.remove_vertex(2));
  EXPECT_FALSE(graph.has_vertex(2));
  EXPECT_EQ(graph.vertex_count(), 2u);
  EXPECT_EQ(graph.edge_count(), 0u);
  EXPECT_FALSE(graph.has_edge(1, 2));
  EXPECT_FALSE(graph.has_edge(3, 2));
}

TEST(UndirectedGraph, TraversalsAndPathQueriesWork) {
  UndirectedGraph<int> graph;
  graph.add_edge(1, 2);
  graph.add_edge(1, 3);
  graph.add_edge(2, 4);
  graph.add_edge(3, 5);

  EXPECT_EQ(graph.bfs(1), (std::vector<int>{1, 2, 3, 4, 5}));
  EXPECT_TRUE(graph.contains_path(4, 5));
  EXPECT_TRUE(graph.contains_path(1, 5));
}

TEST(UndirectedGraph, ConnectedComponentsFindsEachDisjointGroup) {
  UndirectedGraph<int> graph;
  graph.add_edge(1, 2);
  graph.add_edge(2, 3);
  graph.add_edge(10, 11);
  graph.add_vertex(99); // isolated vertex, its own component

  const auto components = graph.connected_components();
  ASSERT_EQ(components.size(), 3u);
  EXPECT_EQ(components[0], (std::vector<int>{1, 2, 3}));
  EXPECT_EQ(components[1], (std::vector<int>{10, 11}));
  EXPECT_EQ(components[2], (std::vector<int>{99}));
}

TEST(UndirectedGraph, HasCycleDetectsATriangleButNotATree) {
  UndirectedGraph<int> tree;
  tree.add_edge(1, 2);
  tree.add_edge(2, 3);
  tree.add_edge(2, 4);
  EXPECT_FALSE(tree.has_cycle());

  UndirectedGraph<int> triangle;
  triangle.add_edge(1, 2);
  triangle.add_edge(2, 3);
  triangle.add_edge(3, 1);
  EXPECT_TRUE(triangle.has_cycle());
}

TEST(UndirectedGraph, HasCycleDetectsACycleInADisconnectedGraph) {
  // The cycle is in the SECOND component -- has_cycle must not stop after checking the
  // acyclic first component.
  UndirectedGraph<int> graph;
  graph.add_edge(1, 2); // acyclic component
  graph.add_edge(10, 11);
  graph.add_edge(11, 12);
  graph.add_edge(12, 10); // cyclic component

  EXPECT_TRUE(graph.has_cycle());
}

TEST(UndirectedGraph, HasCycleDetectsASelfLoop) {
  UndirectedGraph<int> graph;
  graph.add_edge(1, 2);
  graph.add_edge(2, 2);

  EXPECT_TRUE(graph.has_cycle());
}

TEST(UndirectedGraph, HasCycleDetectsADiamond) {
  // A-B, A-C, B-D, C-D: two distinct paths from A to D via B and via C.
  UndirectedGraph<std::string> graph;
  graph.add_edge("A", "B");
  graph.add_edge("A", "C");
  graph.add_edge("B", "D");
  graph.add_edge("C", "D");

  EXPECT_TRUE(graph.has_cycle());
}

TEST(UndirectedGraph, MissingVerticesThrowForLookupOperations) {
  UndirectedGraph<int> graph;
  graph.add_vertex(1);

  EXPECT_THROW(static_cast<void>(graph.neighbors(9)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(graph.bfs(9)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(graph.contains_path(1, 9)), std::out_of_range);
}

TEST(UndirectedGraph, ClearResetsGraph) {
  UndirectedGraph<int> graph;
  graph.add_edge(1, 2);
  graph.add_edge(2, 3);

  graph.clear();

  EXPECT_TRUE(graph.empty());
  EXPECT_EQ(graph.vertex_count(), 0u);
  EXPECT_EQ(graph.edge_count(), 0u);
}
