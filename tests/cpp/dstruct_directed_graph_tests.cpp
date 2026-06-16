#include <gtest/gtest.h>

#include <datamunge/dstruct/dstruct.hpp>

#include <string>
#include <vector>

using datamunge::dstruct::DirectedGraph;

TEST(DirectedGraph, SupportsVertexAndEdgeMutation) {
  DirectedGraph<int> graph;

  EXPECT_TRUE(graph.add_vertex(1));
  EXPECT_FALSE(graph.add_vertex(1));
  EXPECT_TRUE(graph.add_edge(1, 2));
  EXPECT_TRUE(graph.add_edge(2, 3));
  EXPECT_FALSE(graph.add_edge(1, 2));

  EXPECT_EQ(graph.vertex_count(), 3u);
  EXPECT_EQ(graph.edge_count(), 2u);
  EXPECT_TRUE(graph.has_vertex(2));
  EXPECT_TRUE(graph.has_edge(1, 2));
  EXPECT_FALSE(graph.has_edge(3, 1));
}

TEST(DirectedGraph, PreservesInsertionOrderForVerticesAndNeighbors) {
  DirectedGraph<std::string> graph;
  graph.add_edge("a", "b");
  graph.add_edge("a", "c");
  graph.add_edge("d", "a");

  EXPECT_EQ(graph.vertices(), (std::vector<std::string>{"a", "b", "c", "d"}));
  EXPECT_EQ(graph.outgoing_neighbors("a"), (std::vector<std::string>{"b", "c"}));
  EXPECT_EQ(graph.incoming_neighbors("a"), (std::vector<std::string>{"d"}));
}

TEST(DirectedGraph, ReportsDegreesAndSupportsEdgeRemoval) {
  DirectedGraph<int> graph;
  graph.add_edge(1, 2);
  graph.add_edge(1, 3);
  graph.add_edge(4, 1);

  EXPECT_EQ(graph.out_degree(1), 2u);
  EXPECT_EQ(graph.in_degree(1), 1u);
  EXPECT_TRUE(graph.remove_edge(1, 2));
  EXPECT_FALSE(graph.remove_edge(1, 2));
  EXPECT_EQ(graph.out_degree(1), 1u);
  EXPECT_EQ(graph.edge_count(), 2u);
}

TEST(DirectedGraph, RemovingVertexRemovesIncidentEdges) {
  DirectedGraph<int> graph;
  graph.add_edge(1, 2);
  graph.add_edge(2, 3);
  graph.add_edge(3, 2);
  graph.add_edge(2, 2);

  EXPECT_TRUE(graph.remove_vertex(2));
  EXPECT_FALSE(graph.has_vertex(2));
  EXPECT_EQ(graph.vertex_count(), 2u);
  EXPECT_EQ(graph.edge_count(), 0u);
  EXPECT_FALSE(graph.has_edge(1, 2));
  EXPECT_FALSE(graph.has_edge(3, 2));
}

TEST(DirectedGraph, TraversalsAndPathQueriesWork) {
  DirectedGraph<int> graph;
  graph.add_edge(1, 2);
  graph.add_edge(1, 3);
  graph.add_edge(2, 4);
  graph.add_edge(3, 5);

  EXPECT_EQ(graph.bfs(1), (std::vector<int>{1, 2, 3, 4, 5}));
  EXPECT_EQ(graph.dfs(1), (std::vector<int>{1, 2, 4, 3, 5}));
  EXPECT_TRUE(graph.contains_path(1, 5));
  EXPECT_FALSE(graph.contains_path(4, 5));
}

TEST(DirectedGraph, TransposeReversesEdges) {
  DirectedGraph<int> graph;
  graph.add_edge(1, 2);
  graph.add_edge(2, 3);

  const auto reversed = graph.transpose();
  EXPECT_TRUE(reversed.has_edge(2, 1));
  EXPECT_TRUE(reversed.has_edge(3, 2));
  EXPECT_FALSE(reversed.has_edge(1, 2));
  EXPECT_EQ(reversed.vertices(), (std::vector<int>{1, 2, 3}));
}

TEST(DirectedGraph, TopologicalSortOrdersDag) {
  DirectedGraph<std::string> graph;
  graph.add_edge("parse", "analyze");
  graph.add_edge("analyze", "emit");
  graph.add_edge("parse", "lint");

  EXPECT_EQ(graph.topological_sort(), (std::vector<std::string>{"parse", "analyze", "lint", "emit"}));
}

TEST(DirectedGraph, TopologicalSortRejectsCycles) {
  DirectedGraph<int> graph;
  graph.add_edge(1, 2);
  graph.add_edge(2, 1);

  EXPECT_THROW(static_cast<void>(graph.topological_sort()), std::invalid_argument);
}

TEST(DirectedGraph, MissingVerticesThrowForLookupOperations) {
  DirectedGraph<int> graph;
  graph.add_vertex(1);

  EXPECT_THROW(static_cast<void>(graph.outgoing_neighbors(9)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(graph.incoming_neighbors(9)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(graph.bfs(9)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(graph.contains_path(1, 9)), std::out_of_range);
}

TEST(DirectedGraph, ClearResetsGraph) {
  DirectedGraph<int> graph;
  graph.add_edge(1, 2);
  graph.add_edge(2, 3);

  graph.clear();

  EXPECT_TRUE(graph.empty());
  EXPECT_EQ(graph.vertex_count(), 0u);
  EXPECT_EQ(graph.edge_count(), 0u);
}
