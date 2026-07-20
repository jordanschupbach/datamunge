#include <gtest/gtest.h>

#include <datamunge/dstruct/dstruct.hpp>

#include <algorithm>
#include <set>
#include <string>
#include <vector>

using datamunge::dstruct::WeightedGraph;

namespace {
// Order-agnostic comparison for connected-/strongly-connected-component results: checks the
// same SET of vertex-sets was returned, without over-specifying traversal-dependent element
// or component order.
template <typename Vertex>
std::set<std::set<Vertex>> as_sets(const std::vector<std::vector<Vertex>>& components) {
  std::set<std::set<Vertex>> result;
  for (const auto& component : components) result.emplace(component.begin(), component.end());
  return result;
}
} // namespace

TEST(WeightedGraph, SupportsVertexAndEdgeMutationWithReplaceableWeights) {
  WeightedGraph<int> graph; // directed by default

  EXPECT_TRUE(graph.add_vertex(1));
  EXPECT_FALSE(graph.add_vertex(1));
  EXPECT_TRUE(graph.add_edge(1, 2, 5.0));
  EXPECT_DOUBLE_EQ(graph.edge_weight(1, 2), 5.0);

  EXPECT_FALSE(graph.add_edge(1, 2, 9.0)); // re-adding updates the weight, doesn't count as new
  EXPECT_DOUBLE_EQ(graph.edge_weight(1, 2), 9.0);
  EXPECT_EQ(graph.edge_count(), 1u);
}

TEST(WeightedGraph, DirectedEdgesAreOneWay) {
  WeightedGraph<int> graph(/*directed=*/true);
  graph.add_edge(1, 2, 1.0);

  EXPECT_TRUE(graph.has_edge(1, 2));
  EXPECT_FALSE(graph.has_edge(2, 1));
  EXPECT_EQ(graph.neighbors(1), (std::vector<int>{2}));
  EXPECT_TRUE(graph.neighbors(2).empty());
}

TEST(WeightedGraph, UndirectedEdgesAreSymmetric) {
  WeightedGraph<int> graph(/*directed=*/false);
  graph.add_edge(1, 2, 3.5);

  EXPECT_TRUE(graph.has_edge(1, 2));
  EXPECT_TRUE(graph.has_edge(2, 1));
  EXPECT_DOUBLE_EQ(graph.edge_weight(2, 1), 3.5);
  EXPECT_EQ(graph.edge_count(), 1u);

  EXPECT_TRUE(graph.remove_edge(2, 1));
  EXPECT_FALSE(graph.has_edge(1, 2));
}

TEST(WeightedGraph, RemovingVertexUpdatesEdgeCountForDirectedInAndOutEdges) {
  WeightedGraph<int> graph(/*directed=*/true);
  graph.add_edge(1, 2, 1.0); // outgoing from 2's perspective: none; incoming to 2
  graph.add_edge(2, 3, 1.0); // outgoing from 2
  graph.add_edge(4, 2, 1.0); // incoming to 2

  EXPECT_EQ(graph.edge_count(), 3u);
  EXPECT_TRUE(graph.remove_vertex(2));
  EXPECT_EQ(graph.edge_count(), 0u);
  EXPECT_EQ(graph.vertex_count(), 3u);
}

// ---- Dijkstra ----

TEST(WeightedGraph, DijkstraFindsShortestPathAndReconstructsIt) {
  WeightedGraph<char> graph(/*directed=*/true);
  graph.add_edge('A', 'B', 1.0);
  graph.add_edge('A', 'C', 4.0);
  graph.add_edge('B', 'C', 1.0);
  graph.add_edge('B', 'D', 5.0);
  graph.add_edge('C', 'D', 1.0);

  const auto result = graph.dijkstra('A');
  EXPECT_DOUBLE_EQ(result.distance.at('A'), 0.0);
  EXPECT_DOUBLE_EQ(result.distance.at('B'), 1.0);
  EXPECT_DOUBLE_EQ(result.distance.at('C'), 2.0); // via B (1+1), not direct A-C (4)
  EXPECT_DOUBLE_EQ(result.distance.at('D'), 3.0); // via B, C (1+1+1), not via B, D (1+5) or C, D (4+1)

  EXPECT_EQ(result.path_to('D'), (std::vector<char>{'A', 'B', 'C', 'D'}));
}

TEST(WeightedGraph, DijkstraDistancesMatchesMapBasedResultInVerticesOrder) {
  WeightedGraph<char> graph(/*directed=*/true);
  graph.add_edge('A', 'B', 1.0);
  graph.add_edge('A', 'C', 4.0);
  graph.add_edge('B', 'C', 1.0);

  const auto pairs = graph.dijkstra_distances('A');
  const std::vector<std::pair<char, double>> expected{{'A', 0.0}, {'B', 1.0}, {'C', 2.0}};
  EXPECT_EQ(pairs, expected);
}

TEST(WeightedGraph, BellmanFordDistancesMatchesMapBasedResultAndIsEmptyOnNegativeCycle) {
  WeightedGraph<char> graph(/*directed=*/true);
  graph.add_edge('A', 'B', 4.0);
  graph.add_edge('A', 'C', 1.0);
  graph.add_edge('C', 'B', -3.0);

  const auto pairs = graph.bellman_ford_distances('A');
  const std::vector<std::pair<char, double>> expected{{'A', 0.0}, {'B', -2.0}, {'C', 1.0}};
  EXPECT_EQ(pairs, expected);
  EXPECT_FALSE(graph.bellman_ford_has_negative_cycle('A'));

  WeightedGraph<char> cyclic(/*directed=*/true);
  cyclic.add_edge('A', 'B', 1.0);
  cyclic.add_edge('B', 'C', -3.0);
  cyclic.add_edge('C', 'A', 1.0);
  EXPECT_TRUE(cyclic.bellman_ford_distances('A').empty());
  EXPECT_TRUE(cyclic.bellman_ford_has_negative_cycle('A'));
}

TEST(WeightedGraph, DijkstraLeavesUnreachableVerticesAbsent) {
  WeightedGraph<int> graph(/*directed=*/true);
  graph.add_edge(1, 2, 1.0);
  graph.add_vertex(3); // isolated

  const auto result = graph.dijkstra(1);
  EXPECT_EQ(result.distance.find(3), result.distance.end());
  EXPECT_TRUE(result.path_to(3).empty());
}

TEST(WeightedGraph, DijkstraRejectsNegativeWeights) {
  WeightedGraph<int> graph(/*directed=*/true);
  graph.add_edge(1, 2, -1.0);
  EXPECT_THROW(static_cast<void>(graph.dijkstra(1)), std::invalid_argument);
}

// ---- Bellman-Ford ----

TEST(WeightedGraph, BellmanFordHandlesNegativeEdgesWithoutNegativeCycle) {
  // A->B:4, A->C:1, C->B:-3, B->D:2, C->D:5. Shortest A->D is via A,C,B,D: 1-3+2=0, not the
  // direct A,B,D (4+2=6) or A,C,D (1+5=6) routes.
  WeightedGraph<char> graph(/*directed=*/true);
  graph.add_edge('A', 'B', 4.0);
  graph.add_edge('A', 'C', 1.0);
  graph.add_edge('C', 'B', -3.0);
  graph.add_edge('B', 'D', 2.0);
  graph.add_edge('C', 'D', 5.0);

  const auto result = graph.bellman_ford('A');
  EXPECT_FALSE(result.has_negative_cycle);
  EXPECT_DOUBLE_EQ(result.distance.at('A'), 0.0);
  EXPECT_DOUBLE_EQ(result.distance.at('C'), 1.0);
  EXPECT_DOUBLE_EQ(result.distance.at('B'), -2.0);
  EXPECT_DOUBLE_EQ(result.distance.at('D'), 0.0);
}

TEST(WeightedGraph, BellmanFordDetectsANegativeCycle) {
  WeightedGraph<char> graph(/*directed=*/true);
  graph.add_edge('A', 'B', 1.0);
  graph.add_edge('B', 'C', -3.0);
  graph.add_edge('C', 'A', 1.0); // cycle total: 1 - 3 + 1 = -1

  const auto result = graph.bellman_ford('A');
  EXPECT_TRUE(result.has_negative_cycle);
}

// ---- Floyd-Warshall ----

TEST(WeightedGraph, FloydWarshallMatchesDijkstraOnAllPairs) {
  WeightedGraph<char> graph(/*directed=*/false);
  graph.add_edge('A', 'B', 1.0);
  graph.add_edge('B', 'C', 2.0);
  graph.add_edge('A', 'C', 5.0);

  const auto all_pairs = graph.floyd_warshall();
  EXPECT_DOUBLE_EQ(all_pairs.at('A').at('C'), 3.0); // via B, not the direct 5.0 edge
  EXPECT_DOUBLE_EQ(all_pairs.at('A').at('B'), 1.0);
  EXPECT_DOUBLE_EQ(all_pairs.at('A').at('A'), 0.0);

  const auto dijkstra_result = graph.dijkstra('A');
  EXPECT_DOUBLE_EQ(all_pairs.at('A').at('C'), dijkstra_result.distance.at('C'));
}

// ---- Kruskal MST ----

TEST(WeightedGraph, MinimumSpanningTreeMatchesHandComputedKruskal) {
  // A-B:1, B-C:2, A-C:3, C-D:4, B-D:5. Kruskal picks A-B, B-C, C-D (skipping A-C and B-D,
  // both of which would close a cycle) for a total weight of 1+2+4=7.
  WeightedGraph<char> graph(/*directed=*/false);
  graph.add_edge('A', 'B', 1.0);
  graph.add_edge('B', 'C', 2.0);
  graph.add_edge('A', 'C', 3.0);
  graph.add_edge('C', 'D', 4.0);
  graph.add_edge('B', 'D', 5.0);

  const auto mst = graph.minimum_spanning_tree();
  EXPECT_EQ(mst.vertex_count(), 4u);
  EXPECT_EQ(mst.edge_count(), 3u);

  double total_weight = 0.0;
  for (const auto& v : mst.vertices())
    for (const auto& n : mst.neighbors(v))
      if (v < n) total_weight += mst.edge_weight(v, n);
  EXPECT_DOUBLE_EQ(total_weight, 7.0);

  EXPECT_TRUE(mst.has_edge('A', 'B'));
  EXPECT_TRUE(mst.has_edge('B', 'C'));
  EXPECT_TRUE(mst.has_edge('C', 'D'));
  EXPECT_FALSE(mst.has_edge('A', 'C'));
  EXPECT_FALSE(mst.has_edge('B', 'D'));
}

TEST(WeightedGraph, MinimumSpanningTreeRejectsDirectedGraphs) {
  WeightedGraph<int> graph(/*directed=*/true);
  graph.add_edge(1, 2, 1.0);
  EXPECT_THROW(static_cast<void>(graph.minimum_spanning_tree()), std::logic_error);
}

// ---- Connected components / strongly connected components ----

TEST(WeightedGraph, ConnectedComponentsGroupsUndirectedVertices) {
  WeightedGraph<int> graph(/*directed=*/false);
  graph.add_edge(1, 2, 1.0);
  graph.add_edge(10, 11, 1.0);
  graph.add_vertex(99);

  const auto components = as_sets(graph.connected_components());
  const std::set<std::set<int>> expected{{1, 2}, {10, 11}, {99}};
  EXPECT_EQ(components, expected);
}

TEST(WeightedGraph, ConnectedComponentsRejectsDirectedGraphs) {
  WeightedGraph<int> graph(/*directed=*/true);
  graph.add_edge(1, 2, 1.0);
  EXPECT_THROW(static_cast<void>(graph.connected_components()), std::logic_error);
}

TEST(WeightedGraph, StronglyConnectedComponentsFindsEachCycleGroup) {
  // {1,2,3} is one SCC (1->2->3->1), 3->4 is a one-way bridge into the second SCC {4,5}
  // (4->5->4).
  WeightedGraph<int> graph(/*directed=*/true);
  graph.add_edge(1, 2, 1.0);
  graph.add_edge(2, 3, 1.0);
  graph.add_edge(3, 1, 1.0);
  graph.add_edge(3, 4, 1.0);
  graph.add_edge(4, 5, 1.0);
  graph.add_edge(5, 4, 1.0);

  const auto components = as_sets(graph.strongly_connected_components());
  const std::set<std::set<int>> expected{{1, 2, 3}, {4, 5}};
  EXPECT_EQ(components, expected);
}

TEST(WeightedGraph, StronglyConnectedComponentsRejectsUndirectedGraphs) {
  WeightedGraph<int> graph(/*directed=*/false);
  graph.add_edge(1, 2, 1.0);
  EXPECT_THROW(static_cast<void>(graph.strongly_connected_components()), std::logic_error);
}

// ---- Cycle detection ----

TEST(WeightedGraph, HasCycleWorksForDirectedGraphs) {
  WeightedGraph<int> acyclic(/*directed=*/true);
  acyclic.add_edge(1, 2, 1.0);
  acyclic.add_edge(2, 3, 1.0);
  EXPECT_FALSE(acyclic.has_cycle());

  WeightedGraph<int> cyclic(/*directed=*/true);
  cyclic.add_edge(1, 2, 1.0);
  cyclic.add_edge(2, 3, 1.0);
  cyclic.add_edge(3, 1, 1.0);
  EXPECT_TRUE(cyclic.has_cycle());
}

TEST(WeightedGraph, HasCycleWorksForUndirectedGraphs) {
  WeightedGraph<int> tree(/*directed=*/false);
  tree.add_edge(1, 2, 1.0);
  tree.add_edge(2, 3, 1.0);
  EXPECT_FALSE(tree.has_cycle());

  WeightedGraph<int> triangle(/*directed=*/false);
  triangle.add_edge(1, 2, 1.0);
  triangle.add_edge(2, 3, 1.0);
  triangle.add_edge(3, 1, 1.0);
  EXPECT_TRUE(triangle.has_cycle());
}

TEST(WeightedGraph, MissingVerticesThrowForLookupOperations) {
  WeightedGraph<int> graph;
  graph.add_vertex(1);

  EXPECT_THROW(static_cast<void>(graph.neighbors(9)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(graph.dijkstra(9)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(graph.bellman_ford(9)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(graph.edge_weight(1, 9)), std::out_of_range);
}

TEST(WeightedGraph, ClearResetsGraph) {
  WeightedGraph<int> graph;
  graph.add_edge(1, 2, 1.0);

  graph.clear();

  EXPECT_TRUE(graph.empty());
  EXPECT_EQ(graph.vertex_count(), 0u);
  EXPECT_EQ(graph.edge_count(), 0u);
}
