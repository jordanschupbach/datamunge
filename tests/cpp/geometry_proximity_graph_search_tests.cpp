#include <datamunge/geometry/best_bin_first.hpp>
#include <datamunge/geometry/proximity_graph_search.hpp>
#include <datamunge/stats/knn_graph.hpp>

#include <gtest/gtest.h>

#include <random>
#include <vector>

namespace geo = datamunge::geometry;

TEST(ProximityGraphSearch, LargeEfOnAConnectedGraphIsExactAndEfOneIsGreedy) {
  std::mt19937 rng(4);
  std::normal_distribution<double> z;
  std::vector<std::vector<double>> pts(400, std::vector<double>(6));
  for (auto& p : pts)
    for (auto& v : p) v = z(rng);
  const datamunge::stats::KnnGraph graph(pts, 8);
  ASSERT_EQ(graph.components(), 1u);
  std::vector<std::vector<std::size_t>> adjacency(pts.size());
  for (std::size_t i = 0; i < pts.size(); ++i) adjacency[i] = graph.neighbors(i);
  for (int q = 0; q < 30; ++q) {
    std::vector<double> query(6);
    for (auto& v : query) v = z(rng);
    const auto truth = geo::knn_brute_force(pts, query, 3);
    const auto all = geo::proximity_graph_search(pts, adjacency, query, 0, 3, 400);
    ASSERT_EQ(all.indices.size(), 3u);
    for (int i = 0; i < 3; ++i) EXPECT_EQ(all.indices[i], truth[i]);
    EXPECT_EQ(all.evaluations, 400);  // ef = n explores the whole graph
    // ef = 1: greedy routing ends at a vertex with no closer neighbour.
    const auto greedy = geo::proximity_graph_search(pts, adjacency, query, 0, 1, 1);
    const auto end = static_cast<std::size_t>(greedy.indices[0]);
    for (const auto v : adjacency[end]) {
      double dv = 0;
      for (int a = 0; a < 6; ++a) dv += (pts[v][a] - query[a]) * (pts[v][a] - query[a]);
      EXPECT_GE(std::sqrt(dv), greedy.distances[0] - 1e-12);
    }
    EXPECT_LE(greedy.evaluations, all.evaluations);
  }
  EXPECT_THROW((void)geo::proximity_graph_search(pts, adjacency, pts[0], 400, 1, 4), std::invalid_argument);
}
