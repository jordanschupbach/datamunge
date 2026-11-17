#include <gtest/gtest.h>

#include <algorithm>
#include <datamunge/algorithms/minimum_degree.hpp>
#include <numeric>
#include <stdexcept>
using namespace datamunge::algorithms;
TEST(MinimumDegreeTrace, ExhaustiveSimpleGraphsAndActualFill) {
  for (int n = 0; n <= 5; ++n) {
    int edges = n * (n - 1) / 2;
    for (unsigned mask = 0; mask < (1U << edges); ++mask) {
      std::vector<std::vector<int>>  a(n);
      std::vector<std::vector<bool>> matrix(n, std::vector<bool>(n));
      unsigned                       bit = 0;
      for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j, ++bit)
          if (mask & (1U << bit)) {
            a[i].push_back(j);
            a[j].push_back(i);
            matrix[i][j] = matrix[j][i] = true;
          }
      auto t = minimum_degree_trace(a);
      EXPECT_EQ(t.order, minimum_degree_ordering(a));
      std::vector<bool> gone(n);
      std::size_t       fill = 0, trials = 0;
      for (int step = 0; step < n; ++step) {
        std::vector<int> degrees(n);
        int              v = -1;
        for (int i = 0; i < n; ++i)
          if (!gone[i]) {
            for (int j = 0; j < n; ++j)
              degrees[i] += !gone[j] && matrix[i][j];
            if (v < 0 || degrees[i] < degrees[v])
              v = i;
          }
        auto& s = t.steps[step];
        EXPECT_EQ(s.vertex, v);
        EXPECT_EQ(t.order[step], v);
        std::vector<int> neighbors;
        for (int j = 0; j < n; ++j)
          if (!gone[j] && matrix[v][j])
            neighbors.push_back(j);
        EXPECT_EQ(s.neighbors, neighbors);
        std::vector<std::pair<int, int>> added;
        for (std::size_t i = 0; i < neighbors.size(); ++i)
          for (std::size_t j = i + 1; j < neighbors.size(); ++j) {
            ++trials;
            int u = neighbors[i], w = neighbors[j];
            if (!matrix[u][w]) {
              matrix[u][w] = matrix[w][u] = true;
              added.emplace_back(u, w);
              ++fill;
            }
          }
        EXPECT_EQ(s.fill, added);
        gone[v] = true;
      }
      EXPECT_EQ(t.fill_edges, fill);
      EXPECT_EQ(t.pair_trials, trials);
      EXPECT_NO_THROW(minimum_degree_trace(a, t.captured_entries));
      if (t.captured_entries > 1)
        EXPECT_THROW(minimum_degree_trace(a, t.captured_entries - 1), std::length_error);
    }
  }
}
TEST(MinimumDegreeTrace, FixedOrdersNormalizationGuardsAndOriginalArrow) {
  auto a       = minimum_degree_arrow();
  auto greedy  = minimum_degree_trace(a);
  auto natural = elimination_trace(a, elimination_natural_order(16));
  EXPECT_EQ(greedy.fill_edges, 0);
  EXPECT_EQ(natural.fill_edges, 105);
  auto t = minimum_degree_trace({{0, 1, 1, 2}, {}, {}});
  EXPECT_EQ(t.order, (std::vector<int>{1, 0, 2}));
  EXPECT_EQ(t.input[0].size(), 4);
  EXPECT_EQ(t.fill_edges, 0);
  EXPECT_THROW(minimum_degree_trace({{1}}), std::invalid_argument);
  EXPECT_THROW(minimum_degree_trace({{-1}}), std::invalid_argument);
  EXPECT_THROW(elimination_trace(a, {0}), std::invalid_argument);
  EXPECT_THROW(elimination_trace({{}, {}}, {0, 0}), std::invalid_argument);
  EXPECT_THROW(elimination_trace({{}, {}}, {0, 2}), std::invalid_argument);
  EXPECT_THROW(minimum_degree_arrow(-1), std::invalid_argument);
  EXPECT_THROW(elimination_natural_order(-1), std::invalid_argument);
  auto copy = t.input;
  a.clear();
  EXPECT_EQ(t.input, copy);
  EXPECT_NO_THROW(minimum_degree_trace(minimum_degree_arrow(), 0));
}
