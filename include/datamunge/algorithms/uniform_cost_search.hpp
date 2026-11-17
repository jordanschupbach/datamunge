#include <tuple>
#pragma once
/// @file uniform_cost_search.hpp
/// @brief Checked native uniform cost search and actual producing diagnostics.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual goal-directed Dijkstra result.
/// @par Example
/// @code
/// auto r=dm::algorithms::uniform_cost_search(2,{{{1,2}},{}},0,1);
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
struct UniformCostResult {
  double              cost{0};       ///< Goal distance when found;0 otherwise, consult found.
  std::vector<int>    path;          ///< Actual predecessor path when found; empty otherwise.
  bool                found{false};  ///< Goal was settled.
  int                 expanded{0};   ///< Actual first-time vertex settlements, including goal.
  std::vector<int>    settled;       ///< Actual settlement order.
  std::vector<double> settled_costs; ///< Actual distances aligned to settled.
  std::size_t edge_checks{0}; ///< Actual outgoing relaxation checks; goal edges not processed.
  std::size_t relaxations{0}; ///< Actual successful distance decreases.
};
/// @brief Find a least-cost path using a minimum-distance frontier.
/// All edges validated before early stopping. Equal queue distances settle smaller vertex ID first; strict improvements retain earlier parent ties. O((n+m)log(n+m)) time, O(n+m) storage; no shared state.
/// @param n Positive vertex count<=1000000, exactly adj.size().
/// @param adj Complete directed adjacency; finite nonnegative weights and in-range endpoints, <=10000000 edges.
/// @param start In-range source.
/// @param goal In-range goal.
/// @return Actual path, cost, settlements and work; unreachable has found=false.
/// @throws std::invalid_argument Invalid dimensions, endpoint or weight.
/// @throws std::length_error Vertex/edge budget exceeded.
/// @throws std::overflow_error A considered path sum is nonfinite.
/// @par Example
/// @code
/// auto r=dm::algorithms::uniform_cost_search(2,{{{1,2}},{}},0,1); //cost2
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
inline UniformCostResult uniform_cost_search(
    int                                                     n,
    const std::vector<std::vector<std::pair<int, double>>>& adj,
    int                                                     start,
    int                                                     goal) {
  if (n <= 0 || adj.size() != static_cast<std::size_t>(n) || start < 0 || start >= n || goal < 0 ||
      goal >= n)
    throw std::invalid_argument("UCS shape/endpoints");
  if (n > 1000000)
    throw std::length_error("UCS vertices");
  std::size_t edges = 0;
  for (auto& row : adj)
    for (auto [v, w] : row) {
      if (v < 0 || v >= n || !std::isfinite(w) || w < 0)
        throw std::invalid_argument("UCS edge");
      if (++edges > 10000000)
        throw std::length_error("UCS edges");
    }
  std::vector<double> dist(n, std::numeric_limits<double>::infinity());
  std::vector<int>    parent(n, -1);
  std::vector<bool>   done(n);
  using Item = std::pair<double, int>;
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;
  dist[start] = 0;
  pq.emplace(0, start);
  UniformCostResult r;
  while (!pq.empty()) {
    auto [c, u] = pq.top();
    pq.pop();
    if (done[u])
      continue;
    done[u] = true;
    ++r.expanded;
    r.settled.push_back(u);
    r.settled_costs.push_back(c);
    if (u == goal) {
      r.found = true;
      r.cost  = c;
      for (int v = goal; v != -1; v = parent[v])
        r.path.push_back(v);
      std::reverse(r.path.begin(), r.path.end());
      return r;
    }
    for (auto [v, w] : adj[u]) {
      ++r.edge_checks;
      if (!done[v]) {
        double next = c + w;
        if (!std::isfinite(next))
          throw std::overflow_error("UCS cost overflow");
        if (next < dist[v]) {
          dist[v]   = next;
          parent[v] = u;
          pq.emplace(next, v);
          ++r.relaxations;
        }
      }
    }
  }
  return r;
}
/// @brief Return the original undirected ten-node UCS experiment.
/// Vertices6..9 lie beyond goal5; no labels or answers are baked into search.
/// @return Owned adjacency with the original12 undirected edges.
/// @par Example
/// @code
/// auto graph=dm::algorithms::uniform_cost_example_graph();
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
inline std::vector<std::vector<std::pair<int, double>>> uniform_cost_example_graph() {
  std::vector<std::vector<std::pair<int, double>>> adj(10);
  for (auto [u, v, w] : std::vector<std::tuple<int, int, double>>{
           {0, 1, 2},
           {0, 2, 3},
           {1, 3, 2},
           {2, 3, 1},
           {3, 4, 4},
           {3, 5, 2},
           {4, 5, 5},
           {5, 6, 3},
           {6, 7, 2},
           {7, 8, 2},
           {8, 9, 2},
           {6, 9, 10}}) {
    adj[u].push_back({v, w});
    adj[v].push_back({u, w});
  }
  return adj;
}

} // namespace datamunge::algorithms
