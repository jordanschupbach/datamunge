/// @file proximity_graph_search.cpp
/// @brief Best-first search on proximity graphs (the NSW/HNSW search loop).

#include <datamunge/geometry/proximity_graph_search.hpp>

#include <algorithm>
#include <cmath>
#include <queue>
#include <stdexcept>
#include <utility>

namespace datamunge::geometry {

GraphSearchResult proximity_graph_search(const std::vector<std::vector<double>>& points,
                                         const std::vector<std::vector<std::size_t>>& adjacency,
                                         const std::vector<double>& query, std::size_t entry, int k, int ef) {
  if (points.size() != adjacency.size())
    throw std::invalid_argument("proximity_graph_search: one neighbour list per point");
  if (entry >= points.size()) throw std::invalid_argument("proximity_graph_search: entry out of range");
  if (k < 1) throw std::invalid_argument("proximity_graph_search: k must be at least 1");
  ef = std::max(ef, k);
  GraphSearchResult out;
  const auto distance = [&](std::size_t i) {
    if (points[i].size() != query.size())
      throw std::invalid_argument("proximity_graph_search: the query has the wrong dimension");
    double s = 0.0;
    for (std::size_t a = 0; a < query.size(); ++a) s += (points[i][a] - query[a]) * (points[i][a] - query[a]);
    ++out.evaluations;
    return std::sqrt(s);
  };
  using Item = std::pair<double, std::size_t>;
  std::priority_queue<Item, std::vector<Item>, std::greater<>> candidates;  // closest first
  std::priority_queue<Item> results;                                        // farthest first
  std::vector<char> visited(points.size(), 0);
  const double d0 = distance(entry);
  visited[entry] = 1;
  candidates.emplace(d0, entry);
  results.emplace(d0, entry);
  while (!candidates.empty()) {
    const auto [d, c] = candidates.top();
    if (d > results.top().first) break;  // nothing unexpanded can improve the result list
    candidates.pop();
    ++out.expansions;
    for (const std::size_t v : adjacency[c]) {
      if (visited[v]) continue;
      visited[v] = 1;
      const double dv = distance(v);
      if (static_cast<int>(results.size()) < ef || dv < results.top().first) {
        candidates.emplace(dv, v);
        results.emplace(dv, v);
        if (static_cast<int>(results.size()) > ef) results.pop();
      }
    }
  }
  std::vector<Item> found;
  while (!results.empty()) {
    found.push_back(results.top());
    results.pop();
  }
  std::ranges::reverse(found);
  const std::size_t kk = std::min<std::size_t>(static_cast<std::size_t>(k), found.size());
  for (std::size_t i = 0; i < kk; ++i) {
    out.indices.push_back(static_cast<int>(found[i].second));
    out.distances.push_back(found[i].first);
  }
  return out;
}

}  // namespace datamunge::geometry
