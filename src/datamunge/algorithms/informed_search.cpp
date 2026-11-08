/// @file informed_search.cpp
/// @brief Shared reopening A* implementation for one goal or a goal set.
#include <datamunge/algorithms/informed_search.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Private validation and queue traversal for directed informed search.
namespace {

/// @brief Validate and mark a nonempty goal list, retaining duplicate-goal semantics.
/// @param n Vertex count.
/// @param goals Requested goals.
/// @return Length-n goal membership mask.
/// @throws std::invalid_argument If the list is empty or a goal is outside the graph.
std::vector<bool> goal_mask(std::size_t n, const std::vector<std::size_t>& goals) {
  if (goals.empty()) throw std::invalid_argument("informed search: no goals");
  std::vector<bool> mask(n);
  for (auto v : goals) {
    if (v >= n) throw std::invalid_argument("informed search: invalid goal");
    mask[v] = true;
  }
  return mask;
}

/// @brief Traverse by g+h with reopening; optionally retain non-stale removals.
/// @param n Number of vertices.
/// @param edges Directed finite nonnegative edges.
/// @param start Initial vertex.
/// @param goal Goal mask of length n.
/// @param h Finite heuristic of length n.
/// @param trace Nullable borrowed output vector, appended to during this call.
/// @return Owned result, with zero cost and an empty path for an unreachable goal set.
/// @throws std::invalid_argument For invalid vertices, weights or heuristic.
/// @throws std::overflow_error If a path sum or queue key overflows.
InformedSearchResult search(std::size_t n,
    const std::vector<std::tuple<std::size_t,std::size_t,double>>& edges,
    std::size_t start, const std::vector<bool>& goal, const std::vector<double>& h,
    std::vector<InformedSearchExpansion>* trace) {
  if (start >= n || goal.size() != n || h.size() != n)
    throw std::invalid_argument("informed search: invalid input dimensions or start");
  for (double value : h)
    if (!std::isfinite(value)) throw std::invalid_argument("informed search: nonfinite heuristic");
  std::vector<std::vector<std::pair<std::size_t,double>>> adjacency(n);
  for (auto [u,v,w] : edges) {
    if (u >= n || v >= n || !std::isfinite(w) || w < 0)
      throw std::invalid_argument("informed search: invalid edge");
    adjacency[u].push_back({v,w});
  }
  std::vector<double> g(n, std::numeric_limits<double>::infinity());
  std::vector<std::size_t> parent(n, n);
  /// @brief Priority, vertex tie-break and original g label for exact stale-entry checks.
  using Entry = std::tuple<double,std::size_t,double>;
  std::priority_queue<Entry,std::vector<Entry>,std::greater<Entry>> queue;
  g[start] = 0;
  queue.emplace(h[start], start, 0);
  InformedSearchResult result;
  while (!queue.empty()) {
    const auto [priority,u,label] = queue.top();
    queue.pop();
    if (label != g[u]) continue;
    ++result.expanded;
    if (trace) trace->push_back({u,g[u],priority,queue.size()});
    if (goal[u]) {
      result.found = true;
      result.cost = g[u];
      for (auto v = u; v < n; v = parent[v]) result.path.push_back(v);
      std::reverse(result.path.begin(), result.path.end());
      return result;
    }
    for (auto [v,w] : adjacency[u]) {
      const double candidate = g[u] + w;
      if (!std::isfinite(candidate)) throw std::overflow_error("informed search: path cost overflow");
      if (candidate >= g[v]) continue;
      const double key = candidate + h[v];
      if (!std::isfinite(key)) throw std::overflow_error("informed search: priority overflow");
      g[v] = candidate;
      parent[v] = u;
      queue.emplace(key,v,candidate);
    }
  }
  return result;
}
} // namespace

InformedSearchResult a_star_search(std::size_t n,
    const std::vector<std::tuple<std::size_t,std::size_t,double>>& edges,
    std::size_t start, std::size_t goal, const std::vector<double>& heuristic) {
  return search(n,edges,start,goal_mask(n,{goal}),heuristic,nullptr);
}

InformedSearchResult b_star_search(std::size_t n,
    const std::vector<std::tuple<std::size_t,std::size_t,double>>& edges,
    std::size_t start, const std::vector<std::size_t>& goals, const std::vector<double>& h) {
  return search(n,edges,start,goal_mask(n,goals),h,nullptr);
}

InformedSearchTrace b_star_search_trace(std::size_t n,
    const std::vector<std::tuple<std::size_t,std::size_t,double>>& edges,
    std::size_t start, const std::vector<std::size_t>& goals, const std::vector<double>& h) {
  InformedSearchTrace trace;
  trace.result = search(n,edges,start,goal_mask(n,goals),h,&trace.expansions);
  return trace;
}
} // namespace datamunge::algorithms
