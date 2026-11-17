/// @file topological_sort.cpp
/// @brief Actual checked Kahn emission/decrement kernel and independent order witness validation.
#include <algorithm>
#include <datamunge/algorithms/topological_sort.hpp>
#include <queue>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private complete-geometry validation and actual queue producer.
namespace {
/// @brief Execute actual FIFO emissions, capture decisions and retain partial cyclic progress.
TopologicalRun execute(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    const TopologicalOptions&                               o,
    bool                                                    capture) {
  for (auto [u, v] : edges)
    if (u >= n || v >= n)
      throw std::invalid_argument("topological endpoint");
  if ((o.max_vertices && n > o.max_vertices) || (o.max_edges && edges.size() > o.max_edges))
    throw std::length_error("topological complete geometry");
  TopologicalRun r;
  r.initial_indegree.assign(n, 0);
  std::vector<std::vector<std::size_t>> succ(n);
  for (auto [u, v] : edges) {
    succ[u].push_back(v);
    ++r.initial_indegree[v];
  }
  r.remaining_indegree = r.initial_indegree;
  r.positions.assign(n, n);
  r.levels.assign(n, 0);
  std::queue<std::size_t> ready;
  /// @brief Capture a real queue/edge event using its current producing counts.
  auto record =
      [&](const char* action, std::size_t u, std::size_t v, std::size_t before, std::size_t after) {
        if (!capture)
          return;
        if (o.max_records && r.steps.size() >= o.max_records)
          throw std::length_error("topological history");
        r.steps.push_back({action, u, v, before, after, ready.size(), r.emitted.size()});
      };
  for (std::size_t u = 0; u < n; ++u)
    if (!r.remaining_indegree[u]) {
      ready.push(u);
      ++r.enqueues;
      record("source", u, n, 0, 0);
    }
  while (!ready.empty()) {
    auto u = ready.front();
    ready.pop();
    r.positions[u] = r.emitted.size();
    r.emitted.push_back(u);
    record("emit", u, n, 0, 0);
    for (auto v : succ[u]) {
      auto before = r.remaining_indegree[v];
      --r.remaining_indegree[v];
      ++r.edge_visits;
      r.levels[v] = std::max(r.levels[v], r.levels[u] + 1);
      if (!r.remaining_indegree[v]) {
        ready.push(v);
        ++r.enqueues;
      }
      record("edge", u, v, before, r.remaining_indegree[v]);
    }
  }
  r.result.is_dag = r.emitted.size() == n;
  if (r.result.is_dag)
    r.result.order = r.emitted;
  else
    record("cycle", n, n, 0, 0);
  return r;
}
} // namespace
TopologicalRun topological_sort_run(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    const TopologicalOptions&                               options) {
  return execute(n, edges, options, false);
}
TopologicalRun topological_sort_trace(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    const TopologicalOptions&                               options) {
  return execute(n, edges, options, true);
}
TopologicalSortResult topological_sort(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges) {
  return topological_sort_run(n, edges).result;
}
bool is_topological_order(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    const std::vector<std::size_t>&                         order) {
  if (order.size() != n)
    return false;

  // position[v] = index of v within `order`; the sentinel n marks "not yet seen".
  std::vector<std::size_t> position(n, n);
  for (std::size_t p = 0; p < order.size(); ++p) {
    const std::size_t v = order[p];
    if (v >= n || position[v] != n)
      return false; // out of range, or not a permutation
    position[v] = p;
  }

  // Every edge u -> v must place u strictly before v.
  for (const auto& [u, v] : edges) {
    if (u >= n || v >= n)
      return false;
    if (position[u] >= position[v])
      return false;
  }
  return true;
}
} // namespace datamunge::algorithms
