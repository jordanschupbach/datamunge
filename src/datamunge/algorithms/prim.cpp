/// @file prim.cpp
/// @brief Single checked lazy-heap producer with optional actual frontier capture.
#include <datamunge/algorithms/prim.hpp>
#include <functional>
#include <queue>
#include <cmath>
#include <stdexcept>
#include <limits>
namespace datamunge::algorithms {
/// @brief Local checked event policies and the actual heap producer.
namespace prim_detail {
/// @brief Admit one actual selected loop/heap action before execution.
/// @param r Mutable actual selected counters.
/// @param counter Counter for this action kind.
/// @param options Positive selected cap or zero.
/// @throws std::length_error If a selected-operation/history policy or representable storage is exceeded.
void event(PrimRun& r, std::size_t& counter, PrimOptions options) {
  if (r.operations == std::numeric_limits<std::size_t>::max() ||
      (options.max_operations && r.operations >= options.max_operations))
    throw std::length_error("prim selected work cap");
  ++r.operations;
  ++counter;
}
/// @brief Retain one actual completed frontier event under the history policy.
/// @param out Owned trace under construction.
/// @param record Immediate actual event state.
/// @param options Retained-record policy.
/// @throws std::length_error If a selected-operation/history policy or representable storage is exceeded.
void record(PrimTrace& out, PrimEvent record, PrimOptions options) {
  if (options.max_trace_records && out.events.size() >= options.max_trace_records)
    throw std::length_error("prim history cap");
  out.events.push_back(record);
}
/// @brief Validate the whole graph then execute one native lazy Prim search.
/// @param n Vertex count.
/// @param edges Borrowed complete undirected weighted graph.
/// @param start Chosen vertex.
/// @param options Selected work/history policies.
/// @param capture Whether actual events are retained.
/// @return Owned actual tree/work and optional events.
/// @throws std::invalid_argument If start/endpoints are out of range or any weight is nonfinite.
/// @throws std::length_error If a selected-operation/history policy or representable storage is exceeded.
/// @throws std::overflow_error If an actual total-weight addition is nonfinite.
PrimTrace produce(
    std::size_t                      n,
    const std::vector<WeightedEdge>& edges,
    std::size_t                      start,
    PrimOptions                      options,
    bool                             capture) {
  if (start >= n)
    throw std::invalid_argument("prim start out of range");
  for (const auto& [u, v, w] : edges)
    if (u >= n || v >= n || !std::isfinite(w))
      throw std::invalid_argument("prim invalid edge");
  std::vector<std::vector<std::pair<std::size_t, double>>> adjacency(n);
  for (const auto& [u, v, w] : edges) {
    adjacency[u].emplace_back(v, w);
    adjacency[v].emplace_back(u, w);
  }
  PrimTrace out;
  out.start           = start;
  auto&             r = out.run;
  std::vector<char> joined(n);
  joined[start] = 1;
  r.reached     = 1;
  /// @brief Actual heap key ordered by weight, then source and target.
  using Candidate = std::tuple<double, std::size_t, std::size_t>;
  std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> frontier;
  for (const auto& [v, w] : adjacency[start]) {
    event(r, r.adjacency_entries, options);
    event(r, r.pushes, options);
    frontier.emplace(w, start, v);
    if (capture)
      record(out, {true, start, v, w, frontier.size(), false, r.reached}, options);
  }
  while (!frontier.empty()) {
    event(r, r.pops, options);
    auto [w, u, v] = frontier.top();
    frontier.pop();
    bool accept = !joined[v];
    if (accept) {
      double total = r.tree.total_weight + w;
      if (!std::isfinite(total))
        throw std::overflow_error("prim total overflow");
      joined[v] = 1;
      ++r.reached;
      r.tree.edges.emplace_back(u, v, w);
      r.tree.total_weight = total;
    } else
      ++r.stale;
    if (capture)
      record(out, {false, u, v, w, frontier.size(), accept, r.reached}, options);
    if (!accept)
      continue;
    for (const auto& [target, weight] : adjacency[v]) {
      event(r, r.adjacency_entries, options);
      if (!joined[target]) {
        event(r, r.pushes, options);
        frontier.emplace(weight, v, target);
        if (capture)
          record(out, {true, v, target, weight, frontier.size(), false, r.reached}, options);
      }
    }
  }
  r.tree.is_connected = r.reached == n;
  return out;
}
} // namespace prim_detail
MinimumSpanningTree prim(std::size_t n, const std::vector<WeightedEdge>& edges, std::size_t start) {
  return prim_detail::produce(n, edges, start, {}, false).run.tree;
}
PrimRun prim_run(
    std::size_t                      n,
    const std::vector<WeightedEdge>& edges,
    std::size_t                      start,
    PrimOptions                      options) {
  return prim_detail::produce(n, edges, start, options, false).run;
}
PrimTrace prim_trace(
    std::size_t                      n,
    const std::vector<WeightedEdge>& edges,
    std::size_t                      start,
    PrimOptions                      options) {
  return prim_detail::produce(n, edges, start, options, true);
}
} // namespace datamunge::algorithms
