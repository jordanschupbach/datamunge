/// @file push_relabel.cpp
/// @brief One checked FIFO preflow producer with independently identified residual pairs.
#include <datamunge/algorithms/push_relabel.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Actual residual graph, checked arithmetic and optional producing capture.
namespace preflow_detail {
/// @brief One actual oriented residual arc with its original input identity.
struct Arc {
  std::size_t to;       ///< Actual target vertex.
  std::size_t reverse;  ///< Actual paired reverse adjacency index.
  std::size_t input;    ///< Original input edge id.
  double      capacity; ///< Actual nonnegative residual.
  bool        forward;  ///< Original direction rather than cancellation.
};
/// @brief Owned adjacency of independent original-edge residual pairs.
using Residual = std::vector<std::vector<Arc>>;
/// @brief Admit one selected actual event before execution.
/// @param work Mutable local selected work.
/// @param counter Actual event-kind count.
/// @param options Selected operation policy.
/// @throws std::length_error If selected work/history policies or representable residual/storage geometry are exceeded.
void tick(PushRelabelWork& work, std::size_t& counter, PushRelabelOptions options) {
  if (work.operations == std::numeric_limits<std::size_t>::max() ||
      (options.max_operations && work.operations >= options.max_operations))
    throw std::length_error("preflow work cap");
  ++work.operations;
  ++counter;
}
/// @brief Admit exact actual logical history entries before copies.
/// @param out Owned trace under construction.
/// @param amount Next actual entries.
/// @param options Retained entry policy.
/// @throws std::length_error If selected work/history policies or representable residual/storage geometry are exceeded.
void retain(PushRelabelTrace& out, std::size_t amount, PushRelabelOptions options) {
  if (amount > std::numeric_limits<std::size_t>::max() - out.captured_entries ||
      (options.max_trace_entries && (out.captured_entries > options.max_trace_entries ||
                                     amount > options.max_trace_entries - out.captured_entries)))
    throw std::length_error("preflow history cap");
  out.captured_entries += amount;
}
/// @brief Require a positive finite representable addition/subtraction.
/// @param before Actual finite old value.
/// @param amount Actual positive finite push.
/// @param add Choose addition rather than subtraction.
/// @return Actual updated finite value making strict progress.
/// @throws std::overflow_error If an actual residual/excess update is nonfinite.
/// @throws std::runtime_error If a positive update cannot make representable progress or a checked residual/height invariant fails.
double update(double before, double amount, bool add) {
  double after = add ? before + amount : before - amount;
  if (!std::isfinite(after))
    throw std::overflow_error("preflow nonfinite update");
  if (add ? !(after > before) : !(after < before))
    throw std::runtime_error("preflow nonrepresentable progress");
  return after;
}
/// @brief Validate and perform one native FIFO preflow search.
/// @param n Vertex count.
/// @param edges Borrowed complete network.
/// @param source Source id.
/// @param sink Sink id.
/// @param options Policies.
/// @param capture Whether actual initialization/updates are retained.
/// @return Owned actual flow/work/final state and optional history.
/// @throws std::invalid_argument If terminals/endpoints are invalid or capacities are negative/nonfinite.
/// @throws std::length_error If selected work/history policies or representable residual/storage geometry are exceeded.
/// @throws std::overflow_error If an actual residual/excess update is nonfinite.
/// @throws std::runtime_error If a positive update cannot make representable progress or a checked residual/height invariant fails.
PushRelabelTrace produce(
    std::size_t                      n,
    const std::vector<CapacityEdge>& edges,
    std::size_t                      source,
    std::size_t                      sink,
    PushRelabelOptions               options,
    bool                             capture) {
  if (source >= n || sink >= n || source == sink)
    throw std::invalid_argument("preflow terminals");
  if (n > std::numeric_limits<std::size_t>::max() / 2 ||
      edges.size() > std::numeric_limits<std::size_t>::max() / 2)
    throw std::length_error("preflow residual geometry");
  for (const auto& [u, v, c] : edges)
    if (u >= n || v >= n || c < 0 || !std::isfinite(c))
      throw std::invalid_argument("preflow invalid edge");
  Residual g(n);
  for (std::size_t i = 0; i < edges.size(); ++i) {
    auto [u, v, c] = edges[i];
    auto forward = g[u].size(), reverse = g[v].size() + (u == v ? 1 : 0);
    g[u].push_back({v, reverse, i, c, true});
    g[v].push_back({u, forward, i, 0, false});
  }
  PushRelabelTrace out;
  auto&            r    = out.run;
  auto&            work = r.work;
  r.heights.assign(n, 0);
  r.excess.assign(n, 0);
  auto& height   = r.heights;
  auto& excess   = r.excess;
  height[source] = n;
  std::vector<std::size_t> current(n);
  if (capture) {
    if (edges.size() > std::numeric_limits<std::size_t>::max() - 1 - 2 * n)
      throw std::length_error("preflow history geometry");
    retain(out, 1 + edges.size() + 2 * n, options);
    out.network         = {n, edges, source, sink};
    out.initial_heights = height;
    out.initial_excess  = excess;
  }
  /// @brief Perform one actual paired residual/excess update after checking all arithmetic.
  /// @param u Actual tail vertex.
  /// @param index Actual outgoing residual index.
  /// @param amount Actual strictly positive admissible amount.
  /// @param initial Whether this is source initialization.
  /// @throws std::length_error On selected work/history exhaustion.
  /// @throws std::overflow_error On nonfinite updates.
  /// @throws std::runtime_error On nonrepresentable progress.
  auto push = [&](std::size_t u, std::size_t index, double amount, bool initial) {
    auto& a    = g[u][index];
    auto& back = g[a.to][a.reverse];
    auto  v    = a.to;
    tick(work, work.pushes, options);
    if (capture)
      retain(out, 1, options);
    double cap = update(a.capacity, amount, false), reverse = update(back.capacity, amount, true),
           eu = update(excess[u], amount, false), ev = update(excess[v], amount, true);
    PreflowEvent e{
        PreflowEventKind::Push,
        initial,
        u,
        v,
        a.input,
        a.forward,
        height[u],
        height[u],
        height[v],
        amount,
        a.capacity,
        cap,
        back.capacity,
        reverse,
        excess[u],
        eu,
        excess[v],
        ev};
    a.capacity    = cap;
    back.capacity = reverse;
    excess[u]     = eu;
    excess[v]     = ev;
    if (capture)
      out.events.push_back(e);
  };
  for (std::size_t i = 0; i < g[source].size(); ++i) {
    tick(work, work.arc_inspections, options);
    auto& a = g[source][i];
    if (a.to != source && a.capacity > 0)
      push(source, i, a.capacity, true);
  }
  std::queue<std::size_t> queue;
  std::vector<bool>       queued(n);
  for (std::size_t v = 0; v < n; ++v)
    if (v != source && v != sink && excess[v] > 0) {
      queue.push(v);
      queued[v] = true;
    }
  while (!queue.empty()) {
    tick(work, work.queue_pops, options);
    auto u = queue.front();
    queue.pop();
    queued[u] = false;
    while (excess[u] > 0) {
      if (current[u] == g[u].size()) {
        auto minimum = std::numeric_limits<std::size_t>::max();
        for (const auto& a : g[u]) {
          tick(work, work.arc_inspections, options);
          if (a.capacity > 0)
            minimum = std::min(minimum, height[a.to]);
        }
        if (minimum == std::numeric_limits<std::size_t>::max() || minimum + 1 <= height[u])
          throw std::runtime_error("preflow relabel invariant");
        tick(work, work.relabels, options);
        if (capture) {
          retain(out, 1, options);
          PreflowEvent e{};
          e.kind               = PreflowEventKind::Relabel;
          e.from               = u;
          e.to                 = u;
          e.input              = std::numeric_limits<std::size_t>::max();
          e.height_before      = height[u];
          e.height_after       = minimum + 1;
          e.target_height      = height[u];
          e.from_excess_before = e.from_excess_after = e.to_excess_before = e.to_excess_after =
              excess[u];
          out.events.push_back(e);
        }
        height[u]  = minimum + 1;
        current[u] = 0;
        continue;
      }
      tick(work, work.arc_inspections, options);
      auto& a = g[u][current[u]];
      if (a.capacity > 0 && height[a.to] != std::numeric_limits<std::size_t>::max() &&
          height[u] == height[a.to] + 1) {
        auto v = a.to;
        push(u, current[u], std::min(excess[u], a.capacity), false);
        if (v != source && v != sink && !queued[v] && excess[v] > 0) {
          queue.push(v);
          queued[v] = true;
        }
      } else
        ++current[u];
    }
  }
  r.result.max_flow = excess[sink];
  r.result.edge_flow.assign(edges.size(), 0);
  for (const auto& row : g)
    for (const auto& a : row)
      if (a.forward)
        r.result.edge_flow[a.input] = g[a.to][a.reverse].capacity;
  std::vector<bool>       seen(n);
  std::queue<std::size_t> reach;
  reach.push(source);
  seen[source] = true;
  while (!reach.empty()) {
    auto u = reach.front();
    reach.pop();
    for (const auto& a : g[u]) {
      tick(work, work.arc_inspections, options);
      if (a.capacity > 0 && !seen[a.to]) {
        seen[a.to] = true;
        reach.push(a.to);
      }
    }
  }
  for (std::size_t v = 0; v < n; ++v)
    if (seen[v])
      r.result.min_cut_source_side.push_back(v);
  return out;
}
} // namespace preflow_detail
MaxFlowResult push_relabel_max_flow(
    std::size_t                      vertex_count,
    const std::vector<CapacityEdge>& edges,
    std::size_t                      source,
    std::size_t                      sink) {
  return preflow_detail::produce(vertex_count, edges, source, sink, {}, false).run.result;
}
PushRelabelRun push_relabel_run(
    std::size_t                      vertex_count,
    const std::vector<CapacityEdge>& edges,
    std::size_t                      source,
    std::size_t                      sink,
    PushRelabelOptions               options) {
  return preflow_detail::produce(vertex_count, edges, source, sink, options, false).run;
}
PushRelabelTrace push_relabel_trace(
    std::size_t                      vertex_count,
    const std::vector<CapacityEdge>& edges,
    std::size_t                      source,
    std::size_t                      sink,
    PushRelabelOptions               options) {
  return preflow_detail::produce(vertex_count, edges, source, sink, options, true);
}
} // namespace datamunge::algorithms
