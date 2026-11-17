/// @file iterative_deepening.cpp
/// @brief Single explicit-frame IDDFS kernel preserving recursive input-order search.
#include <algorithm>
#include <datamunge/algorithms/iterative_deepening.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private ordered adjacency, continuation frames and actual-record execution.
namespace {
/// @brief Neighbor and original input arc id, retaining loops and duplicates.
using Arc = std::pair<std::size_t, std::size_t>;
/// @brief Input-order outgoing arc lists for every complete vertex.
using Adjacency = std::vector<std::vector<Arc>>;
/// @brief Heap continuation corresponding to one recursive path activation.
struct Frame {
  std::size_t vertex;   ///< Active vertex.
  std::size_t next{0};  ///< Next outgoing adjacency position.
  std::size_t incoming; ///< Original incoming edge id, E for root.
};
/// @brief Increment an actual selected count without wrapping size_t.
/// @param counter Mutable count.
/// @throws std::length_error On exhausted count range.
void increment(std::size_t& counter) {
  if (counter == std::numeric_limits<std::size_t>::max())
    throw std::length_error("IDDFS count overflow");
  ++counter;
}
/// @brief Validate every requested terminal and original directed endpoint before allocation.
/// @param n Complete vertex count.
/// @param edges Complete borrowed arcs.
/// @param start Requested source.
/// @param goal Requested target.
/// @param options Complete admission policies.
/// @throws std::invalid_argument On any invalid graph-domain value.
/// @throws std::length_error On an enabled complete graph cap.
void validate(
    std::size_t             n,
    const std::vector<Arc>& edges,
    std::size_t             start,
    std::size_t             goal,
    DeepeningOptions        options) {
  if (start >= n || goal >= n)
    throw std::invalid_argument("IDDFS terminal outside graph");
  for (auto [u, v] : edges)
    if (u >= n || v >= n)
      throw std::invalid_argument("IDDFS endpoint outside graph");
  if ((options.max_vertices && n > options.max_vertices) ||
      (options.max_edges && edges.size() > options.max_edges))
    throw std::length_error("IDDFS complete graph cap");
}
/// @brief Execute actual finite simple-path iterative deepening, optionally capturing events.
/// @tparam Record Whether to own original arcs and append actual transition records.
/// @param n Complete vertex count.
/// @param edges Complete original directed arcs.
/// @param start Source vertex.
/// @param goal Target vertex.
/// @param maximum Requested maximum depth, safely bounded by n-1 after validation.
/// @param options Complete graph and online scalar-event policies.
/// @return Actual final path, pass summaries and optional owned chronological history.
/// @throws std::invalid_argument On invalid complete graph domain.
/// @throws std::length_error On count overflow, cap or capacity exhaustion.
/// @throws std::bad_alloc On allocation failure.
template<bool Record>
DeepeningTrace execute(
    std::size_t             n,
    const std::vector<Arc>& edges,
    std::size_t             start,
    std::size_t             goal,
    std::size_t             maximum,
    DeepeningOptions        options) {
  validate(n, edges, start, goal, options);
  DeepeningTrace out;
  out.n     = n;
  out.start = start;
  out.goal  = goal;
  if constexpr (Record)
    out.edges = edges;
  Adjacency adjacency(n);
  for (std::size_t i = 0; i < edges.size(); ++i)
    adjacency[edges[i].first].push_back({edges[i].second, i});
  std::vector<unsigned char> on(n, 0);
  std::vector<Frame>         stack;
  out.run.effective_max_depth = std::min(maximum, n - 1);
  // Append one actual event; no input snapshots or synthetic transition replay.
  const auto emit = [&](DeepeningEventKind kind,
                        std::size_t        limit,
                        std::size_t        u,
                        std::size_t        depth,
                        std::size_t        neighbor,
                        std::size_t        edge,
                        bool               blocked = false) {
    if constexpr (Record) {
      if (options.max_events && out.events.size() >= options.max_events)
        throw std::length_error("IDDFS actual event cap");
      out.events.push_back({kind, limit, u, depth, neighbor, edge, blocked});
    }
  };
  for (std::size_t limit = 0;; ++limit) {
    DeepeningPass pass;
    pass.limit = limit;
    emit(DeepeningEventKind::pass_start, limit, start, 0, n, edges.size());
    // Enter is the actual recursive-prefix entry, including goal and depth cutoffs.
    const auto enter = [&](std::size_t v, std::size_t edge) {
      increment(pass.entered);
      increment(out.run.result.expanded);
      on[v] = 1;
      stack.push_back({v, 0, edge});
      pass.maximum_stack = std::max(pass.maximum_stack, stack.size());
      emit(DeepeningEventKind::enter, limit, v, stack.size() - 1, n, edge);
    };
    enter(start, edges.size());
    while (!stack.empty()) {
      auto&      f = stack.back();
      const auto u = f.vertex, depth = stack.size() - 1;
      if (u == goal) {
        emit(DeepeningEventKind::goal, limit, u, depth, n, f.incoming);
        pass.found           = true;
        out.run.result.found = true;
        out.run.result.path.reserve(stack.size());
        for (const auto& frame : stack)
          out.run.result.path.push_back(frame.vertex);
        break;
      }
      if (depth == limit || f.next == adjacency[u].size()) {
        if (depth == limit) {
          increment(pass.cutoffs);
          emit(DeepeningEventKind::cutoff, limit, u, depth, n, f.incoming);
        }
        const auto incoming = f.incoming;
        stack.pop_back();
        on[u] = 0;
        emit(DeepeningEventKind::leave, limit, u, depth, n, incoming);
        continue;
      }
      const auto [v, id] = adjacency[u][f.next++];
      increment(pass.edges_examined);
      const bool blocked = on[v] != 0;
      emit(DeepeningEventKind::examine, limit, u, depth, v, id, blocked);
      if (blocked)
        increment(pass.cycle_skips);
      else
        enter(v, id);
    }
    out.run.passes.push_back(pass);
    emit(DeepeningEventKind::pass_end, limit, start, 0, n, edges.size());
    if (pass.found || limit == out.run.effective_max_depth)
      break;
  }
  return out;
}
} // namespace
BasicPathResult iterative_deepening_dfs(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    std::size_t                                             start,
    std::size_t                                             goal,
    std::size_t                                             maximum) {
  return execute<false>(n, edges, start, goal, maximum, {0, 0, 0}).run.result;
}
DeepeningRun iterative_deepening_dfs_run(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    std::size_t                                             start,
    std::size_t                                             goal,
    std::size_t                                             maximum,
    DeepeningOptions                                        options) {
  return execute<false>(n, edges, start, goal, maximum, options).run;
}
DeepeningTrace iterative_deepening_dfs_trace(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    std::size_t                                             start,
    std::size_t                                             goal,
    std::size_t                                             maximum,
    DeepeningOptions                                        options) {
  return execute<true>(n, edges, start, goal, maximum, options);
}
} // namespace datamunge::algorithms
