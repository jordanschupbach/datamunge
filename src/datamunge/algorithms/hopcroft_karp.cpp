/// @file hopcroft_karp.cpp
/// @brief Shared shortest-layer, iterative blocking-path kernel and actual histories.
#include <algorithm>
#include <datamunge/algorithms/hopcroft_karp.hpp>
#include <limits>
#include <queue>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Internal checked admission and shared producing execution.
namespace hopcroft_karp_detail {
/// @brief Infinite BFS distance, never a valid retained vertex index.
constexpr std::size_t infinity = std::numeric_limits<std::size_t>::max();
/// @brief Add logical units with representation checking.
/// @param a Existing count.
/// @param b Additional units.
/// @return Exact sum.
/// @throws std::length_error On overflow.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > infinity - a)
    throw std::length_error("Hopcroft-Karp count overflow");
  return a + b;
}
/// @brief Increment a selected actual visit count without wrapping.
/// @param count Owned selected counter updated in place.
/// @param amount Additional actual visits, default1.
/// @throws std::length_error On overflow.
void bump(std::size_t& count, std::size_t amount = 1) {
  count = add(count, amount);
}
/// @brief Admit a complete logical graph/history count against an optional policy.
/// @param value Complete required units.
/// @param cap Maximum units, zero disabling policy.
/// @throws std::length_error On policy excess.
void admit(std::size_t value, std::size_t cap) {
  if (cap && value > cap)
    throw std::length_error("Hopcroft-Karp policy limit");
}
/// @brief Admit known complete record/value units before capture or its producing work.
/// @param trace Optional owned history destination.
/// @param units Additional complete logical units.
/// @param options Capture policy.
/// @throws std::length_error On count/capture excess.
void capture(HopcroftKarpTrace* trace, std::size_t units, HopcroftKarpOptions options) {
  if (trace) {
    const auto total = add(trace->captured_entries, units);
    admit(total, options.max_trace_entries);
    trace->captured_entries = total;
  }
}
/// @brief Execute one actual shortest-layer kernel, optionally retaining producing arrays/paths.
/// @param nl Complete left vertex count.
/// @param nr Complete right vertex count.
/// @param edges Borrowed ordered endpoints.
/// @param options Complete graph/online capture policies.
/// @param trace Optional owned capture destination.
/// @return Owned actual matching, terminal cover, phase count and work.
/// @throws std::invalid_argument On endpoints before allocation/policy checks.
/// @throws std::length_error On graph/history/count/container limits.
/// @throws std::bad_alloc On allocation failure.
HopcroftKarpRun execute(
    std::size_t                          nl,
    std::size_t                          nr,
    const std::vector<HopcroftKarpEdge>& edges,
    HopcroftKarpOptions                  options,
    HopcroftKarpTrace*                   trace) {
  for (auto [u, v] : edges)
    if (u >= nl || v >= nr)
      throw std::invalid_argument("hopcroft_karp: edge endpoint out of range");
  const auto vertices = add(nl, nr);
  admit(vertices, options.max_vertices);
  admit(edges.size(), options.max_edges);
  if (nl > std::vector<std::vector<std::size_t>>{}.max_size() ||
      nl > std::vector<std::size_t>{}.max_size() || nr > std::vector<std::size_t>{}.max_size())
    throw std::length_error("Hopcroft-Karp vector size");
  std::vector<std::vector<std::size_t>> adj(nl);
  HopcroftKarpRun                       run;
  for (auto [u, v] : edges) {
    adj[u].push_back(v);
    bump(run.work.adjacency_edges);
  }
  run.result.match_left.assign(nl, kBipartiteUnmatched);
  run.result.match_right.assign(nr, kBipartiteUnmatched);
  auto&                    ml = run.result.match_left;
  auto&                    mr = run.result.match_right;
  std::vector<std::size_t> dist(nl), cursor(nl), stack, path_right;
  std::vector<bool>        seen_right(nr);
  stack.reserve(nl);
  path_right.reserve(nl);
  while (true) {
    if (trace)
      capture(trace, add(1, nl), options);
    bump(run.work.bfs_calls);
    HopcroftKarpPhase phase;
    phase.bfs_index = run.work.bfs_calls;
    std::queue<std::size_t> queue;
    for (std::size_t u = 0; u < nl; ++u) {
      dist[u] = ml[u] == kBipartiteUnmatched ? 0 : infinity;
      if (dist[u] == 0)
        queue.push(u);
      bump(run.work.initialized_left);
    }
    std::fill(seen_right.begin(), seen_right.end(), false);
    std::size_t shortest = infinity;
    while (!queue.empty()) {
      const auto u = queue.front();
      queue.pop();
      bump(run.work.queue_pops);
      if (dist[u] >= shortest)
        continue;
      for (auto v : adj[u]) {
        bump(run.work.bfs_edges);
        seen_right[v]   = true;
        const auto w    = mr[v];
        const auto next = add(dist[u], 1);
        if (w == kBipartiteUnmatched)
          shortest = std::min(shortest, next);
        else if (dist[w] == infinity && next < shortest) {
          dist[w] = next;
          queue.push(w);
        }
      }
    }
    if (trace) {
      phase.left_distance  = dist;
      phase.shortest_pairs = shortest == infinity ? 0 : shortest;
    }
    if (shortest == infinity) {
      if (trace)
        trace->phases.push_back(std::move(phase));
      for (std::size_t u = 0; u < nl; ++u)
        if (dist[u] == infinity)
          run.cover_left.push_back(u);
      for (std::size_t v = 0; v < nr; ++v)
        if (seen_right[v])
          run.cover_right.push_back(v);
      break;
    }
    bump(run.augmenting_phases);
    std::fill(cursor.begin(), cursor.end(), 0);
    for (std::size_t root = 0; root < nl; ++root) {
      if (ml[root] != kBipartiteUnmatched)
        continue;
      bump(run.work.dfs_roots);
      stack.clear();
      path_right.clear();
      stack.push_back(root);
      run.work.max_path_left_vertices = std::max(run.work.max_path_left_vertices, stack.size());
      bool augmented                  = false;
      while (!stack.empty() && !augmented) {
        const auto u         = stack.back();
        bool       descended = false;
        while (cursor[u] < adj[u].size()) {
          const auto v = adj[u][cursor[u]++];
          bump(run.work.dfs_edges);
          const auto w = mr[v];
          if (w == kBipartiteUnmatched) {
            if (add(dist[u], 1) != shortest)
              continue;
            path_right.push_back(v);
            if (trace) {
              capture(trace, add(1, add(stack.size(), path_right.size())), options);
              phase.augmentations.push_back({stack, path_right});
            }
            for (std::size_t i = 0; i < stack.size(); ++i) {
              ml[stack[i]]      = path_right[i];
              mr[path_right[i]] = stack[i];
              bump(run.work.path_pairs_written);
            }
            bump(run.result.size);
            augmented = true;
            break;
          }
          if (dist[w] != infinity && dist[w] == add(dist[u], 1) && dist[w] < shortest) {
            stack.push_back(w);
            path_right.push_back(v);
            descended = true;
            run.work.max_path_left_vertices =
                std::max(run.work.max_path_left_vertices, stack.size());
            break;
          }
        }
        if (!augmented && !descended) {
          dist[u] = infinity;
          stack.pop_back();
          if (!path_right.empty())
            path_right.pop_back();
        }
      }
    }
    if (trace)
      trace->phases.push_back(std::move(phase));
  }
  return run;
}
} // namespace hopcroft_karp_detail
BipartiteMatchingResult
hopcroft_karp(std::size_t nl, std::size_t nr, const std::vector<HopcroftKarpEdge>& edges) {
  return hopcroft_karp_detail::execute(nl, nr, edges, {0, 0, 0}, nullptr).result;
}
HopcroftKarpRun hopcroft_karp_run(
    std::size_t                          nl,
    std::size_t                          nr,
    const std::vector<HopcroftKarpEdge>& edges,
    HopcroftKarpOptions                  options) {
  return hopcroft_karp_detail::execute(nl, nr, edges, options, nullptr);
}
HopcroftKarpTrace hopcroft_karp_trace(
    std::size_t                          nl,
    std::size_t                          nr,
    const std::vector<HopcroftKarpEdge>& edges,
    HopcroftKarpOptions                  options) {
  HopcroftKarpTrace t;
  t.run = hopcroft_karp_detail::execute(nl, nr, edges, options, &t);
  return t;
}
} // namespace datamunge::algorithms
