/// @file subgraph_isomorphism.cpp
/// @brief One normalized actual degree/assigned-edge backtracker with safe complete input checks.
#include <algorithm>
#include <datamunge/algorithms/subgraph_isomorphism.hpp>
#include <limits>
#include <stdexcept>
#include <utility>
namespace datamunge::algorithms {
/// @brief Translation-unit-private checked producing kernels; no shared mutable state.
namespace {
/// @brief Validate complete adjacency shape/ids independently of matching outcome.
/// @param n Nonnegative vertex count, checked here.
/// @param adjacency Complete supplied rows.
void validate(int n, const std::vector<std::vector<int>>& adjacency) {
  if (n < 0 || adjacency.size() != static_cast<std::size_t>(n))
    throw std::invalid_argument("subgraph: complete rows");
  for (const auto& row : adjacency)
    for (int v : row)
      if (v < 0 || v >= n)
        throw std::invalid_argument("subgraph: endpoint");
}
/// @brief Normalize actual symmetric adjacency and degrees.
/// @param adjacency Complete validated lists.
/// @param degrees Mutable actual degree destination.
/// @return Actual normalized zero/one matrix.
std::vector<std::vector<char>> matrix(
    const std::vector<std::vector<int>>& adjacency,
    std::vector<int>&                    degrees) {
  const auto                     n = adjacency.size();
  std::vector<std::vector<char>> m(n, std::vector<char>(n));
  for (std::size_t u = 0; u < n; ++u)
    for (int v : adjacency[u])
      if (static_cast<std::size_t>(v) != u)
        m[u][v] = m[v][u] = 1;
  degrees.assign(n, 0);
  for (std::size_t u = 0; u < n; ++u)
    for (char edge : m[u])
      degrees[u] += edge;
  return m;
}
/// @brief Invocation-local normalized search state and actual history accounting.
struct State {
  const SubgraphOptions&                o;       ///< Borrowed policies for this call.
  SubgraphRun&                          r;       ///< Borrowed actual result/counter state.
  const std::vector<std::vector<char>>& p;       ///< Borrowed actual normalized pattern matrix.
  const std::vector<std::vector<char>>& g;       ///< Borrowed actual normalized target matrix.
  std::vector<int>                      mapping; ///< Actual current assignment, -1 for unassigned.
  std::vector<char>                     used;    ///< Actual target-use marks.
  bool                                  capture; ///< Retain actual decisions.
  std::size_t retained{}; ///< Actual retained record/mapping/matrix scalar units.

  /// @brief Bound actual selected counters before overflow.
  /// @param counter Actual selected event accumulator.
  void count(std::size_t& counter) {
    if (counter == std::numeric_limits<std::size_t>::max())
      throw std::length_error("subgraph: counter capacity");
    ++counter;
  }
  /// @brief Capture the actual current mapping after a producing event.
  /// @param depth Current pattern position.
  /// @param target Actual candidate, -1 at success.
  /// @param action Actual branch label.
  void record(std::size_t depth, int target, const std::string& action) {
    if (capture) {
      const auto n = mapping.size(), limit = std::numeric_limits<std::size_t>::max();
      if (n == limit || retained > limit - n - 1)
        throw std::length_error("subgraph: history capacity");
      retained += n + 1;
      if (o.max_history_values && retained > o.max_history_values)
        throw std::length_error("subgraph: history");
      r.steps.push_back({depth, target, action, mapping});
    }
  }
};
/// @brief Recurse actual increasing-id candidate decisions.
/// @param depth Actual next pattern position.
/// @param state Invocation-local state.
/// @return First complete successful assignment was reached.
bool search(std::size_t depth, State& state) {
  auto& r = state.r;
  if (depth == state.mapping.size()) {
    state.record(depth, -1, "success");
    return true;
  }
  for (std::size_t v = 0; v < state.used.size(); ++v) {
    if (state.o.max_candidate_trials && r.candidate_trials >= state.o.max_candidate_trials)
      throw std::length_error("subgraph: candidate policy");
    state.count(r.candidate_trials);
    if (state.used[v]) {
      state.count(r.used_prunes);
      state.record(depth, static_cast<int>(v), "used-prune");
      continue;
    }
    if (r.target_degrees[v] < r.pattern_degrees[depth]) {
      state.count(r.degree_prunes);
      state.record(depth, static_cast<int>(v), "degree-prune");
      continue;
    }
    bool ok = true;
    for (std::size_t w = 0; w < depth; ++w) {
      state.count(r.assigned_pair_visits);
      if (state.p[depth][w]) {
        state.count(r.required_edge_tests);
        if (!state.g[v][state.mapping[w]]) {
          ok = false;
          break;
        }
      }
    }
    if (!ok) {
      state.count(r.adjacency_prunes);
      state.record(depth, static_cast<int>(v), "edge-prune");
      continue;
    }
    state.mapping[depth] = static_cast<int>(v);
    state.used[v]        = 1;
    state.count(r.assignments);
    state.record(depth, static_cast<int>(v), "assign");
    if (search(depth + 1, state))
      return true;
    state.used[v]        = 0;
    state.mapping[depth] = -1;
    state.count(r.backtracks);
    state.record(depth, static_cast<int>(v), "backtrack");
  }
  return false;
}
/// @brief Validate, normalize and execute one actual search.
/// @param np Pattern count.
/// @param p Pattern rows.
/// @param ng Target count.
/// @param g Target rows.
/// @param o Policies.
/// @param capture Retain actual normalized matrices/events.
/// @return Owned actual matching result.
SubgraphRun run(
    int                                  np,
    const std::vector<std::vector<int>>& p,
    int                                  ng,
    const std::vector<std::vector<int>>& g,
    const SubgraphOptions&               o,
    bool                                 capture) {
  validate(np, p);
  validate(ng, g);
  const auto a = static_cast<std::size_t>(np), b = static_cast<std::size_t>(ng),
             limit = std::numeric_limits<std::size_t>::max();
  if (a > 1024 || (o.max_pattern_vertices && a > o.max_pattern_vertices) || (a && a > limit / a) ||
      (b && b > limit / b) || a * a > limit - b * b)
    throw std::length_error("subgraph: complete geometry/depth");
  const auto cells = a * a + b * b;
  if (o.max_matrix_cells && cells > o.max_matrix_cells)
    throw std::length_error("subgraph: matrix policy");
  SubgraphRun r;
  r.target_matched.assign(b, false);
  if (np > ng)
    return r;
  auto  pm = matrix(p, r.pattern_degrees), gm = matrix(g, r.target_degrees);
  State state{o, r, pm, gm, std::vector<int>(a, -1), std::vector<char>(b), capture};
  if (capture) {
    if (cells > limit - a || cells + a > limit - b)
      throw std::length_error("subgraph: capture capacity");
    state.retained = cells + a + b;
    if (o.max_history_values && state.retained > o.max_history_values)
      throw std::length_error("subgraph: capture matrices");
    r.pattern_matrix = pm;
    r.target_matrix  = gm;
  }
  r.match.found = search(0, state);
  if (r.match.found) {
    r.match.mapping = std::move(state.mapping);
    for (int v : r.match.mapping)
      r.target_matched[v] = true;
  }
  return r;
}
} // namespace
SubgraphRun subgraph_isomorphism_run(
    int                                  np,
    const std::vector<std::vector<int>>& p,
    int                                  ng,
    const std::vector<std::vector<int>>& g,
    const SubgraphOptions&               o) {
  return run(np, p, ng, g, o, false);
}
SubgraphRun subgraph_isomorphism_trace(
    int                                  np,
    const std::vector<std::vector<int>>& p,
    int                                  ng,
    const std::vector<std::vector<int>>& g,
    const SubgraphOptions&               o) {
  return run(np, p, ng, g, o, true);
}
SubgraphMatch subgraph_isomorphism(
    int                                  np,
    const std::vector<std::vector<int>>& p,
    int                                  ng,
    const std::vector<std::vector<int>>& g) {
  return subgraph_isomorphism_run(np, p, ng, g).match;
}
std::vector<std::vector<int>> subgraph_cycle_graph(std::size_t n) {
  if (n < 3)
    throw std::invalid_argument("subgraph: cycle order");
  if (n > 100000)
    throw std::length_error("subgraph: cycle count");
  std::vector<std::vector<int>> g(n);
  for (std::size_t i = 0; i < n; ++i)
    g[i] = {static_cast<int>((i + n - 1) % n), static_cast<int>((i + 1) % n)};
  return g;
}
} // namespace datamunge::algorithms
