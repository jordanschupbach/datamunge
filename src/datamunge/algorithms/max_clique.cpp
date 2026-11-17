/// @file max_clique.cpp
/// @brief Canonical max clique API, producing histories and checked native contracts.

#include <algorithm>
#include <cstdint>
#include <datamunge/algorithms/max_clique.hpp>
#include <numeric>
#include <random>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Implementation-only helpers for canonical max clique production and capture.
namespace {
/// @brief Advance checked operation or capture geometry.
/// @param a Existing count.
/// @param b Additional count.
/// @return Exact sum.
/// @throws std::length_error On size_t overflow.
std::size_t add(std::size_t a, std::size_t b = 1) {
  if (b > SIZE_MAX - a)
    throw std::length_error("maximum-clique count geometry");
  return a + b;
}
/// @brief Complete validated canonical matrix and original logical input count.
struct Model {
  std::vector<std::vector<char>> matrix; ///< Actual simple symmetric dense matrix.
  std::size_t entries{0};                ///< Original N vertex slots plus every listed neighbour.
  std::size_t edges{0};                  ///< Actual distinct non-self undirected edge count.
};
/// @brief Validate every original list/id and construct the retained symmetric-union graph.
/// @param n Original vertex count.
/// @param adj Original lists, including possible duplicate/self/asymmetric entries.
/// @return Actual canonical matrix and complete counts.
/// @throws std::invalid_argument On malformed original input before indexing.
/// @throws std::length_error On complete input/dense geometry overflow.
Model model(int n, const std::vector<std::vector<int>>& adj) {
  if (n < 0 || adj.size() != static_cast<std::size_t>(n))
    throw std::invalid_argument("maximum-clique row count");
  Model out;
  out.entries = n;
  for (auto& row : adj) {
    out.entries = add(out.entries, row.size());
    for (int v : row)
      if (v < 0 || v >= n)
        throw std::invalid_argument("maximum-clique neighbour");
  }
  auto size = static_cast<std::size_t>(n);
  if (size && size > SIZE_MAX / size)
    throw std::length_error("maximum-clique matrix geometry");
  out.matrix.assign(size, std::vector<char>(size));
  for (int u = 0; u < n; ++u)
    for (int v : adj[u])
      if (u != v && !out.matrix[u][v]) {
        out.matrix[u][v] = out.matrix[v][u] = 1;
        ++out.edges;
      }
  return out;
}
/// @brief Complete actual logical capture admission.
/// @param trace Optional owned capture target.
/// @param limit Optional cap, zero off.
/// @param count Actual next contribution.
/// @throws std::length_error On overflow or exceeded cap.
void admit(MaxCliqueTrace* trace, std::size_t limit, std::size_t count = 1) {
  if (!trace)
    return;
  auto next = add(trace->captured_entries, count);
  if (limit && next > limit)
    throw std::length_error("maximum-clique complete capture");
  trace->captured_entries = next;
}
/// @brief One actual DFS frame, explicitly retaining the legacy recursive continuation state.
struct Frame {
  std::vector<int> candidates; ///< Actual entry candidate sequence.
  std::vector<int> order;      ///< Actual greedy class-grouped sequence.
  std::vector<int> colours;    ///< Actual paired one-based class bounds.
  std::size_t      next{0};    ///< Number of reverse-order vertices still eligible for processing.
  std::size_t      id{0};      ///< Actual entered frame id.
  bool             entered{false}; ///< Whether work admission/colouring occurred.
  bool owns_vertex{false}; ///< Whether popping this frame must remove its selected active vertex.
};
/// @brief Canonical greedy colouring with actual class-member conflict capture.
/// @param frame Actual entered frame whose order/bounds are produced.
/// @param matrix Canonical simple graph.
/// @param work Actual operation counts.
/// @param trace Optional owned capture.
/// @param limit Optional complete capture cap.
/// @throws std::length_error On count/admission overflow.
void colour(
    Frame&                                frame,
    const std::vector<std::vector<char>>& matrix,
    MaxCliqueWork&                        work,
    MaxCliqueTrace*                       trace,
    std::size_t                           limit) {
  std::vector<std::vector<int>> classes;
  for (int vertex : frame.candidates) {
    std::size_t c = 0;
    for (; c < classes.size(); ++c) {
      bool conflict = false;
      for (int against : classes[c]) {
        conflict           = matrix[vertex][against];
        work.colour_checks = add(work.colour_checks);
        if (trace) {
          admit(trace, limit);
          trace->colour_checks.push_back({frame.id, vertex, against, c + 1, conflict});
        }
        if (conflict)
          break;
      }
      if (!conflict)
        break;
    }
    if (c == classes.size())
      classes.emplace_back();
    classes[c].push_back(vertex);
  }
  for (std::size_t c = 0; c < classes.size(); ++c)
    for (int vertex : classes[c]) {
      frame.order.push_back(vertex);
      frame.colours.push_back(static_cast<int>(c + 1));
    }
  frame.next = frame.order.size();
}
/// @brief One iterative canonical colouring-bound search preserving original DFS/tie order.
/// @param n Original vertex count.
/// @param adj Fully checked original lists, retained in capture.
/// @param graph Canonical matrix/counts.
/// @param max_nodes Optional actual entered-frame cap.
/// @param trace Optional owned actual history.
/// @param limit Optional complete logical cap.
/// @return Actual best witness, completion/bound and work.
/// @throws std::length_error On complete actual capture or operation geometry overflow.
MaxCliqueAnalysis produce(
    int                                  n,
    const std::vector<std::vector<int>>& adj,
    const Model&                         graph,
    std::size_t                          max_nodes,
    MaxCliqueTrace*                      trace,
    std::size_t                          limit) {
  MaxCliqueAnalysis out;
  out.edges = graph.edges;
  if (trace) {
    admit(trace, limit, graph.entries);
    trace->n     = n;
    trace->input = adj;
  }
  Frame root;
  root.candidates.resize(n);
  std::iota(root.candidates.begin(), root.candidates.end(), 0);
  std::vector<Frame> stack;
  stack.push_back(std::move(root));
  std::vector<int> active, best;
  /// @brief Pop actual frame and restore its owning chosen vertex.
  auto pop = [&]() {
    bool owns = stack.back().owns_vertex;
    stack.pop_back();
    if (owns)
      active.pop_back();
  };
  /// @brief Capture one actual already-decided search event before owned-record insertion.
  /// @param kind Actual event kind.
  /// @param frame Actual owning entered frame.
  /// @param vertex Actual branch/cutoff vertex, or -1 at leaf.
  /// @param bound Actual local bound.
  /// @param before Actual best size before event.
  auto event =
      [&](unsigned kind, const Frame& frame, int vertex, std::size_t bound, std::size_t before) {
        if (trace) {
          admit(trace, limit);
          trace->decisions.push_back(
              {kind, frame.id, vertex, active.size(), bound, before, best.size()});
        }
      };
  while (!stack.empty()) {
    auto& frame = stack.back();
    if (!frame.entered) {
      if (max_nodes && out.work.nodes >= max_nodes) {
        out.complete = false;
        break;
      }
      frame.entered      = true;
      frame.id           = out.work.nodes;
      out.work.nodes     = add(out.work.nodes);
      out.work.max_depth = std::max(out.work.max_depth, stack.size());
      auto before        = best.size();
      if (!frame.candidates.empty())
        colour(frame, graph.matrix, out.work, trace, limit);
      if (frame.id == 0)
        out.upper_bound = frame.colours.empty() ? 0 : frame.colours.back();
      if (trace) {
        auto count =
            add(add(add(add(1, active.size()), frame.candidates.size()), frame.order.size()),
                frame.colours.size());
        admit(trace, limit, count);
        trace->nodes.push_back(
            {frame.id, active, frame.candidates, frame.order, frame.colours, before});
      }
      if (frame.candidates.empty()) {
        if (active.size() > best.size()) {
          best                  = active;
          out.work.improvements = add(out.work.improvements);
        }
        event(0, frame, -1, active.size(), before);
        pop();
        continue;
      }
    }
    if (!frame.next) {
      pop();
      continue;
    }
    auto i      = frame.next - 1;
    auto bound  = active.size() + static_cast<std::size_t>(frame.colours[i]);
    int  vertex = frame.order[i];
    if (bound <= best.size()) {
      out.work.prunes = add(out.work.prunes);
      event(2, frame, vertex, bound, best.size());
      pop();
      continue;
    }
    event(1, frame, vertex, bound, best.size());
    --frame.next;
    out.work.branches = add(out.work.branches);
    Frame child;
    child.owns_vertex = true;
    for (std::size_t j = 0; j < i; ++j) {
      out.work.candidate_checks = add(out.work.candidate_checks);
      if (graph.matrix[vertex][frame.order[j]])
        child.candidates.push_back(frame.order[j]);
    }
    active.push_back(vertex);
    stack.push_back(std::move(child));
  }
  std::sort(best.begin(), best.end());
  out.clique = std::move(best);
  if (out.complete)
    out.upper_bound = out.clique.size();
  return out;
}
} // namespace
std::vector<int> max_clique_dyn(int n, const std::vector<std::vector<int>>& adj) {
  return max_clique_analyze(n, adj).clique;
}
MaxCliqueAnalysis
max_clique_analyze(int n, const std::vector<std::vector<int>>& adj, std::size_t max_nodes) {
  auto graph = model(n, adj);
  return produce(n, adj, graph, max_nodes, nullptr, 0);
}
MaxCliqueTrace max_clique_trace(
    int                                  n,
    const std::vector<std::vector<int>>& adj,
    std::size_t                          max_nodes,
    std::size_t                          limit) {
  auto           graph = model(n, adj);
  MaxCliqueTrace out;
  out.analysis = produce(n, adj, graph, max_nodes, &out, limit);
  return out;
}
std::vector<std::vector<int>> max_clique_normalize(
    int                                  n,
    const std::vector<std::vector<int>>& adj) {
  auto                          graph = model(n, adj);
  std::vector<std::vector<int>> out(n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      if (graph.matrix[i][j])
        out[i].push_back(j);
  return out;
}
std::vector<std::vector<int>> max_clique_demo_graph() {
  return {{1, 2, 3}, {0, 2, 3}, {0, 1, 3, 4}, {0, 1, 2}, {2, 5, 6}, {4, 6}, {4, 5}, {6}};
}
std::vector<MaxCliqueDensityRow> max_clique_density_study() {
  std::mt19937_64                  rng(31);
  std::vector<MaxCliqueDensityRow> out;
  for (int pct = 10; pct <= 90; pct += 10) {
    MaxCliqueDensityRow row;
    row.percent = pct;
    row.graph.resize(40);
    for (int u = 0; u < 40; ++u)
      for (int v = u + 1; v < 40; ++v)
        if (static_cast<int>(rng() % 100) < pct) {
          row.graph[u].push_back(v);
          row.graph[v].push_back(u);
        }
    row.analysis = max_clique_analyze(40, row.graph);
    out.push_back(std::move(row));
  }
  return out;
}
} // namespace datamunge::algorithms
