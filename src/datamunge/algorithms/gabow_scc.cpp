/// @file gabow_scc.cpp
/// @brief Shared validated iterative SCC kernel, owned primitive states and quotient summaries.
#include <algorithm>
#include <cstdint>
#include <datamunge/algorithms/gabow_scc.hpp>
#include <limits>
#include <map>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private checked counts, validation and actual DFS state.
namespace gabow_detail {
/// @brief Add retained-entry counts.
/// @param a First count.
/// @param b Second count.
/// @return Checked sum.
/// @throws std::length_error Count overflow.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > SIZE_MAX - a)
    throw std::length_error("Gabow count overflow");
  return a + b;
}
/// @brief Multiply retained-entry counts.
/// @param a First count.
/// @param b Second count.
/// @return Checked product.
/// @throws std::length_error Count overflow.
std::size_t mul(std::size_t a, std::size_t b) {
  if (a && b > SIZE_MAX / a)
    throw std::length_error("Gabow count overflow");
  return a * b;
}
/// @brief Validate the entire graph before allocating traversal state.
/// @param n Nonnegative vertex count.
/// @param adj Exactly n raw rows.
/// @return Checked raw edge count.
/// @throws std::invalid_argument Negative count, wrong row count or invalid endpoint.
/// @throws std::length_error Edge/transition count overflow.
std::size_t validate(int n, const SccAdjacency& adj) {
  if (n < 0 || adj.size() != static_cast<std::size_t>(n))
    throw std::invalid_argument("Gabow requires n exact adjacency rows");
  std::size_t edges = 0;
  for (const auto& row : adj) {
    edges = add(edges, row.size());
    for (const int v : row)
      if (v < 0 || v >= n)
        throw std::invalid_argument("Gabow endpoint out of range");
  }
  add(edges, mul(4, static_cast<std::size_t>(n)));
  return edges;
}
/// @brief Validate all supplied nonnegative labels.
/// @param labels Complete partition labels.
/// @return Nothing.
/// @throws std::invalid_argument A negative label denotes an incomplete partition.
void labels_valid(const std::vector<int>& labels) {
  for (int c : labels)
    if (c < 0)
      throw std::invalid_argument("SCC labels must be nonnegative");
}
/// @brief Private current kernel state, without retained history.
struct State {
  std::vector<int>           comp;      ///< Actual current component IDs; -1 until assigned.
  std::vector<int>           preorder;  ///< Actual monotonic1-based discovery numbers.
  std::vector<int>           path;      ///< Actual discovered-unassigned S, not only active calls.
  std::vector<int>           roots;     ///< Actual candidate root vertex IDs B.
  std::vector<GabowDfsFrame> stack;     ///< Actual active heap frames.
  int                        counter{}; ///< Number of discovered vertices.
  int                        ncomp{};   ///< Number of completely closed components.
  GabowWork                  work{};    ///< Actual operations and peaks.
};
/// @brief Count one actual transition and optionally own its current state.
/// @param state Current kernel state after the primitive.
/// @param trace Optional admitted history destination.
/// @param kind Transition name.
/// @param u Current or finished vertex.
/// @param v Edge target/assigned vertex, otherwise-1.
/// @return Nothing; captured snapshots own independent vectors.
/// @throws std::length_error Retained count overflow; conservative preflight rules it out for valid
/// calls.
/// @throws std::bad_alloc Allocation of owned or working storage fails.
void observe(State& state, GabowTrace* trace, const char* kind, int u, int v = -1) {
  ++state.work.event_count;
  if (!trace)
    return;
  std::size_t count = 1;
  for (const auto size :
       {state.comp.size(),
        state.preorder.size(),
        state.path.size(),
        state.roots.size(),
        state.stack.size()})
    count = add(count, size);
  trace->captured_entries = add(trace->captured_entries, count);
  trace->events.push_back(
      {kind,
       u,
       v,
       state.counter,
       state.ncomp,
       state.comp,
       state.preorder,
       state.path,
       state.roots,
       state.stack});
}
/// @brief Push a real heap DFS frame and update its actual peak.
/// @param state Private current state.
/// @param u Undiscovered vertex.
/// @return Nothing.
/// @throws std::bad_alloc Allocation of owned or working storage fails.
void push_frame(State& state, int u) {
  state.stack.push_back({u, 0});
  ++state.work.dfs_pushes;
  state.work.peak_dfs = std::max(state.work.peak_dfs, state.stack.size());
}
/// @brief Execute the same ordered kernel for ordinary/work/captured calls.
/// @param n Vertex count.
/// @param adj Raw directed adjacency rows.
/// @param capture Whether actual primitive snapshots are requested.
/// @param max_entries Optional whole retained-entry quota,0 disables only the cap.
/// @return Raw labels, actual work and optional owned history.
/// @throws std::invalid_argument Invalid graph shape/endpoints.
/// @throws std::length_error Count/capacity or optional capture quota exceeded.
/// @throws std::bad_alloc Allocation of owned or working storage fails.
GabowTrace run(int n, const SccAdjacency& adj, bool capture, std::size_t max_entries) {
  const auto edges = validate(n, adj);
  if (capture) {
    const auto count = gabow_scc_trace_entries(n, edges);
    if (max_entries && count > max_entries)
      throw std::length_error("Gabow trace admission quota");
  }
  GabowTrace out;
  if (capture) {
    out.adjacency        = adj;
    out.captured_entries = add(static_cast<std::size_t>(n), edges);
  }
  State state;
  state.comp.assign(static_cast<std::size_t>(n), -1);
  state.preorder.assign(static_cast<std::size_t>(n), 0);
  auto* history = capture ? &out : nullptr;
  for (int start = 0; start < n; ++start) {
    if (state.preorder[start] != 0)
      continue;
    push_frame(state, start);
    while (!state.stack.empty()) {
      auto& [u, ci] = state.stack.back();
      if (ci == 0) {
        state.preorder[u] = ++state.counter;
        ++state.work.visited_vertices;
        state.path.push_back(u);
        ++state.work.open_pushes;
        state.roots.push_back(u);
        ++state.work.root_pushes;
        state.work.peak_open  = std::max(state.work.peak_open, state.path.size());
        state.work.peak_roots = std::max(state.work.peak_roots, state.roots.size());
        observe(state, history, "enter", u);
      }
      if (ci < adj[u].size()) {
        const int v = adj[u][ci];
        ++ci;
        ++state.work.edges_examined;
        observe(state, history, "edge", u, v);
        if (state.preorder[v] == 0) {
          ++state.work.tree_edges;
          push_frame(state, v);
        } else if (state.comp[v] == -1) {
          ++state.work.open_edges;
          while (!state.roots.empty() && state.preorder[state.roots.back()] > state.preorder[v]) {
            state.roots.pop_back();
            ++state.work.root_pops;
            ++state.work.root_collapses;
            observe(state, history, "collapse", u, v);
          }
        } else
          ++state.work.closed_edges;
      } else {
        if (!state.roots.empty() && state.roots.back() == u) {
          state.roots.pop_back();
          ++state.work.root_pops;
          observe(state, history, "close-root", u);
          int w;
          do {
            w = state.path.back();
            state.path.pop_back();
            ++state.work.open_pops;
            state.comp[w] = state.ncomp;
            ++state.work.assignments;
            observe(state, history, "assign", u, w);
          } while (w != u);
          ++state.ncomp;
          ++state.work.component_closures;
        }
        const int finished = u;
        state.stack.pop_back();
        ++state.work.dfs_pops;
        observe(state, history, "finish", finished);
      }
    }
  }
  out.components = std::move(state.comp);
  out.work       = state.work;
  return out;
}
} // namespace gabow_detail
std::vector<int> gabow_scc(int n, const SccAdjacency& adj) {
  return gabow_detail::run(n, adj, false, 0).components;
}
GabowResult gabow_scc_work(int n, const SccAdjacency& adj) {
  auto out = gabow_detail::run(n, adj, false, 0);
  return std::move(static_cast<GabowResult&>(out));
}
std::size_t gabow_scc_trace_entries(int n, std::size_t edges) {
  if (n < 0 || (n == 0 && edges))
    throw std::invalid_argument("Gabow virtual graph dimensions");
  const auto vertices = static_cast<std::size_t>(n);
  const auto events   = gabow_detail::add(edges, gabow_detail::mul(4, vertices));
  if (vertices > std::vector<int>().max_size() ||
      vertices > std::vector<GabowDfsFrame>().max_size() || vertices > SccAdjacency().max_size() ||
      events > std::vector<GabowEvent>().max_size())
    throw std::length_error("Gabow trace vector capacity");
  return gabow_detail::add(
      gabow_detail::add(vertices, edges),
      gabow_detail::mul(events, gabow_detail::add(1, gabow_detail::mul(5, vertices))));
}
GabowTrace gabow_scc_trace(int n, const SccAdjacency& adj, std::size_t max_entries) {
  return gabow_detail::run(n, adj, true, max_entries);
}
bool scc_same_partition(const std::vector<int>& a, const std::vector<int>& b) {
  gabow_detail::labels_valid(a);
  gabow_detail::labels_valid(b);
  if (a.size() != b.size())
    return false;
  std::map<int, int> ab, ba;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const auto [ia, inserted_a] = ab.emplace(a[i], b[i]);
    if (!inserted_a && ia->second != b[i])
      return false;
    const auto [ib, inserted_b] = ba.emplace(b[i], a[i]);
    if (!inserted_b && ib->second != a[i])
      return false;
  }
  return true;
}
SccSummary scc_summary(const SccAdjacency& adj, const std::vector<int>& components) {
  if (adj.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("SCC vertex IDs exceed int");
  gabow_detail::validate(static_cast<int>(adj.size()), adj);
  gabow_detail::labels_valid(components);
  if (components.size() != adj.size())
    throw std::invalid_argument("SCC labels must match vertex count");
  SccSummary         out;
  std::map<int, int> index;
  for (int c : components)
    index.emplace(c, 0);
  int next = 0;
  for (auto& [label, id] : index) {
    id = next++;
    out.groups.push_back({label, {}});
  }
  out.adjacency.resize(out.groups.size());
  for (std::size_t i = 0; i < components.size(); ++i)
    out.groups[static_cast<std::size_t>(index.at(components[i]))].vertices.push_back(
        static_cast<int>(i));
  for (std::size_t u = 0; u < adj.size(); ++u)
    for (int v : adj[u]) {
      if (components[u] == components[v])
        ++out.internal_edges;
      else {
        ++out.cross_edges;
        out.adjacency[static_cast<std::size_t>(index.at(components[u]))].push_back(
            index.at(components[v]));
        if (components[u] <= components[v])
          out.reverse_topological = false;
      }
    }
  for (auto& row : out.adjacency) {
    std::sort(row.begin(), row.end());
    row.erase(std::unique(row.begin(), row.end()), row.end());
    out.unique_cross_edges = gabow_detail::add(out.unique_cross_edges, row.size());
  }
  return out;
}
SccAdjacency gabow_scc_example_graph() {
  return {{1}, {0, 2}, {3}, {4}, {2, 5}, {6}, {5, 7}, {}};
}
} // namespace datamunge::algorithms
