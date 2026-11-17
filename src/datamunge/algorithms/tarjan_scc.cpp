/// @file tarjan_scc.cpp
/// @brief Actual checked explicit-stack Tarjan producer and low-link state capture.
#include <algorithm>
#include <datamunge/algorithms/tarjan_scc.hpp>
#include <limits>
#include <stdexcept>
#include <utility>
namespace datamunge::algorithms {
/// @brief Private graph validation and invocation kernel.
namespace {
/// @brief Validate the complete directed graph and execute actual low-link traversal.
TarjanSCCRun
execute(int n, const std::vector<std::vector<int>>& adj, const TarjanSCCOptions& o, bool capture) {
  if (n < 0 || adj.size() != std::size_t(n))
    throw std::invalid_argument("Tarjan SCC rows");
  if (o.max_vertices && std::size_t(n) > o.max_vertices)
    throw std::length_error("Tarjan SCC vertices");
  std::size_t edges = 0;
  for (const auto& row : adj) {
    if (row.size() > std::numeric_limits<std::size_t>::max() - edges)
      throw std::length_error("Tarjan SCC edge capacity");
    edges += row.size();
    for (int v : row)
      if (v < 0 || v >= n)
        throw std::invalid_argument("Tarjan SCC neighbor");
  }
  if (o.max_edges && edges > o.max_edges)
    throw std::length_error("Tarjan SCC edges");
  TarjanSCCRun r;
  r.discovery.assign(n, -1);
  r.low.assign(n, 0);
  r.components.assign(n, -1);
  std::vector<char> onstack(n);
  std::vector<int>  stack;
  int               timer = 0, component = 0;
  /// @brief Capture actual current vertex state after one real event.
  auto record = [&](const char* action, int u, int other = -1) {
    if (!capture)
      return;
    if (o.max_records && r.steps.size() >= o.max_records)
      throw std::length_error("Tarjan SCC history");
    r.steps.push_back({action, u, other, r.discovery[u], r.low[u], r.components[u], stack.size()});
  };
  /// @brief Assign the actual next discovery number and push one active vertex.
  auto discover = [&](int u) {
    r.discovery[u] = r.low[u] = timer++;
    ++r.discoveries;
    stack.push_back(u);
    onstack[u] = 1;
    record("discover", u);
  };
  for (int root = 0; root < n; ++root) {
    if (r.discovery[root] != -1)
      continue;
    std::vector<std::pair<int, std::size_t>> call{{root, 0}};
    discover(root);
    r.peak_frames = std::max(r.peak_frames, call.size());
    while (!call.empty()) {
      int   u = call.back().first;
      auto& i = call.back().second;
      if (i < adj[u].size()) {
        int v = adj[u][i++];
        ++r.edge_visits;
        record("edge", u, v);
        if (r.discovery[v] == -1) {
          discover(v);
          call.push_back({v, 0});
          r.peak_frames = std::max(r.peak_frames, call.size());
        } else if (onstack[v]) {
          r.low[u] = std::min(r.low[u], r.discovery[v]);
          ++r.active_edge_updates;
          record("back-edge", u, v);
        }
      } else {
        if (r.low[u] == r.discovery[u]) {
          while (true) {
            int w = stack.back();
            stack.pop_back();
            onstack[w]      = 0;
            r.components[w] = component;
            record("pop", w);
            if (w == u)
              break;
          }
          ++component;
          ++r.component_count;
        }
        record("finish", u);
        call.pop_back();
        if (!call.empty()) {
          int parent    = call.back().first;
          r.low[parent] = std::min(r.low[parent], r.low[u]);
          ++r.child_propagations;
          record("propagate", parent, u);
        }
      }
    }
  }
  return r;
}
} // namespace
TarjanSCCRun
tarjan_scc_run(int n, const std::vector<std::vector<int>>& adj, const TarjanSCCOptions& options) {
  return execute(n, adj, options, false);
}
TarjanSCCRun
tarjan_scc_trace(int n, const std::vector<std::vector<int>>& adj, const TarjanSCCOptions& options) {
  return execute(n, adj, options, true);
}
std::vector<int> tarjan_scc(int n, const std::vector<std::vector<int>>& adj) {
  return tarjan_scc_run(n, adj).components;
}
} // namespace datamunge::algorithms
