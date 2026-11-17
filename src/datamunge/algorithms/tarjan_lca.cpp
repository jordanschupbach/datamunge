/// @file tarjan_lca.cpp
/// @brief Actual checked iterative offline LCA and invocation-owned union-find.
#include <datamunge/algorithms/tarjan_lca.hpp>
#include <limits>
#include <numeric>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private actual union-find and tree-validation kernels.
namespace {
/// @brief Count one representable actual event.
void increment(std::size_t& n) {
  if (n == std::numeric_limits<std::size_t>::max())
    throw std::length_error("LCA counter");
  ++n;
}
/// @brief Invocation-owned rank/compression forest with producing counters.
struct DisjointSet {
  std::vector<std::size_t> parent; ///< Actual parent per node.
  std::vector<std::size_t> rank;   ///< Actual rank at representatives.
  TarjanLCARun&            run;    ///< Borrowed current invocation counters.

  /// @brief Initialize actual singleton sets.
  DisjointSet(std::size_t n, TarjanLCARun& r) : parent(n), rank(n), run(r) {
    std::iota(parent.begin(), parent.end(), 0);
  }
  /// @brief Find an actual representative and compress its path.
  std::size_t find(std::size_t x) {
    increment(run.find_calls);
    auto root = x;
    while (parent[root] != root) {
      increment(run.parent_hops);
      root = parent[root];
    }
    while (parent[x] != root) {
      auto next = parent[x];
      parent[x] = root;
      x         = next;
    }
    return root;
  }
  /// @brief Merge actual child/parent sets by rank and return the resulting representative.
  std::size_t unite(std::size_t a, std::size_t b) {
    auto ra = find(a), rb = find(b);
    if (ra == rb)
      return ra;
    if (rank[ra] < rank[rb])
      std::swap(ra, rb);
    parent[rb] = ra;
    if (rank[ra] == rank[rb])
      ++rank[ra];
    increment(run.unions);
    return ra;
  }
};
/// @brief Validate all tree geometry then execute actual finish, query and union events.
TarjanLCARun execute(
    std::size_t                                             n,
    std::size_t                                             root,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    const std::vector<std::pair<std::size_t, std::size_t>>& queries,
    const TarjanLCAOptions&                                 o,
    bool                                                    capture) {
  if (!n || root >= n || edges.size() != n - 1)
    throw std::invalid_argument("LCA tree/root count");
  if ((o.max_vertices && n > o.max_vertices) || (o.max_queries && queries.size() > o.max_queries) ||
      n == std::numeric_limits<std::size_t>::max())
    throw std::length_error("LCA geometry");
  for (auto [u, v] : edges)
    if (u >= n || v >= n || u == v)
      throw std::invalid_argument("LCA tree endpoints/loop");
  for (auto [u, v] : queries)
    if (u >= n || v >= n)
      throw std::invalid_argument("LCA query endpoint");
  std::vector<std::vector<std::size_t>> adj(n);
  for (auto [u, v] : edges) {
    adj[u].push_back(v);
    adj[v].push_back(u);
  }
  TarjanLCARun                                     r;
  std::vector<char>                                seen(n);
  std::vector<std::pair<std::size_t, std::size_t>> validation{{root, n}};
  seen[root] = 1;
  while (!validation.empty()) {
    auto [u, p] = validation.back();
    validation.pop_back();
    ++r.preflight_nodes;
    for (auto v : adj[u]) {
      increment(r.preflight_edges);
      if (v == p)
        continue;
      if (seen[v])
        throw std::invalid_argument("LCA cycle/duplicate");
      seen[v] = 1;
      validation.emplace_back(v, u);
    }
  }
  if (r.preflight_nodes != n)
    throw std::invalid_argument("LCA disconnected tree");
  std::vector<std::vector<std::pair<std::size_t, std::size_t>>> at(n);
  for (std::size_t k = 0; k < queries.size(); ++k) {
    auto [a, b] = queries[k];
    at[a].push_back({b, k});
    at[b].push_back({a, k});
  }
  r.lca.resize(queries.size());
  std::vector<char>        black(n);
  std::vector<std::size_t> ancestor(n);
  std::iota(ancestor.begin(), ancestor.end(), 0);
  DisjointSet              dsu(n, r);
  std::vector<std::size_t> representatives(n);
  std::iota(representatives.begin(), representatives.end(), 0);
  /// @brief Capture a real event using an already computed representative, without extra find work.
  auto record = [&](const char* action,
                    std::size_t u,
                    std::size_t other,
                    std::size_t query,
                    std::size_t rep) {
    if (!capture)
      return;
    if (o.max_records && r.steps.size() >= o.max_records)
      throw std::length_error("LCA history");
    r.steps.push_back({action, u, other, query, rep, ancestor[rep]});
  };
  /// @brief One actual iterative producer DFS frame.
  struct Frame {
    std::size_t node;
    std::size_t parent;
    std::size_t next;
  };
  std::vector<Frame> stack{{root, n, 0}};
  r.entered.push_back(root);
  record("enter", root, n, queries.size(), root);
  while (!stack.empty()) {
    auto& f = stack.back();
    if (f.next < adj[f.node].size()) {
      auto v = adj[f.node][f.next++];
      increment(r.edge_visits);
      if (v == f.parent)
        continue;
      auto u      = f.node;
      ancestor[v] = v;
      r.entered.push_back(v);
      record("enter", v, n, queries.size(), v);
      stack.push_back({v, u, 0});
    } else {
      auto u = f.node, p = f.parent;
      black[u] = 1;
      r.finished.push_back(u);
      record("black", u, n, queries.size(), representatives[u]);
      for (auto [w, k] : at[u]) {
        increment(r.query_visits);
        if (black[w]) {
          auto rep = dsu.find(w);
          r.lca[k] = ancestor[rep];
          increment(r.answer_writes);
          record("answer", u, w, k, rep);
        }
      }
      stack.pop_back();
      if (p != n) {
        auto rep           = dsu.unite(p, u);
        ancestor[rep]      = p;
        representatives[p] = rep;
        record("union", p, u, queries.size(), rep);
      }
    }
  }
  return r;
}
} // namespace
TarjanLCARun tarjan_lca_run(
    std::size_t                                             n,
    std::size_t                                             root,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    const std::vector<std::pair<std::size_t, std::size_t>>& queries,
    const TarjanLCAOptions&                                 options) {
  return execute(n, root, edges, queries, options, false);
}
TarjanLCARun tarjan_lca_trace(
    std::size_t                                             n,
    std::size_t                                             root,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    const std::vector<std::pair<std::size_t, std::size_t>>& queries,
    const TarjanLCAOptions&                                 options) {
  return execute(n, root, edges, queries, options, true);
}
std::vector<std::size_t> tarjan_offline_lca(
    std::size_t                                             n,
    std::size_t                                             root,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    const std::vector<std::pair<std::size_t, std::size_t>>& queries) {
  return tarjan_lca_run(n, root, edges, queries).lca;
}
} // namespace datamunge::algorithms
