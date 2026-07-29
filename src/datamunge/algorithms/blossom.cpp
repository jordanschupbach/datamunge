#include <datamunge/algorithms/blossom.hpp>

#include <algorithm>
#include <numeric>
#include <queue>
#include <stdexcept>

namespace datamunge::algorithms {

GeneralMatchingResult blossom_maximum_matching(
    std::size_t vertex_count,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges) {
  const int n = static_cast<int>(vertex_count);
  std::vector<std::vector<int>> graph(vertex_count);
  for (const auto& [u, v] : edges) {
    if (u >= vertex_count || v >= vertex_count)
      throw std::invalid_argument("blossom_maximum_matching: edge endpoint out of range");
    if (u == v) continue;
    graph[u].push_back(static_cast<int>(v));
    graph[v].push_back(static_cast<int>(u));
  }

  std::vector<int> match(n, -1), parent(n), base(n);
  std::vector<bool> used(n), blossom(n);

  auto lca = [&](int a, int b) {
    std::vector<bool> seen(n, false);
    while (true) {
      a = base[a];
      seen[a] = true;
      if (match[a] == -1) break;
      a = parent[match[a]];
    }
    while (true) {
      b = base[b];
      if (seen[b]) return b;
      b = parent[match[b]];
    }
  };

  auto mark_path = [&](int v, int blossom_base, int child) {
    while (base[v] != blossom_base) {
      blossom[base[v]] = blossom[base[match[v]]] = true;
      parent[v] = child;
      child = match[v];
      v = parent[match[v]];
    }
  };

  auto find_path = [&](int root) {
    std::fill(used.begin(), used.end(), false);
    std::fill(parent.begin(), parent.end(), -1);
    std::iota(base.begin(), base.end(), 0);
    std::queue<int> queue;
    queue.push(root);
    used[root] = true;

    while (!queue.empty()) {
      const int v = queue.front();
      queue.pop();
      for (int u : graph[v]) {
        if (base[v] == base[u] || match[v] == u) continue;
        if (u == root || (match[u] != -1 && parent[match[u]] != -1)) {
          const int current_base = lca(v, u);
          std::fill(blossom.begin(), blossom.end(), false);
          mark_path(v, current_base, u);
          mark_path(u, current_base, v);
          for (int i = 0; i < n; ++i) {
            if (blossom[base[i]]) {
              base[i] = current_base;
              if (!used[i]) {
                used[i] = true;
                queue.push(i);
              }
            }
          }
        } else if (parent[u] == -1) {
          parent[u] = v;
          if (match[u] == -1) return u;
          u = match[u];
          used[u] = true;
          queue.push(u);
        }
      }
    }
    return -1;
  };

  for (int root = 0; root < n; ++root) {
    if (match[root] != -1) continue;
    int v = find_path(root);
    while (v != -1) {
      const int previous = parent[v];
      const int next = previous == -1 ? -1 : match[previous];
      match[v] = previous;
      if (previous != -1) match[previous] = v;
      v = next;
    }
  }

  GeneralMatchingResult result;
  result.mate.resize(vertex_count, kGeneralMatchingUnmatched);
  for (int v = 0; v < n; ++v) {
    if (match[v] != -1) result.mate[v] = static_cast<std::size_t>(match[v]);
  }
  result.size = static_cast<std::size_t>(
      std::count_if(match.begin(), match.end(), [](int mate) { return mate != -1; }) / 2);
  return result;
}

} // namespace datamunge::algorithms
