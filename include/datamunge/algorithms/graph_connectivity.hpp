#pragma once

// Graph structure algorithms:
//   - kosaraju_scc / tarjan_scc : strongly connected components of a directed
//     graph, by two different classic methods (they agree on the partition)
//   - bron_kerbosch             : all maximal cliques of an undirected graph
//
// Graphs are given as adjacency lists over vertices 0..n-1.

#include <datamunge/algorithms/bron_kerbosch.hpp>
#include <algorithm>
#include <vector>

namespace datamunge::algorithms {

namespace detail {

// Iterative DFS pushing vertices onto `order` in finishing order.
inline void kosaraju_dfs1(int u, const std::vector<std::vector<int>>& adj, std::vector<char>& vis,
                          std::vector<int>& order) {
    std::vector<std::pair<int, std::size_t>> stack{{u, 0}};
    vis[u] = 1;
    while (!stack.empty()) {
        auto& [v, i] = stack.back();
        if (i < adj[v].size()) {
            const int w = adj[v][i++];
            if (!vis[w]) { vis[w] = 1; stack.push_back({w, 0}); }
        } else {
            order.push_back(v);
            stack.pop_back();
        }
    }
}

inline void kosaraju_dfs2(int u, const std::vector<std::vector<int>>& radj, std::vector<int>& comp, int c) {
    std::vector<int> stack{u};
    comp[u] = c;
    while (!stack.empty()) {
        const int v = stack.back();
        stack.pop_back();
        for (int w : radj[v])
            if (comp[w] == -1) { comp[w] = c; stack.push_back(w); }
    }
}

} // namespace detail

// Kosaraju's algorithm: DFS to get a finishing order, then DFS the transpose in
// reverse finishing order; each transpose-DFS tree is one SCC. Returns a
// component id (0..k-1) for each vertex.
inline std::vector<int> kosaraju_scc(int n, const std::vector<std::vector<int>>& adj) {
    std::vector<char> vis(n, 0);
    std::vector<int>  order;
    order.reserve(n);
    for (int u = 0; u < n; ++u)
        if (!vis[u]) detail::kosaraju_dfs1(u, adj, vis, order);

    std::vector<std::vector<int>> radj(n);
    for (int u = 0; u < n; ++u)
        for (int v : adj[u]) radj[v].push_back(u);

    std::vector<int> comp(n, -1);
    int              c = 0;
    for (int i = n - 1; i >= 0; --i) {
        const int u = order[i];
        if (comp[u] == -1) detail::kosaraju_dfs2(u, radj, comp, c++);
    }
    return comp;
}

// Tarjan's algorithm: a single DFS tracking discovery times and low-links; a
// vertex whose low-link equals its discovery time is the root of an SCC, which is
// popped off a stack. Returns a component id per vertex.
inline std::vector<int> tarjan_scc(int n, const std::vector<std::vector<int>>& adj) {
    std::vector<int>  disc(n, -1), low(n, 0), comp(n, -1);
    std::vector<char> onstk(n, 0);
    std::vector<int>  stk;
    int               timer = 0, c = 0;

    // Explicit-stack DFS to avoid recursion depth limits.
    for (int s = 0; s < n; ++s) {
        if (disc[s] != -1) continue;
        std::vector<std::pair<int, std::size_t>> call{{s, 0}};
        disc[s] = low[s] = timer++;
        stk.push_back(s);
        onstk[s] = 1;
        while (!call.empty()) {
            auto& [u, i] = call.back();
            if (i < adj[u].size()) {
                const int v = adj[u][i++];
                if (disc[v] == -1) {
                    disc[v] = low[v] = timer++;
                    stk.push_back(v);
                    onstk[v] = 1;
                    call.push_back({v, 0});
                } else if (onstk[v]) {
                    low[u] = std::min(low[u], disc[v]);
                }
            } else {
                if (low[u] == disc[u]) {
                    while (true) {
                        const int w = stk.back();
                        stk.pop_back();
                        onstk[w] = 0;
                        comp[w]  = c;
                        if (w == u) break;
                    }
                    ++c;
                }
                call.pop_back();
                if (!call.empty()) low[call.back().first] = std::min(low[call.back().first], low[u]);
            }
        }
    }
    return comp;
}

} // namespace datamunge::algorithms
