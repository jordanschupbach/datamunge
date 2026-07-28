#pragma once

// Graph structure algorithms:
//   - kosaraju_scc / tarjan_scc : strongly connected components of a directed
//     graph, by two different classic methods (they agree on the partition)
//   - bron_kerbosch             : all maximal cliques of an undirected graph
//
// Graphs are given as adjacency lists over vertices 0..n-1.

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

namespace detail {

inline void bron_kerbosch_rec(std::vector<int> R, std::vector<int> P, std::vector<int> X,
                              const std::vector<std::vector<char>>& adjmat,
                              std::vector<std::vector<int>>&         cliques) {
    if (P.empty() && X.empty()) {
        cliques.push_back(R);
        return;
    }
    // Choose a pivot u from P union X maximising |P ∩ N(u)| to prune branches.
    int pivot = -1, best = -1;
    for (int u : P) { int cnt = 0; for (int v : P) if (adjmat[u][v]) ++cnt; if (cnt > best) { best = cnt; pivot = u; } }
    for (int u : X) { int cnt = 0; for (int v : P) if (adjmat[u][v]) ++cnt; if (cnt > best) { best = cnt; pivot = u; } }

    std::vector<int> candidates;
    for (int v : P)
        if (pivot < 0 || !adjmat[pivot][v]) candidates.push_back(v);

    for (int v : candidates) {
        std::vector<int> R2 = R;
        R2.push_back(v);
        std::vector<int> P2, X2;
        for (int w : P) if (adjmat[v][w]) P2.push_back(w);
        for (int w : X) if (adjmat[v][w]) X2.push_back(w);
        bron_kerbosch_rec(std::move(R2), std::move(P2), std::move(X2), adjmat, cliques);
        // move v from P to X
        P.erase(std::remove(P.begin(), P.end(), v), P.end());
        X.push_back(v);
    }
}

} // namespace detail

// Bron-Kerbosch with pivoting: all maximal cliques of an undirected graph, each
// returned as a sorted vertex list.
inline std::vector<std::vector<int>> bron_kerbosch(int n, const std::vector<std::vector<int>>& adj) {
    std::vector<std::vector<char>> adjmat(n, std::vector<char>(n, 0));
    for (int u = 0; u < n; ++u)
        for (int v : adj[u])
            if (u != v) { adjmat[u][v] = 1; adjmat[v][u] = 1; }

    std::vector<int> P(n);
    for (int i = 0; i < n; ++i) P[i] = i;
    std::vector<std::vector<int>> cliques;
    detail::bron_kerbosch_rec({}, P, {}, adjmat, cliques);
    for (auto& cl : cliques) std::sort(cl.begin(), cl.end());
    return cliques;
}

} // namespace datamunge::algorithms
