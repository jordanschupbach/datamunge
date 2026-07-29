#pragma once

// Two NP-hard graph problems solved exactly by pruned backtracking:
//   - max_clique_dyn        : the maximum (largest) clique, by branch and bound
//                             with a greedy-colouring upper bound (Tomita/MaxCliqueDyn)
//   - subgraph_isomorphism  : does a pattern graph occur as a subgraph of a
//                             target graph? (backtracking edge-consistency search)

#include <algorithm>
#include <vector>

namespace datamunge::algorithms {

namespace detail {

class MaxCliqueSolver {
public:
    MaxCliqueSolver(int n, const std::vector<std::vector<char>>& adjmat) : n_(n), adj_(adjmat) {}

    std::vector<int> solve() {
        std::vector<int> R;
        std::vector<int> P(n_);
        for (int i = 0; i < n_; ++i) P[i] = i;
        expand(R, P);
        return best_;
    }

private:
    // Greedy colouring of the candidate set P: returns vertices ordered by colour
    // (ascending) with each vertex's colour number (an upper bound on the clique
    // extendable through it).
    void colour_sort(const std::vector<int>& P, std::vector<int>& order, std::vector<int>& col) {
        std::vector<std::vector<int>> classes;
        for (int v : P) {
            std::size_t c = 0;
            for (; c < classes.size(); ++c) {
                bool conflict = false;
                for (int u : classes[c])
                    if (adj_[v][u]) { conflict = true; break; }
                if (!conflict) break;
            }
            if (c == classes.size()) classes.emplace_back();
            classes[c].push_back(v);
        }
        order.clear();
        col.clear();
        for (std::size_t c = 0; c < classes.size(); ++c)
            for (int v : classes[c]) { order.push_back(v); col.push_back(static_cast<int>(c) + 1); }
    }

    void expand(std::vector<int>& R, const std::vector<int>& P) {
        if (P.empty()) {
            if (R.size() > best_.size()) best_ = R;
            return;
        }
        std::vector<int> order, col;
        colour_sort(P, order, col);
        for (int i = static_cast<int>(order.size()) - 1; i >= 0; --i) {
            if (R.size() + static_cast<std::size_t>(col[i]) <= best_.size()) return; // bound
            const int v = order[i];
            R.push_back(v);
            std::vector<int> P2;
            for (int j = 0; j < i; ++j)
                if (adj_[v][order[j]]) P2.push_back(order[j]);
            expand(R, P2);
            R.pop_back();
        }
    }

    int                                  n_;
    const std::vector<std::vector<char>>& adj_;
    std::vector<int>                     best_;
};

} // namespace detail

// Maximum clique of an undirected graph (adjacency lists), returned sorted.
inline std::vector<int> max_clique_dyn(int n, const std::vector<std::vector<int>>& adj) {
    std::vector<std::vector<char>> adjmat(n, std::vector<char>(n, 0));
    for (int u = 0; u < n; ++u)
        for (int v : adj[u])
            if (u != v) { adjmat[u][v] = 1; adjmat[v][u] = 1; }
    detail::MaxCliqueSolver solver(n, adjmat);
    auto                    best = solver.solve();
    std::sort(best.begin(), best.end());
    return best;
}

struct SubgraphMatch {
    bool             found{false};
    std::vector<int> mapping; // mapping[patternVertex] = targetVertex
};

namespace detail {

inline bool subgraph_backtrack(int depth, int np, const std::vector<std::vector<char>>& pmat,
                               const std::vector<int>& pdeg, int ng, const std::vector<std::vector<char>>& gmat,
                               const std::vector<int>& gdeg, std::vector<int>& map, std::vector<char>& used) {
    if (depth == np) return true;
    const int u = depth; // assign pattern vertices in index order
    for (int v = 0; v < ng; ++v) {
        if (used[v] || gdeg[v] < pdeg[u]) continue;
        bool ok = true;
        for (int w = 0; w < depth; ++w) // check edges to already-mapped pattern vertices
            if (pmat[u][w] && !gmat[v][map[w]]) { ok = false; break; }
        if (!ok) continue;
        map[u]  = v;
        used[v] = 1;
        if (subgraph_backtrack(depth + 1, np, pmat, pdeg, ng, gmat, gdeg, map, used)) return true;
        used[v] = 0;
        map[u]  = -1;
    }
    return false;
}

} // namespace detail

// Subgraph isomorphism (monomorphism): find an injective map from the pattern's
// vertices to the target's that preserves every pattern edge. Returns the mapping
// if one exists.
inline SubgraphMatch subgraph_isomorphism(int np, const std::vector<std::vector<int>>& padj, int ng,
                                          const std::vector<std::vector<int>>& gadj) {
    if (np > ng) return {};
    std::vector<std::vector<char>> pmat(np, std::vector<char>(np, 0)), gmat(ng, std::vector<char>(ng, 0));
    std::vector<int>               pdeg(np, 0), gdeg(ng, 0);
    for (int u = 0; u < np; ++u)
        for (int v : padj[u])
            if (u != v && !pmat[u][v]) { pmat[u][v] = pmat[v][u] = 1; }
    for (int u = 0; u < ng; ++u)
        for (int v : gadj[u])
            if (u != v && !gmat[u][v]) { gmat[u][v] = gmat[v][u] = 1; }
    for (int u = 0; u < np; ++u) for (int v = 0; v < np; ++v) pdeg[u] += pmat[u][v];
    for (int u = 0; u < ng; ++u) for (int v = 0; v < ng; ++v) gdeg[u] += gmat[u][v];

    std::vector<int>  map(np, -1);
    std::vector<char> used(ng, 0);
    if (detail::subgraph_backtrack(0, np, pmat, pdeg, ng, gmat, gdeg, map, used)) return {true, map};
    return {};
}

} // namespace datamunge::algorithms
