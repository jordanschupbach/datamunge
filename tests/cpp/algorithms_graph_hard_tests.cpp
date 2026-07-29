#include <gtest/gtest.h>

#include <datamunge/algorithms/graph_hard.hpp>

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

using namespace datamunge::algorithms;

namespace {

int brute_max_clique_size(int n, const std::vector<std::vector<char>>& m) {
    int best = 0;
    for (unsigned mask = 0; mask < (1u << n); ++mask) {
        std::vector<int> s;
        for (int i = 0; i < n; ++i) if (mask & (1u << i)) s.push_back(i);
        bool clq = true;
        for (std::size_t i = 0; i < s.size() && clq; ++i)
            for (std::size_t j = i + 1; j < s.size(); ++j)
                if (!m[s[i]][s[j]]) { clq = false; break; }
        if (clq) best = std::max(best, static_cast<int>(s.size()));
    }
    return best;
}

bool is_clique(const std::vector<int>& c, const std::vector<std::vector<char>>& m) {
    for (std::size_t i = 0; i < c.size(); ++i)
        for (std::size_t j = i + 1; j < c.size(); ++j)
            if (!m[c[i]][c[j]]) return false;
    return true;
}

// Brute-force subgraph monomorphism: try all injective maps.
bool brute_subgraph_iso(int np, const std::vector<std::vector<char>>& pm, int ng,
                        const std::vector<std::vector<char>>& gm) {
    if (np > ng) return false;
    std::vector<int> perm(ng);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end());
    do {
        bool ok = true;
        for (int u = 0; u < np && ok; ++u)
            for (int w = 0; w < np; ++w)
                if (pm[u][w] && !gm[perm[u]][perm[w]]) { ok = false; break; }
        if (ok) return true;
    } while (std::next_permutation(perm.begin(), perm.end()));
    return false;
}

} // namespace

TEST(MaxCliqueDyn, MatchesBruteForce) {
    std::mt19937_64 rng(1);
    for (int t = 0; t < 3000; ++t) {
        const int                      n = 1 + static_cast<int>(rng() % 14);
        std::vector<std::vector<char>> m(n, std::vector<char>(n, 0));
        std::vector<std::vector<int>>  adj(n);
        for (int u = 0; u < n; ++u)
            for (int v = u + 1; v < n; ++v)
                if (rng() & 1) { m[u][v] = m[v][u] = 1; adj[u].push_back(v); adj[v].push_back(u); }
        const auto cl = max_clique_dyn(n, adj);
        EXPECT_TRUE(is_clique(cl, m)) << "t=" << t;
        EXPECT_EQ(static_cast<int>(cl.size()), brute_max_clique_size(n, m)) << "t=" << t;
    }
}

TEST(MaxCliqueDyn, KnownGraphs) {
    // Complete graph K5: the whole graph is the maximum clique.
    std::vector<std::vector<int>> k5(5);
    for (int u = 0; u < 5; ++u) for (int v = 0; v < 5; ++v) if (u != v) k5[u].push_back(v);
    EXPECT_EQ(max_clique_dyn(5, k5).size(), 5u);
    // A 4-cycle has no triangle: maximum clique is an edge (size 2).
    std::vector<std::vector<int>> c4 = {{1, 3}, {0, 2}, {1, 3}, {0, 2}};
    EXPECT_EQ(max_clique_dyn(4, c4).size(), 2u);
}

TEST(SubgraphIsomorphism, MatchesBruteForce) {
    std::mt19937_64 rng(2);
    for (int t = 0; t < 3000; ++t) {
        const int ng = 2 + static_cast<int>(rng() % 7);
        const int np = 1 + static_cast<int>(rng() % ng);
        auto      build = [&](int n) {
            std::vector<std::vector<char>> m(n, std::vector<char>(n, 0));
            std::vector<std::vector<int>>  adj(n);
            for (int u = 0; u < n; ++u)
                for (int v = u + 1; v < n; ++v)
                    if (rng() & 1) { m[u][v] = m[v][u] = 1; adj[u].push_back(v); adj[v].push_back(u); }
            return std::make_pair(m, adj);
        };
        auto [pm, padj] = build(np);
        auto [gm, gadj] = build(ng);
        const auto r = subgraph_isomorphism(np, padj, ng, gadj);
        EXPECT_EQ(r.found, brute_subgraph_iso(np, pm, ng, gm)) << "t=" << t;
        if (r.found) {
            // Verify the returned mapping preserves pattern edges and is injective.
            std::vector<char> used(ng, 0);
            for (int v : r.mapping) { EXPECT_FALSE(used[v]); used[v] = 1; }
            for (int u = 0; u < np; ++u)
                for (int w = 0; w < np; ++w)
                    if (pm[u][w]) EXPECT_TRUE(gm[r.mapping[u]][r.mapping[w]]);
        }
    }
}

TEST(SubgraphIsomorphism, KnownCases) {
    // Triangle pattern in K4 -> found.
    std::vector<std::vector<int>> tri = {{1, 2}, {0, 2}, {0, 1}};
    std::vector<std::vector<int>> k4(4);
    for (int u = 0; u < 4; ++u) for (int v = 0; v < 4; ++v) if (u != v) k4[u].push_back(v);
    EXPECT_TRUE(subgraph_isomorphism(3, tri, 4, k4).found);
    // Triangle in a 4-cycle (no triangle) -> not found.
    std::vector<std::vector<int>> c4 = {{1, 3}, {0, 2}, {1, 3}, {0, 2}};
    EXPECT_FALSE(subgraph_isomorphism(3, tri, 4, c4).found);
    // A single edge is a subgraph of any graph with an edge.
    std::vector<std::vector<int>> edge = {{1}, {0}};
    EXPECT_TRUE(subgraph_isomorphism(2, edge, 4, c4).found);
}
