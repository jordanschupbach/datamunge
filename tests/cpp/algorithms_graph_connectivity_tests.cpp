#include <gtest/gtest.h>

#include <datamunge/algorithms/graph_connectivity.hpp>

#include <algorithm>
#include <random>
#include <set>
#include <vector>

using namespace datamunge::algorithms;

namespace {

// Canonicalise a component labelling: map each vertex to the smallest vertex in
// its component, so two labellings of the same partition compare equal.
std::vector<int> canon(const std::vector<int>& comp) {
    const int        n = static_cast<int>(comp.size());
    std::vector<int> rep(n, -1), out(n);
    for (int v = 0; v < n; ++v)
        if (rep[comp[v]] == -1) rep[comp[v]] = v;
    for (int v = 0; v < n; ++v) out[v] = rep[comp[v]];
    return out;
}

// Brute-force SCC via transitive-closure reachability.
std::vector<int> brute_scc(int n, const std::vector<std::vector<int>>& adj) {
    std::vector<std::vector<char>> reach(n, std::vector<char>(n, 0));
    for (int u = 0; u < n; ++u) { reach[u][u] = 1; for (int v : adj[u]) reach[u][v] = 1; }
    for (int k = 0; k < n; ++k)
        for (int i = 0; i < n; ++i)
            if (reach[i][k])
                for (int j = 0; j < n; ++j)
                    if (reach[k][j]) reach[i][j] = 1;
    std::vector<int> comp(n, -1);
    int              c = 0;
    for (int u = 0; u < n; ++u)
        if (comp[u] == -1) {
            for (int v = 0; v < n; ++v)
                if (reach[u][v] && reach[v][u]) comp[v] = c;
            ++c;
        }
    return comp;
}

} // namespace

TEST(GraphConnectivity, SccMethodsAgree) {
    std::mt19937_64 rng(1);
    for (int t = 0; t < 3000; ++t) {
        const int                     n = 1 + static_cast<int>(rng() % 40);
        std::vector<std::vector<int>> adj(n);
        const int                     m = static_cast<int>(rng() % (n * 3 + 1));
        for (int e = 0; e < m; ++e) {
            const int u = static_cast<int>(rng() % n), v = static_cast<int>(rng() % n);
            adj[u].push_back(v);
        }
        const auto k = canon(kosaraju_scc(n, adj));
        const auto ta = canon(tarjan_scc(n, adj));
        const auto b = canon(brute_scc(n, adj));
        EXPECT_EQ(k, b) << "kosaraju t=" << t;
        EXPECT_EQ(ta, b) << "tarjan t=" << t;
    }
}

TEST(GraphConnectivity, SccKnownGraph) {
    // 0->1->2->0 (a 3-cycle), 3->4, 4 alone otherwise: components {0,1,2},{3},{4}.
    std::vector<std::vector<int>> adj = {{1}, {2}, {0}, {4}, {}};
    const auto                    comp = canon(kosaraju_scc(5, adj));
    EXPECT_EQ(comp[0], comp[1]);
    EXPECT_EQ(comp[1], comp[2]);
    EXPECT_NE(comp[0], comp[3]);
    EXPECT_NE(comp[3], comp[4]);
    EXPECT_EQ(canon(tarjan_scc(5, adj)), comp);
}

// ---------------- Bron-Kerbosch ----------------

namespace {

std::set<std::vector<int>> brute_maximal_cliques(int n, const std::vector<std::vector<char>>& adjmat) {
    std::set<std::vector<int>> maximal;
    for (unsigned mask = 1; mask < (1u << n); ++mask) {
        std::vector<int> s;
        for (int i = 0; i < n; ++i) if (mask & (1u << i)) s.push_back(i);
        bool clique = true;
        for (std::size_t i = 0; i < s.size() && clique; ++i)
            for (std::size_t j = i + 1; j < s.size(); ++j)
                if (!adjmat[s[i]][s[j]]) { clique = false; break; }
        if (!clique) continue;
        bool max = true; // maximal if no outside vertex is adjacent to all of s
        for (int v = 0; v < n && max; ++v) {
            if (mask & (1u << v)) continue;
            bool all = true;
            for (int u : s) if (!adjmat[v][u]) { all = false; break; }
            if (all) max = false;
        }
        if (max) maximal.insert(s);
    }
    return maximal;
}

} // namespace

TEST(BronKerbosch, MatchesBruteForce) {
    std::mt19937_64 rng(2);
    for (int t = 0; t < 3000; ++t) {
        const int                      n = 1 + static_cast<int>(rng() % 12);
        std::vector<std::vector<char>> adjmat(n, std::vector<char>(n, 0));
        std::vector<std::vector<int>>  adj(n);
        for (int u = 0; u < n; ++u)
            for (int v = u + 1; v < n; ++v)
                if (rng() & 1) { adjmat[u][v] = adjmat[v][u] = 1; adj[u].push_back(v); adj[v].push_back(u); }

        auto got = bron_kerbosch(n, adj);
        std::set<std::vector<int>> gotset(got.begin(), got.end());
        EXPECT_EQ(gotset, brute_maximal_cliques(n, adjmat)) << "t=" << t;
    }
}

TEST(BronKerbosch, TriangleAndPath) {
    // Triangle 0-1-2: one maximal clique {0,1,2}.
    auto tri = bron_kerbosch(3, {{1, 2}, {0, 2}, {0, 1}});
    ASSERT_EQ(tri.size(), 1u);
    EXPECT_EQ(tri[0], (std::vector<int>{0, 1, 2}));
    // Path 0-1-2: maximal cliques are the edges {0,1} and {1,2}.
    auto path = bron_kerbosch(3, {{1}, {0, 2}, {1}});
    std::set<std::vector<int>> s(path.begin(), path.end());
    EXPECT_EQ(s, (std::set<std::vector<int>>{{0, 1}, {1, 2}}));
}
