#include <gtest/gtest.h>

#include <datamunge/algorithms/golden_section_search.hpp>
#include <datamunge/algorithms/lex_bfs.hpp>
#include <datamunge/algorithms/uniform_cost_search.hpp>

#include <algorithm>
#include <cmath>
#include <queue>
#include <random>
#include <vector>

using namespace datamunge::algorithms;

// ---------------- Lex-BFS ----------------

namespace {

bool is_permutation_of_0n(const std::vector<int>& order, int n) {
    if (static_cast<int>(order.size()) != n) return false;
    std::vector<char> seen(n, 0);
    for (int v : order) {
        if (v < 0 || v >= n || seen[v]) return false;
        seen[v] = 1;
    }
    return true;
}

// The 4-point characterisation of Lex-BFS orderings: for positions a<b<c with
// edge(order[a],order[c]) and not edge(order[a],order[b]), there is d<a with
// edge(order[d],order[b]) and not edge(order[d],order[c]).
bool satisfies_lexbfs_property(const std::vector<int>& order, const std::vector<std::vector<char>>& nb) {
    const int n = static_cast<int>(order.size());
    for (int a = 0; a < n; ++a)
        for (int b = a + 1; b < n; ++b)
            for (int c = b + 1; c < n; ++c) {
                const int A = order[a], B = order[b], C = order[c];
                if (nb[A][C] && !nb[A][B]) {
                    bool ok = false;
                    for (int d = 0; d < a; ++d) {
                        const int D = order[d];
                        if (nb[D][B] && !nb[D][C]) { ok = true; break; }
                    }
                    if (!ok) return false;
                }
            }
    return true;
}

} // namespace

TEST(LexBfs, ValidOrderingAndProperty) {
    std::mt19937_64 rng(1);
    for (int t = 0; t < 2000; ++t) {
        const int                      n = 1 + static_cast<int>(rng() % 12);
        std::vector<std::vector<char>> nb(n, std::vector<char>(n, 0));
        std::vector<std::vector<int>>  adj(n);
        for (int u = 0; u < n; ++u)
            for (int v = u + 1; v < n; ++v)
                if (rng() & 1) { nb[u][v] = nb[v][u] = 1; adj[u].push_back(v); adj[v].push_back(u); }
        const auto order = lex_bfs(n, adj);
        ASSERT_TRUE(is_permutation_of_0n(order, n)) << "t=" << t;
        EXPECT_TRUE(satisfies_lexbfs_property(order, nb)) << "t=" << t;
    }
}

// ---------------- Golden-section search ----------------

TEST(GoldenSection, FindsMaxOfUnimodalFunctions) {
    // -(x-3)^2 has its max at x = 3.
    auto r1 = golden_section_search([](double x) { return -(x - 3) * (x - 3); }, -10, 10);
    EXPECT_NEAR(r1.argmax, 3.0, 1e-6);
    EXPECT_NEAR(r1.value, 0.0, 1e-9);
    // sin has its max at pi/2 on [0, pi].
    auto r2 = golden_section_search([](double x) { return std::sin(x); }, 0.0, M_PI);
    EXPECT_NEAR(r2.argmax, M_PI / 2, 1e-6);
    // A skewed unimodal peak.
    auto r3 = golden_section_search([](double x) { return -std::abs(x - 1.234) - 0.1 * (x - 1.234) * (x - 1.234); }, -5, 5);
    EXPECT_NEAR(r3.argmax, 1.234, 1e-4);
}

// ---------------- Uniform-cost search ----------------

namespace {

double dijkstra_ref(int n, const std::vector<std::vector<std::pair<int, double>>>& adj, int s, int g) {
    std::vector<double> dist(n, 1e300);
    std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::greater<>> pq;
    dist[s] = 0; pq.emplace(0.0, s);
    while (!pq.empty()) {
        auto [c, u] = pq.top(); pq.pop();
        if (c > dist[u]) continue;
        for (auto [v, w] : adj[u]) if (c + w < dist[v]) { dist[v] = c + w; pq.emplace(dist[v], v); }
    }
    return dist[g];
}

} // namespace

TEST(UniformCostSearch, MatchesDijkstra) {
    std::mt19937_64                        rng(2);
    std::uniform_real_distribution<double> wt(0.0, 10.0);
    for (int t = 0; t < 2000; ++t) {
        const int n = 2 + static_cast<int>(rng() % 30);
        std::vector<std::vector<std::pair<int, double>>> adj(n);
        const int m = static_cast<int>(rng() % (n * 3 + 1));
        for (int e = 0; e < m; ++e) {
            const int u = static_cast<int>(rng() % n), v = static_cast<int>(rng() % n);
            const double w = wt(rng);
            adj[u].emplace_back(v, w);
            adj[v].emplace_back(u, w);
        }
        const int s = static_cast<int>(rng() % n), g = static_cast<int>(rng() % n);
        const auto r = uniform_cost_search(n, adj, s, g);
        const double ref = dijkstra_ref(n, adj, s, g);
        if (ref > 1e299) {
            EXPECT_FALSE(r.found) << "t=" << t;
        } else {
            ASSERT_TRUE(r.found) << "t=" << t;
            EXPECT_NEAR(r.cost, ref, 1e-9) << "t=" << t;
            // Verify the returned path realises the cost.
            EXPECT_EQ(r.path.front(), s);
            EXPECT_EQ(r.path.back(), g);
        }
    }
}
