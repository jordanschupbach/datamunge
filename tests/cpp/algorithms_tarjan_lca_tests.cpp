#include <gtest/gtest.h>

#include <datamunge/algorithms/tarjan_lca.hpp>

#include <numeric>
#include <random>
#include <utility>
#include <vector>

using datamunge::algorithms::tarjan_offline_lca;

namespace {

using Edge  = std::pair<std::size_t, std::size_t>;
using Query = std::pair<std::size_t, std::size_t>;

// Naive reference LCA: walk parent pointers, lifting the deeper node until depths match, then
// lifting both in lockstep until they coincide. O(depth) per query -- the cross-check oracle.
std::size_t naive_lca(const std::vector<std::size_t>& parent, const std::vector<std::size_t>& depth,
                      std::size_t root, std::size_t a, std::size_t b) {
    while (depth[a] > depth[b]) a = parent[a];
    while (depth[b] > depth[a]) b = parent[b];
    while (a != b) { a = parent[a]; b = parent[b]; }
    (void)root;
    return a;
}

// Build parent/depth arrays for a tree given as undirected edges rooted at `root`, via BFS.
void root_tree(std::size_t n, std::size_t root, const std::vector<Edge>& edges,
               std::vector<std::size_t>& parent, std::vector<std::size_t>& depth) {
    std::vector<std::vector<std::size_t>> adj(n);
    for (const auto& [u, v] : edges) { adj[u].push_back(v); adj[v].push_back(u); }
    parent.assign(n, root);
    depth.assign(n, 0);
    std::vector<char> seen(n, 0);
    std::vector<std::size_t> stack{root};
    seen[root] = 1;
    while (!stack.empty()) {
        const std::size_t u = stack.back();
        stack.pop_back();
        for (const std::size_t v : adj[u])
            if (!seen[v]) { seen[v] = 1; parent[v] = u; depth[v] = depth[u] + 1; stack.push_back(v); }
    }
}

} // namespace

TEST(TarjanOfflineLca, HandVerifiedSmallTree) {
    //           0
    //          / \
    //         1   2
    //        /|    \
    //       3 4     5
    //      /         \
    //     6           7
    const std::size_t n = 8, root = 0;
    const std::vector<Edge> edges = {{0, 1}, {0, 2}, {1, 3}, {1, 4}, {3, 6}, {2, 5}, {5, 7}};
    const std::vector<Query> queries = {
        {3, 3}, // a node with itself           -> 3
        {6, 1}, // node and its ancestor        -> 1
        {5, 7}, // ancestor and its descendant  -> 5
        {6, 7}, // two leaves, different subtrees-> 0
        {4, 7}, // two leaves, different subtrees-> 0
        {6, 4}, // cousins under 1              -> 1
        {4, 3}, // siblings under 1             -> 1
        {6, 5}, // deep nodes in disjoint halves-> 0
        {2, 0}, // node and the root           -> 0
    };
    const std::vector<std::size_t> expected = {3, 1, 5, 0, 0, 1, 1, 0, 0};

    const auto got = tarjan_offline_lca(n, root, edges, queries);
    ASSERT_EQ(got.size(), queries.size());
    for (std::size_t k = 0; k < queries.size(); ++k)
        EXPECT_EQ(got[k], expected[k]) << "query " << k << " = (" << queries[k].first << ","
                                       << queries[k].second << ")";
}

TEST(TarjanOfflineLca, SingleNodeTree) {
    const auto got = tarjan_offline_lca(1, 0, {}, {{0, 0}});
    ASSERT_EQ(got.size(), 1u);
    EXPECT_EQ(got[0], 0u);
}

TEST(TarjanOfflineLca, MatchesNaiveOnRandomTreesAndQueries) {
    std::mt19937_64 rng(20260727);
    for (int trial = 0; trial < 400; ++trial) {
        const std::size_t n = 1 + rng() % 60;

        // Random labelled rooted tree: a random permutation fixes an insertion order; every node
        // after the first attaches to a uniformly random earlier node -> always a valid tree.
        std::vector<std::size_t> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        const std::size_t root = perm[0];
        std::vector<Edge> edges;
        for (std::size_t i = 1; i < n; ++i) {
            const std::size_t p = perm[rng() % i]; // some already-inserted node
            edges.push_back({perm[i], p});
        }
        std::shuffle(edges.begin(), edges.end(), rng); // order must not matter

        std::vector<std::size_t> parent, depth;
        root_tree(n, root, edges, parent, depth);

        std::vector<Query> queries;
        const std::size_t q = rng() % 40;
        for (std::size_t k = 0; k < q; ++k)
            queries.push_back({rng() % n, rng() % n});

        const auto got = tarjan_offline_lca(n, root, edges, queries);
        ASSERT_EQ(got.size(), queries.size()) << "trial " << trial;
        for (std::size_t k = 0; k < queries.size(); ++k) {
            const std::size_t want = naive_lca(parent, depth, root, queries[k].first, queries[k].second);
            EXPECT_EQ(got[k], want) << "trial " << trial << " query " << k << " = ("
                                    << queries[k].first << "," << queries[k].second << ")";
        }
    }
}

TEST(TarjanOfflineLca, HandlesDeepPathWithoutStackOverflow) {
    // Degenerate line 0-1-2-...-(n-1): would blow a naive recursive DFS; the explicit stack copes.
    const std::size_t n = 20000, root = 0;
    std::vector<Edge> edges;
    for (std::size_t i = 1; i < n; ++i) edges.push_back({i - 1, i});
    const std::vector<Query> queries = {{n - 1, n / 2}, {n - 1, 0}, {123, 4567}};
    const auto got = tarjan_offline_lca(n, root, edges, queries);
    ASSERT_EQ(got.size(), 3u);
    EXPECT_EQ(got[0], n / 2); // shallower node is the ancestor on a path
    EXPECT_EQ(got[1], 0u);    // the root
    EXPECT_EQ(got[2], 123u);  // min index on a path
}

TEST(TarjanOfflineLca, RejectsMalformedInput) {
    EXPECT_THROW(tarjan_offline_lca(0, 0, {}, {}), std::invalid_argument);            // root >= n (empty)
    EXPECT_THROW(tarjan_offline_lca(3, 3, {{0, 1}, {1, 2}}, {}), std::invalid_argument); // root >= n
    EXPECT_THROW(tarjan_offline_lca(3, 0, {{0, 1}}, {}), std::invalid_argument);      // too few edges
    EXPECT_THROW(tarjan_offline_lca(3, 0, {{0, 1}, {1, 2}, {0, 2}}, {}), std::invalid_argument); // too many
    EXPECT_THROW(tarjan_offline_lca(3, 0, {{0, 1}, {1, 9}}, {}), std::invalid_argument); // bad edge endpoint
    EXPECT_THROW(tarjan_offline_lca(3, 0, {{0, 1}, {1, 2}}, {{0, 9}}), std::invalid_argument); // bad query
}
