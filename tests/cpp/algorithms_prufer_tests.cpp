#include <gtest/gtest.h>

#include <datamunge/algorithms/prufer.hpp>

#include <algorithm>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

using datamunge::algorithms::prufer_to_tree;
using datamunge::algorithms::tree_to_prufer;

namespace {

using Edge = std::pair<std::size_t, std::size_t>;

// Canonicalize an undirected edge list (sort each endpoint pair, then the list) so two edge
// lists can be compared regardless of edge order or endpoint orientation.
std::set<Edge> canonical_edges(const std::vector<Edge>& edges) {
    std::set<Edge> s;
    for (auto [u, v] : edges) s.insert({std::min(u, v), std::max(u, v)});
    return s;
}

// True iff `edges` form a tree on n vertices: exactly n-1 edges, connected, and acyclic.
// Verified with union-find -- a cycle shows up as an edge whose endpoints already share a root,
// and connectedness as a single remaining component.
bool is_tree(std::size_t n, const std::vector<Edge>& edges) {
    if (edges.size() != n - 1) return false;
    std::vector<std::size_t> parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    auto find = [&](std::size_t x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    std::size_t components = n;
    for (auto [u, v] : edges) {
        const std::size_t ru = find(u);
        const std::size_t rv = find(v);
        if (ru == rv) return false; // edge closes a cycle
        parent[ru] = rv;
        --components;
    }
    return components == 1;
}

} // namespace

TEST(Prufer, KnownTreeEncodesToKnownCode) {
    // Deleting leaves 0,1,2 (each adjacent to 3) then leaf 3 (adjacent to 4) leaves {4,5}.
    const std::size_t n = 6;
    const std::vector<Edge> tree = {{0, 3}, {1, 3}, {2, 3}, {3, 4}, {4, 5}};
    const std::vector<std::size_t> expected = {3, 3, 3, 4};
    EXPECT_EQ(tree_to_prufer(n, tree), expected);
}

TEST(Prufer, KnownCodeDecodesToKnownTree) {
    const std::vector<std::size_t> code = {3, 3, 3, 4};
    const std::vector<Edge> expected = {{0, 3}, {1, 3}, {2, 3}, {3, 4}, {4, 5}};
    EXPECT_EQ(canonical_edges(prufer_to_tree(code)), canonical_edges(expected));
}

TEST(Prufer, EmptyCodeIsTheSingleEdgeOnTwoVertices) {
    const auto edges = prufer_to_tree({});
    ASSERT_EQ(edges.size(), 1u);
    EXPECT_EQ(canonical_edges(edges), (std::set<Edge>{{0, 1}}));
    EXPECT_TRUE(tree_to_prufer(2, edges).empty());
}

TEST(Prufer, RoundTripsRandomSequences) {
    // Any sequence over {0..n-1} of length n-2 is itself a valid Prüfer code, so
    // encode(decode(seq)) must return seq unchanged -- the cleanest test of the bijection.
    std::mt19937_64 rng(2024);
    for (int trial = 0; trial < 2000; ++trial) {
        const std::size_t n = 2 + rng() % 30;
        std::vector<std::size_t> seq(n - 2);
        std::uniform_int_distribution<std::size_t> pick(0, n - 1);
        for (auto& s : seq) s = pick(rng);
        const auto tree = prufer_to_tree(seq);
        ASSERT_TRUE(is_tree(n, tree)) << "trial " << trial << " n " << n;
        EXPECT_EQ(tree_to_prufer(n, tree), seq) << "trial " << trial;
    }
}

TEST(Prufer, DecodedGraphIsAlwaysASpanningTree) {
    std::mt19937_64 rng(7);
    for (int trial = 0; trial < 1000; ++trial) {
        const std::size_t n = 2 + rng() % 40;
        std::vector<std::size_t> seq(n - 2);
        std::uniform_int_distribution<std::size_t> pick(0, n - 1);
        for (auto& s : seq) s = pick(rng);
        const auto tree = prufer_to_tree(seq);
        EXPECT_EQ(tree.size(), n - 1) << "trial " << trial;
        EXPECT_TRUE(is_tree(n, tree)) << "trial " << trial;
    }
}

TEST(Prufer, RejectsMalformedInput) {
    EXPECT_THROW(tree_to_prufer(1, {}), std::invalid_argument);               // n < 2
    EXPECT_THROW(tree_to_prufer(4, {{0, 1}, {1, 2}}), std::invalid_argument); // not n-1 edges
    EXPECT_THROW(tree_to_prufer(3, {{0, 1}, {1, 3}}), std::invalid_argument); // endpoint >= n
    EXPECT_THROW(prufer_to_tree({4, 0}), std::invalid_argument);              // entry >= n (n = 4)
}
