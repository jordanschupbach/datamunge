#include <gtest/gtest.h>

#include <datamunge/algorithms/hopcroft_karp.hpp>

#include <algorithm>
#include <cstddef>
#include <random>
#include <set>
#include <utility>
#include <vector>

using datamunge::algorithms::BipartiteMatchingResult;
using datamunge::algorithms::hopcroft_karp;
using datamunge::algorithms::kBipartiteUnmatched;

namespace {

using Edge = std::pair<std::size_t, std::size_t>;

// Independent reference: maximum bipartite matching by the simple augmenting-path method (Kuhn's
// algorithm). Returns only the matching *size*, which the tests cross-check against Hopcroft-Karp.
std::size_t kuhn_matching_size(std::size_t n_left, std::size_t n_right, const std::vector<Edge>& edges) {
    std::vector<std::vector<std::size_t>> adj(n_left);
    for (const auto& [l, r] : edges) adj[l].push_back(r);

    std::vector<std::size_t> match_right(n_right, kBipartiteUnmatched);
    std::vector<char> visited(n_right, 0);

    auto try_augment = [&](auto&& self, std::size_t u) -> bool {
        for (const std::size_t v : adj[u]) {
            if (visited[v]) continue;
            visited[v] = 1;
            if (match_right[v] == kBipartiteUnmatched || self(self, match_right[v])) {
                match_right[v] = u;
                return true;
            }
        }
        return false;
    };

    std::size_t size = 0;
    for (std::size_t u = 0; u < n_left; ++u) {
        std::fill(visited.begin(), visited.end(), 0);
        if (try_augment(try_augment, u)) ++size;
    }
    return size;
}

// Asserts that `m` is a structurally valid matching of the given graph: consistent orientations,
// every vertex used at most once, every matched pair is an actual edge, and size == pair count.
void expect_valid_matching(std::size_t n_left, std::size_t n_right, const std::vector<Edge>& edges,
                           const BipartiteMatchingResult& m) {
    ASSERT_EQ(m.match_left.size(), n_left);
    ASSERT_EQ(m.match_right.size(), n_right);

    std::set<Edge> edge_set(edges.begin(), edges.end());

    std::size_t matched_left = 0;
    for (std::size_t l = 0; l < n_left; ++l) {
        const std::size_t r = m.match_left[l];
        if (r == kBipartiteUnmatched) continue;
        ++matched_left;
        ASSERT_LT(r, n_right) << "left " << l;
        EXPECT_TRUE(edge_set.count({l, r})) << "pair (" << l << "," << r << ") is not an edge";
        EXPECT_EQ(m.match_right[r], l) << "orientation mismatch at right " << r; // each vertex once
    }

    std::size_t matched_right = 0;
    for (std::size_t r = 0; r < n_right; ++r) {
        const std::size_t l = m.match_right[r];
        if (l == kBipartiteUnmatched) continue;
        ++matched_right;
        ASSERT_LT(l, n_left) << "right " << r;
        EXPECT_EQ(m.match_left[l], r) << "orientation mismatch at left " << l;
    }

    EXPECT_EQ(matched_left, m.size);
    EXPECT_EQ(matched_right, m.size);
}

} // namespace

TEST(HopcroftKarp, KnownSmallGraphMaximumSize) {
    // 5 applicants (left) vs 4 jobs (right). By hand the maximum matching saturates all four jobs
    // (size 4) and leaves one applicant unmatched.
    const std::size_t n_left = 5, n_right = 4;
    const std::vector<Edge> edges = {{0, 0}, {0, 1}, {1, 0}, {2, 1}, {2, 2}, {3, 2}, {3, 3}, {4, 3}};

    const auto m = hopcroft_karp(n_left, n_right, edges);
    EXPECT_EQ(m.size, 4u);
    expect_valid_matching(n_left, n_right, edges, m);
    // All four jobs are filled; exactly one applicant is left out.
    for (std::size_t r = 0; r < n_right; ++r) EXPECT_NE(m.match_right[r], kBipartiteUnmatched);
    std::size_t unmatched_left = 0;
    for (std::size_t l = 0; l < n_left; ++l) unmatched_left += (m.match_left[l] == kBipartiteUnmatched);
    EXPECT_EQ(unmatched_left, 1u);
}

TEST(HopcroftKarp, AdmitsPerfectMatching) {
    // A 4x4 graph with an obvious perfect matching l<->l, plus extra edges that could tempt a
    // greedy method into a non-perfect matching.
    const std::size_t n = 4;
    std::vector<Edge> edges = {{0, 1}, {0, 0}, {1, 0}, {1, 2}, {2, 1}, {2, 3}, {3, 2}, {3, 3}};
    const auto m = hopcroft_karp(n, n, edges);
    EXPECT_EQ(m.size, n); // perfect: every left vertex matched
    for (std::size_t l = 0; l < n; ++l) EXPECT_NE(m.match_left[l], kBipartiteUnmatched);
    expect_valid_matching(n, n, edges, m);
}

TEST(HopcroftKarp, EmptyGraphHasEmptyMatching) {
    const auto m = hopcroft_karp(5, 3, {});
    EXPECT_EQ(m.size, 0u);
    for (const auto x : m.match_left) EXPECT_EQ(x, kBipartiteUnmatched);
    for (const auto x : m.match_right) EXPECT_EQ(x, kBipartiteUnmatched);
    expect_valid_matching(5, 3, {}, m);
    // Isolated vertices with no edges at all.
    const auto empty = hopcroft_karp(0, 0, {});
    EXPECT_EQ(empty.size, 0u);
}

TEST(HopcroftKarp, MatchesKuhnSizeOnRandomInstances) {
    std::mt19937_64 rng(2024);
    for (int trial = 0; trial < 500; ++trial) {
        const std::size_t n_left = 1 + rng() % 9;
        const std::size_t n_right = 1 + rng() % 9;
        // Sample each possible edge independently with probability ~0.35.
        std::vector<Edge> edges;
        for (std::size_t l = 0; l < n_left; ++l)
            for (std::size_t r = 0; r < n_right; ++r)
                if (rng() % 100 < 35) edges.push_back({l, r});

        const auto m = hopcroft_karp(n_left, n_right, edges);
        const std::size_t ref = kuhn_matching_size(n_left, n_right, edges);
        EXPECT_EQ(m.size, ref) << "trial " << trial;             // maximum cardinality agrees
        expect_valid_matching(n_left, n_right, edges, m);        // and the matching itself is valid
    }
}

TEST(HopcroftKarp, ToleratesDuplicateEdges) {
    // Duplicate and repeated edges must not inflate the matching or corrupt validity.
    const std::vector<Edge> edges = {{0, 0}, {0, 0}, {1, 0}, {1, 0}, {0, 1}};
    const auto m = hopcroft_karp(2, 2, edges);
    EXPECT_EQ(m.size, 2u);
    expect_valid_matching(2, 2, edges, m);
}

TEST(HopcroftKarp, RejectsOutOfRangeEndpoints) {
    EXPECT_THROW(hopcroft_karp(2, 2, {{2, 0}}), std::invalid_argument); // left index == n_left
    EXPECT_THROW(hopcroft_karp(2, 2, {{0, 2}}), std::invalid_argument); // right index == n_right
    EXPECT_THROW(hopcroft_karp(3, 3, {{0, 0}, {5, 1}}), std::invalid_argument);
}
