#include <gtest/gtest.h>

#include <datamunge/algorithms/edmonds_karp.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <tuple>
#include <vector>

using datamunge::algorithms::edmonds_karp;
using datamunge::algorithms::MaxFlowResult;

namespace {

using Edge = std::tuple<std::size_t, std::size_t, double>;

// The classic CLRS Figure 26.1 network. Vertices: s=0, v1=1, v2=2, v3=3, v4=4, t=5.
// Its maximum flow value is 23, with minimum cut S = {s, v1, v2, v4}.
std::vector<Edge> clrs_network() {
    return {
        {0, 1, 16}, // s  -> v1
        {0, 2, 13}, // s  -> v2
        {1, 2, 10}, // v1 -> v2
        {2, 1, 4},  // v2 -> v1
        {1, 3, 12}, // v1 -> v3
        {3, 2, 9},  // v3 -> v2
        {2, 4, 14}, // v2 -> v4
        {4, 3, 7},  // v4 -> v3
        {3, 5, 20}, // v3 -> t
        {4, 5, 4},  // v4 -> t
    };
}

// Capacity of the cut whose source side is `result.min_cut_source_side`: the total capacity of the
// input edges that cross from the source side to its complement.
double cut_capacity(std::size_t n, const std::vector<Edge>& edges, const MaxFlowResult& result) {
    std::vector<char> in_s(n, 0);
    for (const std::size_t v : result.min_cut_source_side) in_s[v] = 1;
    double cap = 0.0;
    for (const auto& [u, v, c] : edges)
        if (in_s[u] && !in_s[v]) cap += c;
    return cap;
}

// Assert 0 <= flow <= capacity per edge and flow conservation at every non-terminal vertex.
void check_feasibility(std::size_t n, const std::vector<Edge>& edges, std::size_t source,
                       std::size_t sink, const MaxFlowResult& result) {
    ASSERT_EQ(result.edge_flow.size(), edges.size());
    std::vector<double> net(n, 0.0); // out - in
    for (std::size_t i = 0; i < edges.size(); ++i) {
        const auto& [u, v, c] = edges[i];
        const double f = result.edge_flow[i];
        EXPECT_GE(f, -1e-9) << "edge " << i;
        EXPECT_LE(f, c + 1e-9) << "edge " << i;
        net[u] += f;
        net[v] -= f;
    }
    for (std::size_t v = 0; v < n; ++v) {
        if (v == source || v == sink) continue;
        EXPECT_NEAR(net[v], 0.0, 1e-9) << "conservation at vertex " << v;
    }
    // Net out of the source and net into the sink both equal the reported max flow.
    EXPECT_NEAR(net[source], result.max_flow, 1e-9);
    EXPECT_NEAR(-net[sink], result.max_flow, 1e-9);
}

// A random directed network on `n` vertices with source 0 and sink n-1.
std::vector<Edge> random_network(std::size_t n, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> cap(0.0, 10.0);
    std::bernoulli_distribution keep(0.45);
    std::vector<Edge> edges;
    for (std::size_t u = 0; u < n; ++u)
        for (std::size_t v = 0; v < n; ++v)
            if (u != v && keep(rng)) edges.emplace_back(u, v, std::round(cap(rng) * 4.0) / 4.0);
    return edges;
}

} // namespace

TEST(EdmondsKarp, ClrsNetworkHasKnownMaxFlowAndMinCut) {
    const auto edges = clrs_network();
    const auto r = edmonds_karp(6, edges, 0, 5);
    EXPECT_NEAR(r.max_flow, 23.0, 1e-9);

    // The source side of the minimum cut is exactly {s, v1, v2, v4} = {0, 1, 2, 4}.
    auto side = r.min_cut_source_side;
    std::sort(side.begin(), side.end());
    EXPECT_EQ(side, (std::vector<std::size_t>{0, 1, 2, 4}));

    // The cut those vertices induce has capacity equal to the max flow (max-flow min-cut).
    EXPECT_NEAR(cut_capacity(6, edges, r), 23.0, 1e-9);
    check_feasibility(6, edges, 0, 5, r);
}

TEST(EdmondsKarp, TrivialSingleEdge) {
    const std::vector<Edge> edges = {{0, 1, 7.5}};
    const auto r = edmonds_karp(2, edges, 0, 1);
    EXPECT_NEAR(r.max_flow, 7.5, 1e-9);
    ASSERT_EQ(r.edge_flow.size(), 1u);
    EXPECT_NEAR(r.edge_flow[0], 7.5, 1e-9);
    // Only the source is reachable in the residual graph once the single edge saturates.
    EXPECT_EQ(r.min_cut_source_side, (std::vector<std::size_t>{0}));
    EXPECT_NEAR(cut_capacity(2, edges, r), 7.5, 1e-9);
}

TEST(EdmondsKarp, NoSourceToSinkPathGivesZeroFlow) {
    // Edges exist but none of them lets flow reach the sink (vertex 3 is isolated as a target).
    const std::vector<Edge> edges = {{0, 1, 5}, {1, 0, 2}, {2, 3, 9}};
    const auto r = edmonds_karp(4, edges, 0, 3);
    EXPECT_NEAR(r.max_flow, 0.0, 1e-9);
    for (const double f : r.edge_flow) EXPECT_NEAR(f, 0.0, 1e-9);
    // Sink is unreachable, so it is on the complement side; the cut it induces has capacity 0.
    EXPECT_NEAR(cut_capacity(4, edges, r), 0.0, 1e-9);
    const auto& side = r.min_cut_source_side;
    EXPECT_EQ(std::find(side.begin(), side.end(), 3u), side.end());
}

TEST(EdmondsKarp, ParallelAndAntiparallelEdges) {
    // Two parallel s->t edges (3 + 2) plus an antiparallel t->s edge that cannot carry s-t flow.
    const std::vector<Edge> edges = {{0, 1, 3}, {0, 1, 2}, {1, 0, 100}};
    const auto r = edmonds_karp(2, edges, 0, 1);
    EXPECT_NEAR(r.max_flow, 5.0, 1e-9);
    EXPECT_NEAR(r.edge_flow[0], 3.0, 1e-9);
    EXPECT_NEAR(r.edge_flow[1], 2.0, 1e-9);
    EXPECT_NEAR(r.edge_flow[2], 0.0, 1e-9);
    check_feasibility(2, edges, 0, 1, r);
}

TEST(EdmondsKarp, DiamondNetworkKnownValue) {
    // s=0 -> a=1 (3), s -> b=2 (2), a -> t=3 (2), b -> t (3), a -> b (1). Max flow = 5:
    // 2 via s-a-t, 2 via s-b-t, 1 via s-a-b-t; the source cut {s} has capacity 3 + 2 = 5.
    const std::vector<Edge> edges = {{0, 1, 3}, {0, 2, 2}, {1, 3, 2}, {2, 3, 3}, {1, 2, 1}};
    const auto r = edmonds_karp(4, edges, 0, 3);
    EXPECT_NEAR(r.max_flow, 5.0, 1e-9);
    EXPECT_NEAR(cut_capacity(4, edges, r), 5.0, 1e-9);
    check_feasibility(4, edges, 0, 3, r);
}

TEST(EdmondsKarp, MaxFlowEqualsMinCutOnManyRandomNetworks) {
    std::mt19937_64 rng(20240927);
    for (int trial = 0; trial < 400; ++trial) {
        const std::size_t n = 2 + rng() % 8;
        const auto edges = random_network(n, rng);
        const std::size_t source = 0, sink = n - 1;
        const auto r = edmonds_karp(n, edges, source, sink);

        // Max-flow min-cut: the reported flow equals the capacity of the residual reachable cut.
        EXPECT_NEAR(r.max_flow, cut_capacity(n, edges, r), 1e-7) << "trial " << trial;
        // The source is always on its own side; the sink never is (else an augmenting path remains).
        const auto& side = r.min_cut_source_side;
        EXPECT_NE(std::find(side.begin(), side.end(), source), side.end()) << "trial " << trial;
        EXPECT_EQ(std::find(side.begin(), side.end(), sink), side.end()) << "trial " << trial;

        check_feasibility(n, edges, source, sink, r);
    }
}

TEST(EdmondsKarp, RejectsMalformedInput) {
    EXPECT_THROW(edmonds_karp(3, {{0, 1, 1}}, 2, 2), std::invalid_argument);      // source == sink
    EXPECT_THROW(edmonds_karp(3, {{0, 1, 1}}, 0, 5), std::invalid_argument);      // sink >= n
    EXPECT_THROW(edmonds_karp(3, {{0, 1, 1}}, 7, 1), std::invalid_argument);      // source >= n
    EXPECT_THROW(edmonds_karp(3, {{0, 9, 1}}, 0, 2), std::invalid_argument);      // endpoint >= n
    EXPECT_THROW(edmonds_karp(3, {{0, 1, -2}}, 0, 2), std::invalid_argument);     // negative capacity
}
