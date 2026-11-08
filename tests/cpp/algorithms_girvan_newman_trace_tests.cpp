#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <datamunge/algorithms/girvan_newman.hpp>
#include <limits>
#include <map>
#include <queue>
#include <random>
#include <set>

#include "../oracles/girvan_newman_exact.hpp"
namespace gn = datamunge::algorithms;
using Edge   = std::pair<std::size_t, std::size_t>;
namespace {
double pair_modularity(
    std::size_t                                  n,
    const std::vector<Edge>&                     es,
    const std::vector<std::vector<std::size_t>>& groups) {
  if (es.empty())
    return 0;
  std::vector<std::vector<int>> a(n, std::vector<int>(n));
  std::vector<int>              k(n), labels(n);
  for (auto [u, v] : es) {
    a[u][v] = a[v][u] = 1;
    ++k[u];
    ++k[v];
  }
  for (std::size_t c = 0; c < groups.size(); ++c)
    for (auto v : groups[c])
      labels[v] = int(c);
  double sum = 0, twom = 2. * es.size();
  for (std::size_t u = 0; u < n; ++u)
    for (std::size_t v = 0; v < n; ++v)
      if (labels[u] == labels[v])
        sum += a[u][v] - k[u] * k[v] / twom;
  return sum / twom;
}
} // namespace
TEST(GirvanNewmanChecked, AllSimpleGraphsThroughFiveExactPathFixtures) {
  using namespace girvan_newman_exact;
  ASSERT_EQ(graphs.size(), 1100u);
  ASSERT_EQ(partitions.size(), 5406u);
  for (const auto& graph : graphs) {
    std::vector<::Edge> es;
    for (std::size_t j = 0; j < graph.edge_count; ++j) {
      const auto& e = edges[graph.edge_offset + j];
      es.emplace_back(e.u, e.v);
    }
    const auto b = gn::edge_betweenness(graph.n, es);
    ASSERT_EQ(b.edges, es);
    ASSERT_EQ(b.raw_scores.size(), graph.edge_count);
    for (std::size_t j = 0; j < es.size(); ++j) {
      EXPECT_NEAR(b.raw_scores[j], edges[graph.edge_offset + j].raw, 1e-12);
      EXPECT_EQ(b.scores[j], b.raw_scores[j] / 2);
    }
    EXPECT_TRUE(b.path_counts_integer_guaranteed);
    for (std::size_t j = 0; j < graph.partition_count; ++j) {
      const auto&                           p = partitions[graph.partition_offset + j];
      std::vector<std::vector<std::size_t>> expected;
      for (std::size_t c = 0; c < p.group_count; ++c) {
        const auto& g = groups[p.group_offset + c];
        expected.emplace_back(
            vertices.begin() + g.vertex_offset,
            vertices.begin() + g.vertex_offset + g.vertex_count);
      }
      const auto r = gn::girvan_newman_run(graph.n, es, p.target);
      EXPECT_EQ(r.groups, expected);
      EXPECT_EQ(gn::girvan_newman(graph.n, es, p.target), expected);
      EXPECT_NEAR(
          gn::graph_modularity(graph.n, es, expected),
          pair_modularity(graph.n, es, expected),
          1e-12);
    }
  }
}
TEST(GirvanNewmanChecked, OriginalBridgeRawScoresAndBatchOvershoot) {
  const auto graph = gn::clique_bridge_graph();
  const auto b     = gn::edge_betweenness(graph.vertex_count, graph.edges);
  const auto pos   = std::lower_bound(b.edges.begin(), b.edges.end(), Edge{2, 3}) - b.edges.begin();
  EXPECT_EQ(b.raw_scores[pos], 18);
  EXPECT_EQ(b.scores[pos], 9);
  const auto r = gn::girvan_newman_run(graph.vertex_count, graph.edges, 2);
  EXPECT_EQ(r.groups, (std::vector<std::vector<std::size_t>>{{0, 1, 2}, {3, 4, 5}}));
  EXPECT_EQ(r.work.rounds, 1u);
  EXPECT_EQ(r.work.removed_edges, 1u);
  EXPECT_EQ(r.work.component_passes, 2u);
  EXPECT_NEAR(gn::graph_modularity(6, graph.edges, r.groups), 5. / 14, 1e-15);
  const auto cycle = gn::girvan_newman_run(4, {{0, 1}, {1, 2}, {2, 3}, {3, 0}}, 2);
  EXPECT_EQ(cycle.groups.size(), 4u);
  EXPECT_EQ(cycle.work.removed_edges, 4u);
  EXPECT_EQ(cycle.work.rounds, 1u);
  const auto stopped = gn::girvan_newman_run(4, {{0, 1}, {2, 3}}, 2);
  EXPECT_EQ(stopped.work.rounds, 0u);
  EXPECT_EQ(stopped.work.source_initializations, 0u);
}
TEST(GirvanNewmanChecked, ActualTraceBfsPathDependencyAndCutCertificates) {
  using K = gn::GirvanNewmanEventKind;
  std::mt19937 rng(134);
  for (int trial = 0; trial < 160; ++trial) {
    const std::size_t   n = 2 + rng() % 8;
    std::vector<::Edge> es;
    for (std::size_t u = 0; u < n; ++u)
      for (std::size_t v = u + 1; v < n; ++v)
        if (rng() % 3 == 0)
          es.emplace_back(u, v);
    const auto                  t = gn::girvan_newman_trace(n, es, 1 + rng() % n, 0);
    std::vector<Edge>           active, kept;
    std::vector<int>            d(n);
    std::vector<double>         paths(n), deps(n), scores;
    std::array<std::size_t, 16> count{};
    std::size_t                 source  = 0;
    double                      maximum = 0;
    bool                        pending = false;
    for (const auto& e : t.events) {
      ++count[static_cast<std::size_t>(e.kind)];
      switch (e.kind) {
        case K::canonical:
          EXPECT_EQ(e.count, es.size());
          break;
        case K::edge:
          active.emplace_back(e.u, e.v);
          break;
        case K::round:
          if (pending) {
            active  = kept;
            pending = false;
          }
          EXPECT_EQ(active.size(), e.count);
          scores.assign(active.size(), 0);
          break;
        case K::source:
          source = e.source;
          std::fill(d.begin(), d.end(), -1);
          std::fill(paths.begin(), paths.end(), 0);
          std::fill(deps.begin(), deps.end(), 0);
          d[source]     = 0;
          paths[source] = 1;
          break;
        case K::pop:
          EXPECT_EQ(e.source, source);
          EXPECT_EQ(e.distance, d[e.u]);
          EXPECT_EQ(e.paths_u, paths[e.u]);
          break;
        case K::scan:
          EXPECT_EQ(e.distance, d[e.v]);
          EXPECT_EQ(e.paths_v, paths[e.v]);
          EXPECT_TRUE(
              std::binary_search(active.begin(), active.end(), Edge(std::minmax(e.u, e.v))));
          break;
        case K::discover:
          EXPECT_EQ(d[e.v], -1);
          d[e.v] = d[e.u] + 1;
          EXPECT_EQ(e.distance, d[e.v]);
          break;
        case K::path:
          EXPECT_EQ(d[e.v], d[e.u] + 1);
          paths[e.v] += paths[e.u];
          EXPECT_EQ(e.paths_u, paths[e.u]);
          EXPECT_EQ(e.paths_v, paths[e.v]);
          break;
        case K::dependency: {
          EXPECT_EQ(e.paths_u, paths[e.u]);
          EXPECT_EQ(e.paths_v, paths[e.v]);
          EXPECT_EQ(e.dependency_v, deps[e.v]);
          const double c = (paths[e.u] / paths[e.v]) * (1 + deps[e.v]);
          EXPECT_EQ(e.contribution, c);
          deps[e.u] += c;
          EXPECT_EQ(e.dependency_u, deps[e.u]);
          EXPECT_EQ(active[e.index], Edge(std::minmax(e.u, e.v)));
          scores[e.index] += c;
          EXPECT_EQ(e.score, scores[e.index]);
          break;
        }
        case K::edge_score:
          EXPECT_EQ(e.score, scores[e.index]);
          break;
        case K::maximum:
          maximum = *std::max_element(scores.begin(), scores.end());
          EXPECT_EQ(e.maximum, maximum);
          kept.clear();
          pending = true;
          break;
        case K::keep:
          EXPECT_EQ(e.accepted, scores[e.index] < maximum - 1e-12);
          if (e.accepted)
            kept.push_back(active[e.index]);
          break;
        default:
          break;
      }
    }
    const auto& w = t.run.work;
    EXPECT_EQ(w.source_initializations, count[6]);
    EXPECT_EQ(w.source_slots, n * count[6]);
    EXPECT_EQ(w.bfs_pops, count[7]);
    EXPECT_EQ(w.neighbor_checks, count[8]);
    EXPECT_EQ(w.discoveries, count[9]);
    EXPECT_EQ(w.path_updates, count[10]);
    EXPECT_EQ(w.dependency_updates, count[11]);
    EXPECT_EQ(w.edge_scores, count[12]);
    EXPECT_EQ(w.rounds, count[13]);
    EXPECT_EQ(w.edge_decisions, count[14]);
    EXPECT_EQ(w.component_passes, count[2]);
    EXPECT_EQ(t.run.groups.size(), t.events.back().count);
    EXPECT_EQ(t.events.back().kind, K::stop);
  }
}
TEST(GirvanNewmanChecked, ExactOnlineQuotasEmptyAndDeepOwnership) {
  const auto t = gn::girvan_newman_trace(2, {{0, 1}}, 2, 33);
  ASSERT_EQ(t.events.size(), 33u);
  EXPECT_EQ(t.run.work.source_slots, 4u);
  EXPECT_EQ(t.run.work.bfs_pops, 4u);
  EXPECT_EQ(t.run.work.neighbor_checks, 4u);
  EXPECT_EQ(t.run.work.dependency_updates, 2u);
  EXPECT_EQ(t.run.work.edge_search_comparisons, 2u);
  EXPECT_THROW(gn::girvan_newman_trace(2, {{0, 1}}, 2, 32), std::length_error);
  EXPECT_EQ(gn::girvan_newman_trace(2, {{0, 1}}, 2, 0).events.size(), 33u);
  auto copy             = t;
  copy.events[0].count  = 999;
  copy.run.groups[0][0] = 999;
  EXPECT_NE(t.events[0].count, 999u);
  EXPECT_NE(t.run.groups[0][0], 999u);
  const auto empty = gn::girvan_newman_trace(0, {}, 1, 4);
  EXPECT_EQ(empty.events.size(), 4u);
  EXPECT_TRUE(empty.run.groups.empty());
  EXPECT_THROW(gn::girvan_newman_trace(0, {}, 1, 3), std::length_error);
  const auto b = gn::edge_betweenness_trace(2, {{0, 1}}, 19);
  EXPECT_EQ(b.events.size(), 19u);
  EXPECT_THROW(gn::edge_betweenness_trace(2, {{0, 1}}, 18), std::length_error);
  EXPECT_EQ(gn::edge_betweenness_trace(0, {}, 1).events.size(), 1u);
}
TEST(GirvanNewmanChecked, CanonicalizationBfsOrderAndCompleteDomainsBeforeQuota) {
  const std::vector<Edge> raw{{2, 0}, {0, 2}, {0, 0}, {2, 0}, {0, 1}, {2, 3}, {3, 3}};
  const auto              r = gn::edge_betweenness(4, raw);
  EXPECT_EQ(r.edges, (std::vector<Edge>{{0, 1}, {0, 2}, {2, 3}}));
  EXPECT_EQ(gn::girvan_newman(4, raw, 1), (std::vector<std::vector<std::size_t>>{{0, 1, 2, 3}}));
  EXPECT_EQ(
      gn::girvan_newman(4, {{0, 3}, {3, 1}, {1, 2}}, 1),
      (std::vector<std::vector<std::size_t>>{{0, 3, 1, 2}}));
  EXPECT_THROW(gn::girvan_newman_trace(2, {{0, 1}, {0, 99}}, 1, 1), std::invalid_argument);
  EXPECT_THROW(gn::girvan_newman_trace(0, {}, 2, 1), std::invalid_argument);
  EXPECT_THROW(gn::girvan_newman(3, {}, 0), std::invalid_argument);
  EXPECT_THROW(gn::girvan_newman(3, {}, 4), std::invalid_argument);
  EXPECT_THROW(
      gn::girvan_newman_run(std::size_t(std::numeric_limits<int>::max()) + 1, {}, 1),
      std::length_error);
  for (const auto& bad : std::vector<std::vector<std::vector<std::size_t>>>{
           {{0}}, {{0}, {0, 1}}, {{0, 1}, {}}, {{0}, {2}}})
    EXPECT_THROW(gn::graph_modularity(2, {{0, 1}}, bad), std::invalid_argument);
  EXPECT_EQ(gn::graph_modularity(0, {}, {}), 0);
  EXPECT_EQ(gn::graph_modularity(2, {}, {{0}, {1}}), 0);
}
TEST(GirvanNewmanChecked, NumericIntegerGuaranteeAndNonfiniteGuard) {
  const auto small = gn::layered_graph(60, 2);
  const auto r     = gn::girvan_newman_run(small.vertex_count, small.edges, 2);
  EXPECT_FALSE(r.path_counts_integer_guaranteed);
  EXPECT_TRUE(std::isfinite(r.max_paths));
  EXPECT_GT(r.max_paths, 9007199254740992.0);
  EXPECT_EQ(r.groups.size(), 2u);
  const auto large  = gn::layered_graph(1030, 2);
  const auto before = large.edges;
  EXPECT_THROW(gn::edge_betweenness(large.vertex_count, large.edges), std::overflow_error);
  EXPECT_THROW(gn::girvan_newman(large.vertex_count, large.edges, 2), std::overflow_error);
  EXPECT_EQ(large.edges, before);
}
TEST(GirvanNewmanChecked, GeneratedWholeGraphCountsAndAdmission) {
  const auto a = gn::clique_bridge_graph(3, 4, 2, 7);
  EXPECT_EQ(a.vertex_count, 7u);
  EXPECT_EQ(a.edges.size(), 11u);
  EXPECT_THROW(gn::clique_bridge_graph(3, 4, 2, 6), std::length_error);
  EXPECT_EQ(gn::clique_bridge_graph(3, 4, 0).edges.size(), 9u);
  EXPECT_THROW(gn::clique_bridge_graph(3, 4, 4), std::invalid_argument);
  EXPECT_THROW(gn::clique_bridge_graph(0, 3), std::invalid_argument);
  EXPECT_THROW(gn::layered_graph(3, 2, 5), std::length_error);
  EXPECT_EQ(gn::layered_graph(3, 2, 6).edges.size(), 8u);
  EXPECT_EQ(gn::layered_graph(1, 2).edges.size(), 0u);
  EXPECT_THROW(gn::layered_graph(0, 2), std::invalid_argument);
  EXPECT_THROW(gn::layered_graph(std::numeric_limits<std::size_t>::max(), 2), std::length_error);
  EXPECT_THROW(
      gn::clique_bridge_graph(std::numeric_limits<std::size_t>::max(), 1), std::length_error);
}

TEST(GirvanNewmanChecked, IntegerGuaranteeChecksBeforeRoundedBoundaryAddition) {
  std::vector<Edge> edges;
  std::size_t       n = 1, last = 0;
  for (int stage = 0; stage < 53; ++stage) {
    const auto a = n++, b = n++, c = n++;
    edges.emplace_back(last, a);
    edges.emplace_back(last, b);
    edges.emplace_back(a, c);
    edges.emplace_back(b, c);
    last = c;
  }
  std::size_t previous = 0;
  for (int i = 1; i < 106; ++i) {
    const auto v = n++;
    edges.emplace_back(previous, v);
    previous = v;
  }
  edges.emplace_back(previous, last);
  ASSERT_EQ(n, 265u);
  ASSERT_EQ(edges.size(), 318u);
  std::vector<std::vector<std::size_t>> adj(n);
  for (auto [u, v] : edges) {
    adj[u].push_back(v);
    adj[v].push_back(u);
  }
  std::uint64_t exact_max = 0;
  for (std::size_t source = 0; source < n; ++source) {
    std::vector<int>           distance(n, -1);
    std::vector<std::uint64_t> count(n);
    std::queue<std::size_t>    q;
    distance[source] = 0;
    count[source]    = 1;
    q.push(source);
    while (!q.empty()) {
      const auto u = q.front();
      q.pop();
      for (auto v : adj[u]) {
        if (distance[v] < 0) {
          distance[v] = distance[u] + 1;
          q.push(v);
        }
        if (distance[v] == distance[u] + 1) {
          ASSERT_LE(count[u], std::numeric_limits<std::uint64_t>::max() - count[v]);
          count[v] += count[u];
          exact_max = std::max(exact_max, count[v]);
        }
      }
    }
  }
  EXPECT_EQ(exact_max, (std::uint64_t{1} << 53) + 1);
  const auto actual = gn::edge_betweenness(n, edges);
  EXPECT_EQ(actual.max_paths, 9007199254740992.0);
  EXPECT_FALSE(actual.path_counts_integer_guaranteed);
}
