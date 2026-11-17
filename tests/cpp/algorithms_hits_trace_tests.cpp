#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/hits.hpp>
#include <limits>
using namespace datamunge::algorithms;
TEST(HitsTrace, ExhaustiveDenseOracle) {
  for (unsigned mask = 0; mask < 512; ++mask) {
    std::vector<HitsEdge> edges;
    double                matrix[3][3]{};
    for (unsigned u = 0; u < 3; ++u)
      for (unsigned v = 0; v < 3; ++v)
        if (mask & (1u << (3 * u + v))) {
          edges.emplace_back(u, v);
          matrix[u][v] += 1;
        }
    const auto t      = hits_trace(3, edges, 1e-15, 12, {0, 0, 0});
    const auto scalar = hits(3, edges, 1e-15, 12);
    EXPECT_EQ(scalar.hubs, t.run.result.hubs);
    EXPECT_EQ(scalar.authorities, t.run.result.authorities);
    EXPECT_EQ(scalar.iterations, t.steps.size());
    std::vector<double> h(3, 1 / std::sqrt(3.0)), a = h;
    std::size_t         divisions = 0;
    for (const auto& step : t.steps) {
      std::vector<double> nexta(3), nexth(3);
      for (unsigned v = 0; v < 3; ++v)
        for (unsigned u = 0; u < 3; ++u)
          nexta[v] += matrix[u][v] * h[u];
      for (unsigned v = 0; v < 3; ++v)
        EXPECT_NEAR(nexta[v], step.raw_authorities[v], 1e-15);
      double an = 0;
      for (double x : nexta)
        an += x * x;
      an = std::sqrt(an);
      EXPECT_NEAR(an, step.authority_norm, 1e-15);
      if (an) {
        for (auto& x : nexta)
          x /= an;
        divisions += 3;
      }
      for (unsigned u = 0; u < 3; ++u)
        for (unsigned v = 0; v < 3; ++v)
          nexth[u] += matrix[u][v] * nexta[v];
      for (unsigned u = 0; u < 3; ++u)
        EXPECT_NEAR(nexth[u], step.raw_hubs[u], 2e-15);
      double hn = 0;
      for (double x : nexth)
        hn += x * x;
      hn = std::sqrt(hn);
      EXPECT_NEAR(hn, step.hub_norm, 2e-15);
      if (hn) {
        for (auto& x : nexth)
          x /= hn;
        divisions += 3;
      }
      double change = 0;
      for (unsigned i = 0; i < 3; ++i) {
        change = std::max({change, std::abs(nexta[i] - a[i]), std::abs(nexth[i] - h[i])});
        EXPECT_NEAR(nexta[i], step.authorities[i], 2e-15);
        EXPECT_NEAR(nexth[i], step.hubs[i], 2e-15);
      }
      EXPECT_NEAR(change, step.change, 2e-15);
      a = nexta;
      h = nexth;
    }
    const auto k = t.steps.size();
    EXPECT_EQ(t.captured_entries, 6 + 13 * k);
    EXPECT_EQ(t.run.work.authority_edges, edges.size() * k);
    EXPECT_EQ(t.run.work.hub_edges, edges.size() * k);
    EXPECT_EQ(t.run.work.norm_values, 6 * k);
    EXPECT_EQ(t.run.work.divisions, divisions);
    EXPECT_EQ(t.run.work.change_vertices, 3 * k);
  }
}
TEST(HitsTrace, DomainsAndAdmission) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(hits(2, {}, nan), std::invalid_argument);
  EXPECT_THROW(hits(2, {}, std::numeric_limits<double>::infinity()), std::invalid_argument);
  EXPECT_THROW(hits(2, {}, 0), std::invalid_argument);
  EXPECT_THROW(hits_run(3, {{0, 3}}, 1e-12, 1, {1, 1, 1}), std::invalid_argument);
  EXPECT_THROW(hits_run(3, {}, 1e-12, 1, {2, 0, 0}), std::length_error);
  EXPECT_THROW(hits_run(2, {{0, 1}, {0, 1}}, 1e-12, 1, {0, 1, 0}), std::length_error);
  EXPECT_THROW(hits_trace(2, {{0, 1}}, 1e-12, 0, {0, 0, 3}), std::length_error);
  EXPECT_THROW(hits_trace(2, {{0, 1}}, 1e-12, 1, {0, 0, 12}), std::length_error);
  EXPECT_EQ(hits_trace(2, {{0, 1}}, 1e-12, 1, {2, 1, 13}).captured_entries, 13);
  EXPECT_EQ(hits_run(2, {{0, 1}}, 1e-12, 1, {2, 1, 1}).result.iterations, 1);
  auto owned            = hits_trace(2, {{0, 1}}, 1e-12, 1);
  auto copy             = owned;
  copy.steps[0].hubs[0] = 42;
  EXPECT_EQ(owned.steps[0].hubs[0], 1);
}
TEST(HitsTrace, DegenerateAndAnalyticGraphs) {
  EXPECT_EQ(hits(0, {}).iterations, 1);
  EXPECT_EQ(hits(0, {}, 1e-12, 0).iterations, 0);
  auto z = hits_run(4, {});
  EXPECT_EQ(z.result.iterations, 2);
  EXPECT_EQ(z.result.hubs, std::vector<double>(4, 0));
  auto init = hits_run(4, {}, 1e-12, 0);
  EXPECT_FALSE(init.has_step);
  EXPECT_EQ(init.result.hubs, std::vector<double>(4, .5));
  EXPECT_EQ(init.stop, HitsStop::IterationLimit);
  const double phi = (1 + std::sqrt(5.0)) / 2, scale = std::sqrt(phi * phi + 1);
  auto         r = hits(4, {{0, 2}, {0, 3}, {1, 2}});
  EXPECT_NEAR(r.hubs[0], phi / scale, 1e-12);
  EXPECT_NEAR(r.hubs[1], 1 / scale, 1e-12);
  EXPECT_NEAR(r.authorities[2], phi / scale, 1e-12);
  EXPECT_NEAR(r.authorities[3], 1 / scale, 1e-12);
  auto repeated = hits(4, {{0, 2}, {0, 2}, {0, 3}, {1, 2}});
  EXPECT_GT(repeated.hubs[0], r.hubs[0]);
  auto ties = hits(4, {{0, 2}, {1, 3}});
  EXPECT_DOUBLE_EQ(ties.hubs[0], ties.hubs[1]);
  auto loop = hits(1, {{0, 0}});
  EXPECT_EQ(loop.hubs[0], 1);
  EXPECT_EQ(loop.iterations, 1);
}
