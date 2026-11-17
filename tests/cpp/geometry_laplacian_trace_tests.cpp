#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/geometry/laplacian_analysis.hpp>
#include <limits>
using namespace datamunge::geometry;
namespace {
void verify(const LaplacianTrace& t) {
  const auto  n = t.result.size();
  std::size_t E = 0, R = 0, pinned = 0, isolated = 0, reads = 0;
  for (std::size_t i = 0; i < n; ++i) {
    E += t.neighbors[i].size();
    if (t.fixed[i])
      ++pinned;
    else if (t.neighbors[i].empty())
      ++isolated;
    else {
      ++R;
      reads += t.neighbors[i].size();
    }
  }
  const auto I = t.states.size() - 1;
  EXPECT_EQ(t.energies.size(), I + 1);
  EXPECT_EQ(t.updates.size(), I * R);
  EXPECT_EQ(t.captured_entries, 2 * n + E + 1 + I * (n + 1 + R));
  EXPECT_EQ(t.work.sweeps, I);
  EXPECT_EQ(t.work.updated_vertices, I * R);
  EXPECT_EQ(t.work.pinned_skips, I * pinned);
  EXPECT_EQ(t.work.isolated_skips, I * isolated);
  EXPECT_EQ(t.work.neighbor_reads, I * reads);
  for (std::size_t pass = 0; pass < I; ++pass) {
    const auto& before = t.states[pass];
    const auto& after  = t.states[pass + 1];
    for (std::size_t i = 0; i < n; ++i) {
      if (t.fixed[i] || t.neighbors[i].empty()) {
        EXPECT_EQ(after[i], before[i]);
        continue;
      }
      long double cx = 0, cy = 0;
      for (int j : t.neighbors[i]) {
        cx += before[j].x;
        cy += before[j].y;
      }
      cx /= t.neighbors[i].size();
      cy /= t.neighbors[i].size();
      EXPECT_NEAR(
          after[i].x, static_cast<double>((1 - t.lambda) * before[i].x + t.lambda * cx), 1e-12);
      EXPECT_NEAR(
          after[i].y, static_cast<double>((1 - t.lambda) * before[i].y + t.lambda * cy), 1e-12);
    }
    EXPECT_LE(t.energies[pass + 1], t.energies[pass] + 1e-10L * std::max(1.L, t.energies[pass]));
  }
  for (const auto& e : t.updates) {
    EXPECT_EQ(e.before, t.states[e.iteration][e.vertex]);
    EXPECT_EQ(e.after, t.states[e.iteration + 1][e.vertex]);
    EXPECT_EQ(e.neighbor_count, t.neighbors[e.vertex].size());
  }
  EXPECT_EQ(t.result, t.states.back());
}
} // namespace
TEST(LaplacianTrace, AllFourVertexGraphsMasksAndRelaxationsAgainstIndependentMatrixUpdate) {
  const std::vector<Point2D> v = {{0, 0}, {2, 5}, {4, -1}, {-2, 3}};
  for (int mask = 0; mask < 64; ++mask) {
    std::vector<std::vector<int>> a(4);
    int                           bit = 0;
    for (int i = 0; i < 4; ++i)
      for (int j = i + 1; j < 4; ++j, ++bit)
        if (mask & (1 << bit)) {
          a[i].push_back(j);
          a[j].push_back(i);
        }
    for (int pins = 0; pins < 16; ++pins) {
      std::vector<char> fixed(4);
      for (int i = 0; i < 4; ++i)
        fixed[i] = bool(pins & (1 << i));
      for (double lambda : {0., .25, .5, 1.})
        verify(laplacian_smooth_trace(v, a, fixed, lambda, 4));
    }
  }
}
TEST(LaplacianTrace, TinyPinnedEnergyAndJacobiOldState) {
  const auto t =
      laplacian_smooth_trace({{0, 0}, {2, 5}, {4, 0}}, {{1}, {0, 2}, {1}}, {1, 0, 1}, 1, 1);
  verify(t);
  ASSERT_EQ(t.updates.size(), 1U);
  EXPECT_EQ(t.updates[0].centroid, (Point2D{2, 0}));
  EXPECT_EQ(t.energies, (std::vector<long double>{58, 8}));
  EXPECT_EQ(t.captured_entries, 16U);
  const auto all = laplacian_smooth({{0, 0}, {2, 5}, {4, 0}}, {{1}, {0, 2}, {1}}, {}, 1, 1);
  EXPECT_EQ(all, (std::vector<Point2D>{{2, 5}, {2, 0}, {2, 5}}));
}
TEST(LaplacianTrace, BipartiteOscillationAndOriginalOversmoothing) {
  const auto oscillation = laplacian_smooth_trace({{0, 0}, {2, 0}}, {{1}, {0}}, {}, 1, 4);
  verify(oscillation);
  for (std::size_t i = 0; i < oscillation.states.size(); ++i) {
    EXPECT_EQ(oscillation.states[i][0].x, i % 2 ? 2 : 0);
    EXPECT_EQ(oscillation.energies[i], 4);
  }
  const auto collapsed = laplacian_smooth({{0, 0}, {2, 0}}, {{1}, {0}}, {}, .5, 1);
  EXPECT_EQ(collapsed, (std::vector<Point2D>{{1, 0}, {1, 0}}));
  const auto d = laplacian_demo_curve();
  const auto t = laplacian_smooth_trace(d.vertices, d.neighbors, d.fixed, .5, 2000);
  EXPECT_EQ(t.result.front(), d.vertices.front());
  EXPECT_EQ(t.result.back(), d.vertices.back());
  for (std::size_t i = 1; i < t.energies.size(); ++i)
    EXPECT_LE(t.energies[i], t.energies[i - 1] + 1e-9L);
  EXPECT_GT(
      laplacian_point_mse(t.states.back(), d.reference),
      laplacian_point_mse(t.states[40], d.reference));
}
TEST(LaplacianTrace, CompleteGraphFlagParameterQuotaAndOwnershipGuards) {
  EXPECT_THROW((void)laplacian_smooth({{0, 0}}, {}, {}), std::invalid_argument);
  EXPECT_THROW((void)laplacian_smooth({{0, 0}}, {{1}}, {}), std::invalid_argument);
  EXPECT_THROW((void)dirichlet_energy({{0, 0}, {1, 1}}, {{1}, {}}), std::invalid_argument);
  EXPECT_THROW((void)dirichlet_energy({{0, 0}, {1, 1}}, {{1, 1}, {0}}), std::invalid_argument);
  EXPECT_THROW((void)laplacian_smooth({{0, 0}}, {{}}, {0, 0}), std::invalid_argument);
  EXPECT_THROW((void)laplacian_smooth({{0, 0}}, {{}}, {}, 1.1), std::invalid_argument);
  EXPECT_THROW((void)laplacian_smooth({{0, 0}}, {{}}, {}, .5, -1), std::invalid_argument);
  EXPECT_THROW(
      (void)laplacian_smooth({{std::numeric_limits<double>::quiet_NaN(), 0}}, {{}}, {}),
      std::invalid_argument);
  const auto t =
      laplacian_smooth_trace({{0, 0}, {2, 5}, {4, 0}}, {{1}, {0, 2}, {1}}, {1, 0, 1}, 1, 1);
  EXPECT_NO_THROW((void)laplacian_smooth_trace(t.states[0], t.neighbors, t.fixed, 1, 1, 16));
  EXPECT_THROW(
      (void)laplacian_smooth_trace(t.states[0], t.neighbors, t.fixed, 1, 1, 15), std::length_error);
  EXPECT_NO_THROW((void)laplacian_smooth_trace(t.states[0], t.neighbors, t.fixed, 1, 1, 0));
  auto c               = t;
  c.states[0][0].x     = 99;
  c.result[0].x        = 99;
  c.updates[0].after.x = 99;
  c.neighbors[0].clear();
  EXPECT_EQ(t.states[0][0].x, 0);
  EXPECT_EQ(t.result[0].x, 0);
  EXPECT_EQ(t.updates[0].after.x, 2);
  verify(laplacian_smooth_trace({{1, 2}}, {{}}, {}));
  verify(laplacian_smooth_trace({}, {}, {}, .5, 3));
}
TEST(LaplacianTrace, ReciprocalLoopsMultiplicityAndFiniteCoordinateFallbacks) {
  verify(laplacian_smooth_trace({{0, 0}, {2, 2}}, {{0, 1, 1}, {0, 0, 1}}, {}, .5, 4));
  const double m     = std::numeric_limits<double>::max();
  const auto   state = laplacian_smooth({{-m, 0}, {m, 0}}, {{1}, {0}}, {}, .5, 1);
  EXPECT_EQ(state, (std::vector<Point2D>{{0, 0}, {0, 0}}));
  EXPECT_THROW((void)dirichlet_energy({{-m, 0}, {m, 0}}, {{1}, {0}}), std::overflow_error);
  const auto large =
      laplacian_smooth_trace({{m, 0}, {m, 0}, {m, 0}}, {{1, 2}, {0, 2}, {0, 1}}, {}, .5, 1);
  EXPECT_TRUE(large.updates[0].widened_centroid);
  EXPECT_EQ(large.result[0].x, m);
  EXPECT_EQ(large.energies.back(), 0);
  EXPECT_EQ(laplacian_point_mse({{0, 0}}, {{3, 4}}), 25);
}
