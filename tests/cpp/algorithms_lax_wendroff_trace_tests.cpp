#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/algorithms/pde_solvers.hpp>
#include <limits>
#include <numbers>
using namespace datamunge::algorithms;
TEST(LaxWendroffTrace, ActualImpulseStencilsAndCapture) {
  auto t = lax_wendroff_trace({0, 0, 1, 0, 0}, 1, 1, .5, 1, false, 17);
  EXPECT_EQ(t.analysis.values, (std::vector<double>{0, -.125, .75, .375, 0}));
  EXPECT_EQ(t.captured_entries, 17);
  EXPECT_EQ(t.states.size(), 2);
  EXPECT_EQ(t.cells.size(), 5);
  for (auto& c : t.cells) {
    EXPECT_EQ(c.left, t.states[0][c.left_index]);
    EXPECT_EQ(c.center, t.states[0][c.index]);
    EXPECT_EQ(c.right, t.states[0][c.right_index]);
    EXPECT_EQ(c.value, t.states[1][c.index]);
    EXPECT_EQ(c.value, c.center - c.advective_term + c.diffusive_term);
  }
  EXPECT_EQ(t.diagnostics.back().mass, 1);
  EXPECT_LT(t.diagnostics.back().l2_norm, 1);
  EXPECT_EQ(t.analysis.work.cell_updates, 5);
  EXPECT_EQ(t.analysis.work.stencil_reads, 15);
  EXPECT_EQ(t.analysis.work.diagnostics_cells, 10);
  EXPECT_THROW(lax_wendroff_trace({0, 0, 1, 0, 0}, 1, 1, .5, 1, false, 16), std::length_error);
}
TEST(LaxWendroffTrace, ExactCopyBranchesAndWideFallback) {
  const double m = std::numeric_limits<double>::max();
  EXPECT_EQ(lax_wendroff_advection({1, 2, 3}, 1, 1, 1, 1), (std::vector<double>{3, 1, 2}));
  EXPECT_EQ(lax_wendroff_advection({1, 2, 3}, -1, 1, 1, 1), (std::vector<double>{2, 3, 1}));
  EXPECT_EQ(lax_wendroff_advection({m, -m}, 0, 1, 1, 2), (std::vector<double>{m, -m}));
  auto t = lax_wendroff_trace({m, m, m}, 1, 1, .8, 1);
  EXPECT_EQ(t.analysis.values, (std::vector<double>{m, m, m}));
  EXPECT_EQ(t.analysis.work.widened_updates, 3);
  EXPECT_EQ(lax_wendroff_advection({3}, 1, 1, .5, 3), (std::vector<double>{3}));
  EXPECT_EQ(lax_wendroff_analyze({1, 2}, 0, 1, 1, 2).work.exact_copies, 4);
}
TEST(LaxWendroffTrace, FourierModeAndExplicitInstability) {
  for (double nu : {.8, 1.2}) {
    std::vector<double> u(32);
    for (std::size_t j = 0; j < u.size(); ++j)
      u[j] = j % 2 ? -1 : 1;
    auto       t = lax_wendroff_trace(u, 1, 1, nu, 12, true);
    const auto g = lax_wendroff_amplification(nu, std::numbers::pi);
    for (std::size_t k = 0; k < t.states.size(); ++k) {
      EXPECT_NEAR(t.states[k][0], std::pow(g.real(), k), 1e-10);
      EXPECT_NEAR(
          double(t.diagnostics[k].l2_norm / t.diagnostics[0].l2_norm),
          std::pow(std::abs(g), k),
          1e-10);
    }
  }
  EXPECT_THROW(lax_wendroff_advection({1, -1}, 1, 1, 1.2, 1), std::invalid_argument);
  for (double nu : {-1., -.8, 0., .3, 1.})
    for (int k = 0; k <= 100; ++k) {
      double theta = k * std::numbers::pi / 100;
      auto   g     = lax_wendroff_amplification(nu, theta);
      EXPECT_NEAR(
          std::norm(g), 1 - 4 * nu * nu * (1 - nu * nu) * std::pow(std::sin(theta / 2), 4), 2e-15);
    }
}
TEST(LaxWendroffTrace, SmoothRefinementAndOriginalGaussian) {
  double previous = 0;
  for (int n : {40, 80, 160, 320}) {
    double              dx = 1. / n, dt = .8 * dx;
    int                 steps = 5 * n / 4;
    std::vector<double> u(n);
    for (int j = 0; j < n; ++j)
      u[j] = std::sin(2 * std::numbers::pi * j * dx);
    auto   a = lax_wendroff_analyze(u, 1, dx, dt, steps);
    double e = 0;
    for (int j = 0; j < n; ++j)
      e = std::max(
          e, std::abs(a.values[j] - std::sin(2 * std::numbers::pi * (j * dx - steps * dt))));
    if (previous)
      EXPECT_GT(previous / e, 3.8);
    previous = e;
    EXPECT_EQ(a.work.cell_updates, std::size_t(n) * steps);
  }
  std::vector<double> u(200);
  for (int j = 0; j < 200; ++j)
    u[j] = std::exp(-120 * std::pow(j / 200. - .5, 2));
  auto t = lax_wendroff_trace(u, 1, .005, .004, 250);
  EXPECT_EQ(t.captured_entries, 100451);
  EXPECT_NEAR(double(t.diagnostics.back().mass - t.diagnostics.front().mass), 0, 1e-14);
}
TEST(LaxWendroffTrace, GuardsZeroStepsAndOwnership) {
  double q = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(lax_wendroff_advection({}, 1, 1, .5, 0), std::invalid_argument);
  EXPECT_THROW(lax_wendroff_advection({1, q}, 1, 1, .5, 0), std::invalid_argument);
  EXPECT_THROW(lax_wendroff_advection({1}, q, 1, .5, 0), std::invalid_argument);
  EXPECT_THROW(lax_wendroff_advection({1}, 1, 0, .5, 0), std::invalid_argument);
  EXPECT_THROW(lax_wendroff_advection({1}, 1, 1, -.5, 0), std::invalid_argument);
  EXPECT_THROW(lax_wendroff_advection({1}, 1, 1, .5, -1), std::invalid_argument);
  EXPECT_THROW(lax_wendroff_amplification(q, 0), std::invalid_argument);
  std::vector<double> u{1, 2};
  auto                t = lax_wendroff_trace(u, 1, 1, .5, 0, false, 0);
  u[0]                  = 9;
  EXPECT_EQ(t.states[0][0], 1);
  EXPECT_EQ(t.analysis.values, (std::vector<double>{1, 2}));
  EXPECT_EQ(t.analysis.work.cell_updates, 0);
}
