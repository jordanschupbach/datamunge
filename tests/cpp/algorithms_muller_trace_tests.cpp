#include <gtest/gtest.h>

#include <climits>
#include <cmath>
#include <datamunge/algorithms/muller.hpp>
#include <limits>
#include <stdexcept>
using namespace datamunge::algorithms;
TEST(MullerTrace, ActualQuadraticProposalsCallbackSchedulingAndResidualChecks) {
  for (int target = 1; target <= 100; ++target) {
    std::vector<std::pair<double, double>> calls;
    auto                                   f = [&](double x) {
      double value = x * x - target;
      calls.emplace_back(x, value);
      return value;
    };
    auto t = muller_trace(f, 0, 1, std::sqrt(target) + 1);
    EXPECT_TRUE(t.run.result.converged);
    EXPECT_NEAR(t.run.result.root, std::sqrt(target), 1e-10);
    EXPECT_EQ(t.evaluations.size(), calls.size());
    EXPECT_EQ(t.run.evaluations, 3 + t.run.updates);
    EXPECT_EQ(t.run.inspections, t.steps.size());
    EXPECT_EQ(t.captured_entries, 5 + t.evaluations.size() + t.steps.size());
    for (std::size_t i = 0; i < calls.size(); ++i) {
      EXPECT_EQ(t.evaluations[i].x, calls[i].first);
      EXPECT_EQ(t.evaluations[i].value, calls[i].second);
      EXPECT_EQ(t.evaluations[i].initial, i < 3);
    }
    std::size_t next = 3;
    for (auto& s : t.steps) {
      for (int i = 0; i < 3; ++i)
        EXPECT_EQ(s.f[i], s.x[i] * s.x[i] - target);
      if (!s.quadratic)
        continue;
      long double h0 = static_cast<long double>(s.x[1]) - s.x[0],
                  h1 = static_cast<long double>(s.x[2]) - s.x[1];
      long double d0 = (static_cast<long double>(s.f[1]) - s.f[0]) / h0,
                  d1 = (static_cast<long double>(s.f[2]) - s.f[1]) / h1, a = (d1 - d0) / (h0 + h1),
                  b = a * h1 + d1;
      EXPECT_NEAR(s.a, a, 1e-10);
      EXPECT_NEAR(s.b, b, 1e-10);
      EXPECT_EQ(s.c, s.f[2]);
      EXPECT_NEAR(s.discriminant_raw, b * b - 4 * a * s.f[2], 1e-8);
      if (s.updated) {
        EXPECT_EQ(t.evaluations[next++].x, s.next);
        EXPECT_NEAR(
            s.a * (s.next - s.x[2]) * (s.next - s.x[2]) + s.b * (s.next - s.x[2]) + s.c, 0, 1e-8);
      }
    }
    EXPECT_EQ(next, t.evaluations.size());
    auto ordinary =
        muller([target](double x) { return x * x - target; }, 0, 1, std::sqrt(target) + 1);
    EXPECT_EQ(ordinary.root, t.run.result.root);
    EXPECT_EQ(ordinary.iterations, t.run.result.iterations);
    EXPECT_EQ(ordinary.residual, t.run.result.residual);
  }
}
TEST(MullerTrace, CapsNumericalStopsControlsAndNonfiniteArithmetic) {
  auto linear = [](double x) { return x - 1; };
  auto t      = muller_trace(linear, 0, .5, 1);
  EXPECT_EQ(t.run.evaluations, 3);
  EXPECT_EQ(t.run.updates, 0);
  EXPECT_EQ(t.run.result.iterations, 1);
  EXPECT_EQ(t.captured_entries, 9);
  EXPECT_NO_THROW(muller_trace(linear, 0, .5, 1, 1e-10, 100, 9));
  EXPECT_THROW(muller_trace(linear, 0, .5, 1, 1e-10, 100, 8), std::length_error);
  auto limit = muller_run(linear, 0, .5, 2, 1e-10, 0);
  EXPECT_EQ(limit.stop, MullerStop::IterationLimit);
  EXPECT_EQ(limit.result.iterations, 0);
  EXPECT_EQ(limit.evaluations, 3);
  EXPECT_EQ(muller_run([](double) { return 1.; }, 0, 1, 2).stop, MullerStop::ZeroDenominator);
  EXPECT_EQ(muller_run([](double) { return 1.; }, 0, 0, 1).stop, MullerStop::CoincidentNodes);
  EXPECT_TRUE(muller(linear, 1, 1, 1).converged);
  auto no_real = muller_run([](double x) { return x * x + 1; }, -1, 0, 1);
  EXPECT_FALSE(no_real.result.converged);
  EXPECT_GT(no_real.clamped_discriminants, 0);
  int  count = 0;
  auto f     = [&](double) {
    ++count;
    return 1.;
  };
  EXPECT_THROW(muller(f, 0, 1, 2, -1), std::invalid_argument);
  EXPECT_THROW(muller(f, 0, 1, 2, 1e-10, -1), std::invalid_argument);
  EXPECT_THROW(muller(f, 0, 1, 2, 1e-10, INT_MAX), std::invalid_argument);
  EXPECT_THROW(muller_trace(f, 0, 1, 2, 1e-10, 100, 5), std::length_error);
  EXPECT_EQ(count, 0);
  EXPECT_THROW(muller({}, 0, 1, 2), std::invalid_argument);
  EXPECT_THROW(muller([](double) { return std::nan(""); }, 0, 1, 2), std::domain_error);
  auto max = std::numeric_limits<double>::max();
  EXPECT_THROW(muller([](double) { return 1.; }, -max, 0, max), std::overflow_error);
  EXPECT_FALSE(muller_run(linear, 0, .5, 1, 0).result.converged);
}
