#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/algorithms/halley.hpp>
#include <limits>
#include <vector>
using namespace datamunge::algorithms;
namespace {
auto f  = [](double x) { return x * x - 2.; };
auto df = [](double x) { return 2 * x; };
auto d2 = [](double) { return 2.; };
} // namespace
TEST(HalleyTrace, ActualCallbacksUpdatesAndAllBudgets) {
  for (double x0 : {.1, .5, 1., 2., 10., 100., -2.})
    for (int budget = 0; budget < 10; ++budget) {
      std::vector<double> visited;
      int                 d = 0, d2calls = 0;
      auto                t = halley_trace(
          [&](double x) {
            visited.push_back(x);
            return f(x);
          },
          [&](double x) {
            ++d;
            return df(x);
          },
          [&](double x) {
            ++d2calls;
            return d2(x);
          },
          x0,
          1e-12,
          budget);
      const auto scalar = halley(f, df, d2, x0, 1e-12, budget);
      const auto run    = halley_run(f, df, d2, x0, 1e-12, budget);
      EXPECT_EQ(t.run.result.root, scalar.root);
      EXPECT_EQ(t.run.result.residual, scalar.residual);
      EXPECT_EQ(t.run.result.iterations, scalar.iterations);
      EXPECT_EQ(t.run.result.converged, scalar.converged);
      EXPECT_EQ(t.run.result.root, run.result.root);
      EXPECT_EQ(t.run.work.f_calls, visited.size());
      EXPECT_EQ(t.steps.size(), visited.size());
      EXPECT_EQ(t.run.work.df_calls, d);
      EXPECT_EQ(t.run.work.d2f_calls, d2calls);
      std::size_t updates = 0;
      for (std::size_t i = 0; i < t.steps.size(); ++i) {
        const auto& s = t.steps[i];
        EXPECT_EQ(s.x, visited[i]);
        EXPECT_EQ(s.fx, f(s.x));
        EXPECT_EQ(s.inspection, i + 1);
        if (s.derivatives_evaluated) {
          EXPECT_EQ(s.derivative, df(s.x));
          EXPECT_EQ(s.second_derivative, 2.);
          EXPECT_EQ(s.denominator, 2 * df(s.x) * df(s.x) - f(s.x) * 2);
        }
        if (s.updated) {
          ++updates;
          EXPECT_EQ(s.next, s.x - 2 * f(s.x) * df(s.x) / s.denominator);
          ASSERT_LT(i + 1, t.steps.size());
          EXPECT_EQ(s.next, t.steps[i + 1].x);
        } else
          EXPECT_EQ(s.next, s.x);
      }
      EXPECT_EQ(updates, t.run.work.updates);
      auto copy       = t;
      copy.steps[0].x = 999;
      EXPECT_EQ(t.steps[0].x, x0);
      EXPECT_NO_THROW(halley_trace(f, df, d2, x0, 1e-12, budget, t.steps.size()));
      if (t.steps.size() > 1)
        EXPECT_THROW(
            halley_trace(f, df, d2, x0, 1e-12, budget, t.steps.size() - 1), std::length_error);
    }
}
TEST(HalleyTrace, StoppingAndMultiplicity) {
  auto t = halley_trace(f, df, d2, 2.);
  EXPECT_TRUE(t.run.result.converged);
  EXPECT_EQ(t.run.result.iterations, 4);
  EXPECT_EQ(t.run.work.updates, 3);
  EXPECT_EQ(t.steps.size(), 4);
  auto exact = halley_run(f, df, d2, 2., 1., 0);
  EXPECT_FALSE(exact.result.converged);
  EXPECT_EQ(exact.work.f_calls, 1);
  EXPECT_EQ(exact.result.iterations, 0);
  auto zero = halley_run([](double) { return 0.; }, df, d2, 2., 0, 2);
  EXPECT_EQ(zero.stop, HalleyStop::IterationLimit);
  EXPECT_FALSE(zero.result.converged);
  EXPECT_EQ(zero.work.updates, 2);
  auto flat = halley_run(
      [](double) { return 1.; }, [](double) { return 0.; }, [](double) { return 0.; }, 0.);
  EXPECT_EQ(flat.stop, HalleyStop::ZeroDenominator);
  EXPECT_EQ(flat.work.updates, 0);
  auto stuck = halley_trace(
      [](double) { return 1.; },
      [](double) { return 0.; },
      [](double) { return 2.; },
      0.,
      1e-10,
      3);
  EXPECT_EQ(stuck.run.stop, HalleyStop::IterationLimit);
  EXPECT_EQ(stuck.run.work.updates, 3);
  EXPECT_EQ(stuck.run.result.root, 0.);
  auto repeated = halley_trace(
      [](double x) { return x * x * x; },
      [](double x) { return 3 * x * x; },
      [](double x) { return 6 * x; },
      1.,
      0.,
      8);
  for (const auto& s : repeated.steps)
    EXPECT_EQ(s.x, std::ldexp(1., 1 - s.inspection));
}
TEST(HalleyTrace, OnlineAdmissionDomainsAndNumericalErrors) {
  int  calls   = 0;
  auto counted = [&](double x) {
    ++calls;
    return f(x);
  };
  EXPECT_THROW(halley_trace(counted, df, d2, 2., 1e-10, 10, 1), std::length_error);
  EXPECT_EQ(calls, 1);
  calls = 0;
  EXPECT_THROW(halley_trace(counted, df, d2, 2., -1, 10, 1), std::invalid_argument);
  EXPECT_EQ(calls, 0);
  EXPECT_THROW(halley(f, df, d2, 2., 1e-10, -1), std::invalid_argument);
  EXPECT_THROW(
      halley(f, df, d2, 2., 1e-10, std::numeric_limits<int>::max()), std::invalid_argument);
  EXPECT_THROW(halley({}, df, d2, 2.), std::invalid_argument);
  EXPECT_THROW(halley(f, df, d2, std::numeric_limits<double>::infinity()), std::invalid_argument);
  EXPECT_THROW(
      halley([](double) { return std::numeric_limits<double>::quiet_NaN(); }, df, d2, 2.),
      std::domain_error);
  EXPECT_THROW(halley(f, [](double) { return 1e308; }, d2, 2.), std::overflow_error);
  EXPECT_THROW(
      halley([](double) -> double { throw std::runtime_error("callback"); }, df, d2, 2.),
      std::runtime_error);
  EXPECT_NO_THROW(halley_trace(f, df, d2, 2., 1e-10, 100, 0));
}
