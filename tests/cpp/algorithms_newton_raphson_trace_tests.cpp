#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/algorithms/newton_raphson.hpp>
#include <limits>
#include <stdexcept>
using namespace datamunge::algorithms;
TEST(NewtonRaphsonTrace, ActualCallbackOrderTangentUpdatesAndExactQuotas) {
  for (int a = 1; a <= 20; ++a)
    for (int root = -10; root <= 10; ++root) {
      std::vector<NewtonRaphsonEvaluation> calls;
      auto                                 f = [&](double x) {
        double v = a * (x - root);
        calls.push_back({x, v, false});
        return v;
      };
      auto df = [&](double x) {
        calls.push_back({x, double(a), true});
        return double(a);
      };
      auto t = newton_raphson_trace(f, df, root + 4.);
      EXPECT_EQ(t.run.result.root, root);
      EXPECT_TRUE(t.run.result.converged);
      EXPECT_EQ(t.run.updates, 1);
      EXPECT_EQ(t.run.function_calls, 2);
      EXPECT_EQ(t.run.derivative_calls, 1);
      EXPECT_EQ(t.run.result.iterations, 2);
      EXPECT_EQ(t.evaluations.size(), calls.size());
      for (std::size_t i = 0; i < calls.size(); ++i) {
        EXPECT_EQ(t.evaluations[i].x, calls[i].x);
        EXPECT_EQ(t.evaluations[i].value, calls[i].value);
        EXPECT_EQ(t.evaluations[i].derivative, calls[i].derivative);
      }
      EXPECT_EQ(t.steps[0].next, t.steps[0].x - t.steps[0].fx / t.steps[0].derivative);
      EXPECT_EQ(t.captured_entries, 3 + t.evaluations.size() + t.steps.size());
      EXPECT_NO_THROW(newton_raphson_trace(f, df, root + 4., 1e-10, 100, t.captured_entries));
      EXPECT_THROW(
          newton_raphson_trace(f, df, root + 4., 1e-10, 100, t.captured_entries - 1),
          std::length_error);
    }
  auto s = newton_raphson_square_study();
  EXPECT_EQ(s.trace.run.updates, 4);
  EXPECT_EQ(s.states.size(), 5);
  EXPECT_EQ(
      s.states,
      (std::vector<double>{2., 1.5, 1.4166666666666667, 1.4142156862745099, 1.4142135623746899}));
  for (std::size_t i = 0; i < s.tangent_x.size(); ++i) {
    EXPECT_EQ(s.tangent_x[i], s.trace.steps[i].x);
    EXPECT_EQ(s.tangent_next[i], s.trace.steps[i].next);
  }
  EXPECT_GT(s.reference_errors.back(), 1e-13);
}
TEST(NewtonRaphsonTrace, NumericalStopsStrictToleranceCapAndCompleteGuards) {
  auto f    = [](double x) { return x * x - 2; };
  auto df   = [](double x) { return 2 * x; };
  auto flat = newton_raphson_run(f, df, 0);
  EXPECT_EQ(flat.stop, NewtonRaphsonStop::ZeroDerivative);
  EXPECT_EQ(flat.derivative_calls, 1);
  EXPECT_EQ(flat.updates, 0);
  auto cap = newton_raphson_trace(f, df, 2, 1e-10, 0);
  EXPECT_EQ(cap.run.function_calls, 1);
  EXPECT_EQ(cap.run.derivative_calls, 0);
  EXPECT_EQ(cap.run.result.iterations, 0);
  EXPECT_TRUE(cap.steps[0].at_limit);
  EXPECT_EQ(cap.run.stop, NewtonRaphsonStop::IterationLimit);
  auto initial = newton_raphson_run([](double x) { return x - 1; }, [](double) { return 1.; }, 1);
  EXPECT_EQ(initial.result.iterations, 1);
  EXPECT_EQ(initial.updates, 0);
  auto strict =
      newton_raphson_run([](double x) { return x - 1; }, [](double) { return 1.; }, 1, 0, 4);
  EXPECT_FALSE(strict.result.converged);
  EXPECT_EQ(strict.updates, 4);
  auto cycle = newton_raphson_trace(
      [](double x) { return x * x * x - 2 * x + 2; },
      [](double x) { return 3 * x * x - 2; },
      0,
      1e-10,
      8);
  EXPECT_EQ(cycle.steps.size(), 9);
  for (std::size_t i = 0; i < cycle.steps.size(); ++i)
    EXPECT_EQ(cycle.steps[i].x, i % 2);
  EXPECT_FALSE(cycle.run.result.converged);
  std::size_t calls = 0;
  auto        count = [&](double) {
    ++calls;
    return 1.;
  };
  EXPECT_THROW(newton_raphson(count, {}, 0), std::invalid_argument);
  EXPECT_THROW(newton_raphson(count, count, NAN), std::invalid_argument);
  EXPECT_THROW(newton_raphson(count, count, 0, -1), std::invalid_argument);
  EXPECT_THROW(newton_raphson(count, count, 0, 1, -1), std::invalid_argument);
  EXPECT_THROW(newton_raphson_trace(count, count, 0, 1, 1, 2), std::length_error);
  EXPECT_EQ(calls, 0);
  EXPECT_THROW(newton_raphson([](double) { return NAN; }, count, 0), std::domain_error);
  EXPECT_THROW(newton_raphson(count, [](double) { return INFINITY; }, 0), std::domain_error);
  EXPECT_THROW(
      newton_raphson(
          [](double) { return std::numeric_limits<double>::max(); },
          [](double) { return std::numeric_limits<double>::denorm_min(); },
          0),
      std::overflow_error);
  EXPECT_THROW(
      newton_raphson([](double) -> double { throw std::runtime_error("caller"); }, count, 0),
      std::runtime_error);
}
