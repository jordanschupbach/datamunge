#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/ode/ode_solver.hpp>
#include <datamunge/ode/study.hpp>
#include <limits>
#include <stdexcept>
using namespace datamunge::ode;
namespace {
struct Decay : RHS {
  std::size_t         calls{0};
  std::vector<double> evaluate(double, const std::vector<double>& y) override {
    ++calls;
    return {-y[0]};
  }
};
struct Bad : RHS {
  std::size_t         calls{0};
  bool                shape{true};
  std::vector<double> evaluate(double, const std::vector<double>&) override {
    ++calls;
    return shape ? std::vector<double>{1, 2} : std::vector<double>{NAN};
  }
};
std::size_t units(const ODETrace& t) {
  auto d = t.initial_state.size();
  return 8 + d + t.solution.size() * (1 + d) + t.evaluations.size() * (2 + 2 * d) +
         t.attempts.size() * (1 + 2 * d);
}
} // namespace
TEST(ODETrace, ActualRK4StagesAndCompleteQuota) {
  Decay     rhs;
  ODESolver solver({StepMethod::RK4, 4, .5});
  auto      t = solver.trace(rhs, {1.}, 0, .5);
  ASSERT_EQ(t.evaluations.size(), 4);
  EXPECT_EQ(t.evaluations[0].time, 0);
  EXPECT_EQ(t.evaluations[0].state[0], 1);
  EXPECT_EQ(t.evaluations[1].time, .25);
  EXPECT_EQ(t.evaluations[1].state[0], .75);
  EXPECT_EQ(t.evaluations[2].state[0], .8125);
  EXPECT_EQ(t.evaluations[3].time, .5);
  EXPECT_EQ(t.evaluations[3].state[0], .59375);
  EXPECT_EQ(t.solution.y.back()[0], 1. + (.5 / 6) * (-1. + 2 * (-.75) + 2 * (-.8125) - .59375));
  EXPECT_EQ(t.attempts.size(), 1);
  EXPECT_TRUE(t.attempts[0].accepted);
  EXPECT_EQ(t.attempts[0].evaluation_end - t.attempts[0].evaluation_begin, 4);
  EXPECT_EQ(t.captured_entries, units(t));
  ODEExecutionLimits limits;
  limits.max_trace_entries = t.captured_entries;
  EXPECT_NO_THROW(solver.trace(rhs, {1.}, 0, .5, limits));
  --limits.max_trace_entries;
  EXPECT_THROW(solver.trace(rhs, {1.}, 0, .5, limits), std::length_error);
}
TEST(ODETrace, ActualAdaptiveAcceptedRejectedCallbacksAndAnalyticalDecay) {
  ODEOptions options{StepMethod::RK45, 4, 2., 1e-12, 1e-12, 0};
  ODESolver  solver(options);
  Decay      rhs;
  auto       t = solver.trace(rhs, {1.}, 0, 2);
  EXPECT_GT(t.attempts.size(), t.solution.steps_taken);
  EXPECT_EQ(t.evaluations.size(), 7 * t.attempts.size());
  EXPECT_EQ(t.solution.function_evaluations, rhs.calls);
  EXPECT_EQ(t.solution.t.back(), 2);
  EXPECT_NEAR(t.solution.y.back()[0], std::exp(-2.), 1e-9);
  std::size_t accepted = 0;
  for (auto& a : t.attempts) {
    EXPECT_EQ(a.kind, ODEAttemptKind::rk45_trial);
    EXPECT_EQ(a.evaluation_end - a.evaluation_begin, 7);
    EXPECT_EQ(a.accepted, a.error_norm <= 1.);
    EXPECT_EQ(t.evaluations[a.evaluation_begin].time, a.time_before);
    EXPECT_EQ(t.evaluations[a.evaluation_begin].state, a.before);
    if (a.accepted) {
      ++accepted;
      EXPECT_GT(a.time_after, a.time_before);
    } else
      EXPECT_EQ(a.time_after, a.time_before);
  }
  EXPECT_EQ(accepted, t.solution.steps_taken);
  for (auto& e : t.evaluations) {
    ASSERT_EQ(e.derivative.size(), 1);
    EXPECT_EQ(e.derivative[0], -e.state[0]);
  }
  EXPECT_EQ(t.captured_entries, units(t));
}
TEST(ODETrace, AllSixOriginalMethodsOrdersStudiesAndShortenedAdamsTail) {
  Decay rhs;
  for (auto method :
       {StepMethod::Euler,
        StepMethod::Midpoint,
        StepMethod::RK4,
        StepMethod::RK45,
        StepMethod::AdamsBashforth,
        StepMethod::AdamsMoulton}) {
    ODESolver solver({method, 4, .01});
    auto      t = solver.trace(rhs, {1}, 0, .2);
    EXPECT_EQ(t.solution.t.front(), 0);
    EXPECT_EQ(t.solution.t.back(), .2);
    EXPECT_EQ(t.solution.steps_taken, t.solution.size() - 1);
    EXPECT_EQ(t.captured_entries, units(t));
    EXPECT_EQ(solver.solve_builtin("exponential_decay", {1}, {1}, 0, .2).y, t.solution.y);
  }
  for (auto method : {StepMethod::AdamsBashforth, StepMethod::AdamsMoulton})
    for (std::size_t order : {0, 2, 3, 4, 99}) {
      ODESolver solver({method, order, .3});
      auto      t = solver.trace_builtin("exponential_decay", {1}, {1}, 0, 1);
      EXPECT_EQ(t.solution.t.back(), 1);
      EXPECT_EQ(t.attempts.back().kind, ODEAttemptKind::rk4_tail);
      EXPECT_NEAR(t.solution.y.back()[0], std::exp(-1.), .02);
    }
  auto rows = ode_decay_comparison();
  EXPECT_EQ(rows.size(), 6);
  auto convergence = ode_decay_step_study();
  EXPECT_EQ(convergence.size(), 6);
  EXPECT_GT(convergence.front().euler_error, convergence.back().euler_error);
  EXPECT_GT(convergence.front().rk4_error, convergence.back().rk4_error);
  auto oscillator = ode_harmonic_study();
  EXPECT_EQ(oscillator.solution.t.back(), 20);
  EXPECT_LT(oscillator.max_state_error, 1e-7);
  EXPECT_LT(oscillator.max_energy_drift, 1e-7);
  auto lorenz = ode_lorenz_study();
  EXPECT_EQ(lorenz.solution.t.back(), 25);
  EXPECT_EQ(lorenz.x.size(), lorenz.solution.size());
}
TEST(ODETrace, CompletePreCallbackGuardsFailureEffectsOutputLimitsAndTinyIntervals) {
  Decay rhs;
  EXPECT_THROW((ODESolver({StepMethod::RK4, 4, 0.})), std::invalid_argument);
  EXPECT_THROW((ODESolver({static_cast<StepMethod>(99), 4, .1})), std::invalid_argument);
  ODESolver solver;
  EXPECT_THROW(solver.solve(rhs, {}, 0, 1), std::invalid_argument);
  EXPECT_THROW(solver.solve(rhs, {NAN}, 0, 1), std::invalid_argument);
  EXPECT_THROW(solver.solve(rhs, {1}, 0, INFINITY), std::invalid_argument);
  EXPECT_EQ(rhs.calls, 0);
  Bad bad;
  EXPECT_THROW(solver.solve(bad, {1}, 0, 1), std::invalid_argument);
  EXPECT_EQ(bad.calls, 1);
  bad.shape = false;
  EXPECT_THROW(solver.solve(bad, {1}, 0, 1), std::overflow_error);
  ODEExecutionLimits limits;
  limits.max_evaluations = 3;
  rhs.calls              = 0;
  EXPECT_THROW(solver.solve_with_limits(rhs, {1}, 0, 1, limits), std::length_error);
  EXPECT_EQ(rhs.calls, 3);
  limits              = {};
  limits.max_attempts = 1;
  rhs.calls           = 0;
  EXPECT_THROW(solver.solve_with_limits(rhs, {1}, 0, 1, limits), std::length_error);
  EXPECT_EQ(rhs.calls, 4);
  limits                      = {};
  limits.max_solution_entries = 3;
  EXPECT_THROW(solver.solve_with_limits(rhs, {1}, 0, .01, limits), std::length_error);
  auto tiny = solver.solve_builtin("exponential_decay", {1}, {1}, 0, 1e-16);
  EXPECT_EQ(tiny.size(), 2);
  EXPECT_EQ(tiny.t.back(), 1e-16);
  rhs.calls = 0;
  EXPECT_THROW(solver.solve(rhs, {1}, 1e16, std::nextafter(1e16, INFINITY)), std::overflow_error);
  EXPECT_EQ(rhs.calls, 0);
  EXPECT_THROW(solver.solve_builtin("harmonic_oscillator", {1}, {1}, 0, 1), std::invalid_argument);
  EXPECT_THROW(solver.solve_builtin("logistic_growth", {1, 0}, {1}, 0, 1), std::invalid_argument);
  EXPECT_THROW(
      solver.solve_builtin("exponential_decay", {1, NAN}, {1}, 0, 1), std::invalid_argument);
}
