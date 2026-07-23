#include <gtest/gtest.h>

#include <datamunge/ode/ode.hpp>

#include <cmath>
#include <stdexcept>

using namespace datamunge::ode;

namespace {

class ExpDecay : public RHS {
 public:
  std::vector<double> evaluate(double, const std::vector<double>& y) override { return {-y[0]}; }
};

class HarmonicRHS : public RHS {
 public:
  std::vector<double> evaluate(double, const std::vector<double>& y) override { return {y[1], -y[0]}; }
};

} // namespace

TEST(ODESolver, EulerMatchesExponentialDecayAtFineStepSize) {
  ExpDecay rhs;
  ODESolver solver({StepMethod::Euler, 4, 0.0001});
  const auto sol = solver.solve(rhs, {1.0}, 0.0, 2.0);
  EXPECT_NEAR(sol.y.back()[0], std::exp(-2.0), 1e-3);
  EXPECT_DOUBLE_EQ(sol.t.front(), 0.0);
  EXPECT_DOUBLE_EQ(sol.t.back(), 2.0);
}

TEST(ODESolver, MidpointIsMoreAccurateThanEulerAtEqualStepSize) {
  ExpDecay rhs;
  const double exact = std::exp(-2.0);
  ODESolver euler({StepMethod::Euler, 4, 0.01});
  ODESolver midpoint({StepMethod::Midpoint, 4, 0.01});
  const double euler_error = std::abs(euler.solve(rhs, {1.0}, 0.0, 2.0).y.back()[0] - exact);
  const double midpoint_error = std::abs(midpoint.solve(rhs, {1.0}, 0.0, 2.0).y.back()[0] - exact);
  EXPECT_LT(midpoint_error, euler_error);
}

TEST(ODESolver, RK4MatchesExponentialDecayTightly) {
  ExpDecay rhs;
  ODESolver solver({StepMethod::RK4, 4, 0.01});
  const auto sol = solver.solve(rhs, {1.0}, 0.0, 2.0);
  EXPECT_NEAR(sol.y.back()[0], std::exp(-2.0), 1e-8);
  EXPECT_EQ(sol.function_evaluations, sol.steps_taken * 4);
}

TEST(ODESolver, AdaptiveRK45MatchesExponentialDecayAndUsesFewSteps) {
  ExpDecay rhs;
  ODEOptions options;
  options.method = StepMethod::RK45;
  options.step_size = 0.1;
  options.abs_tol = 1e-8;
  options.rel_tol = 1e-8;
  ODESolver solver(options);
  const auto sol = solver.solve(rhs, {1.0}, 0.0, 2.0);
  EXPECT_NEAR(sol.y.back()[0], std::exp(-2.0), 1e-6);
  EXPECT_LT(sol.steps_taken, 50u) << "adaptive stepping should need far fewer steps than a fine fixed grid";
}

TEST(ODESolver, RK45RespectsMaxStep) {
  ExpDecay rhs;
  ODEOptions options;
  options.method = StepMethod::RK45;
  options.step_size = 1.0;
  options.max_step = 0.05;
  ODESolver solver(options);
  const auto sol = solver.solve(rhs, {1.0}, 0.0, 1.0);
  EXPECT_GE(sol.steps_taken, 19u) << "capped at max_step=0.05 over a unit interval needs at least 20 steps";
}

TEST(ODESolver, AdamsBashforthMatchesExponentialDecay) {
  ExpDecay rhs;
  ODESolver solver({StepMethod::AdamsBashforth, 4, 0.001});
  const auto sol = solver.solve(rhs, {1.0}, 0.0, 2.0);
  EXPECT_NEAR(sol.y.back()[0], std::exp(-2.0), 1e-6);
}

TEST(ODESolver, AdamsMoultonMatchesExponentialDecayAcrossOrders) {
  ExpDecay rhs;
  for (std::size_t order = 2; order <= 4; ++order) {
    ODESolver solver({StepMethod::AdamsMoulton, order, 0.001});
    const auto sol = solver.solve(rhs, {1.0}, 0.0, 2.0);
    EXPECT_NEAR(sol.y.back()[0], std::exp(-2.0), 1e-4) << "order " << order;
  }
}

TEST(ODESolver, HarmonicOscillatorMatchesClosedFormAndConservesEnergy) {
  HarmonicRHS rhs;
  ODESolver solver({StepMethod::RK4, 4, 0.001});
  const auto sol = solver.solve(rhs, {1.0, 0.0}, 0.0, 10.0);
  const double x = sol.y.back()[0];
  const double v = sol.y.back()[1];
  EXPECT_NEAR(x, std::cos(10.0), 1e-6);
  EXPECT_NEAR(v, -std::sin(10.0), 1e-6);
  EXPECT_NEAR(x * x + v * v, 1.0, 1e-6);
}

TEST(ODESolver, SolveBuiltinExponentialDecayMatchesLiveRHS) {
  ExpDecay rhs;
  ODESolver solver({StepMethod::RK4, 4, 0.01});
  const auto live = solver.solve(rhs, {1.0}, 0.0, 2.0);
  const auto builtin = solver.solve_builtin("exponential_decay", {1.0}, {1.0}, 0.0, 2.0);
  EXPECT_DOUBLE_EQ(live.y.back()[0], builtin.y.back()[0]);
}

TEST(ODESolver, SolveBuiltinLogisticGrowthMatchesClosedForm) {
  ODESolver solver({StepMethod::RK4, 4, 0.001});
  const auto sol = solver.solve_builtin("logistic_growth", {1.0, 1.0}, {0.5}, 0.0, 5.0);
  const double expected = 1.0 / (1.0 + ((1.0 - 0.5) / 0.5) * std::exp(-5.0));
  EXPECT_NEAR(sol.y.back()[0], expected, 1e-4);
}

TEST(ODESolver, SolveBuiltinCoversEveryNamedSystem) {
  ODESolver solver({StepMethod::RK4, 4, 0.001});
  EXPECT_NO_THROW(solver.solve_builtin("exponential_decay", {1.0}, {1.0}, 0.0, 1.0));
  EXPECT_NO_THROW(solver.solve_builtin("logistic_growth", {1.0, 1.0}, {0.5}, 0.0, 1.0));
  EXPECT_NO_THROW(solver.solve_builtin("harmonic_oscillator", {1.0}, {1.0, 0.0}, 0.0, 1.0));
  EXPECT_NO_THROW(solver.solve_builtin("van_der_pol", {1.0}, {2.0, 0.0}, 0.0, 1.0));
  EXPECT_NO_THROW(solver.solve_builtin("lorenz", {}, {1.0, 1.0, 1.0}, 0.0, 1.0));
}

TEST(ODESolver, SolveBuiltinRejectsUnknownSystem) {
  ODESolver solver;
  EXPECT_THROW(solver.solve_builtin("not_a_system", {}, {1.0}, 0.0, 1.0), std::invalid_argument);
}

TEST(ODESolver, RejectsBackwardIntegration) {
  ExpDecay rhs;
  ODESolver solver;
  EXPECT_THROW(solver.solve(rhs, {1.0}, 1.0, 0.0), std::invalid_argument);
  EXPECT_THROW(solver.solve(rhs, {1.0}, 1.0, 1.0), std::invalid_argument);
}
