/// @file study.cpp
/// @brief Canonical original six-method decay, convergence, harmonic and Lorenz report studies.
#include <algorithm>
#include <cmath>
#include <datamunge/ode/study.hpp>
namespace datamunge::ode {
std::vector<ODEAccuracyRow> ode_decay_comparison() {
  std::vector<std::pair<std::string, ODEOptions>> methods{
      {"Euler", {StepMethod::Euler, 4, .01}},
      {"Midpoint", {StepMethod::Midpoint, 4, .01}},
      {"RK4", {StepMethod::RK4, 4, .01}},
      {"RK45", {StepMethod::RK45, 4, .1, 1e-8, 1e-8, 0}},
      {"AB4", {StepMethod::AdamsBashforth, 4, .01}},
      {"PECE4", {StepMethod::AdamsMoulton, 4, .01}}};
  std::vector<ODEAccuracyRow> rows;
  for (auto& [name, options] : methods) {
    ODEAccuracyRow row;
    row.name           = name;
    row.options        = options;
    row.solution       = ODESolver(options).solve_builtin("exponential_decay", {1}, {1}, 0, 2);
    row.exact          = std::exp(-2.);
    row.absolute_error = std::abs(row.solution.y.back()[0] - row.exact);
    rows.push_back(std::move(row));
  }
  return rows;
}
std::vector<ODEOrderRow> ode_decay_step_study() {
  std::vector<ODEOrderRow> rows;
  for (double h : {.1, .05, .02, .01, .005, .002}) {
    auto euler = ODESolver({StepMethod::Euler, 4, h})
                     .solve_builtin("exponential_decay", {1}, {1}, 0, 2),
         rk4 =
             ODESolver({StepMethod::RK4, 4, h}).solve_builtin("exponential_decay", {1}, {1}, 0, 2);
    rows.push_back(
        {h,
         std::abs(euler.y.back()[0] - std::exp(-2.)),
         std::abs(rk4.y.back()[0] - std::exp(-2.)),
         euler.steps_taken,
         rk4.steps_taken,
         euler.function_evaluations,
         rk4.function_evaluations});
  }
  return rows;
}
ODEOscillatorStudy ode_harmonic_study() {
  ODEOscillatorStudy study;
  study.solution =
      ODESolver({StepMethod::RK4, 4, .01}).solve_builtin("harmonic_oscillator", {1}, {1, 0}, 0, 20);
  for (std::size_t i = 0; i < study.solution.t.size(); ++i) {
    double t = study.solution.t[i], x = study.solution.y[i][0], v = study.solution.y[i][1],
           exact_x = std::cos(t), exact_v = -std::sin(t), energy = x * x + v * v;
    study.position.push_back(x);
    study.velocity.push_back(v);
    study.exact_position.push_back(exact_x);
    study.exact_velocity.push_back(exact_v);
    study.energy_drift.push_back(energy - 1);
    study.max_energy_drift = std::max(study.max_energy_drift, std::abs(energy - 1));
    study.max_state_error  = std::max(study.max_state_error, std::hypot(x - exact_x, v - exact_v));
    study.final_energy     = energy;
  }
  return study;
}
ODELorenzStudy ode_lorenz_study() {
  ODELorenzStudy study;
  study.solution = ODESolver({StepMethod::RK4, 4, .005})
                       .solve_builtin("lorenz", {10, 28, 8. / 3}, {1, 1, 1}, 0, 25);
  study.x.reserve(study.solution.y.size());
  study.z.reserve(study.solution.y.size());
  for (auto& state : study.solution.y) {
    study.x.push_back(state[0]);
    study.z.push_back(state[2]);
  }
  return study;
}
} // namespace datamunge::ode
