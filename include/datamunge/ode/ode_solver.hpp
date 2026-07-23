#pragma once

#include <datamunge/ode/rhs.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace datamunge::ode {

enum class StepMethod { Euler, Midpoint, RK4, RK45, AdamsBashforth, AdamsMoulton };

struct ODEOptions {
  StepMethod method{StepMethod::RK4};
  /// @brief Order (2-4) for AdamsBashforth/AdamsMoulton; ignored by every other method.
  std::size_t multistep_order{4};
  /// @brief Fixed step size for every method except RK45, where it is only the *initial* step
  ///        (subsequent steps are chosen adaptively).
  double step_size{0.01};
  /// @brief RK45-only: local error tolerances driving adaptive step-size control.
  double abs_tol{1e-6};
  double rel_tol{1e-6};
  /// @brief RK45-only: hard cap on step size; 0 means unbounded (capped only by t_end - t_start).
  double max_step{0.0};
};

/// @brief One numerically integrated trajectory: t[i] paired with y[i] (the full state vector
///        at that time), in increasing-time order, always including both endpoints. `t` and `y`
///        are public fields for direct C++ use, but are not exposed to language bindings (a
///        container-typed struct *field getter* -- even a plain vector<double>, not just a
///        nested vector<vector<double>> -- hits a swig-jse R-backend codegen bug that a
///        same-shaped plain *method* return does not) -- time_at()/state_at()/size() are the
///        cross-language-safe way to read it.
struct ODESolution {
  std::vector<double> t;
  std::vector<std::vector<double>> y;
  std::size_t steps_taken{0};
  std::size_t function_evaluations{0};

  [[nodiscard]] std::size_t size() const { return t.size(); }
  [[nodiscard]] double time_at(std::size_t index) const { return t.at(index); }
  /// @brief The full state vector at t[index].
  [[nodiscard]] const std::vector<double>& state_at(std::size_t index) const { return y.at(index); }
};

/// @brief Numerically integrates dy/dt = f(t, y) forward from t_start to t_end (t_end must be
///        greater than t_start; this project's "basic" ODE support does not handle backward
///        integration) given an initial state y0, via one of: explicit Euler, the explicit
///        midpoint rule (RK2), classical Runge-Kutta (RK4), adaptive Dormand-Prince
///        Runge-Kutta 4(5) (RK45), explicit Adams-Bashforth (order 2-4), or
///        Adams-Bashforth-Moulton predictor-corrector (order 2-4 -- the "implicit" linear
///        multistep option; this project uses a PECE predictor-corrector rather than a full
///        Newton-iteration implicit solve, avoiding the need for a Jacobian). Every multistep
///        method self-starts using classical RK4 for its first few steps.
class ODESolver {
 public:
  explicit ODESolver(ODEOptions options = {});

  [[nodiscard]] ODESolution solve(RHS& rhs, const std::vector<double>& y0, double t_start, double t_end) const;

  /// @brief Same as solve(), but the RHS is one of a fixed set of named systems instead of a
  ///        live user-supplied one -- the only way every language binding (not just the ones
  ///        with SWIG director support) can use this solver. `params` is interpreted
  ///        positionally per system (missing entries fall back to the defaults below):
  ///          - "exponential_decay": dy/dt = -k y.                     params = {k=1}, y0 size 1.
  ///          - "logistic_growth":   dy/dt = r y (1 - y/K).            params = {r=1, K=1}, y0 size 1.
  ///          - "harmonic_oscillator": (x,v), dx/dt=v, dv/dt=-omega^2 x. params = {omega=1}, y0 size 2.
  ///          - "van_der_pol":       (x,v), dx/dt=v, dv/dt=mu(1-x^2)v-x. params = {mu=1}, y0 size 2.
  ///          - "lorenz": (x,y,z), the classic chaotic system.         params = {sigma=10, rho=28, beta=8/3}, y0 size 3.
  ///        Throws std::invalid_argument for an unknown system name.
  [[nodiscard]] ODESolution solve_builtin(const std::string& system, const std::vector<double>& params,
                                          const std::vector<double>& y0, double t_start, double t_end) const;

 private:
  ODEOptions options_;
};

} // namespace datamunge::ode
