#include <datamunge/ode/ode_solver.hpp>

#include <algorithm>
#include <cmath>
#include <deque>
#include <stdexcept>

namespace datamunge::ode {

namespace {

using State = std::vector<double>;

State axpy(const State& y, const double h, const State& f) {
  State out(y.size());
  for (std::size_t i = 0; i < y.size(); ++i) out[i] = y[i] + h * f[i];
  return out;
}

State rk4_step(RHS& rhs, const double t, const State& y, const double h, std::size_t& evals) {
  const auto k1 = rhs.evaluate(t, y);
  const auto k2 = rhs.evaluate(t + h / 2.0, axpy(y, h / 2.0, k1));
  const auto k3 = rhs.evaluate(t + h / 2.0, axpy(y, h / 2.0, k2));
  const auto k4 = rhs.evaluate(t + h, axpy(y, h, k3));
  evals += 4;
  State out(y.size());
  for (std::size_t i = 0; i < y.size(); ++i) out[i] = y[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
  return out;
}

State euler_step(RHS& rhs, const double t, const State& y, const double h, std::size_t& evals) {
  const auto f = rhs.evaluate(t, y);
  ++evals;
  return axpy(y, h, f);
}

State midpoint_step(RHS& rhs, const double t, const State& y, const double h, std::size_t& evals) {
  const auto k1 = rhs.evaluate(t, y);
  const auto k2 = rhs.evaluate(t + h / 2.0, axpy(y, h / 2.0, k1));
  evals += 2;
  return axpy(y, h, k2);
}

void require_forward_span(const double t_start, const double t_end) {
  if (!(t_end > t_start)) {
    throw std::invalid_argument("ODESolver: t_end must be greater than t_start (backward integration is not supported)");
  }
}

template <typename StepFn>
ODESolution solve_fixed_step(RHS& rhs, const State& y0, const double t_start, const double t_end, const double h_nominal,
                             StepFn step) {
  ODESolution solution;
  solution.t.push_back(t_start);
  solution.y.push_back(y0);

  double t = t_start;
  State y = y0;
  while (t < t_end - 1e-14) {
    const double h = std::min(h_nominal, t_end - t);
    y = step(rhs, t, y, h, solution.function_evaluations);
    t += h;
    solution.t.push_back(t);
    solution.y.push_back(y);
    ++solution.steps_taken;
  }
  return solution;
}

// Adaptive Dormand-Prince Runge-Kutta 4(5) (the classical "ode45" tableau).
ODESolution solve_rk45(RHS& rhs, const State& y0, const double t_start, const double t_end, const ODEOptions& options) {
  static constexpr double c2 = 1.0 / 5, c3 = 3.0 / 10, c4 = 4.0 / 5, c5 = 8.0 / 9, c6 = 1.0;
  static constexpr double a21 = 1.0 / 5;
  static constexpr double a31 = 3.0 / 40, a32 = 9.0 / 40;
  static constexpr double a41 = 44.0 / 45, a42 = -56.0 / 15, a43 = 32.0 / 9;
  static constexpr double a51 = 19372.0 / 6561, a52 = -25360.0 / 2187, a53 = 64448.0 / 6561, a54 = -212.0 / 729;
  static constexpr double a61 = 9017.0 / 3168, a62 = -355.0 / 33, a63 = 46732.0 / 5247, a64 = 49.0 / 176,
                          a65 = -5103.0 / 18656;
  static constexpr double a71 = 35.0 / 384, a73 = 500.0 / 1113, a74 = 125.0 / 192, a75 = -2187.0 / 6784,
                          a76 = 11.0 / 84;
  // 5th-order solution weights are identical to row 7 (this tableau is FSAL); 4th-order weights
  // (b*) differ, and their difference gives the embedded error estimate.
  static constexpr double b1 = 35.0 / 384, b3 = 500.0 / 1113, b4 = 125.0 / 192, b5 = -2187.0 / 6784, b6 = 11.0 / 84;
  static constexpr double bs1 = 5179.0 / 57600, bs3 = 7571.0 / 16695, bs4 = 393.0 / 640, bs5 = -92097.0 / 339200,
                          bs6 = 187.0 / 2100, bs7 = 1.0 / 40;

  const std::size_t n = y0.size();
  ODESolution solution;
  solution.t.push_back(t_start);
  solution.y.push_back(y0);

  double t = t_start;
  State y = y0;
  double h = options.step_size;
  const double max_step = options.max_step > 0.0 ? options.max_step : (t_end - t_start);

  while (t < t_end - 1e-14) {
    h = std::min({h, max_step, t_end - t});

    const auto k1 = rhs.evaluate(t, y);
    const auto k2 = rhs.evaluate(t + c2 * h, axpy(y, h * a21, k1));
    const auto k3 = rhs.evaluate(t + c3 * h, axpy(axpy(y, h * a31, k1), h * a32, k2));
    const auto k4 = rhs.evaluate(t + c4 * h, axpy(axpy(axpy(y, h * a41, k1), h * a42, k2), h * a43, k3));
    const auto k5 =
        rhs.evaluate(t + c5 * h, axpy(axpy(axpy(axpy(y, h * a51, k1), h * a52, k2), h * a53, k3), h * a54, k4));
    const auto k6 = rhs.evaluate(
        t + c6 * h,
        axpy(axpy(axpy(axpy(axpy(y, h * a61, k1), h * a62, k2), h * a63, k3), h * a64, k4), h * a65, k5));
    State y5_arg = axpy(axpy(axpy(axpy(y, h * a71, k1), h * a73, k3), h * a74, k4), h * a75, k5);
    y5_arg = axpy(y5_arg, h * a76, k6);
    const auto k7 = rhs.evaluate(t + h, y5_arg);
    solution.function_evaluations += 7;

    State y5(n), y4(n);
    for (std::size_t i = 0; i < n; ++i) {
      y5[i] = y[i] + h * (b1 * k1[i] + b3 * k3[i] + b4 * k4[i] + b5 * k5[i] + b6 * k6[i]);
      y4[i] = y[i] + h * (bs1 * k1[i] + bs3 * k3[i] + bs4 * k4[i] + bs5 * k5[i] + bs6 * k6[i] + bs7 * k7[i]);
    }

    double error_norm = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double scale = options.abs_tol + options.rel_tol * std::max(std::abs(y[i]), std::abs(y5[i]));
      const double e = (y5[i] - y4[i]) / (scale > 0.0 ? scale : 1.0);
      error_norm += e * e;
    }
    error_norm = std::sqrt(error_norm / static_cast<double>(n));

    if (error_norm <= 1.0 || h <= 1e-12) {
      t += h;
      y = y5;
      solution.t.push_back(t);
      solution.y.push_back(y);
      ++solution.steps_taken;
    }

    const double safety = 0.9;
    const double growth = error_norm > 0.0 ? safety * std::pow(error_norm, -0.2) : 5.0;
    h *= std::clamp(growth, 0.2, 5.0);
  }
  return solution;
}

// Explicit Adams-Bashforth (order 2-4) and, when `moulton` is true, an Adams-Bashforth-Moulton
// PECE predictor-corrector -- the "implicit" flavored linear multistep option, avoiding a full
// Newton solve. Self-starts using classical RK4 for the first (order - 1) steps.
ODESolution solve_adams(RHS& rhs, const State& y0, const double t_start, const double t_end, const ODEOptions& options,
                        const bool moulton) {
  const std::size_t order = std::clamp<std::size_t>(options.multistep_order, 2, 4);
  const double h = options.step_size;

  // Coefficients index 0 = f_n (most recent), increasing index = further into the past.
  static const std::vector<std::vector<double>> ab_coeffs = {
      {}, {}, {3.0 / 2, -1.0 / 2}, {23.0 / 12, -16.0 / 12, 5.0 / 12}, {55.0 / 24, -59.0 / 24, 37.0 / 24, -9.0 / 24}};
  // am_coeffs[0] applies to the (predicted) f_{n+1}; index k>=1 applies to f_{n+1-k}.
  static const std::vector<std::vector<double>> am_coeffs = {
      {}, {}, {1.0 / 2, 1.0 / 2}, {5.0 / 12, 8.0 / 12, -1.0 / 12}, {9.0 / 24, 19.0 / 24, -5.0 / 24, 1.0 / 24}};

  ODESolution solution;
  solution.t.push_back(t_start);
  solution.y.push_back(y0);

  double t = t_start;
  State y = y0;
  std::deque<State> f_history; // back() = most recent
  f_history.push_back(rhs.evaluate(t, y));
  ++solution.function_evaluations;

  for (std::size_t i = 1; i < order && t < t_end - 1e-14; ++i) {
    const double step_h = std::min(h, t_end - t);
    y = rk4_step(rhs, t, y, step_h, solution.function_evaluations);
    t += step_h;
    solution.t.push_back(t);
    solution.y.push_back(y);
    ++solution.steps_taken;
    f_history.push_back(rhs.evaluate(t, y));
    ++solution.function_evaluations;
  }

  const auto& ab = ab_coeffs[order];
  const auto& am = am_coeffs[order];

  while (t < t_end - 1e-14) {
    const double step_h = std::min(h, t_end - t);
    const std::size_t n = y.size();

    State predictor = y;
    for (std::size_t k = 0; k < order; ++k) {
      const auto& f_k = f_history[f_history.size() - 1 - k];
      for (std::size_t i = 0; i < n; ++i) predictor[i] += step_h * ab[k] * f_k[i];
    }

    State y_next;
    if (moulton) {
      const auto f_pred = rhs.evaluate(t + step_h, predictor);
      ++solution.function_evaluations;
      y_next = y;
      for (std::size_t i = 0; i < n; ++i) y_next[i] += step_h * am[0] * f_pred[i];
      for (std::size_t k = 0; k + 1 < order; ++k) {
        const auto& f_k = f_history[f_history.size() - 1 - k];
        for (std::size_t i = 0; i < n; ++i) y_next[i] += step_h * am[k + 1] * f_k[i];
      }
    } else {
      y_next = predictor;
    }

    t += step_h;
    y = y_next;
    solution.t.push_back(t);
    solution.y.push_back(y);
    ++solution.steps_taken;

    f_history.push_back(rhs.evaluate(t, y));
    ++solution.function_evaluations;
    if (f_history.size() > order) f_history.pop_front();
  }

  return solution;
}

class ExponentialDecayRHS : public RHS {
 public:
  explicit ExponentialDecayRHS(const double k) : k_(k) {}
  std::vector<double> evaluate(double, const std::vector<double>& y) override { return {-k_ * y.at(0)}; }

 private:
  double k_;
};

class LogisticGrowthRHS : public RHS {
 public:
  LogisticGrowthRHS(const double r, const double capacity) : r_(r), capacity_(capacity) {}
  std::vector<double> evaluate(double, const std::vector<double>& y) override {
    return {r_ * y.at(0) * (1.0 - y.at(0) / capacity_)};
  }

 private:
  double r_;
  double capacity_;
};

class HarmonicOscillatorRHS : public RHS {
 public:
  explicit HarmonicOscillatorRHS(const double omega) : omega_sq_(omega * omega) {}
  std::vector<double> evaluate(double, const std::vector<double>& y) override { return {y.at(1), -omega_sq_ * y.at(0)}; }

 private:
  double omega_sq_;
};

class VanDerPolRHS : public RHS {
 public:
  explicit VanDerPolRHS(const double mu) : mu_(mu) {}
  std::vector<double> evaluate(double, const std::vector<double>& y) override {
    return {y.at(1), mu_ * (1.0 - y.at(0) * y.at(0)) * y.at(1) - y.at(0)};
  }

 private:
  double mu_;
};

class LorenzRHS : public RHS {
 public:
  LorenzRHS(const double sigma, const double rho, const double beta) : sigma_(sigma), rho_(rho), beta_(beta) {}
  std::vector<double> evaluate(double, const std::vector<double>& y) override {
    return {sigma_ * (y.at(1) - y.at(0)), y.at(0) * (rho_ - y.at(2)) - y.at(1), y.at(0) * y.at(1) - beta_ * y.at(2)};
  }

 private:
  double sigma_;
  double rho_;
  double beta_;
};

double param_or(const std::vector<double>& params, const std::size_t index, const double fallback) {
  return index < params.size() ? params[index] : fallback;
}

} // namespace

ODESolver::ODESolver(ODEOptions options) : options_(options) {}

ODESolution ODESolver::solve(RHS& rhs, const std::vector<double>& y0, const double t_start, const double t_end) const {
  require_forward_span(t_start, t_end);
  switch (options_.method) {
    case StepMethod::Euler:
      return solve_fixed_step(rhs, y0, t_start, t_end, options_.step_size, euler_step);
    case StepMethod::Midpoint:
      return solve_fixed_step(rhs, y0, t_start, t_end, options_.step_size, midpoint_step);
    case StepMethod::RK4:
      return solve_fixed_step(rhs, y0, t_start, t_end, options_.step_size, rk4_step);
    case StepMethod::RK45:
      return solve_rk45(rhs, y0, t_start, t_end, options_);
    case StepMethod::AdamsBashforth:
      return solve_adams(rhs, y0, t_start, t_end, options_, false);
    case StepMethod::AdamsMoulton:
      return solve_adams(rhs, y0, t_start, t_end, options_, true);
  }
  throw std::invalid_argument("ODESolver: unknown method");
}

ODESolution ODESolver::solve_builtin(const std::string& system, const std::vector<double>& params,
                                     const std::vector<double>& y0, const double t_start, const double t_end) const {
  if (system == "exponential_decay") {
    ExponentialDecayRHS rhs(param_or(params, 0, 1.0));
    return solve(rhs, y0, t_start, t_end);
  }
  if (system == "logistic_growth") {
    LogisticGrowthRHS rhs(param_or(params, 0, 1.0), param_or(params, 1, 1.0));
    return solve(rhs, y0, t_start, t_end);
  }
  if (system == "harmonic_oscillator") {
    HarmonicOscillatorRHS rhs(param_or(params, 0, 1.0));
    return solve(rhs, y0, t_start, t_end);
  }
  if (system == "van_der_pol") {
    VanDerPolRHS rhs(param_or(params, 0, 1.0));
    return solve(rhs, y0, t_start, t_end);
  }
  if (system == "lorenz") {
    LorenzRHS rhs(param_or(params, 0, 10.0), param_or(params, 1, 28.0), param_or(params, 2, 8.0 / 3.0));
    return solve(rhs, y0, t_start, t_end);
  }
  throw std::invalid_argument("ODESolver::solve_builtin: unknown system '" + system +
                              "' (expected exponential_decay, logistic_growth, harmonic_oscillator, van_der_pol, "
                              "or lorenz)");
}

} // namespace datamunge::ode
