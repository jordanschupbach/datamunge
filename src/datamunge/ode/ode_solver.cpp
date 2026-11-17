/// @file ode_solver.cpp
/// @brief Canonical checked forward ODE kernels with actual callback/trial evidence and finite progress admission.
#include <algorithm>
#include <cmath>
#include <datamunge/ode/ode_solver.hpp>
#include <deque>
#include <limits>
#include <memory>
#include <stdexcept>
namespace datamunge::ode {
/// @brief Local checked execution and original numerical kernels, with no global mutable state.
namespace {
/// @brief Owned native state-vector arithmetic within one integration call.
using State = std::vector<double>;
/// @brief Add actual count/output geometry without wraparound.
/// @param a Existing count.
/// @param b Additional count.
/// @return Checked sum.
/// @throws std::length_error If sum overflows.
std::size_t checked_add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("ODE: count geometry");
  return a + b;
}
/// @brief Validate every finite state component and expected shape before callback/commit.
/// @param state Complete actual state/derivative.
/// @param dimension Expected positive dimension.
/// @throws std::invalid_argument If dimension differs.
/// @throws std::overflow_error If any component is nonfinite.
void finite_state(const State& state, std::size_t dimension) {
  if (state.size() != dimension)
    throw std::invalid_argument("ODE: callback/state dimension mismatch");
  for (double x : state)
    if (!std::isfinite(x))
      throw std::overflow_error("ODE: nonfinite stage or derivative");
}
/// @brief Validate known method and every finite original numerical control.
/// @param options Complete original numerical options.
/// @throws std::invalid_argument If method or a complete control is invalid.
void validate_options(const ODEOptions& options) {
  switch (options.method) {
    case StepMethod::Euler:
    case StepMethod::Midpoint:
    case StepMethod::RK4:
    case StepMethod::RK45:
    case StepMethod::AdamsBashforth:
    case StepMethod::AdamsMoulton:
      break;
    default:
      throw std::invalid_argument("ODE: unknown method");
  }
  if (!std::isfinite(options.step_size) || options.step_size <= 0 ||
      !std::isfinite(options.abs_tol) || options.abs_tol <= 0 || !std::isfinite(options.rel_tol) ||
      options.rel_tol < 0 || !std::isfinite(options.max_step) || options.max_step < 0)
    throw std::invalid_argument("ODE: complete finite numerical controls required");
}
/// @brief Bound one positive step to remaining interval and require representable progress.
/// @param nominal Actual proposed finite positive step.
/// @param time Actual old finite time.
/// @param end Original finite later end.
/// @return Actual selected finite advancing step.
/// @throws std::overflow_error If proposed time cannot advance finitely.
double advancing_step(double nominal, double time, double end) {
  double step = std::min(nominal, end - time);
  if (!std::isfinite(step) || step <= 0 || !std::isfinite(time + step) || !(time + step > time))
    throw std::overflow_error("ODE: step cannot advance representable time");
  return step;
}
/// @brief Actual scoped execution admission and checked live-RHS adapter, with no global state.
class CheckedRHS : public RHS {
  RHS&               inner_;     ///< Borrowed callback only during this executing solve.
  std::size_t        dimension_; ///< Complete fixed state/derivative dimension.
  ODEExecutionLimits limits_;    ///< Actual original optional policies.
  ODETrace*   history_; ///< Optional caller-owned local trace sink, never retained after solve.
  std::size_t evaluations_{0};      ///< Actual invoked callback count.
  std::size_t attempts_{0};         ///< Actual admitted trial count.
  std::size_t solution_entries_{0}; ///< Actual retained solution time/state scalar count.
  ODEAttempt  pending_; ///< Current actual admitted trial observation, moved on completion.
 public:
  /// @brief Bind one complete local solve context without invoking a callback.
  /// @param inner Borrowed original derivative callback.
  /// @param dimension Positive complete state dimension.
  /// @param limits Original optional execution/history policies.
  /// @param history Optional complete local owned trace sink.
  CheckedRHS(RHS& inner, std::size_t dimension, ODEExecutionLimits limits, ODETrace* history)
      : inner_(inner), dimension_(dimension), limits_(limits), history_(history) {}
  /// @brief Admit future actual owned trace units before their copy/append.
  /// @param units Actual next logical scalar/record contribution.
  /// @throws std::length_error If actual count geometry or selected quota fails.
  void charge(std::size_t units) {
    if (!history_)
      return;
    auto next = checked_add(history_->captured_entries, units);
    if (limits_.max_trace_entries && next > limits_.max_trace_entries)
      throw std::length_error("ODE: actual trace quota");
    history_->captured_entries = next;
  }
  /// @brief Validate/admit and invoke one actual derivative, preserving callback effects/order.
  /// @param time Actual finite stage time.
  /// @param state Actual finite fixed-dimensional stage.
  /// @return Owned actual finite matching-dimensional derivative.
  /// @throws std::invalid_argument If callback dimension is wrong.
  /// @throws std::overflow_error If stage/derivative is nonfinite.
  /// @throws std::length_error If actual callback/history policy fails.
  State evaluate(double time, const State& state) override {
    if (!std::isfinite(time))
      throw std::overflow_error("ODE: nonfinite stage time");
    finite_state(state, dimension_);
    auto next = checked_add(evaluations_, 1);
    if (limits_.max_evaluations && next > limits_.max_evaluations)
      throw std::length_error("ODE: actual callback quota");
    if (history_)
      charge(checked_add(2, checked_add(dimension_, dimension_)));
    ODEEvaluation observation;
    if (history_) {
      observation.time  = time;
      observation.state = state;
    }
    evaluations_    = next;
    auto derivative = inner_.evaluate(time, state);
    finite_state(derivative, dimension_);
    if (history_) {
      observation.derivative = derivative;
      history_->evaluations.push_back(std::move(observation));
    }
    return derivative;
  }
  /// @brief Admit one actual accepted time/state sample before its output copy.
  /// @param state Complete finite actual accepted state.
  /// @throws std::invalid_argument If actual state shape differs.
  /// @throws std::overflow_error If an actual accepted component is nonfinite.
  /// @throws std::length_error If actual output/history quota fails.
  void sample(const State& state) {
    finite_state(state, dimension_);
    auto units = checked_add(1, dimension_);
    auto next  = checked_add(solution_entries_, units);
    if (limits_.max_solution_entries && next > limits_.max_solution_entries)
      throw std::length_error("ODE: actual solution-entry quota");
    charge(units);
    solution_entries_ = next;
  }
  /// @brief Admit one actual trial before its stage callbacks and before-state copy.
  /// @param kind Actual selected kernel.
  /// @param time Actual old accepted time.
  /// @param state Actual old finite state.
  /// @param step Actual finite advancing selected step.
  /// @throws std::length_error If actual attempt/history policy fails.
  void begin(ODEAttemptKind kind, double time, const State& state, double step) {
    auto next = checked_add(attempts_, 1);
    if (limits_.max_attempts && next > limits_.max_attempts)
      throw std::length_error("ODE: actual attempt quota");
    attempts_ = next;
    if (history_) {
      charge(checked_add(1, checked_add(dimension_, dimension_)));
      pending_                  = {};
      pending_.kind             = kind;
      pending_.time_before      = time;
      pending_.before           = state;
      pending_.step             = step;
      pending_.evaluation_begin = evaluations_;
    }
  }
  /// @brief Commit one actual proposal/acceptance record after all its callbacks.
  /// @param proposed Actual finite proposed state, also for rejected RK45 trials.
  /// @param time Actual committed time or unchanged old time on rejection.
  /// @param accepted Actual acceptance decision.
  /// @param error Actual scaled RK45 norm,0 for nonadaptive methods.
  /// @param widened Actual widened-error normalization route.
  /// @throws std::invalid_argument If proposal dimension differs.
  /// @throws std::overflow_error If a proposal component is nonfinite.
  void finish(
      const State& proposed,
      double       time,
      bool         accepted,
      double       error   = 0,
      bool         widened = false) {
    finite_state(proposed, dimension_);
    if (history_) {
      pending_.proposed       = proposed;
      pending_.time_after     = time;
      pending_.accepted       = accepted;
      pending_.error_norm     = error;
      pending_.error_widened  = widened;
      pending_.evaluation_end = evaluations_;
      history_->attempts.push_back(std::move(pending_));
    }
  }
  /// @brief Read actual invoked callback count after successful integration.
  /// @return Owned actual callback count.
  std::size_t evaluations() const noexcept { return evaluations_; }
};
/// @brief Compute original safe-double RMS error normalization, widening unsafe arithmetic.
/// @param old Actual previous state.
/// @param y5 Actual finite fifth-order proposal.
/// @param y4 Actual finite embedded fourth-order proposal.
/// @param options Complete validated positive/finite tolerances.
/// @return Actual scaled norm and whether widened fallback was used;+inf means rejection.
std::pair<double, bool>
scaled_error(const State& old, const State& y5, const State& y4, const ODEOptions& options) {
  double sum     = 0;
  bool   widened = false;
  for (std::size_t i = 0; i < old.size(); ++i) {
    double scale = options.abs_tol + options.rel_tol * std::max(std::abs(old[i]), std::abs(y5[i]));
    double e     = (y5[i] - y4[i]) / scale;
    sum += e * e;
    widened |= !std::isfinite(scale) || !std::isfinite(e) || !std::isfinite(sum);
  }
  if (!widened)
    return {std::sqrt(sum / double(old.size())), false};
  long double norm = 0;
  for (std::size_t i = 0; i < old.size(); ++i) {
    long double scale =
        static_cast<long double>(options.abs_tol) +
        static_cast<long double>(options.rel_tol) * std::max(
                                                        std::abs(static_cast<long double>(old[i])),
                                                        std::abs(static_cast<long double>(y5[i])));
    long double error = (static_cast<long double>(y5[i]) - y4[i]) / scale;
    norm              = std::hypot(norm, error);
  }
  norm /= std::sqrt(static_cast<long double>(old.size()));
  return {static_cast<double>(norm), true};
}
/// @brief Compute actual y+h*f with finite-state admission.
/// @param y Actual original fixed-dimensional state.
/// @param h Actual finite scalar step/coefficient.
/// @param f Actual matching finite derivative.
/// @return Actual finite componentwise proposal.
/// @throws std::overflow_error If any produced component is nonfinite.
State axpy(const State& y, const double h, const State& f) {
  State out(y.size());
  for (std::size_t i = 0; i < y.size(); ++i)
    out[i] = y[i] + h * f[i];
  finite_state(out, y.size());
  return out;
}

/// @brief Execute the four original classical RK4 stage callbacks and weighted proposal.
/// @param rhs Checked actual callback/capture context.
/// @param t Actual old finite time.
/// @param y Actual old finite state.
/// @param h Actual finite positive step.
/// @param evals Actual returned-callback counter, updated by selected kernel.
/// @return Actual finite proposal.
/// @throws std::invalid_argument If a callback changes dimension.
/// @throws std::length_error If actual callback/history quota fails.
/// @throws std::overflow_error If a stage/proposal is nonfinite.
State rk4_step(
    CheckedRHS&  rhs,
    const double t,
    const State& y,
    const double h,
    std::size_t& evals) {
  const auto k1 = rhs.evaluate(t, y);
  const auto k2 = rhs.evaluate(t + h / 2.0, axpy(y, h / 2.0, k1));
  const auto k3 = rhs.evaluate(t + h / 2.0, axpy(y, h / 2.0, k2));
  const auto k4 = rhs.evaluate(t + h, axpy(y, h, k3));
  evals += 4;
  State out(y.size());
  for (std::size_t i = 0; i < y.size(); ++i)
    out[i] = y[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
  finite_state(out, y.size());
  return out;
}

/// @brief Execute one original forward-Euler callback and proposal.
/// @param rhs Checked actual callback/capture context.
/// @param t Actual old finite time.
/// @param y Actual old finite state.
/// @param h Actual finite positive step.
/// @param evals Actual returned-callback counter, updated by selected kernel.
/// @return Actual finite proposal.
/// @throws std::invalid_argument If a callback changes dimension.
/// @throws std::length_error If actual callback/history quota fails.
/// @throws std::overflow_error If a stage/proposal is nonfinite.
State euler_step(
    CheckedRHS&  rhs,
    const double t,
    const State& y,
    const double h,
    std::size_t& evals) {
  const auto f = rhs.evaluate(t, y);
  ++evals;
  return axpy(y, h, f);
}

/// @brief Execute the original two explicit-midpoint stage callbacks and proposal.
/// @param rhs Checked actual callback/capture context.
/// @param t Actual old finite time.
/// @param y Actual old finite state.
/// @param h Actual finite positive step.
/// @param evals Actual returned-callback counter, updated by selected kernel.
/// @return Actual finite proposal.
/// @throws std::invalid_argument If a callback changes dimension.
/// @throws std::length_error If actual callback/history quota fails.
/// @throws std::overflow_error If a stage/proposal is nonfinite.
State midpoint_step(
    CheckedRHS&  rhs,
    const double t,
    const State& y,
    const double h,
    std::size_t& evals) {
  const auto k1 = rhs.evaluate(t, y);
  const auto k2 = rhs.evaluate(t + h / 2.0, axpy(y, h / 2.0, k1));
  evals += 2;
  return axpy(y, h, k2);
}

/// @brief Validate a complete finite forward interval before callbacks.
/// @param t_start Original finite start.
/// @param t_end Original finite strictly later end.
/// @throws std::invalid_argument If interval/difference is nonfinite or non-forward.
void require_forward_span(double t_start, double t_end) {
  if (!std::isfinite(t_start) || !std::isfinite(t_end) || !(t_end > t_start) ||
      !std::isfinite(t_end - t_start))
    throw std::invalid_argument("ODE: finite forward span required");
}

/// @brief Execute actual advancing fixed steps, recording every proposal and accepted sample.
template<typename StepFn>
ODESolution solve_fixed_step(
    CheckedRHS&  rhs,
    const State& y0,
    const double t_start,
    const double t_end,
    const double h_nominal,
    StepFn       step) {
  ODESolution solution;
  rhs.sample(y0);
  solution.t.push_back(t_start);
  solution.y.push_back(y0);

  double t = t_start;
  State  y = y0;
  while (t < t_end) {
    const double h = advancing_step(h_nominal, t, t_end);
    rhs.begin(ODEAttemptKind::fixed_step, t, y, h);
    y = step(rhs, t, y, h, solution.function_evaluations);
    t += h;
    if (t > t_end)
      t = t_end;
    rhs.finish(y, t, true);
    rhs.sample(y);
    solution.t.push_back(t);
    solution.y.push_back(y);
    ++solution.steps_taken;
  }
  return solution;
}

// Adaptive Dormand-Prince Runge-Kutta 4(5) (the classical "ode45" tableau).
/// @brief Execute actual seven-stage Dormand-Prince trials without forced-error acceptance.
ODESolution solve_rk45(
    CheckedRHS&       rhs,
    const State&      y0,
    const double      t_start,
    const double      t_end,
    const ODEOptions& options) {
  /// @brief Immutable original Dormand-Prince tableau/embedded coefficient group.
  static constexpr double c2 = 1.0 / 5, c3 = 3.0 / 10, c4 = 4.0 / 5, c5 = 8.0 / 9, c6 = 1.0;
  /// @brief Immutable original Dormand-Prince tableau/embedded coefficient group.
  static constexpr double a21 = 1.0 / 5;
  /// @brief Immutable original Dormand-Prince tableau/embedded coefficient group.
  static constexpr double a31 = 3.0 / 40, a32 = 9.0 / 40;
  /// @brief Immutable original Dormand-Prince tableau/embedded coefficient group.
  static constexpr double a41 = 44.0 / 45, a42 = -56.0 / 15, a43 = 32.0 / 9;
  /// @brief Immutable original Dormand-Prince tableau/embedded coefficient group.
  static constexpr double a51 = 19372.0 / 6561, a52 = -25360.0 / 2187, a53 = 64448.0 / 6561,
                          a54 = -212.0 / 729;
  /// @brief Immutable original Dormand-Prince tableau/embedded coefficient group.
  static constexpr double a61 = 9017.0 / 3168, a62 = -355.0 / 33, a63 = 46732.0 / 5247,
                          a64 = 49.0 / 176, a65 = -5103.0 / 18656;
  /// @brief Immutable original Dormand-Prince tableau/embedded coefficient group.
  static constexpr double a71 = 35.0 / 384, a73 = 500.0 / 1113, a74 = 125.0 / 192,
                          a75 = -2187.0 / 6784, a76 = 11.0 / 84;
  // 5th-order solution weights are identical to row 7 (this tableau is FSAL); 4th-order weights
  // (b*) differ, and their difference gives the embedded error estimate.
  /// @brief Immutable original Dormand-Prince tableau/embedded coefficient group.
  static constexpr double b1 = 35.0 / 384, b3 = 500.0 / 1113, b4 = 125.0 / 192, b5 = -2187.0 / 6784,
                          b6 = 11.0 / 84;
  /// @brief Immutable original Dormand-Prince tableau/embedded coefficient group.
  static constexpr double bs1 = 5179.0 / 57600, bs3 = 7571.0 / 16695, bs4 = 393.0 / 640,
                          bs5 = -92097.0 / 339200, bs6 = 187.0 / 2100, bs7 = 1.0 / 40;

  const std::size_t n = y0.size();
  ODESolution       solution;
  rhs.sample(y0);
  solution.t.push_back(t_start);
  solution.y.push_back(y0);

  double       t        = t_start;
  State        y        = y0;
  double       h        = options.step_size;
  const double max_step = options.max_step > 0.0 ? options.max_step : (t_end - t_start);

  while (t < t_end) {
    h = advancing_step(std::min(h, max_step), t, t_end);
    rhs.begin(ODEAttemptKind::rk45_trial, t, y, h);

    const auto k1 = rhs.evaluate(t, y);
    const auto k2 = rhs.evaluate(t + c2 * h, axpy(y, h * a21, k1));
    const auto k3 = rhs.evaluate(t + c3 * h, axpy(axpy(y, h * a31, k1), h * a32, k2));
    const auto k4 =
        rhs.evaluate(t + c4 * h, axpy(axpy(axpy(y, h * a41, k1), h * a42, k2), h * a43, k3));
    const auto k5 = rhs.evaluate(
        t + c5 * h, axpy(axpy(axpy(axpy(y, h * a51, k1), h * a52, k2), h * a53, k3), h * a54, k4));
    const auto k6 = rhs.evaluate(
        t + c6 * h,
        axpy(
            axpy(axpy(axpy(axpy(y, h * a61, k1), h * a62, k2), h * a63, k3), h * a64, k4),
            h * a65,
            k5));
    State y5_arg  = axpy(axpy(axpy(axpy(y, h * a71, k1), h * a73, k3), h * a74, k4), h * a75, k5);
    y5_arg        = axpy(y5_arg, h * a76, k6);
    const auto k7 = rhs.evaluate(t + h, y5_arg);
    solution.function_evaluations += 7;

    State y5(n), y4(n);
    for (std::size_t i = 0; i < n; ++i) {
      y5[i] = y[i] + h * (b1 * k1[i] + b3 * k3[i] + b4 * k4[i] + b5 * k5[i] + b6 * k6[i]);
      y4[i] = y[i] + h * (bs1 * k1[i] + bs3 * k3[i] + bs4 * k4[i] + bs5 * k5[i] + bs6 * k6[i] +
                          bs7 * k7[i]);
    }

    finite_state(y5, n);
    finite_state(y4, n);
    const auto [error_norm, error_widened] = scaled_error(y, y5, y4, options);
    const bool accepted                    = error_norm <= 1.;

    if (accepted) {
      t += h;
      if (t > t_end)
        t = t_end;
      y = y5;
      rhs.sample(y);
      solution.t.push_back(t);
      solution.y.push_back(y);
      ++solution.steps_taken;
    }

    rhs.finish(y5, t, accepted, error_norm, error_widened);
    const double safety = 0.9;
    const double growth = error_norm > 0.0 ? safety * std::pow(error_norm, -0.2) : 5.0;
    h *= std::clamp(growth, 0.2, 5.0);
  }
  return solution;
}

// Explicit Adams-Bashforth (order 2-4) and, when `moulton` is true, an Adams-Bashforth-Moulton
// PECE predictor-corrector -- the "implicit" flavored linear multistep option, avoiding a full
// Newton solve. Self-starts using classical RK4 for the first (order - 1) steps.
/// @brief Execute actual constant-grid Adams/one-PECE with RK4 startup and shortened tail.
ODESolution solve_adams(
    CheckedRHS&       rhs,
    const State&      y0,
    const double      t_start,
    const double      t_end,
    const ODEOptions& options,
    const bool        moulton) {
  const std::size_t order = std::clamp<std::size_t>(options.multistep_order, 2, 4);
  const double      h     = options.step_size;

  // Coefficients index 0 = f_n (most recent), increasing index = further into the past.
  /// @brief Original constant-step Adams-Bashforth coefficient rows, indexed by order.
  static const std::vector<std::vector<double>> ab_coeffs = {
      {},
      {},
      {3.0 / 2, -1.0 / 2},
      {23.0 / 12, -16.0 / 12, 5.0 / 12},
      {55.0 / 24, -59.0 / 24, 37.0 / 24, -9.0 / 24}};
  // am_coeffs[0] applies to the (predicted) f_{n+1}; index k>=1 applies to f_{n+1-k}.
  /// @brief Original single-PECE Adams-Moulton coefficient rows, indexed by order.
  static const std::vector<std::vector<double>> am_coeffs = {
      {},
      {},
      {1.0 / 2, 1.0 / 2},
      {5.0 / 12, 8.0 / 12, -1.0 / 12},
      {9.0 / 24, 19.0 / 24, -5.0 / 24, 1.0 / 24}};

  ODESolution solution;
  rhs.sample(y0);
  solution.t.push_back(t_start);
  solution.y.push_back(y0);

  double            t = t_start;
  State             y = y0;
  std::deque<State> f_history; // back() = most recent
  f_history.push_back(rhs.evaluate(t, y));
  ++solution.function_evaluations;

  for (std::size_t i = 1; i < order && t < t_end; ++i) {
    const double step_h = advancing_step(h, t, t_end);
    rhs.begin(ODEAttemptKind::rk4_startup, t, y, step_h);
    y = rk4_step(rhs, t, y, step_h, solution.function_evaluations);
    t += step_h;
    if (t > t_end)
      t = t_end;
    rhs.sample(y);
    solution.t.push_back(t);
    solution.y.push_back(y);
    ++solution.steps_taken;
    f_history.push_back(rhs.evaluate(t, y));
    ++solution.function_evaluations;
    rhs.finish(y, t, true);
  }

  const auto& ab = ab_coeffs[order];
  const auto& am = am_coeffs[order];

  while (t < t_end) {
    const double step_h = advancing_step(h, t, t_end);
    const bool   tail   = step_h != h;
    rhs.begin(
        tail ? ODEAttemptKind::rk4_tail
             : (moulton ? ODEAttemptKind::adams_pece : ODEAttemptKind::adams_bashforth),
        t,
        y,
        step_h);
    const std::size_t n = y.size();

    State y_next;
    if (tail)
      y_next = rk4_step(rhs, t, y, step_h, solution.function_evaluations);
    else {
      State predictor = y;
      for (std::size_t k = 0; k < order; ++k) {
        const auto& f_k = f_history[f_history.size() - 1 - k];
        for (std::size_t i = 0; i < n; ++i)
          predictor[i] += step_h * ab[k] * f_k[i];
      }

      if (moulton) {
        const auto f_pred = rhs.evaluate(t + step_h, predictor);
        ++solution.function_evaluations;
        y_next = y;
        for (std::size_t i = 0; i < n; ++i)
          y_next[i] += step_h * am[0] * f_pred[i];
        for (std::size_t k = 0; k + 1 < order; ++k) {
          const auto& f_k = f_history[f_history.size() - 1 - k];
          for (std::size_t i = 0; i < n; ++i)
            y_next[i] += step_h * am[k + 1] * f_k[i];
        }
      } else {
        y_next = predictor;
      }
    }
    t += step_h;
    if (t > t_end)
      t = t_end;
    y = y_next;
    rhs.sample(y);
    solution.t.push_back(t);
    solution.y.push_back(y);
    ++solution.steps_taken;

    f_history.push_back(rhs.evaluate(t, y));
    ++solution.function_evaluations;
    if (f_history.size() > order)
      f_history.pop_front();
    rhs.finish(y, t, true);
  }

  return solution;
}

/// @brief Original finite-parameter named-system derivative implementation.
class ExponentialDecayRHS : public RHS {
 public:
  /// @brief Own the validated original builtin parameter.
  explicit ExponentialDecayRHS(const double k) : k_(k) {}
  /// @brief Return the actual named-system derivative, guarded by CheckedRHS.
  /// @param y Actual original finite state with exact builtin dimension.
  /// @return Owned actual derivative in original component order.
  std::vector<double> evaluate(double, const std::vector<double>& y) override {
    return {-k_ * y.at(0)};
  }

 private:
  double k_; ///< Owned validated original builtin coefficient.
};

/// @brief Original finite-parameter named-system derivative implementation.
class LogisticGrowthRHS : public RHS {
 public:
  /// @brief Own the validated original builtin parameters.
  LogisticGrowthRHS(const double r, const double capacity) : r_(r), capacity_(capacity) {}
  /// @brief Return the actual named-system derivative, guarded by CheckedRHS.
  /// @param y Actual original finite state with exact builtin dimension.
  /// @return Owned actual derivative in original component order.
  std::vector<double> evaluate(double, const std::vector<double>& y) override {
    return {r_ * y.at(0) * (1.0 - y.at(0) / capacity_)};
  }

 private:
  double r_;        ///< Owned validated original builtin coefficient.
  double capacity_; ///< Owned validated original builtin coefficient.
};

/// @brief Original finite-parameter named-system derivative implementation.
class HarmonicOscillatorRHS : public RHS {
 public:
  /// @brief Own the validated original builtin parameter.
  explicit HarmonicOscillatorRHS(const double omega) : omega_sq_(omega * omega) {}
  /// @brief Return the actual named-system derivative, guarded by CheckedRHS.
  /// @param y Actual original finite state with exact builtin dimension.
  /// @return Owned actual derivative in original component order.
  std::vector<double> evaluate(double, const std::vector<double>& y) override {
    return {y.at(1), -omega_sq_ * y.at(0)};
  }

 private:
  double omega_sq_; ///< Owned validated original builtin coefficient.
};

/// @brief Original finite-parameter named-system derivative implementation.
class VanDerPolRHS : public RHS {
 public:
  /// @brief Own the validated original builtin parameter.
  explicit VanDerPolRHS(const double mu) : mu_(mu) {}
  /// @brief Return the actual named-system derivative, guarded by CheckedRHS.
  /// @param y Actual original finite state with exact builtin dimension.
  /// @return Owned actual derivative in original component order.
  std::vector<double> evaluate(double, const std::vector<double>& y) override {
    return {y.at(1), mu_ * (1.0 - y.at(0) * y.at(0)) * y.at(1) - y.at(0)};
  }

 private:
  double mu_; ///< Owned validated original builtin coefficient.
};

/// @brief Original finite-parameter named-system derivative implementation.
class LorenzRHS : public RHS {
 public:
  /// @brief Own the validated original builtin parameters.
  LorenzRHS(const double sigma, const double rho, const double beta)
      : sigma_(sigma), rho_(rho), beta_(beta) {}
  /// @brief Return the actual named-system derivative, guarded by CheckedRHS.
  /// @param y Actual original finite state with exact builtin dimension.
  /// @return Owned actual derivative in original component order.
  std::vector<double> evaluate(double, const std::vector<double>& y) override {
    return {
        sigma_ * (y.at(1) - y.at(0)),
        y.at(0) * (rho_ - y.at(2)) - y.at(1),
        y.at(0) * y.at(1) - beta_ * y.at(2)};
  }

 private:
  double sigma_; ///< Owned validated original builtin coefficient.
  double rho_;   ///< Owned validated original builtin coefficient.
  double beta_;  ///< Owned validated original builtin coefficient.
};

/// @brief Read an existing finite positional parameter or its documented default.
/// @param params Complete finite supplied coefficients.
/// @param index Original positional index.
/// @param fallback Documented default coefficient.
/// @return Supplied coefficient or fallback.
double param_or(const std::vector<double>& params, const std::size_t index, const double fallback) {
  return index < params.size() ? params[index] : fallback;
}

/// @brief Construct one validated original builtin callback before integration.
/// @param system Exact builtin name.
/// @param params Complete finite supplied coefficients; finite extras ignored.
/// @param dimension Complete original initial-state dimension.
/// @return Owned validated named derivative implementation.
/// @throws std::invalid_argument If name/parameter/state dimension validation fails.
std::unique_ptr<RHS>
builtin(const std::string& system, const std::vector<double>& params, std::size_t dimension) {
  for (double x : params)
    if (!std::isfinite(x))
      throw std::invalid_argument("ODE builtin: complete finite parameters required");
  std::size_t expected = 0;
  if (system == "exponential_decay" || system == "logistic_growth")
    expected = 1;
  else if (system == "harmonic_oscillator" || system == "van_der_pol")
    expected = 2;
  else if (system == "lorenz")
    expected = 3;
  else
    throw std::invalid_argument("ODE builtin: unknown system");
  if (dimension != expected)
    throw std::invalid_argument("ODE builtin: exact initial-state dimension required");
  if (system == "exponential_decay")
    return std::make_unique<ExponentialDecayRHS>(param_or(params, 0, 1));
  if (system == "logistic_growth") {
    double capacity = param_or(params, 1, 1);
    if (capacity <= 0)
      throw std::invalid_argument("ODE builtin: positive logistic capacity required");
    return std::make_unique<LogisticGrowthRHS>(param_or(params, 0, 1), capacity);
  }
  if (system == "harmonic_oscillator") {
    double omega = param_or(params, 0, 1);
    if (!std::isfinite(omega * omega))
      throw std::invalid_argument("ODE builtin: finite squared frequency required");
    return std::make_unique<HarmonicOscillatorRHS>(omega);
  }
  if (system == "van_der_pol")
    return std::make_unique<VanDerPolRHS>(param_or(params, 0, 1));
  return std::make_unique<LorenzRHS>(
      param_or(params, 0, 10), param_or(params, 1, 28), param_or(params, 2, 8. / 3));
}
/// @brief Validate complete problem then dispatch exactly one checked original numerical path.
/// @param original Borrowed live derivative only during execution.
/// @param y0 Complete finite nonempty original state.
/// @param start Finite original start.
/// @param end Finite strictly later terminal time.
/// @param options Complete validated numerical controls.
/// @param limits Original optional execution/history policies.
/// @param history Optional complete local actual-history sink.
/// @return Actual complete endpoint-inclusive trajectory.
/// @throws std::invalid_argument If complete controls/state or callback shape fail.
/// @throws std::length_error If complete/actual geometry or selected policies fail.
/// @throws std::overflow_error If stages/time/proposals cannot remain finite and advancing.
ODESolution execute(
    RHS&               original,
    const State&       y0,
    double             start,
    double             end,
    const ODEOptions&  options,
    ODEExecutionLimits limits,
    ODETrace*          history) {
  validate_options(options);
  require_forward_span(start, end);
  if (y0.empty())
    throw std::invalid_argument("ODE: nonempty initial state required");
  for (double x : y0)
    if (!std::isfinite(x))
      throw std::invalid_argument("ODE: complete finite initial state required");
  if (history) {
    auto base = checked_add(8, y0.size());
    if (limits.max_trace_entries && base > limits.max_trace_entries)
      throw std::length_error("ODE: complete root trace quota");
    history->captured_entries = base;
    history->options          = options;
    history->limits           = limits;
    history->initial_state    = y0;
    history->start            = start;
    history->end              = end;
  }
  CheckedRHS  checked(original, y0.size(), limits, history);
  ODESolution solution;
  switch (options.method) {
    case StepMethod::Euler:
      solution = solve_fixed_step(checked, y0, start, end, options.step_size, euler_step);
      break;
    case StepMethod::Midpoint:
      solution = solve_fixed_step(checked, y0, start, end, options.step_size, midpoint_step);
      break;
    case StepMethod::RK4:
      solution = solve_fixed_step(checked, y0, start, end, options.step_size, rk4_step);
      break;
    case StepMethod::RK45:
      solution = solve_rk45(checked, y0, start, end, options);
      break;
    case StepMethod::AdamsBashforth:
      solution = solve_adams(checked, y0, start, end, options, false);
      break;
    case StepMethod::AdamsMoulton:
      solution = solve_adams(checked, y0, start, end, options, true);
      break;
  }
  solution.function_evaluations = checked.evaluations();
  return solution;
}
} // namespace
ODESolver::ODESolver(ODEOptions options) : options_(options) {
  validate_options(options_);
}
ODESolution ODESolver::solve(RHS& rhs, const State& y0, double start, double end) const {
  return execute(rhs, y0, start, end, options_, {}, nullptr);
}
ODESolution ODESolver::solve_with_limits(
    RHS&               rhs,
    const State&       y0,
    double             start,
    double             end,
    ODEExecutionLimits limits) const {
  return execute(rhs, y0, start, end, options_, limits, nullptr);
}
ODETrace ODESolver::trace(
    RHS&               rhs,
    const State&       y0,
    double             start,
    double             end,
    ODEExecutionLimits limits) const {
  ODETrace t;
  t.solution = execute(rhs, y0, start, end, options_, limits, &t);
  return t;
}
ODESolution ODESolver::solve_builtin(
    const std::string&         system,
    const std::vector<double>& params,
    const State&               y0,
    double                     start,
    double                     end) const {
  auto rhs = builtin(system, params, y0.size());
  return execute(*rhs, y0, start, end, options_, {}, nullptr);
}
ODESolution ODESolver::solve_builtin_with_limits(
    const std::string&         system,
    const std::vector<double>& params,
    const State&               y0,
    double                     start,
    double                     end,
    ODEExecutionLimits         limits) const {
  auto rhs = builtin(system, params, y0.size());
  return execute(*rhs, y0, start, end, options_, limits, nullptr);
}
ODETrace ODESolver::trace_builtin(
    const std::string&         system,
    const std::vector<double>& params,
    const State&               y0,
    double                     start,
    double                     end,
    ODEExecutionLimits         limits) const {
  auto     rhs = builtin(system, params, y0.size());
  ODETrace t;
  t.solution = execute(*rhs, y0, start, end, options_, limits, &t);
  return t;
}
} // namespace datamunge::ode
