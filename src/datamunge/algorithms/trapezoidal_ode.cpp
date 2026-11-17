/// @file trapezoidal_ode.cpp
/// @brief Actual finite-difference Newton solves with residual-based step acceptance.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/trapezoidal_ode.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private per-invocation numerical integration producer.
namespace {
/// @brief Narrow an actual finite widened arithmetic result or fail explicitly.
double narrow(long double x) {
  if (!std::isfinite(x) || std::abs(x) > std::numeric_limits<double>::max())
    throw std::overflow_error("trapezoidal arithmetic");
  return static_cast<double>(x);
}
/// @brief Validate and execute the actual implicit-step/Newton producer.
TrapezoidRun execute(
    const std::function<double(double, double)>& f,
    double                                       y0,
    double                                       t0,
    double                                       t1,
    int                                          count,
    const TrapezoidOptions&                      o,
    bool                                         capture) {
  if (!f || !std::isfinite(y0) || !std::isfinite(t0) || !std::isfinite(t1) || count <= 0 ||
      o.max_newton_iterations < 0 || !std::isfinite(o.absolute_tolerance) ||
      o.absolute_tolerance < 0 || !std::isfinite(o.relative_tolerance) || o.relative_tolerance < 0)
    throw std::invalid_argument("trapezoidal complete controls");
  if (o.max_steps && std::size_t(count) > o.max_steps)
    throw std::length_error("trapezoidal grid");
  TrapezoidRun r;
  r.trajectory.emplace_back(t0, y0);
  double y = y0, t = t0;
  /// @brief Invoke the actual borrowed RHS with finite result validation and call accounting.
  auto value = [&](double time, double state) {
    if (r.evaluations == std::numeric_limits<std::size_t>::max() ||
        (o.max_evaluations && r.evaluations >= o.max_evaluations))
      throw std::length_error("trapezoidal evaluations");
    ++r.evaluations;
    double v = f(time, state);
    if (!std::isfinite(v))
      throw std::domain_error("trapezoidal finite RHS");
    return v;
  };
  /// @brief Check the actual equation residual against the configured absolute/relative bound.
  auto small = [&](double g, double previous, double candidate) {
    long double threshold =
        (long double)o.absolute_tolerance +
        (long double)o.relative_tolerance * std::max(std::abs(previous), std::abs(candidate));
    return std::abs((long double)g) <= threshold;
  };
  /// @brief Retain one actual derivative/correction attempt under record policy.
  auto record = [&](TrapezoidNewton s) {
    if (!capture)
      return;
    if (o.max_records && r.newton.size() >= o.max_records)
      throw std::length_error("trapezoidal Newton history");
    r.newton.push_back(s);
  };
  for (int n = 0; n < count; ++n) {
    double        tn = n + 1 == count ? t1 : std::lerp(t0, t1, double(n + 1) / count);
    double        h  = narrow((long double)tn - t);
    TrapezoidStep step;
    step.time_before = t;
    step.time_after  = tn;
    step.previous    = y;
    if (!h) {
      step.predictor = step.candidate = y;
      step.converged                  = true;
      r.steps.push_back(step);
      r.trajectory.emplace_back(tn, y);
      ++r.completed_steps;
      t = tn;
      continue;
    }
    double fn = value(t, y), Y = narrow((long double)y + (long double)h * fn), fY = value(tn, Y);
    /// @brief Evaluate the actual implicit equation using already sampled RHS values.
    auto equation = [&](double candidate, double rhs) {
      return narrow((long double)candidate - y - (long double)h / 2 * ((long double)fn + rhs));
    };
    double g              = equation(Y, fY);
    step.rhs_before       = fn;
    step.predictor        = Y;
    bool        converged = small(g, y, Y);
    std::string failure   = "Newton iteration limit";
    int         updates   = 0;
    for (int it = 0; !converged && it < o.max_newton_iterations; ++it) {
      double eps       = 1e-8 * (std::abs(Y) + 1e-8);
      double perturbed = Y + eps;
      if (!std::isfinite(perturbed) || perturbed == Y)
        perturbed = std::nextafter(Y, 0.);
      if (perturbed == Y) {
        failure = "precision exhausted";
        break;
      }
      double          fp = value(tn, perturbed), delta = perturbed - Y;
      double          gp = narrow(1 - (long double)h / 2 * ((long double)fp - fY) / delta);
      TrapezoidNewton event{std::size_t(n), it + 1, Y, g, perturbed, fp, gp, false, Y, g};
      if (gp == 0) {
        failure = "zero Newton derivative";
        record(event);
        break;
      }
      double next    = narrow((long double)Y - (long double)g / gp);
      bool   stalled = next == Y;
      double fnext = value(tn, next), after = equation(next, fnext);
      ++updates;
      ++r.newton_updates;
      event.corrected      = true;
      event.next           = next;
      event.residual_after = after;
      record(event);
      Y         = next;
      fY        = fnext;
      g         = after;
      converged = small(g, y, Y);
      if (!converged && stalled) {
        failure = "precision exhausted";
        break;
      }
    }
    step.candidate  = Y;
    step.residual   = g;
    step.iterations = updates;
    step.converged  = converged;
    r.steps.push_back(step);
    if (!converged) {
      r.stop = failure;
      return r;
    }
    y = Y;
    t = tn;
    r.trajectory.emplace_back(t, y);
    ++r.completed_steps;
  }
  r.converged = true;
  r.stop      = "complete";
  return r;
}
} // namespace
TrapezoidRun trapezoidal_ode_run(
    const std::function<double(double, double)>& f,
    double                                       y0,
    double                                       t0,
    double                                       t1,
    int                                          steps,
    const TrapezoidOptions&                      options) {
  return execute(f, y0, t0, t1, steps, options, false);
}
TrapezoidRun trapezoidal_ode_trace(
    const std::function<double(double, double)>& f,
    double                                       y0,
    double                                       t0,
    double                                       t1,
    int                                          steps,
    const TrapezoidOptions&                      options) {
  return execute(f, y0, t0, t1, steps, options, true);
}
std::vector<std::pair<double, double>> trapezoidal_ode(
    const std::function<double(double, double)>& f,
    double                                       y0,
    double                                       t0,
    double                                       t1,
    int                                          steps) {
  auto r = trapezoidal_ode_run(f, y0, t0, t1, steps);
  if (!r.converged)
    throw std::runtime_error("trapezoidal unresolved step: " + r.stop);
  return r.trajectory;
}
} // namespace datamunge::algorithms
