/// @file newton_raphson.cpp
/// @brief Canonical checked Newton producer preserving actual legacy callback scheduling.
#include <cmath>
#include <datamunge/algorithms/newton_raphson.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private count admission and single shared callback/update loop.
namespace {
/// @brief Execute complete control admission and actual Newton steps.
/// @param f Original function callback, finite returns required.
/// @param df Original supplied derivative callback, finite returns required.
/// @param x0 Finite original point.
/// @param tol Finite nonnegative absolute residual threshold.
/// @param budget Nonnegative update cap below INT_MAX.
/// @param out Optional actual-history sink.
/// @param cap Actual online history policy, zero disables.
/// @return Actual result, work and first numerical stop.
/// @throws std::invalid_argument If complete controls/callables fail.
/// @throws std::domain_error If a callback returns nonfinite.
/// @throws std::overflow_error If a correction/next point is nonfinite.
/// @throws std::length_error If count geometry/capture policy is exceeded.
NewtonRaphsonRun run(
    const std::function<double(double)>& f,
    const std::function<double(double)>& df,
    double                               x0,
    double                               tol,
    int                                  budget,
    NewtonRaphsonTrace*                  out,
    std::size_t                          cap) {
  if (!f || !df || !std::isfinite(x0) || !std::isfinite(tol) || tol < 0 || budget < 0 ||
      budget == std::numeric_limits<int>::max())
    throw std::invalid_argument("Newton: complete finite controls/callables required");
  /// @brief Admit one actual captured record count before allocation/callback.
  auto admit = [&](std::size_t n) {
    if (!out)
      return;
    if (n > std::numeric_limits<std::size_t>::max() - out->captured_entries)
      throw std::length_error("Newton: count overflow");
    auto next = out->captured_entries + n;
    if (cap && next > cap)
      throw std::length_error("Newton: online capture policy");
    out->captured_entries = next;
  };
  if (out) {
    admit(3);
    out->initial        = x0;
    out->tolerance      = tol;
    out->max_iterations = budget;
  }
  NewtonRaphsonRun r;
  /// @brief Evaluate one actual admitted f/df callback and retain its value once.
  auto evaluate = [&](double x, bool derivative) {
    admit(1);
    double value = derivative ? df(x) : f(x);
    if (derivative)
      ++r.derivative_calls;
    else
      ++r.function_calls;
    if (!std::isfinite(value))
      throw std::domain_error("Newton: nonfinite callback");
    if (out)
      out->evaluations.push_back({x, value, derivative});
    return value;
  };
  double x = x0;
  for (int index = 0;; ++index) {
    admit(1);
    NewtonRaphsonStep s;
    s.x        = x;
    s.next     = x;
    s.at_limit = index == budget;
    s.fx       = evaluate(x, false);
    ++r.inspections;
    r.result.root       = x;
    r.result.residual   = std::abs(s.fx);
    r.result.iterations = s.at_limit ? budget : index + 1;
    if (r.result.residual < tol) {
      r.result.converged = true;
      r.stop             = NewtonRaphsonStop::Residual;
      if (out)
        out->steps.push_back(s);
      return r;
    }
    if (s.at_limit) {
      r.stop = NewtonRaphsonStop::IterationLimit;
      if (out)
        out->steps.push_back(s);
      return r;
    }
    s.derivative           = evaluate(x, true);
    s.derivative_evaluated = true;
    if (s.derivative == 0) {
      r.stop = NewtonRaphsonStop::ZeroDerivative;
      if (out)
        out->steps.push_back(s);
      return r;
    }
    s.correction = s.fx / s.derivative;
    s.next       = x - s.correction;
    if (!std::isfinite(s.correction) || !std::isfinite(s.next))
      throw std::overflow_error("Newton: nonfinite correction/point");
    s.updated = true;
    x         = s.next;
    ++r.updates;
    if (out)
      out->steps.push_back(s);
  }
}
} // namespace
std::string_view newton_raphson_stop_name(NewtonRaphsonStop stop) noexcept {
  switch (stop) {
    case NewtonRaphsonStop::Residual:
      return "residual";
    case NewtonRaphsonStop::ZeroDerivative:
      return "zero-derivative";
    case NewtonRaphsonStop::IterationLimit:
      return "iteration-limit";
  }
  return "unknown";
}
RootResult newton_raphson(
    const std::function<double(double)>& f,
    const std::function<double(double)>& df,
    double                               x,
    double                               tol,
    int                                  max_iter) {
  return run(f, df, x, tol, max_iter, nullptr, 0).result;
}
NewtonRaphsonRun newton_raphson_run(
    const std::function<double(double)>& f,
    const std::function<double(double)>& df,
    double                               x,
    double                               tol,
    int                                  max_iter) {
  return run(f, df, x, tol, max_iter, nullptr, 0);
}
NewtonRaphsonTrace newton_raphson_trace(
    const std::function<double(double)>& f,
    const std::function<double(double)>& df,
    double                               x,
    double                               tol,
    int                                  max_iter,
    std::size_t                          cap) {
  NewtonRaphsonTrace t;
  t.run = run(f, df, x, tol, max_iter, &t, cap);
  return t;
}
NewtonRaphsonSquareStudy newton_raphson_square_study(double x, double tol, int budget) {
  NewtonRaphsonSquareStudy s;
  s.trace = newton_raphson_trace(
      [](double v) { return v * v - 2; }, [](double v) { return 2 * v; }, x, tol, budget);
  s.reference = std::sqrt(2.);
  for (int i = 0; i <= 50; ++i) {
    double v = 1.15 + i / 50.;
    s.curve_x.push_back(v);
    s.curve_y.push_back(v * v - 2);
  }
  for (auto& step : s.trace.steps) {
    s.indices.push_back(s.states.size());
    s.states.push_back(step.x);
    s.values.push_back(step.fx);
    s.reference_errors.push_back(std::abs(step.x - s.reference));
    if (step.updated) {
      s.tangent_x.push_back(step.x);
      s.tangent_y.push_back(step.fx);
      s.tangent_next.push_back(step.next);
    }
  }
  return s;
}
} // namespace datamunge::algorithms
