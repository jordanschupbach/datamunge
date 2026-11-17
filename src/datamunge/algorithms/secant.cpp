/// @file secant.cpp
/// @brief One finite actual secant stream, normalized proposals and honest unresolved outcomes.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/secant.hpp>
#include <limits>
#include <stdexcept>
/// @brief Canonical unbracketed scalar iteration implementation.
namespace datamunge::algorithms {
/// @brief Private checked callback/proposal execution.
/// @brief Private invocation-local checked algorithm helpers.
namespace {
/// @brief Execute actual initial queries and normalized secant proposals once.
/// @param f Borrowed callback.
/// @param x0 Earlier point.
/// @param x1 Latest point.
/// @param o Controls.
/// @param capture Retain actual query/proposal records.
/// @return Owned actual matching diagnostics and histories.
SecantRun solve(
    const std::function<double(double)>& f,
    double                               x0,
    double                               x1,
    const SecantOptions&                 o,
    bool                                 capture) {
  if (!f || !std::isfinite(x0) || !std::isfinite(x1) || !std::isfinite(o.tolerance) ||
      o.tolerance < 0 ||
      o.max_iterations > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::invalid_argument("secant: callback/points/controls");
  SecantRun   out;
  std::size_t retained = 0;
  /// @brief Admit one actual optional query/proposal record before its work.
  const auto admit = [&]() {
    if (!capture)
      return;
    if (retained == std::numeric_limits<std::size_t>::max() ||
        (o.max_trace_entries && retained >= o.max_trace_entries))
      throw std::length_error("secant: capture policy");
    ++retained;
  };
  /// @brief Invoke and validate one real callback return.
  /// @param x Actual finite query.
  /// @return Actual finite result.
  const auto evaluate = [&](double x) {
    if (out.function_evaluations == std::numeric_limits<std::size_t>::max() ||
        (o.max_evaluations && out.function_evaluations >= o.max_evaluations))
      throw std::length_error("secant: callback policy");
    admit();
    ++out.function_evaluations;
    const double value = f(x);
    if (!std::isfinite(value))
      throw std::invalid_argument("secant: nonfinite callback");
    if (capture)
      out.evaluations.push_back({x, value});
    return value;
  };
  /// @brief Test only actual represented callback residuals.
  /// @param value Actual finite return.
  /// @return Whether exact zero or the enabled absolute tolerance passes.
  const auto success = [&](double value) {
    return value == 0 || (o.tolerance > 0 && std::fabs(value) <= o.tolerance);
  };
  /// @brief Finish with the actual currently stored matching diagnostics.
  /// @param stop Actual terminal branch.
  /// @return Owned result with honest convergence flag.
  const auto finish = [&](SecantStop stop) {
    out.stop             = stop;
    out.result.converged = stop == SecantStop::Residual;
    return out;
  };
  double f0 = evaluate(x0), f1 = evaluate(x1);
  out.result.root     = x1;
  out.result.residual = std::fabs(f1);
  if (success(f0)) {
    out.result.root     = x0;
    out.result.residual = std::fabs(f0);
    return finish(SecantStop::Residual);
  }
  if (success(f1))
    return finish(SecantStop::Residual);
  for (std::size_t i = 0; i < o.max_iterations; ++i) {
    admit();
    ++out.proposals;
    SecantStep event;
    event.previous   = x0;
    event.current    = x1;
    event.f_previous = f0;
    event.f_current  = f1;
    if (f0 == f1) {
      event.proposal = x1;
      if (capture)
        out.steps.push_back(event);
      return finish(SecantStop::Horizontal);
    }
    const long double scale = std::max(
                          std::fabs(static_cast<long double>(f0)),
                          std::fabs(static_cast<long double>(f1))),
                      a = f0 / scale, b = f1 / scale, denominator = b - a, ratio = b / denominator;
    event.normalized_denominator = denominator;
    const long double proposal =
        ratio >= 0 && ratio <= 1
            ? (1 - ratio) * x1 + ratio * x0
            : static_cast<long double>(x1) - ratio * (static_cast<long double>(x1) - x0);
    event.proposal = proposal;
    if (!std::isfinite(proposal) || std::fabs(proposal) > std::numeric_limits<double>::max()) {
      if (capture)
        out.steps.push_back(event);
      return finish(SecantStop::ArithmeticRange);
    }
    const double x2 = static_cast<double>(proposal);
    if (x2 == x1) {
      if (capture)
        out.steps.push_back(event);
      return finish(SecantStop::Stagnation);
    }
    const auto f2   = evaluate(x2);
    event.evaluated = true;
    event.value     = f2;
    ++out.result.iterations;
    out.result.root     = x2;
    out.result.residual = std::fabs(f2);
    if (capture)
      out.steps.push_back(event);
    if (success(f2))
      return finish(SecantStop::Residual);
    x0 = x1;
    f0 = f1;
    x1 = x2;
    f1 = f2;
  }
  return finish(SecantStop::Budget);
}
} // namespace
SecantRun
secant_run(const std::function<double(double)>& f, double x0, double x1, const SecantOptions& o) {
  return solve(f, x0, x1, o, false);
}
SecantRun
secant_trace(const std::function<double(double)>& f, double x0, double x1, const SecantOptions& o) {
  return solve(f, x0, x1, o, true);
}
RootResult secant(const std::function<double(double)>& f, double x0, double x1, double tol, int n) {
  if (n < 0)
    throw std::invalid_argument("secant: negative attempt budget");
  SecantOptions o;
  o.max_iterations = static_cast<std::size_t>(n);
  o.tolerance      = tol;
  return solve(f, x0, x1, o, false).result;
}
} // namespace datamunge::algorithms
