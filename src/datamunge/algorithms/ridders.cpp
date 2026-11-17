/// @file ridders.cpp
/// @brief One finite callback stream with normalized interpolation and actual bracket updates.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/ridders.hpp>
#include <limits>
#include <numeric>
#include <stdexcept>
/// @brief Canonical finite scalar bracket implementation.
namespace datamunge::algorithms {
/// @brief Private normalized interpolation execution.
namespace {
/// @brief Compare signs without multiplying callback values.
/// @param a First finite value.
/// @param b Second finite value.
/// @return Whether both are negative or both nonnegative.
bool same(double a, double b) {
  return (a < 0) == (b < 0);
}
/// @brief Execute the actual checked midpoint/correction callback stream.
/// @param f Borrowed callback.
/// @param a First endpoint.
/// @param b Second endpoint.
/// @param o Controls.
/// @param capture Retain actual query/step records.
/// @return Owned final point, diagnostics and optional histories.
RiddersRun solve(
    const std::function<double(double)>& f,
    double                               a,
    double                               b,
    const RiddersOptions&                o,
    bool                                 capture) {
  if (!f || !std::isfinite(a) || !std::isfinite(b) || !std::isfinite(o.residual_tolerance) ||
      !std::isfinite(o.x_tolerance) || o.residual_tolerance < 0 || o.x_tolerance < 0 ||
      o.max_iterations > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::invalid_argument("Ridders: callback/endpoints/controls");
  if (b < a)
    std::swap(a, b);
  RiddersRun  out;
  std::size_t retained = 0;
  /// @brief Reserve one actual optional query/step record before its work.
  const auto admit = [&]() {
    if (!capture)
      return;
    if (retained == std::numeric_limits<std::size_t>::max() ||
        (o.max_trace_entries && retained >= o.max_trace_entries))
      throw std::length_error("Ridders: history policy");
    ++retained;
  };
  /// @brief Query and validate one actual callback result.
  /// @param x Actual finite scalar query.
  /// @return Actual finite callback return.
  const auto evaluate = [&](double x) {
    if (out.function_evaluations == std::numeric_limits<std::size_t>::max() ||
        (o.max_evaluations && out.function_evaluations >= o.max_evaluations))
      throw std::length_error("Ridders: callback policy");
    admit();
    ++out.function_evaluations;
    const auto value = f(x);
    if (!std::isfinite(value))
      throw std::invalid_argument("Ridders: nonfinite callback");
    if (capture)
      out.evaluations.push_back({x, value});
    return value;
  };
  double fa = evaluate(a), fb = evaluate(b);
  out.result.root     = std::fabs(fa) <= std::fabs(fb) ? a : b;
  out.result.residual = std::min(std::fabs(fa), std::fabs(fb));
  /// @brief Update the actual final bracket and stopping branch.
  /// @param stop Actual stop kind.
  /// @return Owned final result.
  const auto finish = [&](RiddersStop stop) {
    out.stop             = stop;
    out.lower            = a;
    out.upper            = b;
    out.result.converged = stop == RiddersStop::Residual || stop == RiddersStop::Interval;
    return out;
  };
  if (fa == 0 || fb == 0) {
    out.bracketed = true;
    return finish(RiddersStop::Residual);
  }
  if (same(fa, fb))
    return finish(RiddersStop::NoBracket);
  out.bracketed = true;
  if (o.residual_tolerance > 0 && out.result.residual <= o.residual_tolerance)
    return finish(RiddersStop::Residual);
  if (o.x_tolerance > 0 && static_cast<long double>(b) - a <= o.x_tolerance)
    return finish(RiddersStop::Interval);
  for (std::size_t iteration = 0; iteration < o.max_iterations; ++iteration) {
    const double mid = std::midpoint(a, b);
    if (mid == a || mid == b)
      return finish(RiddersStop::Stagnation);
    admit();
    RiddersStep event;
    event.a              = a;
    event.b              = b;
    event.fa             = fa;
    event.fb             = fb;
    event.midpoint       = mid;
    const double fm      = evaluate(mid);
    event.midpoint_value = fm;
    double candidate = mid, value = fm;
    bool residual = fm == 0 || (o.residual_tolerance > 0 && std::fabs(fm) <= o.residual_tolerance);
    if (!residual) {
      const long double scale = std::max(
                            {std::fabs(static_cast<long double>(fa)),
                             std::fabs(static_cast<long double>(fb)),
                             std::fabs(static_cast<long double>(fm))}),
                        sa = fa / scale, sb = fb / scale, sm = fm / scale,
                        radicand = sm * sm - sa * sb;
      event.normalized_radicand  = radicand;
      const long double corrected =
          static_cast<long double>(mid) +
          (static_cast<long double>(mid) - a) * (fa >= fb ? 1.L : -1.L) * sm / std::sqrt(radicand);
      candidate = static_cast<double>(corrected);
      if (!std::isfinite(candidate) || !(candidate > a && candidate < b)) {
        candidate         = mid;
        event.safeguarded = true;
      }
      if (candidate != mid) {
        value                      = evaluate(candidate);
        event.correction_evaluated = true;
      }
      residual =
          value == 0 || (o.residual_tolerance > 0 && std::fabs(value) <= o.residual_tolerance);
    }
    event.candidate       = candidate;
    event.candidate_value = value;
    out.result.root       = candidate;
    out.result.residual   = std::fabs(value);
    ++out.result.iterations;
    if (!residual) {
      if (!same(fm, value)) {
        a  = mid;
        fa = fm;
        b  = candidate;
        fb = value;
      } else if (!same(fa, value)) {
        b  = candidate;
        fb = value;
      } else {
        a  = candidate;
        fa = value;
      }
      if (b < a) {
        std::swap(a, b);
        std::swap(fa, fb);
      }
    }
    event.a_after = a;
    event.b_after = b;
    if (capture)
      out.steps.push_back(event);
    if (residual)
      return finish(RiddersStop::Residual);
    if (o.x_tolerance > 0 && static_cast<long double>(b) - a <= o.x_tolerance)
      return finish(RiddersStop::Interval);
  }
  return finish(RiddersStop::Budget);
}
} // namespace
RiddersRun
ridders_run(const std::function<double(double)>& f, double a, double b, const RiddersOptions& o) {
  return solve(f, a, b, o, false);
}
RiddersRun
ridders_trace(const std::function<double(double)>& f, double a, double b, const RiddersOptions& o) {
  return solve(f, a, b, o, true);
}
RootResult ridders(const std::function<double(double)>& f, double a, double b, double tol, int n) {
  if (n < 0)
    throw std::invalid_argument("Ridders: negative budget");
  RiddersOptions o;
  o.max_iterations     = static_cast<std::size_t>(n);
  o.residual_tolerance = tol;
  o.x_tolerance        = tol;
  return solve(f, a, b, o, false).result;
}
} // namespace datamunge::algorithms
