/// @file muller.cpp
/// @brief Shared checked real quadratic root producer with actual callback/proposal capture.
#include <cmath>
#include <datamunge/algorithms/muller.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private finite arithmetic and canonical iteration implementation.
namespace {
/// @brief Reject a produced nonfinite scalar before it can enter a callback or comparison.
/// @param value Actual produced arithmetic value.
/// @return The same finite scalar.
/// @throws std::overflow_error If value is nonfinite.
double finite(double value) {
  if (!std::isfinite(value))
    throw std::overflow_error("muller: nonfinite produced arithmetic");
  return value;
}
/// @brief Execute retained callback/update scheduling with optional actual capture.
/// @param f Validated-in-body finite-valued callable.
/// @param x Three finite seed values, updated locally.
/// @param tol Finite nonnegative tolerance.
/// @param budget Nonnegative update cap below INT_MAX.
/// @param out Optional actual-history sink.
/// @param cap Actual capture quota, zero disables.
/// @return Actual root summary, stop and work.
/// @throws std::invalid_argument If callable/seeds/controls are invalid.
/// @throws std::domain_error If a callback returns nonfinite data.
/// @throws std::overflow_error If produced arithmetic is nonfinite.
/// @throws std::length_error If capture admission exceeds its limit.
MullerRun run(
    const std::function<double(double)>& f,
    std::array<double, 3>                x,
    double                               tol,
    int                                  budget,
    MullerTrace*                         out,
    std::size_t                          cap) {
  if (!f || !std::isfinite(tol) || tol < 0 || budget < 0 ||
      budget == std::numeric_limits<int>::max())
    throw std::invalid_argument("muller: invalid callable or controls");
  for (auto v : x)
    if (!std::isfinite(v))
      throw std::invalid_argument("muller: finite seeds required");
  /// @brief Admit actual history units before recording/evaluating the next observation.
  auto admit = [&](std::size_t amount) {
    if (!out)
      return;
    if (amount > std::numeric_limits<std::size_t>::max() - out->captured_entries ||
        (cap && (out->captured_entries > cap || amount > cap - out->captured_entries)))
      throw std::length_error("muller: capture quota exceeded");
    out->captured_entries += amount;
  };
  if (out) {
    admit(5);
    out->initial        = x;
    out->tolerance      = tol;
    out->max_iterations = budget;
  }
  MullerRun result;
  /// @brief Make one actual finite callback and capture its original coordinate/value.
  auto evaluate = [&](double point, bool initial) {
    if (out)
      admit(1);
    auto value = f(point);
    if (!std::isfinite(value))
      throw std::domain_error("muller: nonfinite callback result");
    ++result.evaluations;
    if (out)
      out->evaluations.push_back({point, value, initial});
    return value;
  };
  std::array<double, 3> values;
  for (std::size_t i = 0; i < 3; ++i)
    values[i] = evaluate(x[i], true);
  /// @brief Append one admitted actual residual/proposal inspection.
  auto capture = [&](const MullerStep& step) {
    if (out) {
      admit(1);
      out->steps.push_back(step);
    }
  };
  for (int i = 1; i <= budget; ++i) {
    ++result.inspections;
    result.result = {false, x[2], i, std::abs(values[2])};
    MullerStep step;
    step.legacy_iteration = i;
    step.x                = x;
    step.f                = values;
    step.next             = x[2];
    if (std::abs(values[2]) < tol) {
      result.result.converged = true;
      result.stop             = MullerStop::Residual;
      capture(step);
      return result;
    }
    step.h0        = finite(x[1] - x[0]);
    step.h1        = finite(x[2] - x[1]);
    const auto sum = finite(step.h1 + step.h0);
    if (step.h0 == 0 || step.h1 == 0 || sum == 0) {
      result.stop = MullerStop::CoincidentNodes;
      capture(step);
      return result;
    }
    const auto d0          = finite(finite(values[1] - values[0]) / step.h0),
               d1          = finite(finite(values[2] - values[1]) / step.h1);
    step.a                 = finite(finite(d1 - d0) / sum);
    step.b                 = finite(finite(step.a * step.h1) + d1);
    step.c                 = values[2];
    step.discriminant_raw  = finite(finite(step.b * step.b) - finite(finite(4. * step.a) * step.c));
    step.discriminant_used = step.discriminant_raw;
    if (step.discriminant_raw < 0) {
      step.clamped           = true;
      step.discriminant_used = 0;
      ++result.clamped_discriminants;
    }
    step.quadratic    = true;
    const auto square = std::sqrt(step.discriminant_used);
    const auto plus = finite(step.b + square), minus = finite(step.b - square);
    step.denominator = std::abs(plus) > std::abs(minus) ? plus : minus;
    if (step.denominator == 0) {
      result.stop = MullerStop::ZeroDenominator;
      capture(step);
      return result;
    }
    step.next    = finite(x[2] - finite(finite(2. * step.c) / step.denominator));
    step.updated = true;
    capture(step);
    x      = {x[1], x[2], step.next};
    values = {values[1], values[2], evaluate(step.next, false)};
    ++result.updates;
  }
  ++result.inspections;
  result.result = {std::abs(values[2]) < tol, x[2], budget, std::abs(values[2])};
  result.stop   = result.result.converged ? MullerStop::Residual : MullerStop::IterationLimit;
  MullerStep final;
  final.legacy_iteration = budget;
  final.at_limit         = true;
  final.x                = x;
  final.f                = values;
  final.next             = x[2];
  capture(final);
  return result;
}
} // namespace
std::string_view muller_stop_name(MullerStop stop) noexcept {
  switch (stop) {
    case MullerStop::Residual:
      return "residual";
    case MullerStop::CoincidentNodes:
      return "coincident-nodes";
    case MullerStop::ZeroDenominator:
      return "zero-denominator";
    case MullerStop::IterationLimit:
      return "iteration-limit";
  }
  return "unknown";
}
RootResult muller(
    const std::function<double(double)>& f,
    double                               x0,
    double                               x1,
    double                               x2,
    double                               tol,
    int                                  budget) {
  return run(f, {x0, x1, x2}, tol, budget, nullptr, 0).result;
}
MullerRun muller_run(
    const std::function<double(double)>& f,
    double                               x0,
    double                               x1,
    double                               x2,
    double                               tol,
    int                                  budget) {
  return run(f, {x0, x1, x2}, tol, budget, nullptr, 0);
}
MullerTrace muller_trace(
    const std::function<double(double)>& f,
    double                               x0,
    double                               x1,
    double                               x2,
    double                               tol,
    int                                  budget,
    std::size_t                          cap) {
  MullerTrace t;
  t.run = run(f, {x0, x1, x2}, tol, budget, &t, cap);
  return t;
}
} // namespace datamunge::algorithms
