/// @file ternary_search.cpp
/// @brief Actual checked finite bracket narrowing with explicit callback order.
#include <cmath>
#include <datamunge/algorithms/ternary_search.hpp>
#include <limits>
#include <numeric>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private invocation-local ternary producer.
namespace {
/// @brief Validate and execute actual sampled bracket reductions.
TernaryRun execute(
    const std::function<double(double)>& f,
    double                               lo,
    double                               hi,
    const TernaryOptions&                o,
    bool                                 maximum,
    bool                                 capture) {
  if (!f || !std::isfinite(lo) || !std::isfinite(hi) || lo > hi || !std::isfinite(o.tolerance) ||
      o.tolerance < 0 || o.max_iterations < 0)
    throw std::invalid_argument("ternary controls");
  TernaryRun r;
  /// @brief Evaluate the actual borrowed objective and count its invocation.
  auto value = [&](double x) {
    if (r.evaluations == std::numeric_limits<std::size_t>::max())
      throw std::length_error("ternary evaluations");
    ++r.evaluations;
    double v = f(x);
    if (!std::isfinite(v))
      throw std::domain_error("ternary finite objective");
    return v;
  };
  while (r.result.iterations < o.max_iterations &&
         (long double)hi - (long double)lo > o.tolerance) {
    double width = hi - lo;
    double m1    = std::isfinite(width) ? lo + width / 3 : std::lerp(lo, hi, 1. / 3);
    double m2    = std::isfinite(width) ? hi - width / 3 : std::lerp(lo, hi, 2. / 3);
    if (!(m1 > lo && m2 < hi && m1 <= m2)) {
      r.precision_limited = true;
      break;
    }
    double      v1 = value(m1), v2 = value(m2);
    bool        left = maximum ? v1 >= v2 : v1 <= v2;
    TernaryStep step{lo, hi, m1, m2, v1, v2, left, left ? lo : m1, left ? m2 : hi};
    lo = step.lo_after;
    hi = step.hi_after;
    ++r.result.iterations;
    if (capture) {
      if (o.max_records && r.steps.size() >= o.max_records)
        throw std::length_error("ternary history");
      r.steps.push_back(step);
    }
  }
  r.lo           = lo;
  r.hi           = hi;
  r.converged    = (long double)hi - (long double)lo <= o.tolerance;
  r.result.x     = std::midpoint(lo, hi);
  r.result.value = value(r.result.x);
  return r;
}
} // namespace
TernaryRun ternary_search_max_run(
    const std::function<double(double)>& f,
    double                               lo,
    double                               hi,
    const TernaryOptions&                options) {
  return execute(f, lo, hi, options, true, false);
}
TernaryRun ternary_search_max_trace(
    const std::function<double(double)>& f,
    double                               lo,
    double                               hi,
    const TernaryOptions&                options) {
  return execute(f, lo, hi, options, true, true);
}
TernaryResult ternary_search_max(
    const std::function<double(double)>& f,
    double                               lo,
    double                               hi,
    double                               tol,
    int                                  max_iter) {
  return ternary_search_max_run(f, lo, hi, {tol, max_iter}).result;
}
TernaryRun ternary_search_min_run(
    const std::function<double(double)>& f,
    double                               lo,
    double                               hi,
    const TernaryOptions&                options) {
  return execute(f, lo, hi, options, false, false);
}
TernaryRun ternary_search_min_trace(
    const std::function<double(double)>& f,
    double                               lo,
    double                               hi,
    const TernaryOptions&                options) {
  return execute(f, lo, hi, options, false, true);
}
TernaryResult ternary_search_min(
    const std::function<double(double)>& f,
    double                               lo,
    double                               hi,
    double                               tol,
    int                                  max_iter) {
  return ternary_search_min_run(f, lo, hi, {tol, max_iter}).result;
}
} // namespace datamunge::algorithms
