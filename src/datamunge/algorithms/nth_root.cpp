/// @file nth_root.cpp
/// @brief Checked positive-root Newton producer with bounded small-degree powers and explicit logarithmic large-degree route.
#include <cmath>
#include <datamunge/algorithms/nth_root.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private complete admission, safe initialization and actual root-update producer.
namespace {
/// @brief Produce finite principal-root states and actual arithmetic-route records.
/// @param a Original finite nonnegative radicand.
/// @param n Original positive degree.
/// @param options Complete actual update/tolerance/history controls.
/// @param out Optional owned actual-history sink.
/// @return Actual finite final state and selected work.
/// @throws std::invalid_argument If complete inputs/options fail.
/// @throws std::overflow_error If seed/ratio/next state is not finite positive.
/// @throws std::length_error If actual capture/count policy is exceeded.
NthRootRun run(double a, int n, NthRootOptions options, NthRootTrace* out) {
  if (!std::isfinite(a) || a < 0 || n < 1 || options.max_iterations < 0 ||
      options.max_iterations == std::numeric_limits<int>::max() ||
      !std::isfinite(options.relative_tolerance) || options.relative_tolerance < 0)
    throw std::invalid_argument("nth root: complete finite nonnegative/positive controls required");
  if (out) {
    if (options.max_trace_entries && options.max_trace_entries < 4)
      throw std::length_error("nth root: capture base");
    out->radicand         = a;
    out->degree           = n;
    out->options          = options;
    out->captured_entries = 4;
  }
  NthRootRun r;
  if (n == 1) {
    r.value = r.initial = a;
    r.identity_shortcut = r.converged = true;
    return r;
  }
  if (a == 0) {
    r.zero_shortcut = r.converged = true;
    return r;
  }
  r.mantissa              = std::frexp(a, &r.exponent);
  const long double log_a = std::log(static_cast<long double>(a));
  r.log_seed              = n > 64;
  if (r.log_seed) {
    r.initial = double(std::exp(log_a / n));
  } else {
    int exponent = r.exponent / n;
    if (r.exponent > 0 && r.exponent % n)
      ++exponent;
    r.initial = std::ldexp(1., exponent);
  }
  if (!std::isfinite(r.initial) || r.initial <= 0)
    throw std::overflow_error("nth root: invalid seed");
  double x = r.initial;
  for (int i = 0; i < options.max_iterations; ++i) {
    if (out) {
      if (out->captured_entries == std::numeric_limits<std::size_t>::max() ||
          (options.max_trace_entries && out->captured_entries == options.max_trace_entries))
        throw std::length_error("nth root: capture quota");
      ++out->captured_entries;
    }
    NthRootStep s;
    s.before    = x;
    s.log_ratio = r.log_seed;
    if (s.log_ratio) {
      s.ratio =
          std::exp(log_a - static_cast<long double>(n - 1) * std::log(static_cast<long double>(x)));
      s.after = double((static_cast<long double>(n - 1) * x + s.ratio) / n);
      ++r.log_ratio_updates;
    } else {
      double power = 1;
      for (int k = 0; k < n - 1; ++k) {
        power *= x;
        ++s.power_multiplications;
      }
      s.power_available = std::isfinite(power) && power > 0;
      if (s.power_available) {
        s.power      = power;
        double ratio = a / power;
        s.ratio      = ratio;
        s.after      = ((n - 1) * x + ratio) / n;
        s.widened    = power < std::numeric_limits<double>::min() ||
                       !std::isfinite(ratio) || !std::isfinite(s.after) || s.after <= 0;
      } else
        s.widened = true;
      if (s.widened) {
        long double power_wide = 1;
        for (int k = 0; k < n - 1; ++k) {
          power_wide *= static_cast<long double>(x);
          ++s.power_multiplications;
        }
        s.ratio = static_cast<long double>(a) / power_wide;
        s.after = double((static_cast<long double>(n - 1) * x + s.ratio) / n);
        ++r.widened_updates;
      }
    }
    if (!std::isfinite(s.ratio) || s.ratio <= 0 || !std::isfinite(s.after) || s.after <= 0)
      throw std::overflow_error("nth root: nonfinite/zero ratio or state");
    s.relative_change = std::abs(s.after - x) / std::abs(s.after);
    if (s.power_multiplications > std::numeric_limits<std::size_t>::max() - r.power_multiplications)
      throw std::length_error("nth root: multiplication count geometry");
    r.power_multiplications += s.power_multiplications;
    ++r.updates;
    if (out)
      out->steps.push_back(s);
    x = s.after;
    if (options.relative_tolerance > 0 && s.relative_change <= options.relative_tolerance) {
      r.converged = true;
      break;
    }
  }
  r.value = x;
  return r;
}
} // namespace
double nth_root(double a, int n) {
  return run(a, n, {}, nullptr).value;
}
NthRootRun nth_root_run(double a, int n, NthRootOptions options) {
  return run(a, n, options, nullptr);
}
NthRootTrace nth_root_trace(double a, int n, NthRootOptions options) {
  NthRootTrace t;
  t.run = run(a, n, options, &t);
  return t;
}
NthRootStudy nth_root_demo_study(NthRootOptions options) {
  NthRootStudy s;
  s.trace     = nth_root_trace(50, 3, options);
  s.reference = std::cbrt(50.);
  s.indices.push_back(0);
  s.values.push_back(s.trace.run.initial);
  s.reference_errors.push_back(std::abs(s.trace.run.initial - s.reference));
  for (auto& step : s.trace.steps) {
    s.indices.push_back(s.values.size());
    s.values.push_back(step.after);
    s.reference_errors.push_back(std::abs(step.after - s.reference));
  }
  return s;
}
} // namespace datamunge::algorithms
