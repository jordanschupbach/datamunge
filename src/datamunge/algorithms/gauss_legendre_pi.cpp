/// @file gauss_legendre_pi.cpp
/// @brief Shared fixed-count AGM recurrence, actual records and unclipped reference diagnostics.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/gauss_legendre_pi.hpp>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace datamunge::algorithms {
static_assert(
    std::numeric_limits<double>::radix == 2,
    "AGM doubling model requires radix-two double");
/// @brief Internal bounded count checks and original ordered scalar execution.
namespace gauss_legendre_detail {
/// @brief Validate a checked count before initialization or allocation.
/// @param n Exact stage count.
/// @throws std::invalid_argument Count is negative or would use an overflowed weight.
void count(int n) {
  if (n < 0 || n > std::numeric_limits<double>::max_exponent)
    throw std::invalid_argument("Gauss-Legendre stage count");
}
/// @brief Check a used scalar arithmetic result, excluding final unused weight overflow.
/// @param value Actual stored result.
/// @throws std::overflow_error Used result is nonfinite.
void finite(double value) {
  if (!std::isfinite(value))
    throw std::overflow_error("Gauss-Legendre used arithmetic");
}
/// @brief Check widened reference diagnostic arithmetic.
/// @param value Computed wide result.
/// @return The same finite value.
/// @throws std::overflow_error Wide arithmetic is nonfinite.
long double wide(long double value) {
  if (!std::isfinite(value))
    throw std::overflow_error("Pi reference arithmetic");
  return value;
}
/// @brief Evaluate and count the actual rounded final-estimate expression.
/// @param state Actual means/t; p is not used.
/// @param work Actual counters to increment.
/// @return Finite computed estimate.
/// @throws std::overflow_error Estimate is nonfinite.
double estimate(const GaussLegendreState& state, GaussLegendreWork& work) {
  ++work.estimate_evaluations;
  const double value = (state.a + state.b) * (state.a + state.b) / (4.0 * state.t);
  finite(value);
  return value;
}
/// @brief Execute all admitted stages once with optional per-stage analysis and recording.
/// Count and any history quota have been checked. The final unused weight may
/// overflow; every used weight and computed mean/t/estimate remains checked.
/// @param n Exact admitted stage count.
/// @param analyse Whether to evaluate initial/per-stage estimates and change flags.
/// @param trace Nullable whole-admitted actual history.
/// @return Owned state and actual work; legacy mode leaves analysis indices at -1.
/// @throws std::overflow_error A used arithmetic result is nonfinite.
/// @throws std::bad_alloc History allocation fails.
GaussLegendreRun run(int n, bool analyse, GaussLegendreTrace* trace) {
  GaussLegendreRun out;
  out.iterations        = n;
  out.initial           = {1.0, 1.0 / std::sqrt(2.0), 0.25, 1.0};
  out.work.square_roots = 1;
  auto   state          = out.initial;
  double previous       = 0;
  if (analyse)
    previous = estimate(state, out.work);
  for (int i = 0; i < n; ++i) {
    const auto before = state;
    finite(state.p);
    const double a_next = 0.5 * (state.a + state.b);
    ++out.work.arithmetic_means;
    const double b_next = std::sqrt(state.a * state.b);
    ++out.work.geometric_means;
    ++out.work.square_roots;
    const double correction = state.p * (state.a - a_next) * (state.a - a_next);
    finite(correction);
    state.t -= correction;
    ++out.work.correction_updates;
    state.a = a_next;
    state.b = b_next;
    state.p *= 2.0;
    ++out.work.power_updates;
    finite(state.a);
    finite(state.b);
    finite(state.t);
    if (analyse) {
      const double value       = estimate(state, out.work);
      const bool   changed     = value != previous;
      const bool means_changed = state.a != before.a || state.b != before.b || state.t != before.t;
      if (!changed && out.first_unchanged_estimate == -1)
        out.first_unchanged_estimate = i + 1;
      if (!means_changed && out.first_unchanged_mean_state == -1)
        out.first_unchanged_mean_state = i + 1;
      if (trace)
        trace->steps.push_back({i + 1, before, state, correction, value, changed, means_changed});
      previous = value;
    }
  }
  out.final = state;
  out.value = analyse ? previous : estimate(state, out.work);
  return out;
}
} // namespace gauss_legendre_detail
/// @copydoc gauss_legendre_pi_limit()
int gauss_legendre_pi_limit() {
  return std::numeric_limits<double>::max_exponent;
}
/// @copydoc gauss_legendre_pi_trace_records()
std::size_t gauss_legendre_pi_trace_records(int iterations) {
  gauss_legendre_detail::count(iterations);
  const auto n = static_cast<std::size_t>(iterations);
  if (n > std::vector<GaussLegendreStep>{}.max_size())
    throw std::length_error("Gauss-Legendre history capacity");
  return n + 1;
}
/// @copydoc gauss_legendre_pi_run()
GaussLegendreRun gauss_legendre_pi_run(int iterations) {
  gauss_legendre_detail::count(iterations);
  return gauss_legendre_detail::run(iterations, true, nullptr);
}
/// @copydoc gauss_legendre_pi_trace()
GaussLegendreTrace gauss_legendre_pi_trace(int iterations, std::size_t max_records) {
  const auto bound = gauss_legendre_pi_trace_records(iterations);
  if (max_records && bound > max_records)
    throw std::length_error("Gauss-Legendre whole record quota");
  GaussLegendreTrace out;
  out.run              = gauss_legendre_detail::run(iterations, true, &out);
  out.captured_records = bound;
  return out;
}
/// @copydoc gauss_legendre_pi()
double gauss_legendre_pi(int iterations) {
  const auto n = std::max(0, iterations);
  gauss_legendre_detail::count(n);
  return gauss_legendre_detail::run(n, false, nullptr).value;
}
/// @copydoc pi_reference_accuracy()
PiReferenceAccuracy pi_reference_accuracy(double value) {
  using namespace gauss_legendre_detail;
  if (!std::isfinite(value))
    throw std::invalid_argument("Pi reference finite candidate");
  PiReferenceAccuracy out;
  out.rounded_reference      = std::numbers::pi;
  out.wider_reference        = std::numbers::pi_v<long double>;
  out.rounded_absolute_error = std::fabs(value - out.rounded_reference);
  finite(out.rounded_absolute_error);
  out.wider_absolute_error = std::fabs(wide(static_cast<long double>(value) - out.wider_reference));
  const double spacing =
      std::nextafter(out.rounded_reference, std::numeric_limits<double>::infinity()) -
      out.rounded_reference;
  out.rounded_ulp_offset =
      wide((static_cast<long double>(value) - out.rounded_reference) / spacing);
  out.has_rounded_log_error = out.rounded_absolute_error > 0;
  out.has_wider_log_error   = out.wider_absolute_error > 0;
  if (out.has_rounded_log_error)
    out.rounded_log10_error = std::log10(out.rounded_absolute_error);
  if (out.has_wider_log_error)
    out.wider_log10_error = wide(std::log10(out.wider_absolute_error));
  out.wider_has_more_bits =
      std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits;
  return out;
}
} // namespace datamunge::algorithms
