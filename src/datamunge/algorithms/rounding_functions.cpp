/// @file rounding_functions.cpp
/// @brief One original-fraction rounding producer and widened actual error summaries.
#include <cmath>
#include <datamunge/algorithms/rounding_functions.hpp>
#include <stdexcept>
/// @brief Canonical scalar rounding implementation.
namespace datamunge::algorithms {
std::string_view round_mode_name(RoundMode mode) {
  switch (mode) {
    case RoundMode::HalfToEven:
      return "half-to-even";
    case RoundMode::HalfAwayFromZero:
      return "half-away-from-zero";
    case RoundMode::HalfUp:
      return "half-up";
    case RoundMode::HalfDown:
      return "half-down";
    case RoundMode::TowardZero:
      return "toward-zero";
    case RoundMode::Ceil:
      return "ceil";
    case RoundMode::Floor:
      return "floor";
  }
  throw std::invalid_argument("rounding: unsupported rule");
}
double round_with(double x, RoundMode mode) {
  round_mode_name(mode);
  if (!std::isfinite(x) || x == 0)
    return x;
  if (mode == RoundMode::TowardZero)
    return std::trunc(x);
  if (mode == RoundMode::Ceil)
    return std::ceil(x);
  if (mode == RoundMode::Floor)
    return std::floor(x);
  // Avoid cancellation around negative half even when long double has double precision.
  if (x < 0 && x > -1) {
    if (x > -.5)
      return 0;
    if (x < -.5)
      return -1;
  }
  const double      lower    = std::floor(x);
  const long double fraction = static_cast<long double>(x) - lower;
  if (fraction < .5)
    return lower;
  if (fraction > .5)
    return lower + 1;
  switch (mode) {
    case RoundMode::HalfToEven:
      return std::fmod(lower, 2) == 0 ? lower : lower + 1;
    case RoundMode::HalfAwayFromZero:
      return x < 0 ? lower : lower + 1;
    case RoundMode::HalfUp:
      return lower + 1;
    case RoundMode::HalfDown:
      return lower;
    default:
      throw std::logic_error("rounding: internal mode");
  }
}
double round_half_to_even(double x) {
  return round_with(x, RoundMode::HalfToEven);
}
double round_half_away_from_zero(double x) {
  return round_with(x, RoundMode::HalfAwayFromZero);
}
double round_half_up(double x) {
  return round_with(x, RoundMode::HalfUp);
}
double round_half_down(double x) {
  return round_with(x, RoundMode::HalfDown);
}
double round_toward_zero(double x) {
  return round_with(x, RoundMode::TowardZero);
}
/// @brief Private original-sample summary producer.
namespace {
/// @brief Validate and round a complete finite sample once.
/// @param values Borrowed observations.
/// @param mode Supported rule.
/// @param cap Complete allocation policy.
/// @param capture Retain actual decisions.
/// @return Owned sample outputs and observed summary.
RoundingRun
solve(const std::vector<double>& values, RoundMode mode, std::size_t cap, bool capture) {
  round_mode_name(mode);
  if (cap && values.size() > cap)
    throw std::length_error("rounding: sample policy");
  for (double x : values)
    if (!std::isfinite(x))
      throw std::invalid_argument("rounding: nonfinite sample");
  RoundingRun out;
  out.mode = mode;
  out.rounded.reserve(values.size());
  if (capture)
    out.steps.reserve(values.size());
  long double total = 0;
  for (double x : values) {
    const double      lower = std::floor(x), value = round_with(x, mode), error = value - x;
    const long double fraction      = static_cast<long double>(x) - lower;
    double            integral_part = 0;
    const bool        tie           = std::modf(std::fabs(x), &integral_part) == .5;
    out.ties += tie;
    out.rounded.push_back(value);
    total += static_cast<long double>(value) - x;
    out.mean_error = static_cast<double>(total / out.rounded.size());
    if (capture)
      out.steps.push_back({x, lower, fraction, tie, value, error, out.mean_error});
  }
  return out;
}
} // namespace
RoundingRun rounding_run(const std::vector<double>& v, RoundMode m, std::size_t cap) {
  return solve(v, m, cap, false);
}
RoundingRun rounding_trace(const std::vector<double>& v, RoundMode m, std::size_t cap) {
  return solve(v, m, cap, true);
}
double rounding_bias(const std::vector<double>& v, RoundMode m) {
  return solve(v, m, 1000000, false).mean_error;
}
std::vector<double> rounding_example_ties() {
  std::vector<double> values;
  for (int i = 0; i < 60; ++i)
    values.push_back(i + .5);
  return values;
}
} // namespace datamunge::algorithms
