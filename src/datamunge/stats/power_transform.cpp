/// @file power_transform.cpp
/// @brief Numerically scaled power transforms and Gaussian profile likelihood.

#include <datamunge/algorithms/golden_section_search.hpp>
#include <datamunge/stats/detail/test_common.hpp>
#include <datamunge/stats/power_transform.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Private power formulas, validation and likelihood calculations.
namespace {

/// @brief Checks the public transform's parameter domain.
/// @param family Requested transformation formula.
/// @param power Requested power.
/// @throws std::invalid_argument Invalid family or nonfinite power.
void check_parameters(PowerTransformFamily family, double power) {
  if ((family != PowerTransformFamily::BoxCox && family != PowerTransformFamily::YeoJohnson) ||
      !std::isfinite(power))
    throw std::invalid_argument("power transform: invalid family or power");
}

/// @brief Computes (exp(power * log_base) - 1) / power or its logarithmic limit.
/// @param log_base Logarithm of the positive base.
/// @param power Power parameter.
/// @return Transformed value in widened precision.
long double power_value(long double log_base, long double power) {
  const auto exponent = power * log_base;
  if (exponent == 0)
    return log_base;
  if (exponent > 1)
    return std::copysign(std::exp(exponent - std::log(std::abs(power)) +
                                 std::log(-std::expm1(-exponent))), power);
  return std::expm1(exponent) / power;
}

/// @brief Inverts the power formula to obtain the logarithm of its positive base.
/// @param value Transformed value.
/// @param power Power parameter.
/// @return Logarithm of the original base.
/// @throws std::invalid_argument Value outside the invertible range.
long double inverse_log_base(long double value, long double power) {
  const auto product = power * value;
  if (product == 0)
    return value;
  if (product < -.5L) {
    const auto base = std::fma(power, value, 1.L);
    if (!(base > 0))
      throw std::invalid_argument("power transform: value is outside the inverse domain");
    return std::log(base) / power;
  }
  if (std::isinf(product))
    return (std::log(std::abs(power)) + std::log(std::abs(value))) / power;
  return std::log1p(product) / power;
}

/// @brief Converts a widened value to finite double precision without rounding a nonzero value to zero.
/// @param value Widened result.
/// @return Representable result.
/// @throws std::overflow_error Nonfinite, overflowing, or totally underflowing result.
double checked_double(long double value) {
  if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
    throw std::overflow_error("power transform: result exceeds the finite double range");
  const auto result = static_cast<double>(value);
  if (result == 0 && value != 0)
    throw std::overflow_error("power transform: result rounds to zero");
  return result;
}

/// @brief Computes the scaled log-distance between two power-transformed bases.
/// @param low Smaller raw magnitude.
/// @param high Larger raw magnitude.
/// @param power Power used on this side of the transform.
/// @param center Log-base offset used to absorb the likelihood Jacobian.
/// @param shifted Whether the positive base is one plus the raw magnitude.
/// @return Log-distance after scaling by exp(-(power-1) * center).
long double power_log_distance(long double low, long double high, long double power, long double center,
                               bool shifted) {
  // Use the raw increment: adding one to both magnitudes can erase a
  // small but nonzero difference on the Yeo–Johnson branch.
  const auto ratio = (high - low) / (shifted ? 1 + low : low);
  const auto gap = std::isfinite(ratio) ? std::log1p(ratio) : std::log(high) - std::log(low);
  const auto amount = std::abs(power) * gap;
  const auto endpoint = power < 0 ? low : high;
  const auto log_endpoint = shifted ? std::log1p(endpoint) : std::log(endpoint);
  const auto scale = power * (log_endpoint - center) + center;
  if (amount < .5L)
    return scale + std::log(gap) + (amount == 0 ? 0 : std::log(-std::expm1(-amount) / amount));
  return scale + std::log(-std::expm1(-amount)) - std::log(std::abs(power));
}

/// @brief Computes a scaled log-distance from the minimum observation to another point.
/// @param origin Minimum sample value.
/// @param value Observation at or above the minimum.
/// @param lambda Main power parameter.
/// @param center Mean signed log-base value.
/// @param family Power transformation family.
/// @return Log-distance, with negative infinity for another occurrence of the minimum.
long double log_distance(long double origin, long double value, long double lambda, long double center,
                         PowerTransformFamily family) {
  if (value == origin)
    return -std::numeric_limits<long double>::infinity();
  if (family == PowerTransformFamily::BoxCox)
    return power_log_distance(origin, value, lambda, center, false);
  if (origin >= 0)
    return power_log_distance(origin, value, lambda, center, true);
  if (value <= 0)
    return power_log_distance(-value, -origin, 2 - lambda, -center, true);
  const auto negative = power_log_distance(0, -origin, 2 - lambda, -center, true);
  const auto positive = power_log_distance(0, value, lambda, center, true);
  return std::max(negative, positive) + std::log1p(std::exp(-std::abs(negative - positive)));
}

/// @brief Fits the bounded Gaussian profile with a logarithmically scaled variance.
/// @param sample Input observations.
/// @param lower Lower search endpoint.
/// @param upper Upper search endpoint.
/// @param family Valid transformation family.
/// @return Fitted parameter, likelihood and boundary indicator.
FittedPowerTransform fit_power(const std::vector<double>& sample, double lower, double upper,
                               PowerTransformFamily family) {
  detail::check_sample(sample, 2);
  if (!std::isfinite(lower) || !std::isfinite(upper) || !(lower < upper))
    throw std::invalid_argument("power transform: search bounds must be finite and increasing");
  const auto [minimum, maximum] = std::ranges::minmax_element(sample);
  const bool box_cox = family == PowerTransformFamily::BoxCox;
  if (*minimum == *maximum || (box_cox && *minimum <= 0))
    throw std::invalid_argument("power transform: need distinct observations, all positive for Box–Cox");

  long double center = 0;
  for (const long double value : sample)
    center += (box_cox ? std::log(value) : std::copysign(std::log1p(std::abs(value)), value)) / sample.size();
  std::vector<long double> distances(sample.size());

  /// @brief Evaluates the Jacobian-adjusted profile objective per observation at a power.
  const auto profile = [&](long double lambda) {
    for (std::size_t i = 0; i < sample.size(); ++i)
      distances[i] = log_distance(*minimum, sample[i], lambda, center, family);
    const auto largest = *std::ranges::max_element(distances);
    long double mean = 0, ss = 0;
    std::size_t n = 0;
    for (const auto log_value : distances) {
      const auto scaled = std::exp(log_value - largest);
      const auto delta = scaled - mean;
      mean += delta / ++n;
      ss += delta * (scaled - mean);
    }
    return -largest - .5L * std::log(ss / sample.size());
  };

  // A monotone parameter coordinate keeps very wide user brackets searchable.
  const auto interior = algorithms::golden_section_search(
      [&](double u) {
        const auto lambda = std::clamp(std::sinh(static_cast<long double>(u)),
                                       static_cast<long double>(lower), static_cast<long double>(upper));
        return static_cast<double>(std::asinh(profile(lambda)));
      },
      std::asinh(lower), std::asinh(upper));
  auto lambda = static_cast<double>(std::clamp(std::sinh(static_cast<long double>(interior.argmax)),
                                               static_cast<long double>(lower), static_cast<long double>(upper)));
  auto value = profile(lambda);
  for (const auto edge : {lower, upper}) {
    const auto candidate = profile(edge);
    if (candidate >= value) {
      lambda = edge;
      value = candidate;
    }
  }
  return {family, lambda, checked_double(value * sample.size()), lambda == lower || lambda == upper};
}

} // namespace

double FittedPowerTransform::apply(double value) const {
  check_parameters(family, lambda);
  const bool box_cox = family == PowerTransformFamily::BoxCox;
  if (!std::isfinite(value) || (box_cox && value <= 0))
    throw std::invalid_argument("power transform: invalid observation");
  const bool negative = !box_cox && value < 0;
  const long double x = value;
  const long double power = negative ? 2.L - lambda : lambda;
  if (!box_cox && power == 1)
    return value;
  const auto log_base = box_cox ? std::log(x) : std::log1p(std::abs(x));
  const auto transformed = checked_double(power == 1 ? x - 1 : power_value(log_base, power));
  if (!(std::fma(power, static_cast<long double>(transformed), 1.L) > 0))
    throw std::overflow_error("power transform: result rounds outside the invertible range");
  return negative ? -transformed : transformed;
}

double FittedPowerTransform::inverse(double value) const {
  check_parameters(family, lambda);
  if (!std::isfinite(value))
    throw std::invalid_argument("power transform: transformed observation must be finite");
  const bool box_cox = family == PowerTransformFamily::BoxCox;
  const bool negative = !box_cox && value < 0;
  const long double power = negative ? 2.L - lambda : lambda;
  if (!box_cox && power == 1)
    return value;
  const auto log_base = inverse_log_base(negative ? -value : value, power);
  if (!box_cox)
    return checked_double(negative ? -std::expm1(log_base) : std::expm1(log_base));
  const auto restored = checked_double(std::exp(log_base));
  if (restored == 0)
    throw std::overflow_error("power transform: inverse underflows the positive domain");
  return restored;
}

FittedPowerTransform fit_box_cox(const std::vector<double>& sample, double lower, double upper) {
  return fit_power(sample, lower, upper, PowerTransformFamily::BoxCox);
}

FittedPowerTransform fit_yeo_johnson(const std::vector<double>& sample, double lower, double upper) {
  return fit_power(sample, lower, upper, PowerTransformFamily::YeoJohnson);
}

} // namespace datamunge::stats
