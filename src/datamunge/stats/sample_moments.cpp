/// @file sample_moments.cpp
/// @brief Native estimators for weighted covariance and standardized sample moments.

#include <datamunge/stats/detail/test_common.hpp>
#include <datamunge/stats/sample_moments.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {

/// @brief Helpers for conversion of widened moment estimates.
namespace {

/// @brief Narrows a moment to double and checks its range.
/// @param value Widened moment.
/// @return Finite double.
/// @throws std::invalid_argument A nonfinite or out-of-range moment.
double check_moment(long double value) {
  if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
    throw std::invalid_argument("sample moments: moment overflow");
  return static_cast<double>(value);
}

} // namespace

WeightedSampleMoments weighted_sample_moments(const linalg::DenseMatrix<double>& values,
                                              const std::vector<double>& weights,
                                              SampleWeightType type) {
  if (values.rows() == 0 || values.cols() == 0 || weights.size() != values.rows())
    throw std::invalid_argument("weighted_sample_moments: weights must match a nonempty matrix");
  const bool analytic = type == SampleWeightType::Analytic;
  if (!analytic && type != SampleWeightType::Frequency)
    throw std::invalid_argument("weighted_sample_moments: unknown weight type");
  for (const double weight : weights) {
    if (!std::isfinite(weight) || weight < 0 || (!analytic && weight != std::floor(weight)))
      throw std::invalid_argument("weighted_sample_moments: invalid weight");
  }
  const auto max_weight = std::max_element(weights.begin(), weights.end());
  if (*max_weight == 0)
    throw std::invalid_argument("weighted_sample_moments: need positive weight");
  const auto origin_row = static_cast<std::size_t>(max_weight - weights.begin());
  const auto p = values.cols();
  std::vector<long double> offsets(p), origin(p);
  for (std::size_t j = 0; j < p; ++j)
    origin[j] = values(origin_row, j);

  long double total = 0, weight_pairs = 0;
  for (std::size_t i = 0; i < values.rows(); ++i) {
    const long double w = static_cast<long double>(weights[i]) / *max_weight;
    weight_pairs += w * total;
    total += w;
    for (std::size_t j = 0; j < p; ++j) {
      const double x = values(i, j);
      if (!std::isfinite(x))
        throw std::invalid_argument("weighted_sample_moments: observations must be finite");
      if (w > 0)
        offsets[j] += w * (x - origin[j]);
    }
  }
  // The pair sum avoids cancellation when one analytic weight dominates.
  const long double divisor =
      analytic ? 2 * weight_pairs / total : total - 1.0L / *max_weight;
  if (!(divisor > 0))
    throw std::invalid_argument("weighted_sample_moments: nonpositive covariance divisor");

  WeightedSampleMoments result{std::vector<double>(p), {p, p, 0.0}};
  for (std::size_t j = 0; j < p; ++j) {
    offsets[j] /= total;
    result.mean[j] = check_moment(origin[j] + offsets[j]);
  }
  for (std::size_t j = 0; j < p; ++j) {
    for (std::size_t k = 0; k <= j; ++k) {
      long double covariance = 0;
      for (std::size_t i = 0; i < values.rows(); ++i) {
        if (weights[i] == 0)
          continue;
        const long double w = static_cast<long double>(weights[i]) / *max_weight;
        covariance += w * ((values(i, j) - origin[j]) - offsets[j]) *
                          ((values(i, k) - origin[k]) - offsets[k]);
      }
      result.covariance(j, k) = result.covariance(k, j) = check_moment(covariance / divisor);
    }
  }
  return result;
}

SampleShape sample_shape(const std::vector<double>& values, bool bias_correct) {
  detail::check_sample(values, bias_correct ? 4 : 2);
  const auto [minimum_it, maximum_it] = std::minmax_element(values.begin(), values.end());
  const long double minimum = *minimum_it, maximum = *maximum_it;
  if (minimum == maximum)
    throw std::invalid_argument("sample_shape: positive variance required");
  const long double n = values.size();
  // Work with unit-scaled observations so higher powers stay representable.
  const long double scale = std::max(std::abs(minimum), std::abs(maximum));
  const long double origin = minimum / scale;
  long double mean = 0;
  for (const auto x : values)
    mean += x / scale - origin;
  mean /= n;
  long double m2 = 0, m3 = 0, m4 = 0;
  for (const auto x : values) {
    const long double centered = (x / scale - origin) - mean;
    const long double square = centered * centered;
    m2 += square;
    m3 += square * centered;
    m4 += square * square;
  }
  m2 /= n;
  m3 /= n;
  m4 /= n;
  long double skewness = m3 / (m2 * std::sqrt(m2));
  long double excess = m4 / (m2 * m2) - 3;
  if (bias_correct) {
    skewness *= std::sqrt(n * (n - 1)) / (n - 2);
    excess = (n - 1) * ((n + 1) * excess + 6) / ((n - 2) * (n - 3));
  }
  return {static_cast<double>(skewness), static_cast<double>(excess)};
}

SampleLMoments sample_l_moments(const std::vector<double>& values) {
  detail::check_sample(values, 4);
  auto sorted = values;
  std::sort(sorted.begin(), sorted.end());
  const long double low = sorted.front(), high = sorted.back();
  if (low == high)
    return {sorted.front(), 0, 0, 0, std::nullopt, std::nullopt};

  const auto origin = std::midpoint(low, high);
  const auto range_scale = std::max(high - origin, origin - low);
  const long double n = sorted.size();
  std::array<long double, 4> pwm{};
  for (std::size_t i = 0; i < sorted.size(); ++i) {
    const auto x = ((sorted[i] - origin) / range_scale) / n;
    const auto rank = static_cast<long double>(i);
    long double weight = 1;
    pwm[0] += x;
    for (std::size_t r = 1; r < pwm.size() && r <= i; ++r) {
      weight *= (rank - (r - 1)) / (n - r);
      pwm[r] += weight * x;
    }
  }
  const auto second = 2 * pwm[1] - pwm[0];
  const auto third = 6 * (pwm[2] - pwm[1]) + pwm[0];
  const auto fourth = 20 * pwm[3] - 30 * pwm[2] + 12 * pwm[1] - pwm[0];
  return {check_moment(origin + pwm[0] * range_scale), check_moment(second * range_scale),
          check_moment(third * range_scale), check_moment(fourth * range_scale),
          check_moment(third / second), check_moment(fourth / second)};
}

LorenzCurve lorenz_curve(const std::vector<double>& values, const std::vector<double>& weights) {
  detail::check_sample(values, 1);
  if (!weights.empty() && weights.size() != values.size())
    throw std::invalid_argument("lorenz_curve: weights must match observations");
  std::vector<std::pair<double, double>> ordered;
  ordered.reserve(values.size());
  double max_weight = 0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    const auto weight = weights.empty() ? 1.0 : weights[i];
    if (values[i] < 0 || !std::isfinite(weight) || weight < 0)
      throw std::invalid_argument("lorenz_curve: expected nonnegative observations and weights");
    if (weight > 0) {
      ordered.emplace_back(values[i], weight);
      max_weight = std::max(max_weight, weight);
    }
  }
  std::sort(ordered.begin(), ordered.end());
  if (ordered.empty() || ordered.back().first == 0)
    throw std::invalid_argument("lorenz_curve: positive total weighted value required");
  const long double max_value = ordered.back().first;
  std::vector<long double> suffix(ordered.size() + 1);
  long double total_value = 0;
  for (std::size_t i = ordered.size(); i > 0; --i) {
    const long double weight = static_cast<long double>(ordered[i - 1].second) / max_weight;
    suffix[i - 1] = suffix[i] + weight;
    total_value += weight * (ordered[i - 1].first / max_value);
  }
  if (!(total_value > 0))
    throw std::invalid_argument("lorenz_curve: weighted value underflow");

  LorenzCurve result{{0}, {0}, 0};
  result.population_share.reserve(ordered.size() + 1);
  result.value_share.reserve(ordered.size() + 1);
  long double population = 0, value_sum = 0, gini = 0, previous = 0;
  for (std::size_t i = 0; i < ordered.size(); ++i) {
    const auto [value, raw_weight] = ordered[i];
    const auto x = value / max_value;
    // Across each adjacent gap, the pairwise absolute differences accrue once
    // for every pair straddling that gap.
    gini += ((x - previous) * suffix[i] / total_value) * (population / suffix.front());
    const auto weight = static_cast<long double>(raw_weight) / max_weight;
    population += weight;
    value_sum += weight * x;
    result.population_share.push_back(static_cast<double>(std::min(1.0L, population / suffix.front())));
    result.value_share.push_back(static_cast<double>(std::min(1.0L, value_sum / total_value)));
    previous = x;
  }
  result.population_share.back() = 1;
  result.value_share.back() = 1;
  result.gini = static_cast<double>(std::clamp(gini, 0.0L, 1.0L));
  return result;
}

std::vector<std::pair<double, double>> fieller_confidence_set(
    double numerator, double denominator, double numerator_variance, double denominator_variance,
    double covariance, double critical_value) {
  if (!std::isfinite(numerator) || !std::isfinite(denominator) || !std::isfinite(covariance) ||
      !(numerator_variance >= 0 && std::isfinite(numerator_variance)) ||
      !(denominator_variance >= 0 && std::isfinite(denominator_variance)) ||
      !(critical_value > 0 && std::isfinite(critical_value)))
    throw std::invalid_argument("fieller_confidence_set: invalid estimates or uncertainty");
  const auto sigma_x = std::sqrt(static_cast<long double>(numerator_variance));
  const auto sigma_y = std::sqrt(static_cast<long double>(denominator_variance));
  const auto smallest_sigma = std::min(sigma_x, sigma_y);
  if (smallest_sigma == 0 && covariance != 0)
    throw std::invalid_argument("fieller_confidence_set: covariance matrix must be positive semidefinite");
  const long double correlation =
      covariance == 0 ? 0 : (covariance / std::max(sigma_x, sigma_y)) / smallest_sigma;
  if (std::abs(correlation) > 1 + 8 * std::numeric_limits<double>::epsilon())
    throw std::invalid_argument("fieller_confidence_set: covariance matrix must be positive semidefinite");

  constexpr auto infinity = std::numeric_limits<double>::infinity();
  if (sigma_x == 0 && sigma_y == 0) {
    if (denominator != 0) {
      const auto ratio = numerator / denominator;
      return {{ratio, ratio}};
    }
    if (numerator == 0)
      return {{-infinity, infinity}};
    return {};
  }
  const auto unit = std::max({std::abs(static_cast<long double>(numerator)),
                             std::abs(static_cast<long double>(denominator)), sigma_x, sigma_y});
  const auto rho = std::clamp(correlation, -1.0L, 1.0L);
  const auto x = (numerator / unit) / std::max(1.0, critical_value);
  const auto y = (denominator / unit) / std::max(1.0, critical_value);
  const auto sx = (sigma_x / unit) * std::min(1.0, critical_value);
  const auto sy = (sigma_y / unit) * std::min(1.0, critical_value);
  const auto a = (y - sy) * (y + sy);
  const auto b = -2 * std::fma(x, y, -rho * sx * sy);
  const auto c = (x - sx) * (x + sx);
  if (a == 0) {
    if (b == 0) {
      if (c <= 0)
        return {{-infinity, infinity}};
      return {};
    }
    const auto endpoint = static_cast<double>(-c / b);
    if (b > 0)
      return {{-infinity, endpoint}};
    return {{endpoint, infinity}};
  }
  // This factorization avoids subtracting two large nearly equal products
  // when the input covariance is close to singular.
  const auto gap = std::fma(x, sy, -rho * y * sx);
  const auto quarter_discriminant = std::fma((1 - rho) * (1 + rho) * sx * sx, a, gap * gap);
  if (quarter_discriminant < 0 || (quarter_discriminant == 0 && a < 0)) {
    if (a < 0)
      return {{-infinity, infinity}};
    return {};
  }
  const auto q = -.5L * b - std::copysign(std::sqrt(quarter_discriminant), b);
  const auto first = q / a;
  const auto second = q == 0 ? first : c / q;
  const auto lower = static_cast<double>(std::min(first, second));
  const auto upper = static_cast<double>(std::max(first, second));
  if (a > 0)
    return {{lower, upper}};
  return {{-infinity, lower}, {upper, infinity}};
}

linalg::DenseMatrix<double> delta_method_covariance(const linalg::DenseMatrix<double>& jacobian,
                                                      const linalg::DenseMatrix<double>& covariance) {
  const auto p = jacobian.cols(), q = jacobian.rows();
  if (p == 0 || q == 0 || covariance.rows() != p || covariance.cols() != p)
    throw std::invalid_argument("delta_method_covariance: incompatible or empty matrix dimensions");
  linalg::DenseMatrix<long double> derivatives(q, p), symmetric(p, p);
  for (std::size_t i = 0; i < q; ++i) {
    for (std::size_t j = 0; j < p; ++j) {
      const auto value = jacobian(i, j);
      if (!std::isfinite(value))
        throw std::invalid_argument("delta_method_covariance: Jacobian entries must be finite");
      derivatives(i, j) = value;
    }
  }
  for (std::size_t i = 0; i < p; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      const long double lower = covariance(i, j), upper = covariance(j, i);
      if (!std::isfinite(lower) || !std::isfinite(upper) || (i == j && lower < 0) ||
          std::abs(lower - upper) > 8 * std::numeric_limits<double>::epsilon() *
              std::max(std::abs(lower), std::abs(upper)))
        throw std::invalid_argument("delta_method_covariance: covariance must be finite and symmetric with nonnegative diagonal");
      symmetric(i, j) = symmetric(j, i) = std::midpoint(lower, upper);
    }
  }
  const auto transformed = (derivatives * symmetric) * derivatives.transpose();
  linalg::DenseMatrix<double> result(q, q);
  for (std::size_t i = 0; i < q; ++i)
    for (std::size_t j = 0; j <= i; ++j)
      result(i, j) = result(j, i) = check_moment(std::midpoint(transformed(i, j), transformed(j, i)));
  return result;
}

} // namespace datamunge::stats
