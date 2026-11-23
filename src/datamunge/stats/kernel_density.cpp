/// @file kernel_density.cpp
/// @brief Gaussian mixture evaluation and unbinned bandwidth objectives.

#include <datamunge/stats/kernel_density.hpp>

#include <datamunge/algorithms/bisection.hpp>
#include <datamunge/stats/detail/test_common.hpp>
#include <datamunge/stats/robust_summary.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Kernels, integration and bandwidth pilot helpers.
namespace {

/// @brief Inverse square root of two.
constexpr auto inverse_root_two = 1 / std::numbers::sqrt2_v<long double>;

/// @brief Standard normal density at zero.
constexpr auto normal_peak = std::numbers::inv_sqrtpi_v<long double> * inverse_root_two;

/// @brief Standard normal probability between standardized endpoints.
/// @param lower Standardized lower endpoint.
/// @param upper Standardized upper endpoint.
/// @param width Standardized width computed before centering.
/// @return Gaussian probability in the interval.
long double normal_interval(long double lower, long double upper, long double width) {
  // With very close endpoints, the midpoint rule avoids cancellation in tail
  // differences. Its relative error here is at most of order 1e-14.
  const auto center = lower + width / 2;
  if (width * (1 + std::abs(center)) < 1e-6L)
    return normal_peak * std::exp(-.5L * center * center) * width;
  if (lower > -1 && upper < 1)
    return .5L * (std::erf(upper * inverse_root_two) - std::erf(lower * inverse_root_two));
  if (lower >= 0)
    return .5L * (std::erfc(lower * inverse_root_two) - std::erfc(upper * inverse_root_two));
  if (upper <= 0)
    return .5L * (std::erfc(-upper * inverse_root_two) - std::erfc(-lower * inverse_root_two));
  return .5L * (std::erf(upper * inverse_root_two) - std::erf(lower * inverse_root_two));
}

/// @brief Sums a scalar kernel over unordered pairs of distinct observations.
/// @tparam Kernel Callable accepting a squared standardized difference.
/// @param sample Finite observations.
/// @param h Standardizing scale in sample units.
/// @param kernel Function of a squared standardized pair difference.
/// @return Unnormalized sum over the strict lower triangle of sample pairs.
template <class Kernel>
long double pair_sum(const std::vector<double>& sample, long double h, Kernel kernel) {
  long double sum = 0;
  for (std::size_t i = 1; i < sample.size(); ++i)
    for (std::size_t j = 0; j < i; ++j) {
      const auto z = (static_cast<long double>(sample[i]) - sample[j]) / h;
      sum += kernel(z * z);
    }
  return sum;
}

/// @brief Estimates a Gaussian-smoothed density derivative functional at a pilot bandwidth.
/// @param sample Finite observations.
/// @param scale Data scale in sample units.
/// @param h Pilot bandwidth in units of scale.
/// @param order Four or six.
/// @return Pilot derivative functional in scaled coordinates.
long double pilot_derivative(const std::vector<double>& sample, long double scale, long double h, int order) {
  /// @brief Evaluates the Hermite factor for the selected derivative and its Gaussian multiplier.
  const auto kernel = [order](long double squared) {
    if (squared > 10000)
      return 0.0L;
    const auto hermite = order == 4 ? (squared - 6) * squared + 3 :
                                      ((squared - 15) * squared + 45) * squared - 15;
    return std::exp(-squared / 2) * hermite;
  };
  const long double n = sample.size();
  const auto sum = n * kernel(0) + 2 * pair_sum(sample, scale * h, kernel);
  return normal_peak * sum / (n * (n - 1) * std::pow(h, order + 1));
}

/// @brief Evaluates convolution and leave-one-out terms for the score or its slope.
/// @tparam Slope Whether to evaluate the slope with respect to log bandwidth.
/// @param sample Finite observations, length at least two.
/// @param scale Data scale used to standardize the objective.
/// @param h Candidate bandwidth in units of scale.
/// @return Objective times scale, or its log-bandwidth slope times scale and h.
template <bool Slope = false>
long double lscv_terms(const std::vector<double>& sample, long double scale, long double h) {
  const long double n = sample.size();
  const auto convolution_weight = std::numbers::inv_sqrtpi_v<long double> / (n * n);
  const auto leave_one_out_weight = 4 * normal_peak / (n * (n - 1));
  /// @brief Sums the off-diagonal convolution and leave-one-out terms for each unordered pair.
  const auto kernel = [&](long double d) {
    const auto e = std::exp(-d / 4);
    if constexpr (Slope) {
      if (d > 10000)
        return 0.0L;
      return e * (convolution_weight * (d / 2 - 1) - leave_one_out_weight * e * (d - 1));
    }
    return e * (convolution_weight - leave_one_out_weight * e);
  };
  const auto diagonal = std::numbers::inv_sqrtpi_v<long double> / (2 * n);
  if constexpr (Slope)
    return -diagonal + pair_sum(sample, scale * h, kernel);
  return (diagonal + pair_sum(sample, scale * h, kernel)) / h;
}

/// @brief Checks that a computed bandwidth is representable as a positive finite double.
/// @param h Bandwidth in sample units.
/// @return Converted bandwidth.
/// @throws std::overflow_error Bandwidth cannot be represented.
double checked_bandwidth(long double h) {
  const auto result = static_cast<double>(h);
  if (!(result > 0 && std::isfinite(result)))
    throw std::overflow_error("kernel density: selected bandwidth out of range");
  return result;
}

/// @brief Calculates sample spread with exact zero for any constant input.
/// @param sample At least two finite observations.
/// @return Sample standard deviation.
long double sample_standard_deviation(const std::vector<double>& sample) {
  const auto moments = detail::test_moments(sample);
  if (std::ranges::all_of(sample, [&](double x) { return x == sample.front(); }))
    return 0;
  return std::sqrt(moments.variance);
}

} // namespace

GaussianKernelDensity::GaussianKernelDensity(std::vector<double> sample, double bandwidth,
                                             std::optional<KdeBoundary> boundary)
    : sample_(std::move(sample)), bandwidth_(bandwidth), boundary_(boundary) {
  detail::check_sample(sample_);
  if (!(bandwidth > 0 && std::isfinite(bandwidth)))
    throw std::invalid_argument("kernel density: bandwidth must be positive and finite");
  if (boundary_ && !std::isfinite(boundary_->location))
    throw std::invalid_argument("kernel density: boundary must be finite");
  if (boundary_ && std::ranges::any_of(sample_, [&](double center) {
        return boundary_->lower ? center < boundary_->location : center > boundary_->location;
      }))
    throw std::invalid_argument("kernel density: observation outside the support");
}

double GaussianKernelDensity::density(double value) const {
  if (std::isnan(value))
    throw std::invalid_argument("kernel density: invalid coordinate");
  if (std::isinf(value) || (boundary_ && (boundary_->lower ? value < boundary_->location :
                                                            value > boundary_->location)))
    return 0;

  const long double point = value;
  long double sum = 0;
  for (const long double center : sample_) {
    const auto z = (point - center) / bandwidth_;
    sum += std::exp(-.5L * z * z);
    if (boundary_) {
      const auto reflected = ((point - boundary_->location) + (center - boundary_->location)) / bandwidth_;
      sum += std::exp(-.5L * reflected * reflected);
    }
  }
  return static_cast<double>((sum / sample_.size()) * normal_peak / bandwidth_);
}

double GaussianKernelDensity::probability(double lower, double upper) const {
  if (!(lower <= upper))
    throw std::invalid_argument("kernel density: invalid interval");
  if (boundary_) {
    if (boundary_->lower)
      lower = std::max(lower, boundary_->location);
    else
      upper = std::min(upper, boundary_->location);
  }
  if (lower >= upper)
    return 0;

  long double sum = 0;
  const long double lo = lower, hi = upper;
  const auto width = (hi - lo) / bandwidth_;
  for (const long double center : sample_) {
    sum += normal_interval((lo - center) / bandwidth_, (hi - center) / bandwidth_, width);
    if (boundary_) {
      const long double boundary = boundary_->location;
      const auto displaced_center = center - boundary;
      sum += normal_interval(((lo - boundary) + displaced_center) / bandwidth_,
                             ((hi - boundary) + displaced_center) / bandwidth_, width);
    }
  }
  return static_cast<double>(std::clamp(sum / sample_.size(), 0.0L, 1.0L));
}

double kde_sheather_jones_bandwidth(const std::vector<double>& sample) {
  const auto sd = sample_standard_deviation(sample);
  const auto quartiles = sample_quantiles(sample, {.25, .75});
  const auto iqr = (static_cast<long double>(quartiles.back()) - quartiles.front()) / 1.349L;
  const auto scale = iqr > 0 ? std::min(sd, iqr) : sd;
  if (!(scale > 0 && std::isfinite(scale)))
    throw std::invalid_argument("Sheather–Jones bandwidth: sample must have positive finite spread");

  const long double n = sample.size();
  const auto sixth = -pilot_derivative(sample, scale, 1.23L * std::pow(n, -1.0L / 9), 6);
  if (!(sixth > 0 && std::isfinite(sixth)))
    throw std::invalid_argument("Sheather–Jones bandwidth: invalid sixth-derivative pilot");
  const auto fourth = pilot_derivative(sample, scale, std::pow(2.394L / (n * sixth), 1.0L / 7), 4);
  if (!(fourth > 0 && std::isfinite(fourth)))
    throw std::invalid_argument("Sheather–Jones bandwidth: invalid fourth-derivative pilot");
  return checked_bandwidth(scale * std::pow(std::numbers::inv_sqrtpi_v<long double> / (2 * n * fourth), .2L));
}

KdeLscvResult kde_lscv_bandwidth(const std::vector<double>& sample,
                                std::optional<std::pair<double, double>> range) {
  const auto sd = sample_standard_deviation(sample);
  if (range && !(range->first > 0 && range->first < range->second && std::isfinite(range->second)))
    throw std::invalid_argument("LSCV bandwidth: require positive finite ordered endpoints");
  if (!range && !(sd > 0 && std::isfinite(sd)))
    throw std::invalid_argument("LSCV bandwidth: explicit search interval required for zero or nonfinite spread");
  const long double scale = range ? range->first : sd;
  const auto upper_default = 1.144L * std::pow(static_cast<long double>(sample.size()), -.2L);
  const long double lo = range ? range->first / scale : .1L * upper_default;
  const long double hi = range ? range->second / scale : upper_default;
  const auto lower_log = static_cast<double>(std::log(lo));
  const auto upper_log = static_cast<double>(std::log(hi));

  /// @brief Converts a search coordinate into a scaled bandwidth inside the requested range.
  const auto scaled_bandwidth = [&](double log_h) {
    return std::clamp(std::exp(static_cast<long double>(log_h)), lo, hi);
  };
  /// @brief Evaluates the slope with the bandwidth factor removed, preserving its sign.
  const auto slope = [&](double log_h) {
    return static_cast<double>(lscv_terms<true>(sample, scale, scaled_bandwidth(log_h)));
  };

  double best_log = lower_log;
  auto best_score = lscv_terms(sample, scale, lo);
  /// @brief Retains a candidate when it has a lower score.
  const auto consider = [&](double log_h) {
    const auto score = lscv_terms(sample, scale, scaled_bandwidth(log_h));
    if (score < best_score) {
      best_log = log_h;
      best_score = score;
    }
  };
  consider(upper_log);
  /// @brief Fixed grid for detecting multiple candidate minima before local refinement.
  constexpr int subdivisions = 64;
  auto previous = lower_log;
  auto previous_slope = slope(previous);
  for (int i = 1; i <= subdivisions; ++i) {
    const auto current = std::lerp(lower_log, upper_log, static_cast<double>(i) / subdivisions);
    const auto current_slope = slope(current);
    if (previous_slope < 0 && current_slope >= 0) {
      const auto bracket = algorithms::bisection_trace(slope, previous, current, {1e-12, 0, 60});
      consider(bracket.result.root);
    }
    previous = current;
    previous_slope = current_slope;
  }
  const bool at_boundary = best_log == lower_log || best_log == upper_log;
  long double h = scaled_bandwidth(best_log);
  auto bandwidth = checked_bandwidth(scale * h);
  if (range && at_boundary)
    bandwidth = best_log == lower_log ? range->first : range->second;
  h = bandwidth / scale;
  const auto score = lscv_terms(sample, scale, h) / scale;
  return {bandwidth, static_cast<double>(score), at_boundary};
}

} // namespace datamunge::stats
