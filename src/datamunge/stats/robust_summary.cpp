/// @file robust_summary.cpp
/// @brief Robust univariate estimators with widened accumulators.

#include <datamunge/random/distributions.hpp>
#include <datamunge/stats/detail/test_common.hpp>
#include <datamunge/stats/robust_summary.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {

/// @brief Helpers for finite univariate summary calculations.
namespace {

/// @brief Checks that every requested quantile lies in the closed unit interval.
/// @param probabilities Requested probabilities.
/// @throws std::invalid_argument A probability is out of range or NaN.
void check_probabilities(const std::vector<double>& probabilities) {
  for (const double p : probabilities) {
    if (!(p >= 0 && p <= 1))
      throw std::invalid_argument("quantile: probability must be in [0, 1]");
  }
}

/// @brief Evaluates one Hyndman–Fan definition on pre-sorted data.
/// @param sorted Nonempty sorted finite observations.
/// @param p Valid probability in [0, 1].
/// @param type Valid type from 1 through 9.
/// @return Selected or interpolated order statistic.
double sample_quantile_sorted(const std::vector<double>& sorted, double p, int type) {
  if (p == 0)
    return sorted.front();
  if (p == 1)
    return sorted.back();
  const double n = static_cast<double>(sorted.size());
  constexpr std::array<double, 6> offsets{0, .5, 0, 1, 1.0 / 3, 3.0 / 8};
  constexpr std::array<double, 6> end_offsets{1, .5, 0, 1, 1.0 / 3, 3.0 / 8};
  const double position = type <= 3
      ? n * p - (type == 3 ? .5 : 0)
      : offsets[type - 4] + p * (n + 1 - offsets[type - 4] - end_offsets[type - 4]);
  // Guard jumps against rounding at integer ranks, using R's default fuzz.
  const double fuzz = type == 7 ? 0 : 4 * std::numeric_limits<double>::epsilon();
  const double rank = std::floor(position * (1 + fuzz));
  if (rank < 1)
    return sorted.front();
  if (rank >= n)
    return sorted.back();
  const auto index = static_cast<std::size_t>(rank);
  double fraction = std::max(0.0, position - rank);
  switch (type) {
  case 1: fraction = fraction > 0 ? 1 : 0; break;
  case 2: fraction = fraction > 0 ? 1 : .5; break;
  case 3: fraction = position == rank && index % 2 == 0 ? 0 : 1; break;
  default: break;
  }
  return std::lerp(sorted[index - 1], sorted[index], fraction);
}

/// @brief Computes beta-weighted order statistics with separate upper-tail evaluation.
/// @param sorted Nonempty sorted finite sample.
/// @param p Valid probability in [0, 1].
/// @return Harrell–Davis estimate.
double harrell_davis_sorted(const std::vector<double>& sorted, double p) {
  if (p == 0)
    return sorted.front();
  if (p == 1 || sorted.size() == 1)
    return sorted.back();
  const double n = static_cast<double>(sorted.size());
  const double a = (n + 1) * p;
  const double b = (n + 1) * (1 - p);
  double previous = 0;
  bool previous_is_upper = false;
  long double total = 0, estimate = 0;
  for (std::size_t i = 0; i < sorted.size(); ++i) {
    const auto rank = static_cast<double>(i + 1);
    const bool upper = rank / n > p;
    const double tail = upper ? random::beta_cdf((n - rank) / n, b, a)
                              : random::beta_cdf(rank / n, a, b);
    const double mass = upper ? (previous_is_upper ? previous - tail : 1 - previous - tail)
                              : tail - previous;
    const long double weight = std::max(0.0, mass);
    if (weight > 0) {
      total += weight;
      estimate = std::lerp(estimate, static_cast<long double>(sorted[i]), weight / total);
    }
    previous = tail;
    previous_is_upper = upper;
  }
  if (!(total > 0))
    throw std::invalid_argument("harrell_davis_quantiles: could not resolve beta weights");
  return static_cast<double>(estimate);
}

/// @brief Finds the median of a nonempty working copy by partial selection.
/// @param values Mutable sample, reordered by the selection.
/// @return Median, averaging the two central values for even size.
long double median_in_place(std::vector<long double>& values) {
  const auto mid = values.begin() + values.size() / 2;
  std::nth_element(values.begin(), mid, values.end());
  if (values.size() % 2 != 0)
    return *mid;
  return std::midpoint(*std::max_element(values.begin(), mid), *mid);
}

/// @brief Converts a widened summary value to the public numeric type.
/// @param value Widened summary.
/// @return Finite double.
/// @throws std::invalid_argument Nonfinite or out-of-range summary.
double checked_summary(long double value) {
  if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
    throw std::invalid_argument("robust summary: result overflow");
  return static_cast<double>(value);
}

/// @brief Copies, widens and sorts a checked sample for distance selection.
/// @param values Finite observations, in any order.
/// @return Owned ascending values.
/// @throws std::invalid_argument Empty sample or nonfinite value.
std::vector<long double> sorted_scale_sample(const std::vector<double>& values) {
  detail::check_sample(values, 1);
  std::vector<long double> sorted(values.begin(), values.end());
  std::sort(sorted.begin(), sorted.end());
  return sorted;
}

/// @brief Selects a pairwise distance rank by counting distances in successive brackets.
/// @param sorted Ascending finite observations.
/// @param rank One-based distance rank, no larger than the number of pairs.
/// @return Pairwise distance with the given rank.
long double pairwise_distance_rank(const std::vector<long double>& sorted, std::size_t rank) {
  long double lo = 0, hi = sorted.back() - sorted.front();
  while (lo < hi) {
    auto probe = std::midpoint(lo, hi);
    if (probe == hi)
      probe = lo;
    std::size_t count = 0, end = 0;
    long double below = 0, above = hi;
    for (std::size_t i = 0; i + 1 < sorted.size(); ++i) {
      end = std::max(end, i);
      while (end + 1 < sorted.size() && sorted[end + 1] - sorted[i] <= probe)
        ++end;
      count += std::min(rank - count, end - i);
      below = std::max(below, sorted[end] - sorted[i]);
      if (end + 1 < sorted.size())
        above = std::min(above, sorted[end + 1] - sorted[i]);
    }
    if (count >= rank)
      hi = below;
    else
      lo = above;
  }
  return lo;
}

/// @brief Finds the high median of distances to one observation in a sorted sample.
/// @param sorted Ascending finite observations.
/// @param index Index of the observation serving as origin.
/// @return Distance at zero-based rank floor(n/2), counting the distance to self.
long double high_distance_median(const std::vector<long double>& sorted, std::size_t index) {
  const auto origin = sorted[index];
  const auto right_size = sorted.size() - index;
  const auto rank = sorted.size() / 2 + 1;
  // Partition the two sorted distance runs into the selected rank smallest
  // values and the remainder. The origin itself starts the right run.
  auto lo = rank > right_size ? rank - right_size : 0;
  auto hi = std::min(rank, index);
  while (lo < hi) {
    const auto left = std::midpoint(lo, hi);
    const auto right = rank - left;
    if (right > 0 && left < index && sorted[index + right - 1] - origin > origin - sorted[index - left - 1])
      lo = left + 1;
    else
      hi = left;
  }
  const auto right = rank - lo;
  const auto last_left = lo == 0 ? 0 : origin - sorted[index - lo];
  const auto last_right = right == 0 ? 0 : sorted[index + right - 1] - origin;
  return std::max(last_left, last_right);
}

/// @brief Validated sample range and arithmetic mean used by the empirical likelihood.
struct MeanProfile {
  double minimum; ///< Smallest observation.
  double maximum; ///< Largest observation.
  double mean;    ///< Unconstrained arithmetic mean.
};

/// @brief Validates a nonconstant sample and computes the unconstrained mean.
/// @param values Observations.
/// @return Sample extrema and mean.
/// @throws std::invalid_argument Invalid values, too few observations or zero variance.
MeanProfile check_mean_profile(const std::vector<double>& values) {
  detail::check_sample(values, 2);
  const auto [minimum, maximum] = std::minmax_element(values.begin(), values.end());
  if (*minimum == *maximum)
    throw std::invalid_argument("empirical likelihood: nonconstant sample required");
  const long double magnitude = std::max(std::abs(*minimum), std::abs(*maximum));
  long double average = 0;
  for (const auto value : values)
    average += (value / magnitude) / values.size();
  return {*minimum, *maximum, static_cast<double>(average * magnitude)};
}

/// @brief Computes the empirical mean likelihood ratio by a scalar multiplier solve.
/// @param values Nonconstant sample used to construct profile.
/// @param profile Validated sample range.
/// @param null_mean Hypothesized mean.
/// @param weights Receives fitted probabilities if supplied and the mean is in the open hull.
/// @return Twice the negative log likelihood ratio, or infinity outside the open hull.
double mean_profile_statistic(const std::vector<double>& values, const MeanProfile& profile,
                              double null_mean, std::vector<double>* weights) {
  if (null_mean <= profile.minimum || null_mean >= profile.maximum)
    return std::numeric_limits<double>::infinity();
  const long double unit = std::max(std::abs(profile.minimum), std::abs(profile.maximum));
  std::vector<long double> residuals;
  residuals.reserve(values.size());
  for (const auto value : values) {
    const auto residual = std::signbit(value) == std::signbit(null_mean)
        ? (static_cast<long double>(value) - null_mean) / unit
        : value / unit - null_mean / unit;
    residuals.push_back(residual);
  }
  const auto score_at_zero = std::accumulate(residuals.begin(), residuals.end(), 0.0L);
  if (score_at_zero < 0) {
    for (auto& residual : residuals)
      residual = -residual;
  }
  /// @brief Evaluates the decreasing score on the positive feasible part of the multiplier domain.
  const auto score = [&](long double multiplier) {
    long double sum = 0;
    for (const auto residual : residuals) {
      const auto denominator = std::fma(multiplier, residual, 1.0L);
      if (denominator <= 0)
        return -std::numeric_limits<long double>::infinity();
      sum += residual / denominator;
    }
    return sum;
  };
  long double multiplier = 0;
  if (score_at_zero != 0) {
    const auto limit = -1 / *std::min_element(residuals.begin(), residuals.end());
    long double lo = 0, hi = std::min(1.0L, limit);
    while (hi < limit && score(hi) > 0) {
      lo = hi;
      hi = std::min(2 * hi, limit);
    }
    while (true) {
      const auto mid = std::midpoint(lo, hi);
      if (mid == lo || mid == hi)
        break;
      if (score(mid) > 0)
        lo = mid;
      else
        hi = mid;
    }
    multiplier = lo;
  }
  long double log_ratio = 0;
  if (weights != nullptr)
    weights->reserve(residuals.size());
  for (const auto residual : residuals) {
    const auto denominator = std::fma(multiplier, residual, 1.0L);
    log_ratio += std::abs(multiplier * residual) < .5L
        ? std::log1p(multiplier * residual) : std::log(denominator);
    if (weights != nullptr)
      weights->push_back(static_cast<double>(1 / (denominator * residuals.size())));
  }
  return static_cast<double>(std::max(0.0L, 2 * log_ratio));
}

} // namespace

std::vector<double> sample_quantiles(const std::vector<double>& values,
                                     const std::vector<double>& probabilities, int type) {
  detail::check_sample(values, 1);
  check_probabilities(probabilities);
  if (type < 1 || type > 9)
    throw std::invalid_argument("sample_quantiles: type must be in [1, 9]");
  auto sorted = values;
  std::sort(sorted.begin(), sorted.end());
  std::vector<double> result;
  result.reserve(probabilities.size());
  for (const double p : probabilities)
    result.push_back(sample_quantile_sorted(sorted, p, type));
  return result;
}

std::vector<double> weighted_empirical_quantiles(const std::vector<double>& values,
                                                  const std::vector<double>& weights,
                                                  const std::vector<double>& probabilities) {
  detail::check_sample(values, 1);
  check_probabilities(probabilities);
  if (weights.size() != values.size())
    throw std::invalid_argument("weighted_empirical_quantiles: values and weights must match");
  std::vector<std::pair<double, long double>> ordered;
  ordered.reserve(values.size());
  long double max_weight = 0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (!(weights[i] >= 0) || !std::isfinite(weights[i]))
      throw std::invalid_argument("weighted_empirical_quantiles: invalid weight");
    if (weights[i] > 0) {
      ordered.emplace_back(values[i], weights[i]);
      max_weight = std::max(max_weight, static_cast<long double>(weights[i]));
    }
  }
  if (ordered.empty())
    throw std::invalid_argument("weighted_empirical_quantiles: need a positive weight");
  std::sort(ordered.begin(), ordered.end());
  long double total = 0;
  for (auto& entry : ordered) {
    total += entry.second / max_weight;
    entry.second = total;
  }
  std::vector<double> result;
  result.reserve(probabilities.size());
  for (const double p : probabilities) {
    const auto it = std::lower_bound(ordered.begin(), ordered.end(), p,
        [total](const auto& entry, double probability) {
          return static_cast<double>(entry.second / total) < probability;
        });
    result.push_back(p == 1 ? ordered.back().first : it->first);
  }
  return result;
}

std::vector<double> harrell_davis_quantiles(const std::vector<double>& values,
                                            const std::vector<double>& probabilities) {
  detail::check_sample(values, 1);
  check_probabilities(probabilities);
  auto sorted = values;
  std::sort(sorted.begin(), sorted.end());
  std::vector<double> result;
  result.reserve(probabilities.size());
  for (const double p : probabilities)
    result.push_back(harrell_davis_sorted(sorted, p));
  return result;
}

ExactQuantileInterval exact_quantile_interval(const std::vector<double>& values,
                                               double probability, double confidence_level) {
  detail::check_sample(values, 1);
  detail::check_confidence(confidence_level);
  if (!(probability > 0 && probability < 1))
    throw std::invalid_argument("exact_quantile_interval: probability must be in (0, 1)");
  const auto n = values.size();
  const auto tail_budget = (1 - confidence_level) / 2;
  // Rank zero and rank n+1 are the unbounded endpoints, each with zero tail probability.
  std::size_t left = 0, right = n + 1;
  double lower_tail = 0, upper_tail = 0;
  while (right - left > 1) {
    const auto rank = std::midpoint(left, right);
    const auto tail = detail::binomial_test_p(rank - 1, n, probability, Alternative::Less);
    if (tail <= tail_budget) {
      left = rank;
      lower_tail = tail;
    } else {
      right = rank;
    }
  }
  const auto lower_rank = left;
  left = 0;
  right = n + 1;
  while (right - left > 1) {
    const auto rank = std::midpoint(left, right);
    const auto tail = detail::binomial_test_p(rank, n, probability, Alternative::Greater);
    if (tail <= tail_budget) {
      right = rank;
      upper_tail = tail;
    } else {
      left = rank;
    }
  }
  const auto upper_rank = right;
  auto sorted = values;
  std::sort(sorted.begin(), sorted.end());
  constexpr auto infinity = std::numeric_limits<double>::infinity();
  return {lower_rank == 0 ? -infinity : sorted[lower_rank - 1],
          upper_rank > n ? infinity : sorted[upper_rank - 1],
          lower_rank, upper_rank, 1 - (lower_tail + upper_tail)};
}

EmpiricalCdfBand::EmpiricalCdfBand(const std::vector<double>& values, double confidence_level) {
  detail::check_sample(values, 1);
  detail::check_confidence(confidence_level);
  half_width_ = std::sqrt((std::log(2.0) - std::log1p(-confidence_level)) / (2.0 * values.size()));
  auto sorted = values;
  std::sort(sorted.begin(), sorted.end());
  for (std::size_t i = 0; i < sorted.size(); ++i) {
    if (i + 1 == sorted.size() || sorted[i] != sorted[i + 1])
      knots_.push_back(band_point(sorted[i], static_cast<double>(i + 1) / sorted.size()));
  }
}

EmpiricalCdfPoint EmpiricalCdfBand::evaluate(double x) const {
  if (std::isnan(x))
    throw std::invalid_argument("EmpiricalCdfBand: query cannot be NaN");
  const auto next = std::upper_bound(knots_.begin(), knots_.end(), x,
      [](double query, const EmpiricalCdfPoint& point) { return query < point.x; });
  return band_point(x, next == knots_.begin() ? 0.0 : (next - 1)->cdf);
}

EmpiricalCdfPoint EmpiricalCdfBand::band_point(double x, double cdf) const noexcept {
  return {x, cdf, std::max(0.0, cdf - half_width_), std::min(1.0, cdf + half_width_)};
}

double median_absolute_deviation(const std::vector<double>& values,
                                 std::optional<double> center, double scale) {
  detail::check_sample(values, 1);
  if ((center && !std::isfinite(*center)) || !std::isfinite(scale) || scale < 0)
    throw std::invalid_argument("median_absolute_deviation: finite center and nonnegative scale required");
  std::vector<long double> sample(values.begin(), values.end());
  const long double location = center ? *center : median_in_place(sample);
  for (auto& value : sample)
    value = std::abs(value - location);
  return checked_summary(scale * median_in_place(sample));
}

TrimmedSummary trimmed_summary(const std::vector<double>& values, double fraction) {
  detail::check_sample(values, 2);
  if (!(fraction >= 0 && fraction < .5))
    throw std::invalid_argument("trimmed_summary: fraction must be in [0, 0.5)");
  auto sorted = values;
  std::sort(sorted.begin(), sorted.end());
  const auto trim = static_cast<std::size_t>(fraction * static_cast<double>(sorted.size()));
  const auto first = sorted.begin() + trim;
  const auto last = sorted.end() - trim;
  const long double origin = *first;
  long double trimmed_sum = 0;
  for (auto it = first; it != last; ++it)
    trimmed_sum += *it - origin;
  const long double winsorized_offset =
      (trimmed_sum + trim * (last[-1] - origin)) / sorted.size();
  long double squared = 0;
  for (const double value : sorted) {
    const auto residual = (std::clamp(value, *first, last[-1]) - origin) - winsorized_offset;
    squared += residual * residual;
  }
  return {checked_summary(origin + trimmed_sum / (last - first)),
          checked_summary(origin + winsorized_offset),
          checked_summary(squared / (sorted.size() - 1))};
}

HuberLocationScale huber_location_scale(const std::vector<double>& values, double tuning_constant,
                                        double tolerance, std::size_t max_iterations) {
  detail::check_sample(values, 2);
  if (!(tuning_constant > 0 && std::isfinite(tuning_constant)) ||
      !(tolerance > 0 && std::isfinite(tolerance)) || max_iterations == 0)
    throw std::invalid_argument("huber_location_scale: invalid tuning, tolerance or iteration count");
  const auto [minimum, maximum] = std::minmax_element(values.begin(), values.end());
  if (*minimum == *maximum)
    return {*minimum, 0, tuning_constant, 0, false};

  const long double unit = std::max(std::abs(*minimum), std::abs(*maximum));
  std::vector<long double> clipped;
  clipped.reserve(values.size());
  for (const auto value : values)
    clipped.push_back(static_cast<long double>(value) / unit);
  auto location = median_in_place(clipped);
  for (auto& value : clipped)
    value = std::abs(value - location);
  auto scale = 1.4826L * median_in_place(clipped);
  const long double k = tuning_constant;
  const auto factor = std::min(k, 1.0L);
  // E[min(Z^2, k^2)] / factor^2. For small k the expansion avoids
  // cancellation between the two terms proportional to k.
  const auto calibration = k < .01L
      ? 1 + random::normal_pdf(0) * k * (-4.0L / 3 + k * k * (2.0L / 15 - k * k / 70))
      : k >= 12 ? 1 : (std::erf(k / std::sqrt(2.0L)) +
          2 * k * k * random::normal_cdf(-tuning_constant) - 2 * k * random::normal_pdf(tuning_constant)) /
          (factor * factor);
  const long double n = values.size();
  std::size_t iterations = 0;
  bool converged = false;
  while (scale > 0 && iterations < max_iterations) {
    const auto cutoff = (k / factor) * scale;
    long double clipped_mean = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
      const auto residual = (static_cast<long double>(values[i]) / unit - location) / factor;
      clipped[i] = std::clamp(residual, -cutoff, cutoff);
      clipped_mean += clipped[i];
    }
    clipped_mean /= n;
    long double sum_squares = 0;
    for (const auto value : clipped) {
      const auto deviation = value - clipped_mean;
      sum_squares += deviation * deviation;
    }
    const auto next_scale = std::sqrt(sum_squares / ((n - 1) * calibration));
    const auto mean_shift = clipped_mean * factor;
    const auto threshold = tolerance * scale;
    converged = next_scale > 0 && std::abs(mean_shift) <= threshold &&
                std::abs(next_scale - scale) <= threshold;
    location += mean_shift;
    scale = next_scale;
    ++iterations;
    if (converged)
      break;
  }
  return {checked_summary(location * unit), checked_summary(scale * unit),
          tuning_constant, iterations, converged};
}

double qn_scale(const std::vector<double>& values, bool finite_correction) {
  const auto sorted = sorted_scale_sample(values);
  const auto size = sorted.size();
  if (size == 1)
    return 0;
  const auto half = size / 2 + 1;
  const auto factor1 = half % 2 == 0 ? half / 2 : (half - 1) / 2;
  const auto factor2 = half % 2 == 0 ? half - 1 : half;
  if (factor1 > std::numeric_limits<std::size_t>::max() / factor2)
    throw std::invalid_argument("qn_scale: distance rank overflow");
  long double correction = 1;
  if (finite_correction) {
    /// @brief Normal calibration corrections for sizes 2 through 12 from robustbase.
    constexpr std::array<long double, 11> small{
        .399356L, .99365L, .51321L, .84401L, .61220L, .85877L,
        .66993L, .87344L, .72014L, .88906L, .75743L};
    const long double n = size;
    const auto adjustment = size % 2 != 0
        ? 1.60188L + (-2.1284L - 5.172L / n) / n
        : 3.67561L + (1.9654L + (6.987L - 77 / n) / n) / n;
    correction = size <= 12 ? small[size - 2] : 1 / (1 + adjustment / n);
  }
  return checked_summary((2.21914L * correction) * pairwise_distance_rank(sorted, factor1 * factor2));
}

double sn_scale(const std::vector<double>& values, bool finite_correction) {
  const auto sorted = sorted_scale_sample(values);
  const auto size = sorted.size();
  if (size == 1)
    return 0;
  std::vector<long double> medians(size);
  for (std::size_t index = 0; index < size; ++index)
    medians[index] = high_distance_median(sorted, index);
  const auto low_median = medians.begin() + (size - 1) / 2;
  std::nth_element(medians.begin(), low_median, medians.end());
  long double correction = 1;
  if (finite_correction) {
    /// @brief Normal calibration corrections for sizes 2 through 9 from robustbase.
    constexpr std::array<long double, 8> small{
        .743L, 1.851L, .954L, 1.351L, .993L, 1.198L, 1.005L, 1.131L};
    const long double n = size;
    correction = size <= 9 ? small[size - 2] : size % 2 != 0 ? n / (n - .9L) : 1;
  }
  return checked_summary((1.1926L * correction) * *low_median);
}

EmpiricalLikelihoodMeanTest empirical_likelihood_mean_test(
    const std::vector<double>& values, double null_mean) {
  if (!std::isfinite(null_mean))
    throw std::invalid_argument("empirical likelihood: null mean must be finite");
  const auto profile = check_mean_profile(values);
  EmpiricalLikelihoodMeanTest result;
  result.statistic = mean_profile_statistic(values, profile, null_mean, &result.weights);
  result.p_value = detail::chi_squared_test_survival(result.statistic, 1);
  return result;
}

EmpiricalLikelihoodMeanInterval empirical_likelihood_mean_interval(
    const std::vector<double>& values, double confidence_level) {
  detail::check_confidence(confidence_level);
  const auto profile = check_mean_profile(values);
  const auto cutoff = random::gamma_quantile(1 - confidence_level, .5, 2, false);
  /// @brief Bisects between the fitted mean and an exterior endpoint.
  const auto endpoint = [&](double outside) {
    double inside = profile.mean;
    while (true) {
      const auto mid = std::midpoint(inside, outside);
      if (mid == inside || mid == outside)
        return inside;
      if (mean_profile_statistic(values, profile, mid, nullptr) <= cutoff)
        inside = mid;
      else
        outside = mid;
    }
  };
  return {profile.mean, endpoint(profile.minimum), endpoint(profile.maximum), cutoff};
}

} // namespace datamunge::stats
