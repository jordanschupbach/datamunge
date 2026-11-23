/// @file standardized_effects.cpp
/// @brief Bias-adjusted Gaussian standardized means and their exact-model intervals.

#include <datamunge/random/noncentral_distributions.hpp>
#include <datamunge/stats/detail/test_common.hpp>
#include <datamunge/stats/standardized_effects.hpp>

#include "../random/beta_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Local scaled moments and inference shared by the three standardizations.
namespace {

/// @brief Chooses a power-of-two scale from the largest observation in the supplied samples.
/// @param first Validated finite sample.
/// @param second Optional validated finite sample on the same measurement scale.
/// @return Exponent of the scale, zero if all values are zero.
int sample_exponent(const std::vector<double>& first, const std::vector<double>& second = {}) {
  double largest = 0;
  for (const auto* sample : {&first, &second})
    for (const auto value : *sample)
      largest = std::max(largest, std::abs(value));
  return largest == 0 ? 0 : std::ilogb(largest);
}

/// @brief Scaled mean represented relative to an observation, with its centered sum of squares.
struct ShiftedMoments {
  long double origin;      ///< First scaled observation.
  long double mean_offset; ///< Mean of scaled deviations from the origin.
  long double squares;     ///< Sum of squared deviations from the scaled mean.
};

/// @brief Computes centered moments without squaring values on the original scale.
/// @tparam Observation Callable returning one scaled observation.
/// @param n Positive sample size.
/// @param observed Callable indexed from zero.
/// @return Mean location components and centered sum of squares.
template <class Observation>
ShiftedMoments shifted_moments(std::size_t n, const Observation& observed) {
  ShiftedMoments fit{observed(0), 0, 0};
  for (std::size_t i = 1; i < n; ++i) {
    const auto value = observed(i) - fit.origin;
    const auto delta = value - fit.mean_offset;
    fit.mean_offset += delta / (i + 1);
    fit.squares += delta * (value - fit.mean_offset);
  }
  return fit;
}

/// @brief Converts the standardized t statistic and computes both estimates and their shared interval.
/// @param contrast Mean contrast in scaled units.
/// @param sd Sample standard deviation in scaled units.
/// @param exponent Exponent converting the SD back to original units.
/// @param df Residual degrees of freedom.
/// @param information Effective sample size multiplying the squared standardized effect.
/// @param standardization Standard deviation convention.
/// @param level Confidence level in (0,1).
/// @return The point estimates, their common population-effect interval, and their standardizer.
/// @throws std::invalid_argument The sample standardizer is zero.
/// @throws std::overflow_error The observed t is not a finite double.
StandardizedMeanDifference effect_result(long double contrast, long double sd, int exponent, std::size_t df,
                                         long double information, MeanDifferenceStandardization standardization,
                                         double level) {
  if (!(sd > 0))
    throw std::invalid_argument("standardized mean difference: standard deviation must be positive");
  const auto root_information = std::sqrt(information);
  const auto cohen = contrast / sd;
  const auto t = cohen * root_information;
  if (std::abs(t) > std::numeric_limits<double>::max())
    throw std::overflow_error("standardized mean difference: observed t exceeds the finite double range");

  std::optional<double> hedges;
  if (df > 1) {
    const auto half_df = df / 2.L;
    const auto log_j = random::detail::gamma_increment(half_df - .5L, .5L) - std::log(half_df) / 2;
    hedges = static_cast<double>(cohen * std::exp(log_j));
  }
  const auto tail = (1 - level) / 2;
  const auto lower_ncp = random::noncentral_t_noncentrality(tail, static_cast<double>(t), static_cast<double>(df), false);
  const auto upper_ncp = tail == .5 ? lower_ncp :
      random::noncentral_t_noncentrality(tail, static_cast<double>(t), static_cast<double>(df));
  const auto raw_sd = std::scalbn(sd, exponent);
  const auto reported_sd = raw_sd > std::numeric_limits<double>::max() ? std::numeric_limits<double>::infinity()
                                                                     : static_cast<double>(raw_sd);
  return {static_cast<double>(cohen), hedges,
          {static_cast<double>(lower_ncp / root_information), static_cast<double>(upper_ncp / root_information)},
          standardization, reported_sd, df};
}

} // namespace

StandardizedMeanDifference one_sample_standardized_mean_difference(
    const std::vector<double>& sample, double reference_mean, double confidence_level) {
  detail::check_sample(sample, 2);
  detail::check_confidence(confidence_level);
  if (!std::isfinite(reference_mean))
    throw std::invalid_argument("standardized mean difference: the reference mean must be finite");
  const auto exponent = sample_exponent(sample);
  const auto n = sample.size();
  const auto moments = shifted_moments(n, [&](std::size_t i) {
    return std::scalbn(static_cast<long double>(sample[i]), -exponent);
  });
  const auto contrast = moments.mean_offset + (moments.origin - std::scalbn(static_cast<long double>(reference_mean), -exponent));
  return effect_result(contrast, std::sqrt(moments.squares / (n - 1)), exponent, n - 1, n,
                       MeanDifferenceStandardization::SampleSd, confidence_level);
}

StandardizedMeanDifference paired_standardized_mean_difference(
    const std::vector<double>& first, const std::vector<double>& second, double confidence_level) {
  detail::check_sample(first, 2);
  detail::check_sample(second, 2);
  detail::check_confidence(confidence_level);
  if (first.size() != second.size())
    throw std::invalid_argument("standardized mean difference: paired samples must have equal lengths");
  const auto exponent = sample_exponent(first, second);
  const auto n = first.size();
  const auto moments = shifted_moments(n, [&](std::size_t i) {
    const auto x = std::scalbn(static_cast<long double>(first[i]), -exponent);
    const auto y = std::scalbn(static_cast<long double>(second[i]), -exponent);
    return x - y;
  });
  return effect_result(moments.origin + moments.mean_offset, std::sqrt(moments.squares / (n - 1)),
                       exponent, n - 1, n, MeanDifferenceStandardization::PairedDifferenceSd, confidence_level);
}

StandardizedMeanDifference pooled_standardized_mean_difference(
    const std::vector<double>& first, const std::vector<double>& second, double confidence_level) {
  detail::check_sample(first, 2);
  detail::check_sample(second, 2);
  detail::check_confidence(confidence_level);
  const auto exponent = sample_exponent(first, second);
  const auto first_fit = shifted_moments(first.size(), [&](std::size_t i) {
    return std::scalbn(static_cast<long double>(first[i]), -exponent);
  });
  const auto second_fit = shifted_moments(second.size(), [&](std::size_t i) {
    return std::scalbn(static_cast<long double>(second[i]), -exponent);
  });
  const auto difference = (first_fit.origin - second_fit.origin) + (first_fit.mean_offset - second_fit.mean_offset);
  const auto df = first.size() - 1 + (second.size() - 1);
  const long double n = first.size(), m = second.size();
  return effect_result(difference, std::sqrt((first_fit.squares + second_fit.squares) / df), exponent, df,
                       (n / (n + m)) * m, MeanDifferenceStandardization::PooledSd, confidence_level);
}

} // namespace datamunge::stats
