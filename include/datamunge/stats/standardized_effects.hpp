#pragma once
/// @file standardized_effects.hpp
/// @brief Standardized Gaussian mean differences and noncentral-t confidence intervals.

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace datamunge::stats {

/// @brief Identifies the sample standard deviation used for both effect estimates.
/// @par Example
/// @code
/// const auto fit = dm::stats::one_sample_standardized_mean_difference({1, 2, 3});
/// std::print("{}\n", fit.standardization == dm::stats::MeanDifferenceStandardization::SampleSd); // true
/// @endcode
/// @see StandardizedMeanDifference
enum class MeanDifferenceStandardization {
  SampleSd,            ///< Standard deviation of one sample.
  PairedDifferenceSd,  ///< Standard deviation of the within-pair differences.
  PooledSd             ///< Residual-degrees-of-freedom-weighted standard deviation of two independent groups.
};

/// @brief Cohen's d, bias-adjusted Hedges' g, and a confidence interval for their population target.
///
/// Both point estimates target the same population-standardized difference.
/// The interval inverts the noncentral t distribution for that parameter.
/// @par Example
/// @code
/// const auto fit = dm::stats::one_sample_standardized_mean_difference({1, 2, 3});
/// std::print("{:.3f} {:.3f}\n", fit.cohen_d, *fit.hedges_g); // 2.000 1.128
/// @endcode
/// @par References
/// - L. V. Hedges (1981), https://doi.org/10.3102/10769986006002107
/// @see one_sample_standardized_mean_difference(), paired_standardized_mean_difference(), pooled_standardized_mean_difference()
struct StandardizedMeanDifference {
  double cohen_d;                                ///< Signed sample mean contrast divided by the sample standardizer.
  std::optional<double> hedges_g;                  ///< Bias-adjusted estimate, absent at one residual degree of freedom.
  std::pair<double, double> confidence_interval;  ///< Equal-tailed interval for the shared population effect.
  MeanDifferenceStandardization standardization;  ///< Standard deviation convention.
  double standardizer;                           ///< Observed SD in original units, rounded to double range.
  std::size_t degrees_of_freedom;                 ///< Residual degrees of freedom for the standardizer.
};

/// @brief Estimates the standardized difference between a Gaussian population mean and a reference.
///
/// Uses the one-sample standard deviation with n-1 denominator. The
/// noncentral-t interval assumes independent Gaussian observations.
/// @param sample At least two finite observations with a positive sample SD.
/// @param reference_mean Finite reference value, default zero.
/// @param confidence_level Two-sided confidence level strictly between zero and one, default .95.
/// @return Standardized point estimates, standardizer and interval for the population effect.
/// @throws std::invalid_argument Invalid sample, reference value or confidence level.
/// @throws std::overflow_error The standardized effect or observed t is outside the finite double range.
/// @par Example
/// @code
/// const auto fit = dm::stats::one_sample_standardized_mean_difference({1, 2, 3}, 1);
/// std::print("{}\n", fit.cohen_d); // 1
/// @endcode
/// @par References
/// - L. V. Hedges (1981), https://doi.org/10.3102/10769986006002107
/// @see paired_standardized_mean_difference(), pooled_standardized_mean_difference()
StandardizedMeanDifference one_sample_standardized_mean_difference(
    const std::vector<double>& sample, double reference_mean = 0, double confidence_level = .95);

/// @brief Estimates a paired mean difference in units of its within-pair standard deviation.
///
/// Forms first minus second. The noncentral-t interval assumes
/// independent Gaussian paired differences.
/// @param first First measurements, at least two finite observations.
/// @param second Second finite measurements in matching order, with variation in paired differences.
/// @param confidence_level Two-sided confidence level strictly between zero and one, default .95.
/// @return Standardized point estimates, difference SD and interval for the population effect.
/// @throws std::invalid_argument Invalid pairs, zero difference SD or invalid confidence level.
/// @throws std::overflow_error The standardized effect or observed t is outside the finite double range.
/// @par Example
/// @code
/// const auto fit = dm::stats::paired_standardized_mean_difference({2, 4, 6}, {1, 2, 3});
/// std::print("{}\n", fit.cohen_d); // 2
/// @endcode
/// @par References
/// - L. V. Hedges (1981), https://doi.org/10.3102/10769986006002107
/// @see one_sample_standardized_mean_difference(), pooled_standardized_mean_difference()
StandardizedMeanDifference paired_standardized_mean_difference(
    const std::vector<double>& first, const std::vector<double>& second, double confidence_level = .95);

/// @brief Estimates the first-minus-second mean difference in units of a common population SD.
///
/// Pools the sample variances with residual degrees-of-freedom weights.
/// The noncentral-t interval assumes independent Gaussian samples with a
/// common population variance.
/// @param first First sample, at least two finite observations.
/// @param second Second sample, at least two finite observations, with positive pooled variance.
/// @param confidence_level Two-sided confidence level strictly between zero and one, default .95.
/// @return Standardized point estimates, pooled SD and interval for the population effect.
/// @throws std::invalid_argument Invalid samples, zero pooled SD or invalid confidence level.
/// @throws std::overflow_error The standardized effect or observed t is outside the finite double range.
/// @par Example
/// @code
/// const auto fit = dm::stats::pooled_standardized_mean_difference({3, 4, 5}, {1, 2, 3});
/// std::print("{}\n", fit.cohen_d); // 2
/// @endcode
/// @par References
/// - L. V. Hedges (1981), https://doi.org/10.3102/10769986006002107
/// @see one_sample_standardized_mean_difference(), paired_standardized_mean_difference()
StandardizedMeanDifference pooled_standardized_mean_difference(
    const std::vector<double>& first, const std::vector<double>& second, double confidence_level = .95);

} // namespace datamunge::stats
