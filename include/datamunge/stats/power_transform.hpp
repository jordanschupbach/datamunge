#pragma once
/// @file power_transform.hpp
/// @brief Gaussian profile-likelihood fits and inverses for Box–Cox and Yeo–Johnson.

#include <vector>

namespace datamunge::stats {

/// @brief Identifies the input domain and formula of a fitted power transformation.
/// @par Example
/// @code
/// std::print("{}\n", dm::stats::PowerTransformFamily::BoxCox ==
///                   dm::stats::PowerTransformFamily::YeoJohnson); // false
/// @endcode
/// @see FittedPowerTransform
enum class PowerTransformFamily {
  BoxCox,     ///< Unshifted Box–Cox for strictly positive real inputs.
  YeoJohnson  ///< Yeo–Johnson for finite real inputs including zero and negatives.
};

/// @brief Owns a power parameter and the Gaussian profile likelihood at its fitted value.
///
/// Reuse the same value to transform held-out observations or invert predictions.
/// There is no fitted shift, centering, or standardization in the returned transform.
/// @par Example
/// @code
/// const auto fit = dm::stats::fit_box_cox({1, 2, 4, 8});
/// std::print("{:.0f}\n", fit.inverse(fit.apply(4))); // 4
/// @endcode
/// @see fit_box_cox(), fit_yeo_johnson()
struct FittedPowerTransform {
  PowerTransformFamily family;   ///< Transformation formula and domain.
  double lambda;                 ///< Power parameter before any centering or scaling.
  double profile_log_likelihood; ///< Maximized Gaussian log likelihood with its common normal constant omitted.
  bool at_search_limit;          ///< Whether the selected lambda is exactly a supplied search endpoint.

  /// @brief Transforms one observation using the fitted family and power.
  /// @param value Finite input, strictly positive for Box–Cox.
  /// @return Transformed observation before any standardization.
  /// @throws std::invalid_argument Invalid input, family, or power.
  /// @throws std::overflow_error The result overflows, underflows to zero, or rounds outside the invertible range.
  /// @par Example
  /// @code
  /// const auto fit = dm::stats::fit_yeo_johnson({-3, 0, 1, 8});
  /// std::print("{}\n", fit.apply(0)); // 0
  /// @endcode
  /// @see inverse()
  double apply(double value) const;

  /// @brief Inverts one value lying in the range of this power transformation.
  ///
  /// Rounding can reduce accuracy close to a finite endpoint of the range.
  /// @param value Finite transformed value in this family's range at lambda.
  /// @return Restored observation on the original scale.
  /// @throws std::invalid_argument Invalid input, family, or power, or a value outside the inverse domain.
  /// @throws std::overflow_error The restored value is outside the representable double range.
  /// @par Example
  /// @code
  /// const auto fit = dm::stats::fit_yeo_johnson({-3, 0, 1, 8});
  /// std::print("{:.0f}\n", fit.inverse(fit.apply(-3))); // -3
  /// @endcode
  /// @see apply()
  double inverse(double value) const;
};

/// @brief Fits an unshifted Box–Cox transformation by maximizing Gaussian profile likelihood.
///
/// Profiles out the Gaussian mean and variance and includes the transformation
/// Jacobian. Searches the closed interval [lower, upper].
/// @param sample At least two distinct finite strictly positive observations.
/// @param lower Finite lower bound for lambda, default -5.
/// @param upper Finite upper bound for lambda, greater than lower, default 5.
/// @return Fitted power, profiled objective and reusable transform.
/// @throws std::invalid_argument Invalid sample or bounds.
/// @throws std::overflow_error Unrepresentable likelihood at the selected power.
/// @par Example
/// @code
/// const auto fit = dm::stats::fit_box_cox({1, 2, 4, 8});
/// std::print("{:.2f}\n", std::abs(fit.lambda)); // 0.00
/// @endcode
/// @par References
/// - G. E. P. Box and D. R. Cox (1964), https://doi.org/10.1111/j.2517-6161.1964.tb00553.x
/// @see fit_yeo_johnson(), FittedPowerTransform
FittedPowerTransform fit_box_cox(const std::vector<double>& sample, double lower = -5, double upper = 5);

/// @brief Fits a Yeo–Johnson transformation by maximizing Gaussian profile likelihood.
///
/// Profiles out the Gaussian mean and variance and includes the signed
/// transformation Jacobian. Searches the closed interval [lower, upper].
/// @param sample At least two distinct finite observations.
/// @param lower Finite lower bound for lambda, default -5.
/// @param upper Finite upper bound for lambda, greater than lower, default 5.
/// @return Fitted power, profiled objective and reusable transform.
/// @throws std::invalid_argument Invalid sample or bounds.
/// @throws std::overflow_error Unrepresentable likelihood at the selected power.
/// @par Example
/// @code
/// const auto fit = dm::stats::fit_yeo_johnson({-8, -1, 0, 1, 8});
/// std::print("{:.2f}\n", fit.lambda); // 1.00
/// @endcode
/// @par References
/// - I. K. Yeo and R. A. Johnson (2000), https://doi.org/10.1093/biomet/87.4.954
/// @see fit_box_cox(), FittedPowerTransform
FittedPowerTransform fit_yeo_johnson(const std::vector<double>& sample, double lower = -5, double upper = 5);

} // namespace datamunge::stats
