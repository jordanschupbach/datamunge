#pragma once

/// @file normality_tests.hpp
/// @brief Omnibus moment tests for normally distributed observations.

#include <vector>

namespace datamunge::stats {

/// @brief Normality statistic with separate skewness and kurtosis contributions.
/// @par Example
/// @code
/// const auto fit = dm::stats::jarque_bera_test({-2, -1, 0, 1, 2});
/// std::print("{:.1f}\n", fit.skewness_contribution); // 0.0
/// @endcode
/// @see dagostino_pearson_test(), jarque_bera_test()
struct MomentNormalityResult {
  double statistic;             ///< Sum of the two nonnegative components.
  double p_value;                ///< Upper-tail probability under chi-square with two degrees of freedom.
  double skewness_contribution;  ///< Omnibus contribution associated with skewness.
  double kurtosis_contribution;  ///< Omnibus contribution associated with kurtosis.
};

/// @brief Tests normality with D'Agostino–Pearson's omnibus K squared statistic.
///
/// The components are the squared normalizing transforms of uncorrected
/// sample skewness and kurtosis. Kurtosis calibration is approximate,
/// especially below 20 observations.
/// @param values At least eight finite, nonconstant observations.
/// @return Omnibus statistic, asymptotic p-value and separate components.
/// @throws std::invalid_argument Too few observations, nonfinite input or constant sample.
/// @par Example
/// @code
/// const auto fit = dm::stats::dagostino_pearson_test({1, 2, 3, 4, 5, 6, 7, 8});
/// std::print("{:.1f}\n", fit.skewness_contribution); // 0.0
/// @endcode
/// @par References
/// - R. B. D'Agostino, A. Belanger and R. B. D'Agostino Jr. (1990), https://doi.org/10.1080/00031305.1990.10475751
/// - F. J. Anscombe and W. J. Glynn (1983), https://doi.org/10.1093/biomet/70.1.227
/// @see jarque_bera_test(), sample_shape()
MomentNormalityResult dagostino_pearson_test(const std::vector<double>& values);

/// @brief Tests normality with the asymptotic Jarque–Bera statistic.
///
/// The contributions are n times squared skewness over six and n times
/// squared excess kurtosis over 24, using uncorrected standardized moments.
/// @param values At least two finite, nonconstant observations.
/// @return Omnibus statistic, asymptotic p-value and separate components.
/// @throws std::invalid_argument Too few observations, nonfinite input or constant sample.
/// @par Example
/// @code
/// const auto fit = dm::stats::jarque_bera_test({-2, -1, 0, 1, 2});
/// std::print("{:.3f}\n", fit.statistic); // 0.352
/// @endcode
/// @par References
/// - C. M. Jarque and A. K. Bera (1987), https://doi.org/10.2307/1403192
/// @see dagostino_pearson_test(), sample_shape()
MomentNormalityResult jarque_bera_test(const std::vector<double>& values);

} // namespace datamunge::stats
