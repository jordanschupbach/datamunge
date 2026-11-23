#pragma once

/// @file multivariate_normality.hpp
/// @brief Affine-invariant tests of multivariate Gaussianity.

#include <datamunge/linalg/dense_matrix.hpp>

namespace datamunge::stats {

/// @brief Mardia skewness and kurtosis statistics with their approximate reference tests.
/// @par Example
/// @code
/// const auto fit = dm::stats::mardia_test(dm::linalg::DenseMatrix<double>(4,1,{-2,-1,1,2}));
/// std::println("{:.1f}", fit.skewness); // 0.0
/// @endcode
struct MardiaTestResult {
    double skewness;         ///< Mean cubed whitened inner product.
    double kurtosis;         ///< Mean fourth power of whitened lengths.
    double skewness_chi_squared; ///< Chi-square reference statistic for skewness.
    double skewness_df;      ///< Dimension-dependent chi-square degrees of freedom.
    double skewness_p_value; ///< Upper chi-square tail.
    double kurtosis_z;       ///< Signed standard normal reference statistic for kurtosis.
    double kurtosis_p_value; ///< Two-sided normal tail.
};

/// @brief Henze–Zirkler statistic with a dimension-dependent lognormal reference law.
/// @par Example
/// @code
/// const auto fit = dm::stats::henze_zirkler_test(dm::linalg::DenseMatrix<double>(4,1,{-2,-1,1,2}));
/// std::println("{}", fit.p_value > 0); // true
/// @endcode
struct HenzeZirklerTestResult {
    double statistic;        ///< Henze–Zirkler statistic.
    double bandwidth;        ///< Smoothing parameter determined by sample size and dimension.
    double mean_log;         ///< Mean of the log reference statistic.
    double standard_deviation_log; ///< Standard deviation on the log scale.
    double p_value;          ///< Upper lognormal tail.
};

/// @brief Tests multivariate skewness and kurtosis using Mardia's measures.
/// @param observations Finite independent rows with at least two more rows than columns.
/// @return Measures and approximate p-values. Skewness uses the small-sample correction below 20 rows.
/// @throws std::invalid_argument The sample has invalid dimensions or nonfinite values.
/// @throws std::domain_error The centered covariance is singular.
/// @throws std::length_error The dense workspace exceeds storage capacity.
/// @throws std::runtime_error The centered factorization failed to converge.
/// @par Example
/// @code
/// const auto fit = dm::stats::mardia_test(dm::linalg::DenseMatrix<double>(4,1,{-2,-1,1,2}));
/// std::println("{:.0f}", fit.skewness_df); // 1
/// @endcode
/// @par References
/// - Mardia (1970), https://doi.org/10.1093/biomet/57.3.519
[[nodiscard]] MardiaTestResult mardia_test(const linalg::DenseMatrix<double>& observations);

/// @brief Tests multivariate Gaussianity with a distance-based statistic and lognormal calibration.
/// @param observations Finite independent rows with at least two more rows than columns.
/// @return Test statistic, bandwidth, reference parameters, and upper-tail p-value.
/// @throws std::invalid_argument The sample has invalid dimensions or nonfinite values.
/// @throws std::domain_error The centered covariance is singular.
/// @throws std::length_error The dense workspace exceeds storage capacity.
/// @throws std::runtime_error The decomposition failed or the reference moments exceed numerical precision.
/// @par Example
/// @code
/// const auto fit = dm::stats::henze_zirkler_test(dm::linalg::DenseMatrix<double>(4,1,{-2,-1,1,2}));
/// std::println("{}", fit.statistic >= 0); // true
/// @endcode
/// @par References
/// - Henze and Zirkler (1990), https://doi.org/10.1080/03610929008830400
[[nodiscard]] HenzeZirklerTestResult henze_zirkler_test(const linalg::DenseMatrix<double>& observations);

} // namespace datamunge::stats
