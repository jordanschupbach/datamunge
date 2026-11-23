#pragma once

/// @file hotelling_test.hpp
/// @brief Hotelling T-squared tests for multivariate means.

#include <datamunge/linalg/dense_matrix.hpp>

#include <vector>

namespace datamunge::stats {

/// @brief Hotelling statistic and the equivalent central F reference test.
/// @par Example
/// @code
/// const auto test = dm::stats::hotelling_t2_one_sample(dm::linalg::DenseMatrix<double>(3,1,{1,2,3}));
/// std::println("{:.1f}", test.t_squared); // 12.0
/// @endcode
struct HotellingTestResult {
    double t_squared;                    ///< Hotelling T-squared statistic.
    double f_statistic;                  ///< Equivalent F statistic.
    std::size_t df_numerator;             ///< Number of mean coordinates tested.
    std::size_t df_denominator;           ///< Denominator degrees of freedom of the reference F law.
    double p_value;                      ///< Upper-tail F probability.
    std::vector<double> mean_difference; ///< Sample mean or first-minus-second estimate, before subtracting the null.
};

/// @brief Tests a multivariate population mean using one sample.
///
/// The reference F distribution is exact for independent Gaussian vectors
/// with an unknown positive-definite covariance.
/// @param observations Finite observations as rows and outcomes as columns, with more rows than outcomes.
/// @param null_mean Tested mean vector in column order. Empty specifies zero.
/// @return Hotelling T-squared, its F test and the sample mean.
/// @throws std::invalid_argument Invalid dimensions, sample size or nonfinite values.
/// @throws std::domain_error The centered observations have singular sample covariance.
/// @throws std::length_error The working matrix exceeds its storage capacity.
/// @throws std::runtime_error The centered sample decomposition failed to converge.
/// @throws std::overflow_error The sample mean exceeds finite double range.
/// @par Example
/// @code
/// const auto test = dm::stats::hotelling_t2_one_sample(dm::linalg::DenseMatrix<double>(3,1,{1,2,3}), {1});
/// std::println("{:.1f}", test.f_statistic); // 3.0
/// @endcode
/// @par References
/// - Hotelling (1931), https://doi.org/10.1214/aoms/1177732979
[[nodiscard]] HotellingTestResult hotelling_t2_one_sample(const linalg::DenseMatrix<double>& observations,
    const std::vector<double>& null_mean = {});

/// @brief Tests a multivariate mean change using paired observations.
///
/// Applies the one-sample test to first-minus-second differences. Pairs are
/// independent of each other, with a Gaussian distribution for the differences
/// under the exact F reference law.
/// @param first Finite observations with one row per pair.
/// @param second Matching finite observations in the same row and column order.
/// @param null_difference Tested mean difference in column order. Empty specifies zero.
/// @return Hotelling T-squared, its F test and the mean paired difference.
/// @throws std::invalid_argument Invalid dimensions, sample size or nonfinite values.
/// @throws std::domain_error The differences have singular sample covariance.
/// @throws std::length_error The working matrix exceeds its storage capacity.
/// @throws std::runtime_error The centered sample decomposition failed to converge.
/// @throws std::overflow_error The sample mean difference exceeds finite double range.
/// @par Example
/// @code
/// using dm::linalg::DenseMatrix;
/// const auto test = dm::stats::hotelling_t2_paired(DenseMatrix<double>(3,1,{3,5,8}), DenseMatrix<double>(3,1,{2,3,5}));
/// std::println("{:.1f}", test.mean_difference[0]); // 2.0
/// @endcode
[[nodiscard]] HotellingTestResult hotelling_t2_paired(const linalg::DenseMatrix<double>& first,
    const linalg::DenseMatrix<double>& second, const std::vector<double>& null_difference = {});

/// @brief Tests the difference between two independent multivariate population means.
///
/// Uses the pooled within-group covariance. The exact F reference assumes
/// independent Gaussian samples with a shared positive-definite covariance.
/// @param first Finite first sample with at least two rows.
/// @param second Finite second sample, in the same outcome order, with at least two rows.
/// @param null_difference Tested first-minus-second mean difference. Empty specifies zero.
/// @return Hotelling T-squared, its F test and the estimated mean difference.
/// @throws std::invalid_argument Invalid dimensions, sample size or nonfinite values.
/// @throws std::domain_error The pooled sample covariance is singular.
/// @throws std::length_error The working matrix exceeds its storage capacity.
/// @throws std::runtime_error The centered sample decomposition failed to converge.
/// @throws std::overflow_error The sample mean difference exceeds finite double range.
/// @par Example
/// @code
/// using dm::linalg::DenseMatrix;
/// const auto test = dm::stats::hotelling_t2_two_sample(DenseMatrix<double>(3,1,{1,2,3}), DenseMatrix<double>(3,1,{2,3,4}));
/// std::println("{:.1f}", test.t_squared); // 1.5
/// @endcode
[[nodiscard]] HotellingTestResult hotelling_t2_two_sample(const linalg::DenseMatrix<double>& first,
    const linalg::DenseMatrix<double>& second, const std::vector<double>& null_difference = {});

} // namespace datamunge::stats
