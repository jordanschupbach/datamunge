#pragma once
/// @file matrix_distributions.hpp
/// @brief Wishart and inverse-Wishart matrix densities and sampling.

#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/random/random.hpp>

namespace datamunge::random {

/// @brief Evaluates the normalized Wishart log density for a positive-definite matrix.
///
/// Uses the scale parameterization in which the mean is df times scale.
/// The density uses the independent entries on and below the diagonal.
/// @param value Finite, symmetric positive-definite matrix of dimension p.
/// @param df Finite degrees of freedom greater than p - 1, including fractional values.
/// @param scale Finite, symmetric positive-definite scale of the same dimension.
/// @return Natural logarithm of the density.
/// @throws std::invalid_argument Invalid degrees of freedom, dimensions, symmetry, or positive definiteness.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(1, 1, {2}), scale(1, 1, {1});
/// std::print("{:.3f}\n", dm::random::wishart_log_pdf(x, 2, scale)); // -1.693
/// @endcode
/// @par References
/// - https://mc-stan.org/docs/functions-reference/covariance_matrix_distributions.html#wishart-distribution
/// @see wishart_sample(), inverse_wishart_log_pdf()
double wishart_log_pdf(const linalg::DenseMatrix<double>& value, double df,
                       const linalg::DenseMatrix<double>& scale);

/// @brief Evaluates the normalized inverse-Wishart log density.
///
/// Uses the scale parameterization in which the mean is scale divided by
/// df - p - 1 when df > p + 1. The density uses the independent entries
/// on and below the diagonal.
/// @param value Finite, symmetric positive-definite matrix of dimension p.
/// @param df Finite degrees of freedom greater than p - 1, including fractional values.
/// @param scale Finite, symmetric positive-definite scale of the same dimension.
/// @return Natural logarithm of the density.
/// @throws std::invalid_argument Invalid degrees of freedom, dimensions, symmetry, or positive definiteness.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(1, 1, {2}), scale(1, 1, {4});
/// std::print("{:.3f}\n", dm::random::inverse_wishart_log_pdf(x, 2, scale)); // -1.693
/// @endcode
/// @par References
/// - https://mc-stan.org/docs/functions-reference/covariance_matrix_distributions.html#inverse-wishart-distribution
/// @see inverse_wishart_sample(), wishart_log_pdf()
double inverse_wishart_log_pdf(const linalg::DenseMatrix<double>& value, double df,
                               const linalg::DenseMatrix<double>& scale);

/// @brief Draws a Wishart matrix using a Bartlett factor and the scale Cholesky factor.
///
/// Equal initial generator states reproduce draws in the same environment.
/// The draw is symmetric and positive definite.
/// @param rng Caller-owned random generator, advanced by sampling.
/// @param df Finite degrees of freedom greater than p - 1.
/// @param scale Finite, symmetric positive-definite p by p scale matrix.
/// @return One Wishart draw on the scale of the input matrix.
/// @throws std::invalid_argument Invalid degrees of freedom or scale matrix.
/// @throws std::overflow_error The draw cannot be represented as a finite positive-definite double matrix.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(4);
/// const dm::linalg::DenseMatrix<double> scale(1, 1, {1});
/// std::print("{}\n", dm::random::wishart_sample(rng, 3.5, scale)(0, 0) > 0); // true
/// @endcode
/// @par References
/// - https://mc-stan.org/docs/functions-reference/covariance_matrix_distributions.html#wishart-distribution
/// @see wishart_log_pdf(), inverse_wishart_sample()
linalg::DenseMatrix<double> wishart_sample(RandomGenerator& rng, double df,
                                           const linalg::DenseMatrix<double>& scale);

/// @brief Draws an inverse-Wishart matrix using triangular solves against a Bartlett factor.
///
/// Equal initial generator states reproduce draws in the same environment.
/// The draw is symmetric and positive definite.
/// @param rng Caller-owned random generator, advanced by sampling.
/// @param df Finite degrees of freedom greater than p - 1.
/// @param scale Finite, symmetric positive-definite p by p scale matrix.
/// @return One inverse-Wishart draw on the scale of the input matrix.
/// @throws std::invalid_argument Invalid degrees of freedom or scale matrix.
/// @throws std::overflow_error The draw cannot be represented as a finite positive-definite double matrix.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(4);
/// const dm::linalg::DenseMatrix<double> scale(1, 1, {1});
/// std::print("{}\n", dm::random::inverse_wishart_sample(rng, 3.5, scale)(0, 0) > 0); // true
/// @endcode
/// @par References
/// - https://mc-stan.org/docs/functions-reference/covariance_matrix_distributions.html#inverse-wishart-distribution
/// @see inverse_wishart_log_pdf(), wishart_sample()
linalg::DenseMatrix<double> inverse_wishart_sample(RandomGenerator& rng, double df,
                                                   const linalg::DenseMatrix<double>& scale);

} // namespace datamunge::random
