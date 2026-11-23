/// @file matrix_distributions.cpp
/// @brief Wishart densities and draws through triangular factorizations.

#include <datamunge/random/matrix_distributions.hpp>
#include <datamunge/linalg/cholesky.hpp>
#include "gamma_sampling.hpp"

#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace datamunge::random {

/// @brief Private factorization and density calculations.
namespace {

/// @brief Widened matrix type for intermediate factorizations.
using Matrix = linalg::DenseMatrix<long double>;

/// @brief Validates the distribution's degrees of freedom for a nonempty dimension.
/// @param df Proposed degrees of freedom.
/// @param dimension Desired matrix dimension.
/// @throws std::invalid_argument An empty dimension or degrees of freedom outside the domain.
void check_df(double df, std::size_t dimension) {
  if (dimension == 0 || !std::isfinite(df) || !(static_cast<long double>(df) > dimension - 1))
    throw std::invalid_argument("Wishart: degrees of freedom must be finite and greater than dimension minus one");
}

/// @brief Validates symmetry and positive definiteness and returns the Cholesky factor.
/// @param matrix Input matrix with symmetric entries on both sides of the diagonal.
/// @return Widened lower triangular Cholesky factor.
/// @throws std::invalid_argument Empty, nonsquare, nonfinite, asymmetric, or non-positive-definite matrix.
Matrix positive_factor(const linalg::DenseMatrix<double>& matrix) {
  const auto n = matrix.rows();
  if (n == 0 || matrix.cols() != n)
    throw std::invalid_argument("Wishart: matrices must be nonempty and square");
  Matrix widened(n, n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      if (!std::isfinite(matrix(i, j)) || matrix(i, j) != matrix(j, i))
        throw std::invalid_argument("Wishart: matrices must have finite symmetric entries");
      widened(i, j) = widened(j, i) = matrix(i, j);
    }
  }
  auto factor = linalg::cholesky(widened);
  if (!factor.ok)
    throw std::invalid_argument("Wishart: matrices must be positive definite");
  return std::move(factor.L);
}

/// @brief Solves a lower triangular system with one or more right-hand sides.
/// @param lower Lower triangular factor with nonzero diagonal.
/// @param rhs Matrix of right-hand sides, overwritten by the solve.
/// @return Overwritten right-hand sides.
Matrix solve_lower(const Matrix& lower, Matrix rhs) {
  for (std::size_t i = 0; i < rhs.rows(); ++i) {
    for (std::size_t j = 0; j < rhs.cols(); ++j) {
      auto value = rhs(i, j);
      for (std::size_t k = 0; k < i; ++k)
        value -= lower(i, k) * rhs(k, j);
      rhs(i, j) = value / lower(i, i);
    }
  }
  return rhs;
}

/// @brief Sums the logarithms of positive Cholesky pivots.
/// @param lower Cholesky factor.
/// @return One half of the log determinant of the represented symmetric matrix.
long double half_log_determinant(const Matrix& lower) {
  long double result = 0;
  for (std::size_t i = 0; i < lower.rows(); ++i)
    result += std::log(lower(i, i));
  return result;
}

/// @brief Evaluates a chi-square log density without canceling large terms near its mean.
/// @param value Positive chi-square variate.
/// @param df Positive degrees of freedom.
/// @return Normalized natural log density.
long double chi_square_log_pdf(long double value, long double df) {
  const auto shape = df / 2;
  const auto log_value = std::log(value);
  if (shape < 10000)
    return (shape - 1) * log_value - value / 2 - std::lgamma(shape) - shape * std::numbers::ln2_v<long double>;
  const auto delta = (value - df) / df;
  const auto deviance = delta < -.5L ? delta - (log_value - std::log(df)) : detail::gamma_divergence(delta);
  const auto inv_shape = 1 / shape;
  return .5L * std::log(shape / (2 * std::numbers::pi_v<long double>)) - log_value -
         shape * deviance - inv_shape * (1.L / 12 - inv_shape * inv_shape / 360);
}

/// @brief Evaluates the unit-scale Wishart density via the entries of its Bartlett factor.
/// @param factor Whitened lower triangular factor.
/// @param df Degrees of freedom of the matrix law.
/// @return Natural log density after the Cholesky change of variables.
long double bartlett_log_pdf(const Matrix& factor, long double df) {
  const auto n = factor.rows();
  const auto log_normal_constant = .5L * std::log(2 * std::numbers::pi_v<long double>);
  long double result = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const auto pivot = factor(i, i);
    result += chi_square_log_pdf(pivot * pivot, df - i) - (n - i - 1) * std::log(pivot);
    for (std::size_t j = 0; j < i; ++j)
      result -= factor(i, j) * factor(i, j) / 2 + log_normal_constant;
  }
  return result;
}

/// @brief Draws the lower Bartlett factor with independent normal and chi-square entries.
/// @param rng Engine used for all draws.
/// @param df Validated degrees of freedom.
/// @param dimension Number of rows and columns.
/// @return Lower Bartlett factor with a positive diagonal in widened precision.
/// @throws std::overflow_error A diagonal draw underflows or overflows.
Matrix bartlett_draw(RandomGenerator& rng, long double df, std::size_t dimension) {
  Matrix factor(dimension, dimension);
  for (std::size_t i = 0; i < dimension; ++i) {
    const auto log_gamma = detail::gamma_draw_log(rng, (df - i) / 2, 1);
    factor(i, i) = std::exp((std::numbers::ln2_v<long double> + log_gamma) / 2);
    if (!(factor(i, i) > 0) || !std::isfinite(factor(i, i)))
      throw std::overflow_error("Wishart: Bartlett diagonal is outside the representable range");
    for (std::size_t j = 0; j < i; ++j)
      factor(i, j) = rng.normal();
  }
  return factor;
}

/// @brief Forms a symmetric row Gram matrix and checks that rounding preserves positive definiteness.
/// @param factor Matrix whose rows are paired in the Gram matrix.
/// @return Finite, positive-definite double-precision matrix.
/// @throws std::overflow_error Overflow or loss of positive definiteness in double precision.
linalg::DenseMatrix<double> rounded_gram(const Matrix& factor) {
  const auto n = factor.rows();
  linalg::DenseMatrix<double> result(n, n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      long double dot = 0;
      for (std::size_t k = 0; k < factor.cols(); ++k)
        dot += factor(i, k) * factor(j, k);
      if (!std::isfinite(dot) || std::abs(dot) > std::numeric_limits<double>::max())
        throw std::overflow_error("Wishart: draw exceeds the finite double range");
      result(i, j) = result(j, i) = static_cast<double>(dot);
    }
  }
  try {
    positive_factor(result);
  } catch (const std::invalid_argument&) {
    throw std::overflow_error("Wishart: draw cannot be represented as a positive-definite double matrix");
  }
  return result;
}

} // namespace

double wishart_log_pdf(const linalg::DenseMatrix<double>& value, double df,
                       const linalg::DenseMatrix<double>& scale) {
  const auto n = scale.rows();
  check_df(df, n);
  if (value.rows() != n)
    throw std::invalid_argument("Wishart: value and scale must have the same dimensions");
  const auto scale_factor = positive_factor(scale);
  const auto value_factor = positive_factor(value);
  const auto log_density = n == 1 ? chi_square_log_pdf(static_cast<long double>(value(0, 0)) / scale(0, 0), df)
                                  : bartlett_log_pdf(solve_lower(scale_factor, value_factor), df);
  return static_cast<double>(log_density - (n + 1) * half_log_determinant(scale_factor));
}

double inverse_wishart_log_pdf(const linalg::DenseMatrix<double>& value, double df,
                               const linalg::DenseMatrix<double>& scale) {
  const auto n = scale.rows();
  check_df(df, n);
  if (value.rows() != n)
    throw std::invalid_argument("inverse-Wishart: value and scale must have the same dimensions");
  const auto scale_factor = positive_factor(scale);
  const auto value_factor = positive_factor(value);
  // The congruence of the inverse has the same determinant and trace
  // as the Gram matrix of the triangular solve.
  const auto log_density = n == 1 ? chi_square_log_pdf(static_cast<long double>(scale(0, 0)) / value(0, 0), df)
                                  : bartlett_log_pdf(solve_lower(value_factor, scale_factor), df);
  return static_cast<double>(log_density + (n + 1) *
      (half_log_determinant(scale_factor) - 2 * half_log_determinant(value_factor)));
}

linalg::DenseMatrix<double> wishart_sample(RandomGenerator& rng, double df,
                                           const linalg::DenseMatrix<double>& scale) {
  check_df(df, scale.rows());
  const auto factor = positive_factor(scale);
  return rounded_gram(factor * bartlett_draw(rng, df, scale.rows()));
}

linalg::DenseMatrix<double> inverse_wishart_sample(RandomGenerator& rng, double df,
                                                   const linalg::DenseMatrix<double>& scale) {
  check_df(df, scale.rows());
  const auto factor = positive_factor(scale);
  const auto bartlett = bartlett_draw(rng, df, scale.rows());
  return rounded_gram(solve_lower(bartlett, factor.transpose()).transpose());
}

} // namespace datamunge::random
