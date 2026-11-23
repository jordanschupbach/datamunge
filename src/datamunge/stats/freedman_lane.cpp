/// @file freedman_lane.cpp
/// @brief Partial F statistics and efficient fixed-design residual permutations.

#include <datamunge/stats/freedman_lane.hpp>
#include <datamunge/linalg/svd.hpp>
#include <datamunge/stats/detail/test_common.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {

/// @brief Fixed-design linear algebra for repeated permutation fits.
namespace {

/// @brief Copies a design into a wide matrix with columns scaled to unit maximum.
/// @param target Destination design with matching rows.
/// @param start Target column offset.
/// @param source Finite design whose columns fit at that offset.
void copy_scaled_columns(linalg::DenseMatrix<long double>& target, std::size_t start,
                         const linalg::DenseMatrix<double>& source) {
  for (std::size_t col = 0; col < source.cols(); ++col) {
    long double scale = 0;
    for (std::size_t row = 0; row < source.rows(); ++row) {
      const auto value = source(row, col);
      if (!std::isfinite(value))
        throw std::invalid_argument("Freedman–Lane: design values must be finite");
      scale = std::max(scale, std::abs(static_cast<long double>(value)));
    }
    if (scale == 0)
      throw std::invalid_argument("Freedman–Lane: design has a zero column");
    for (std::size_t row = 0; row < source.rows(); ++row)
      target(row, start + col) = source(row, col) / scale;
  }
}

/// @brief Finds an orthonormal basis of a full-column-rank design.
/// @param design Scaled matrix with more rows than columns.
/// @return Thin orthonormal basis in columns.
linalg::DenseMatrix<long double> orthogonal_basis(const linalg::DenseMatrix<long double>& design) {
  auto decomposition = linalg::svd(design);
  if (decomposition.rank() != design.cols())
    throw std::invalid_argument("Freedman–Lane: rank-deficient design");
  if (!decomposition.converged)
    throw std::runtime_error("Freedman–Lane: design factorization did not converge");
  return std::move(decomposition.U);
}

/// @brief Subtracts the projection onto the given orthonormal columns.
/// @param response Vector on the same rows as the basis.
/// @param basis Thin orthonormal design basis.
/// @return Least squares residual vector.
std::vector<long double> residuals(const std::vector<long double>& response,
                                   const linalg::DenseMatrix<long double>& basis) {
  auto result = response;
  for (std::size_t col = 0; col < basis.cols(); ++col) {
    long double coefficient = 0;
    for (std::size_t row = 0; row < response.size(); ++row)
      coefficient += basis(row, col) * response[row];
    for (std::size_t row = 0; row < response.size(); ++row)
      result[row] -= coefficient * basis(row, col);
  }
  return result;
}

/// @brief Computes the squared Euclidean norm of a residual vector.
/// @param values Residuals after response rescaling.
/// @return Sum of squares.
long double sum_of_squares(const std::vector<long double>& values) {
  return std::inner_product(values.begin(), values.end(), values.begin(), 0.0L);
}

} // namespace

HypothesisTestResult freedman_lane_test(const std::vector<double>& response,
                                      const linalg::DenseMatrix<double>& nuisance,
                                      const linalg::DenseMatrix<double>& tested,
                                      std::size_t draws, random::RandomGenerator& rng) {
  detail::check_sample(response);
  const auto n = response.size(), q = nuisance.cols(), k = tested.cols();
  if (nuisance.rows() != n || tested.rows() != n || q >= n || k == 0 || k >= n - q || draws == 0)
    throw std::invalid_argument("Freedman–Lane: invalid design dimensions or permutation count");

  linalg::DenseMatrix<long double> reduced(n, q, 0), full(n, q + k, 0);
  copy_scaled_columns(reduced, 0, nuisance);
  for (std::size_t row = 0; row < n; ++row)
    for (std::size_t col = 0; col < q; ++col)
      full(row, col) = reduced(row, col);
  copy_scaled_columns(full, q, tested);
  const auto reduced_basis = orthogonal_basis(reduced);
  const auto full_basis = orthogonal_basis(full);

  long double scale = 0;
  for (const double value : response)
    scale = std::max(scale, std::abs(static_cast<long double>(value)));
  if (scale == 0)
    throw std::invalid_argument("Freedman–Lane: no reduced-model residual variation");
  std::vector<long double> scaled(response.begin(), response.end());
  for (auto& y : scaled)
    y /= scale;

  const auto original_residual = residuals(scaled, reduced_basis);
  const auto df_error = n - q - k;
  const auto roundoff = 64 * std::numeric_limits<long double>::epsilon() * n;
  const auto negligible = roundoff * roundoff;
  if (sum_of_squares(original_residual) <= negligible * sum_of_squares(scaled))
    throw std::invalid_argument("Freedman–Lane: no reduced-model residual variation");

  // The fixed null fit lies in both model spaces and has zero residual from each.
  /// @brief Computes the partial F statistic on a permuted residual response.
  const auto partial_f = [&](const std::vector<long double>& y) {
    const auto rss_reduced = sum_of_squares(residuals(y, reduced_basis));
    const auto rss_full = sum_of_squares(residuals(y, full_basis));
    if (rss_reduced <= negligible * sum_of_squares(y))
      return 0.0L;
    if (rss_full <= negligible * rss_reduced)
      return std::numeric_limits<long double>::infinity();
    return std::max(0.0L, (rss_reduced - rss_full) / k) / (rss_full / df_error);
  };
  const auto observed = partial_f(original_residual);
  const auto cutoff = std::isinf(observed) ? observed : observed - roundoff * std::max(1.0L, observed);
  auto permuted = original_residual;
  std::size_t extreme = 0;
  for (std::size_t draw = 0; draw < draws; ++draw) {
    for (std::size_t remaining = n; remaining > 1; --remaining) {
      const auto choice = static_cast<std::size_t>(rng.uniform_u64(0, remaining - 1));
      std::swap(permuted[remaining - 1], permuted[choice]);
    }
    extreme += partial_f(permuted) >= cutoff;
  }
  HypothesisTestResult result;
  result.statistic = static_cast<double>(observed);
  result.parameter1 = static_cast<double>(k);
  result.parameter2 = static_cast<double>(df_error);
  result.p_value = static_cast<double>((1.0L + extreme) / (1.0L + draws));
  result.alternative = Alternative::Greater;
  result.method = "Freedman–Lane partial F permutation test";
  return result;
}

} // namespace datamunge::stats
