#pragma once

/// @file gls.hpp
/// @brief Generalized least squares with supplied or fitted residual covariance.

#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::stats {

/// @brief Ordinary-scale fitted values and Gaussian inference from generalized least squares.
/// @par Example
/// @code
/// using dm::linalg::DenseMatrix;
/// const auto fit = dm::stats::generalized_least_squares(DenseMatrix<double>(3, 1, 1), {1, 2, 3},
///                                                      DenseMatrix<double>::identity(3));
/// std::println("{:.1f}", fit.coefficients.front()); // 2.0
/// @endcode
/// @see generalized_least_squares(), structured_gls()
struct GeneralizedLeastSquaresResult {
    std::vector<double> coefficients;       ///< In design-column order.
    linalg::DenseMatrix<double> covariance; ///< Model-based coefficient covariance conditional on the residual covariance.
    std::vector<double> fitted;             ///< Fitted values in input row order.
    std::vector<double> residuals;          ///< Observed minus fitted values in input row order.
    double log_likelihood;                 ///< Gaussian log likelihood at the fitted model.
};

/// @brief Fits the coefficients using a known symmetric positive definite residual covariance.
/// The covariance is in absolute response units. The fit whitens the design and
/// response, then solves the transformed least-squares problem.
/// @param design Full-rank finite design, with any intercept column included.
/// @param response Finite response in row order.
/// @param residual_covariance Finite symmetric positive definite n by n covariance.
/// @return Coefficients, covariance, fitted values, residuals, and Gaussian log likelihood.
/// @throws std::invalid_argument Invalid shapes, nonfinite input, covariance, or design rank.
/// @throws std::runtime_error The whitened design factorization reaches its sweep limit.
/// @throws std::overflow_error An output exceeds finite double range.
/// @par Example
/// @code
/// using dm::linalg::DenseMatrix;
/// const auto fit = dm::stats::generalized_least_squares(DenseMatrix<double>(2, 1, 1), {1, 3},
///                                                      DenseMatrix<double>(2, 2, {1, 0, 0, 3}));
/// std::println("{:.1f}", fit.coefficients.front()); // 1.5
/// @endcode
/// @par References
/// - Pinheiro and Bates (2000), Mixed-Effects Models in S and S-PLUS.
/// @see structured_gls()
GeneralizedLeastSquaresResult generalized_least_squares(
    const linalg::DenseMatrix<double>& design, const std::vector<double>& response,
    const linalg::DenseMatrix<double>& residual_covariance);

/// @brief Correlation structures for independent blocks of observations.
/// @par Example
/// @code
/// const auto type = dm::stats::GlsCorrelation::AR1;
/// std::println("{}", type == dm::stats::GlsCorrelation::AR1); // true
/// @endcode
/// @see structured_gls()
enum class GlsCorrelation {
    AR1,              ///< Correlation is a power of rho based on the within-block row distance.
    CompoundSymmetry, ///< Every distinct pair within a block has correlation rho.
};

/// @brief Grouping and numerical controls for profile maximum likelihood.
/// @par Example
/// @code
/// const dm::stats::StructuredGlsOptions options{.groups = {1, 1, 2, 2}};
/// std::println("{}", options.groups.size()); // 4
/// @endcode
/// @see structured_gls()
struct StructuredGlsOptions {
    std::vector<std::int64_t> groups{}; ///< Block label per row. Empty means one block.
    double tolerance{1e-9};            ///< Absolute convergence tolerance in correlation units.
    std::size_t max_iterations{100};   ///< Positive refinement limit per bracket.
    double boundary_margin{1e-8};     ///< Fraction of the admissible correlation range excluded at each end.
};

/// @brief Fitted GLS and diagnostics of the correlation profile search.
/// @par Example
/// @code
/// const auto fit = dm::stats::structured_gls(dm::linalg::DenseMatrix<double>(4, 1, 1), {0, 1, 0, 1},
///                                           dm::stats::GlsCorrelation::AR1);
/// std::println("{}", fit.correlation < 0); // true
/// @endcode
/// @see structured_gls()
struct StructuredGlsResult {
    GeneralizedLeastSquaresResult fit; ///< GLS fitted at the estimated residual covariance.
    double correlation;               ///< Estimated within-block correlation parameter.
    double residual_variance;         ///< Estimated marginal residual variance.
    bool converged;                    ///< All candidate brackets reached tolerance.
    bool at_boundary;                  ///< The selected estimate is near a numerical search endpoint.
};

/// @brief Fits a structured residual covariance by Gaussian profile maximum likelihood.
///
/// Independent blocks share a single variance and correlation. Rows remain
/// in input order within blocks, with unit spacing for AR(1). The search
/// profiles out coefficients and residual variance, samples the admissible
/// correlation interval, and refines its candidate maxima. The marginal
/// variance estimate divides the whitened residual sum of squares by the
/// number of observations. The coefficient covariance conditions on the
/// estimated residual covariance.
/// @param design Full-rank finite design with more rows than columns, including any intercept.
/// @param response Finite response in row order.
/// @param structure AR(1) or compound symmetry.
/// @param options Block labels and numerical controls.
/// @return Fitted model, estimated correlation and residual variance, and search diagnostics.
/// @throws std::invalid_argument Invalid inputs, controls, or design rank.
/// @throws std::domain_error No block contains repeated observations, or the residuals are numerically zero.
/// @throws std::runtime_error A whitened design factorization reaches its sweep limit or loses rank.
/// @throws std::overflow_error An output exceeds finite double range or positive variance rounds to zero.
/// @par Example
/// @code
/// const auto fit = dm::stats::structured_gls(dm::linalg::DenseMatrix<double>(4, 1, 1), {0, 1, 0, 1},
///                                           dm::stats::GlsCorrelation::AR1);
/// std::println("{}", fit.residual_variance > 0); // true
/// @endcode
/// @par References
/// - Pinheiro and Bates (2000), Mixed-Effects Models in S and S-PLUS.
/// - https://stat.ethz.ch/R-manual/R-devel/library/nlme/html/gls.html
/// @see generalized_least_squares()
StructuredGlsResult structured_gls(const linalg::DenseMatrix<double>& design, const std::vector<double>& response,
                                   GlsCorrelation structure, const StructuredGlsOptions& options = {});

} // namespace datamunge::stats
