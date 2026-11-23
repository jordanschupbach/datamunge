#pragma once

/// @file gee.hpp
/// @brief Marginal regression for independent clusters with sandwich uncertainty.

#include <datamunge/linalg/dense_matrix.hpp>

#include <optional>
#include <vector>

namespace datamunge::stats {

/// @brief Response variance and link used in the marginal mean model.
/// @par Example
/// @code
/// const auto f = dm::stats::GEEFamily::Poisson;
/// std::println("{}", f == dm::stats::GEEFamily::Poisson); // true
/// @endcode
enum class GEEFamily {
    Gaussian, ///< Continuous response with identity link and constant variance.
    Binomial, ///< Binomial counts with logit link and optional trial counts.
    Poisson,  ///< Nonnegative integer counts with log link.
    Gamma     ///< Positive continuous response with log link and squared-mean variance.
};

/// @brief Working within-cluster correlation in the estimating equations.
/// @par Example
/// @code
/// const auto c = dm::stats::GEECorrelation::AR1;
/// std::println("{}", c == dm::stats::GEECorrelation::AR1); // true
/// @endcode
enum class GEECorrelation {
    Independence, ///< Working correlation zero across distinct rows.
    Exchangeable, ///< The same pairwise correlation within each cluster.
    AR1           ///< Correlation equal to the parameter raised to the integer visit gap.
};

/// @brief Outcome of the estimating-equation iteration.
/// @par Example
/// @code
/// const auto s = dm::stats::GEEStatus::Converged;
/// std::println("{}", s == dm::stats::GEEStatus::Converged); // true
/// @endcode
enum class GEEStatus {
    Converged,       ///< The working predictor update met tolerance.
    IterationLimit,  ///< The maximum number of accepted mean updates was reached.
    NumericalFailure ///< A finite fit was retained after a failed update or information solve.
};

/// @brief Specifies working correlation, link offsets and estimation controls.
/// @par Example
/// @code
/// const dm::stats::GEEOptions options{.correlation = dm::stats::GEECorrelation::AR1, .fixed_correlation = .5};
/// std::println("{:.1f}", *options.fixed_correlation); // 0.5
/// @endcode
struct GEEOptions {
    GEEFamily family{GEEFamily::Gaussian};                    ///< Marginal response family and link.
    GEECorrelation correlation{GEECorrelation::Independence}; ///< Working correlation pattern.
    std::optional<double> fixed_correlation{}; ///< Known working parameter. Empty estimates it from residuals.
    std::vector<std::size_t> visit{};  ///< Distinct integer visit indices within each cluster for AR(1), or cluster row order.
    std::vector<std::size_t> trials{}; ///< Binomial trial counts, or one trial per row when empty.
    std::vector<double> offset{};     ///< Fixed additive link-scale offsets, or zero when empty.
    std::size_t max_iterations{100};  ///< Maximum accepted mean updates.
    double tolerance{1e-9};           ///< Relative tolerance for the largest fixed predictor update.
};

/// @brief Fitted marginal means and model-based and cluster-robust covariance estimates.
/// @par Example
/// @code
/// const auto fit = dm::stats::fit_gee(dm::linalg::DenseMatrix<double>(4,1,1), {1,2,3,4}, {1,1,2,2});
/// std::println("{:.1f}", fit.coefficients[0]); // 2.5
/// @endcode
struct GEEFit {
    GEEFamily family;                        ///< Fitted family.
    std::vector<double> coefficients;        ///< Slopes in original design units.
    std::vector<double> fitted;              ///< Means in training row order. For binomial, per-trial probabilities.
    double correlation_parameter;            ///< Working exchangeable or lag-one correlation, zero for independence.
    double dispersion;                       ///< Pearson dispersion for Gaussian/Gamma, one otherwise.
    linalg::DenseMatrix<double> model_covariance; ///< Model-based fixed-effect covariance.
    linalg::DenseMatrix<double> sandwich_covariance; ///< Uncorrected cluster-robust covariance.
    GEEStatus status;                         ///< Optimization outcome.
    std::size_t iterations;                   ///< Number of accepted mean updates.

    /// @brief Predicts marginal means, including any specified link offsets.
    /// @param design Finite query rows with the fitted columns.
    /// @param offset Additive link-scale offsets, or empty.
    /// @return Mean response per row, or per-trial probability for binomial.
    /// @throws std::invalid_argument Invalid shapes, family or inputs.
    /// @throws std::overflow_error A prediction exceeds finite reporting range.
    /// @par Example
    /// @code
    /// const dm::linalg::DenseMatrix<double> x(4,1,1);
    /// const auto fit = dm::stats::fit_gee(x, {1,2,3,4}, {1,1,2,2});
    /// std::println("{:.1f}", fit.predict(x)[0]); // 2.5
    /// @endcode
    [[nodiscard]] std::vector<double> predict(const linalg::DenseMatrix<double>& design,
        const std::vector<double>& offset = {}) const;
};

/// @brief Fits marginal regression by generalized estimating equations.
///
/// Independence, exchangeable and AR(1) use the same marginal mean.
/// Exchangeable estimates average standardized Pearson cross-products.
/// AR(1) fits powers of the lag-one correlation to those products at the
/// observed visit gaps. Estimated correlations are capped at 99% of
/// the positive-definite bounds. Cluster sandwich uncertainty uses the
/// separate cluster estimating scores. Covariance matrices are empty
/// when the final information is numerically singular.
/// @param design Finite full-rank fixed predictors. Include any intercept explicitly.
/// @param response Finite observed responses in the chosen family domain.
/// @param cluster One cluster label per row, with at least two distinct labels.
/// @param options Family, working correlation and estimation controls.
/// @return Marginal fit and covariance estimates. Check status before inference.
/// @throws std::invalid_argument Invalid input, rank, family, or correlation.
/// @throws std::length_error A working matrix exceeds storage capacity.
/// @throws std::runtime_error No usable initial mean or information was found.
/// @throws std::overflow_error A fitted output exceeds finite reporting range.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(4,1,1);
/// const auto fit = dm::stats::fit_gee(x, {1,2,3,4}, {1,1,2,2});
/// std::println("{:.2f}", fit.sandwich_covariance(0,0)); // 0.50
/// @endcode
/// @par References
/// - Liang and Zeger (1986), https://doi.org/10.1093/biomet/73.1.13
[[nodiscard]] GEEFit fit_gee(const linalg::DenseMatrix<double>& design, const std::vector<double>& response,
    const std::vector<std::size_t>& cluster, const GEEOptions& options = {});

} // namespace datamunge::stats
