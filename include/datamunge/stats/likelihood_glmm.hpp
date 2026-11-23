#pragma once

/// @file likelihood_glmm.hpp
/// @brief Integrated likelihood for Gaussian random-intercept binomial and Poisson models.

#include <datamunge/stats/glmm.hpp>

namespace datamunge::stats {

/// @brief Optimization outcome for the integrated-likelihood fit.
/// @par Example
/// @code
/// const auto s = dm::stats::LikelihoodGLMMStatus::ZeroVariance;
/// std::println("{}", s == dm::stats::LikelihoodGLMMStatus::ZeroVariance); // true
/// @endcode
enum class LikelihoodGLMMStatus {
    Converged,       ///< Interior fit with observed-information covariance.
    ZeroVariance,    ///< The fixed-effect model was selected at zero random-intercept variance.
    IterationLimit,  ///< The selected candidate exhausted its allowed updates.
    NumericalFailure ///< The selected candidate failed a numerical optimization or information check.
};

/// @brief Family, quadrature order, and controls for fitting a single random intercept.
/// @par Example
/// @code
/// const dm::stats::LikelihoodGLMMOptions options{.quadrature_points = 1};
/// std::println("{}", options.quadrature_points); // 1
/// @endcode
struct LikelihoodGLMMOptions {
    GLMMFamily family{GLMMFamily::Binomial}; ///< Binomial with logit link or Poisson with log link.
    std::size_t quadrature_points{15};       ///< One uses Laplace. Two through 48 use adaptive Gauss–Hermite.
    std::vector<std::size_t> trials{};       ///< Binomial trial counts, empty for one trial per row.
    std::vector<double> offset{};            ///< Fixed additive link-scale offsets, or empty for zero.
    std::size_t max_iterations{300};         ///< Accepted optimization updates per deterministic start.
    double tolerance{1e-9};                  ///< Absolute score tolerance per observation in working units.
};

/// @brief Fixed effects, random-intercept SD, conditional modes, and marginal likelihood.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(4,1,1);
/// const auto fit = dm::stats::fit_likelihood_glmm(x, {0,1,0,1}, {1,1,2,2});
/// std::println("{:.1f}", fit.random_intercept_sd); // 0.0
/// @endcode
struct LikelihoodGLMMFit {
    GLMMFamily family;                       ///< Conditional response family.
    std::vector<double> coefficients;       ///< Fixed effects in the original design order.
    double random_intercept_sd;              ///< Gaussian random-intercept SD on the link scale.
    std::vector<std::size_t> group_levels;    ///< Sorted group labels.
    std::vector<double> group_modes;         ///< Conditional modes in group_levels order.
    std::vector<double> fitted;              ///< Conditional fitted probabilities or means in row order.
    double log_likelihood;                    ///< Integrated log likelihood under the selected approximation.
    linalg::DenseMatrix<double> covariance; ///< Observed covariance, fixed effects then SD if positive. Empty if unavailable.
    LikelihoodGLMMStatus status;              ///< Optimization and information outcome.

    /// @brief Predicts conditional Binomial probabilities or Poisson means.
    /// @param design Finite query predictors with the fitted columns.
    /// @param groups Group per query row, or empty to use a zero random intercept.
    /// Unknown labels also use a zero random intercept.
    /// @param offset Optional additive link-scale offsets.
    /// @return Probabilities or mean counts in query order.
    /// @throws std::invalid_argument Incompatible inputs or invalid fitted values.
    /// @throws std::overflow_error A predicted mean exceeds the finite reporting range.
    /// @par Example
    /// @code
    /// const dm::linalg::DenseMatrix<double> x(4,1,1);
    /// const auto fit = dm::stats::fit_likelihood_glmm(x, {0,1,0,1}, {1,1,2,2});
    /// std::println("{:.1f}", fit.predict(x)[0]); // 0.5
    /// @endcode
    [[nodiscard]] std::vector<double> predict(const linalg::DenseMatrix<double>& design,
        const std::vector<std::size_t>& groups = {}, const std::vector<double>& offset = {}) const;
};

/// @brief Maximizes the marginal likelihood of a Gaussian random-intercept GLMM.
///
/// Each group integral is centered at its conditional mode and scaled by its
/// posterior curvature. One quadrature point gives Laplace's approximation.
/// The fit also evaluates the exact zero-variance boundary. Supply any desired
/// intercept in the design. Returned training predictions are conditional on
/// the fitted group modes, and binomial predictions are per-trial probabilities.
/// @param design Finite full-rank fixed predictors with one row per count.
/// @param counts Binomial successes or Poisson counts.
/// @param groups At least two distinct group labels, one per row.
/// @param options Family, optional binomial trials and offsets, and numerical controls.
/// @return The best candidate found from the deterministic starts and boundary fit.
/// @throws std::invalid_argument Invalid shapes, rank, counts, groups or options.
/// @throws std::length_error Coefficient information matrix exceeds storage capacity.
/// @throws std::overflow_error A fitted output exceeds the finite reporting range.
/// @throws std::runtime_error No usable candidate was found or quadrature failed.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> x(4,1,1);
/// const auto fit = dm::stats::fit_likelihood_glmm(x, {0,1,0,1}, {1,1,2,2},
///     {.quadrature_points = 1});
/// std::println("{:.1f}", fit.log_likelihood); // -2.8
/// @endcode
/// @par References
/// - Liu and Pierce (1994), https://doi.org/10.1093/biomet/81.3.624
/// - Pinheiro and Bates (1995), https://doi.org/10.1080/10618600.1995.10474663
[[nodiscard]] LikelihoodGLMMFit fit_likelihood_glmm(const linalg::DenseMatrix<double>& design,
    const std::vector<std::size_t>& counts, const std::vector<std::size_t>& groups,
    const LikelihoodGLMMOptions& options = {});

} // namespace datamunge::stats
