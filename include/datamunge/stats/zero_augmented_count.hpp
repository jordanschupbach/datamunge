#pragma once

/// @file zero_augmented_count.hpp
/// @brief Zero-inflated and hurdle count regression with separate predictor designs.

#include <datamunge/linalg/dense_matrix.hpp>

#include <vector>

namespace datamunge::stats {

/// @brief Distribution underlying the count component.
/// @par Example
/// @code
/// const auto family = dm::stats::ZeroAugmentedCountFamily::NegativeBinomial;
/// std::println("{}", family == dm::stats::ZeroAugmentedCountFamily::NegativeBinomial); // true
/// @endcode
enum class ZeroAugmentedCountFamily {
    Poisson,         ///< Variance equal to the mean.
    NegativeBinomial ///< NB2 with estimated overdispersion and a Poisson limit.
};

/// @brief Interpretation of the binary zero component.
/// @par Example
/// @code
/// const auto type = dm::stats::ZeroAugmentedCountModel::Hurdle;
/// std::println("{}", type == dm::stats::ZeroAugmentedCountModel::Hurdle); // true
/// @endcode
enum class ZeroAugmentedCountModel {
    ZeroInflated, ///< Mixes structural zeros with an ordinary count distribution.
    Hurdle        ///< Combines a zero probability and a zero-truncated count distribution.
};

/// @brief Numerical outcome of a zero-augmented count fit.
/// @par Example
/// @code
/// const auto status = dm::stats::ZeroAugmentedCountStatus::Converged;
/// std::println("{}", status == dm::stats::ZeroAugmentedCountStatus::Converged); // true
/// @endcode
enum class ZeroAugmentedCountStatus {
    Converged,       ///< Finite fit with observed-information covariance.
    PoissonBoundary, ///< An NB2 fit selected zero dispersion.
    NoZeroComponent, ///< The binary zero probability is zero at the selected boundary.
    IterationLimit,  ///< The best candidate did not reach the score tolerance.
    NumericalFailure ///< The selected fit failed a numerical information or line-search check.
};

/// @brief Family, offsets, weights, and optimization settings shared by the two models.
/// @par Example
/// @code
/// const dm::stats::ZeroAugmentedCountOptions options{.offset = {0, std::log(2.)}};
/// std::println("{}", options.offset.size()); // 2
/// @endcode
struct ZeroAugmentedCountOptions {
    ZeroAugmentedCountFamily family{ZeroAugmentedCountFamily::Poisson}; ///< Count distribution to fit.
    std::vector<double> offset{};   ///< Additive count log-scale offsets. Empty means unit exposure.
    std::vector<double> weights{};  ///< Positive finite case weights. Empty means unit weights.
    std::size_t max_iterations{400}; ///< Accepted updates allowed from each deterministic start.
    double tolerance{1e-9};        ///< Absolute coefficient-score tolerance per unit of case weight, in (0, 1).
};

/// @brief Row-aligned predictions from a zero-augmented count fit.
/// @par Example
/// @code
/// const dm::stats::ZeroAugmentedCountPrediction prediction{{2}, {.5}, {.568}, {1}};
/// std::println("{:.1f}", prediction.mean[0]); // 1.0
/// @endcode
struct ZeroAugmentedCountPrediction {
    std::vector<double> count_mean;                 ///< Mean parameter of the underlying untruncated count distribution.
    std::vector<double> zero_component_probability; ///< Structural-zero probability for mixtures, or zero probability for hurdles.
    std::vector<double> zero_probability;           ///< Marginal probability of an observed zero.
    std::vector<double> mean;                       ///< Marginal expected count.
};

/// @brief Fitted count and zero coefficients, dispersion, and joint covariance.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> intercept(4, 1, 1);
/// const auto fit = dm::stats::hurdle_count_regression(intercept, intercept, {0, 1, 2, 3});
/// std::println("{:.2f}", fit.fitted.zero_probability[0]); // 0.25
/// @endcode
struct ZeroAugmentedCountResult {
    ZeroAugmentedCountModel model;             ///< Fitted zero mechanism.
    std::vector<double> count_coefficients;    ///< Count log-mean coefficients.
    std::vector<double> zero_coefficients;     ///< Logistic zero coefficients, empty at a zero-probability boundary.
    double dispersion;                         ///< NB2 alpha. Zero for the Poisson distribution.
    linalg::DenseMatrix<double> covariance;    ///< Count coefficients, zero coefficients, then alpha if interior. Empty if inference failed.
    ZeroAugmentedCountPrediction fitted;       ///< Predictions at the training designs and offsets.
    double log_likelihood;                      ///< Weighted observed-data log likelihood.
    ZeroAugmentedCountStatus status;           ///< Fit and boundary diagnostic.

    /// @brief Predicts marginal means and zero probabilities at new covariates.
    /// @param count_design Count predictors with the fitted columns.
    /// @param zero_design Zero predictors aligned to the count rows.
    /// @param offset Additive log-scale count offsets.
    /// @return Row-aligned predicted means and probabilities.
    /// @throws std::invalid_argument Incompatible shapes or invalid model values or designs.
    /// @throws std::overflow_error A prediction exceeds the finite reporting range.
    /// @par Example
    /// @code
    /// const dm::linalg::DenseMatrix<double> intercept(4, 1, 1);
    /// const auto fit = dm::stats::hurdle_count_regression(intercept, intercept, {0, 1, 2, 3});
    /// std::println("{:.1f}", fit.predict(intercept, intercept).mean[0]); // 1.5
    /// @endcode
    [[nodiscard]] ZeroAugmentedCountPrediction predict(const linalg::DenseMatrix<double>& count_design,
        const linalg::DenseMatrix<double>& zero_design, const std::vector<double>& offset = {}) const;
};

/// @brief Fits a structural-zero mixture with Poisson or NB2 counts.
/// @param count_design Full-rank finite count predictors, including any desired intercept.
/// @param zero_design Full-rank finite structural-zero predictors with the same rows.
/// @param counts Observed counts, with at least one positive value.
/// @param options Count family, weights, offsets, and optimizer controls.
/// @return Best fit found from deterministic starts and applicable boundary models.
/// @throws std::invalid_argument Incompatible input, aliased design, or invalid options.
/// @throws std::length_error Parameter matrix exceeds representable capacity.
/// @throws std::overflow_error A fitted quantity exceeds finite range.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> intercept(8, 1, 1);
/// const auto fit = dm::stats::zero_inflated_count_regression(intercept, intercept, {0,0,0,0,3,4,5,4});
/// std::println("{:.1f}", fit.fitted.mean[0]); // 2.0
/// @endcode
/// @par References
/// - Lambert (1992), https://doi.org/10.1080/00401706.1992.10485228
/// - Zeileis, Kleiber and Jackman (2008), https://doi.org/10.18637/jss.v027.i08
[[nodiscard]] ZeroAugmentedCountResult zero_inflated_count_regression(
    const linalg::DenseMatrix<double>& count_design, const linalg::DenseMatrix<double>& zero_design,
    const std::vector<std::size_t>& counts, const ZeroAugmentedCountOptions& options = {});

/// @brief Fits a binary zero model and a zero-truncated Poisson or NB2 likelihood.
/// @param count_design Finite count predictors with full rank within positive observations.
/// @param zero_design Full-rank finite zero predictors aligned to the count rows.
/// @param counts Observed counts, with at least one positive value.
/// @param options Count family, weights, offsets, and optimizer controls.
/// @return Fitted hurdle component and truncated count model with joint inference.
/// @throws std::invalid_argument Incompatible input, aliased design, or invalid options.
/// @throws std::length_error Parameter matrix exceeds representable capacity.
/// @throws std::overflow_error A fitted quantity exceeds finite range.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> intercept(4, 1, 1);
/// const auto fit = dm::stats::hurdle_count_regression(intercept, intercept, {0, 0, 2, 5});
/// std::println("{:.1f}", fit.fitted.zero_probability[0]); // 0.5
/// @endcode
/// @par References
/// - Mullahy (1986), https://doi.org/10.1016/0304-4076(86)90002-3
/// - Zeileis, Kleiber and Jackman (2008), https://doi.org/10.18637/jss.v027.i08
[[nodiscard]] ZeroAugmentedCountResult hurdle_count_regression(
    const linalg::DenseMatrix<double>& count_design, const linalg::DenseMatrix<double>& zero_design,
    const std::vector<std::size_t>& counts, const ZeroAugmentedCountOptions& options = {});

} // namespace datamunge::stats
