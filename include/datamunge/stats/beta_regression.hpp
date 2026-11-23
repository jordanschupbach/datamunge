#pragma once

/// @file beta_regression.hpp
/// @brief Beta regression for continuous proportions with variable precision.

#include <datamunge/linalg/dense_matrix.hpp>

#include <vector>

namespace datamunge::stats {

/// @brief Stopping condition for beta maximum likelihood.
/// @par Example
/// @code
/// const auto status = dm::stats::BetaRegressionStatus::Converged;
/// std::println("{}", status == dm::stats::BetaRegressionStatus::Converged); // true
/// @endcode
enum class BetaRegressionStatus {
    Converged,       ///< Fisher scoring reached the specified tolerance.
    IterationLimit,  ///< The permitted number of scoring updates was exhausted.
    NumericalFailure ///< The likelihood or information could not support a reliable update.
};

/// @brief Case weights and convergence settings for beta regression.
/// @par Example
/// @code
/// const dm::stats::BetaRegressionOptions options{.weights = {1, 2, 1}};
/// std::println("{}", options.weights.size()); // 3
/// @endcode
struct BetaRegressionOptions {
    std::vector<double> weights{};     ///< Positive finite case weights. Empty uses one per row.
    std::size_t max_iterations{200};  ///< Maximum accepted Fisher-scoring updates.
    double tolerance{1e-9};           ///< Stopping tolerance for the largest change in either linear predictor.
};

/// @brief Row-aligned beta means and precisions.
/// @par Example
/// @code
/// const dm::stats::BetaRegressionPrediction pred{{.5}, {10}};
/// std::println("{:.1f}", pred.precision[0]); // 10.0
/// @endcode
struct BetaRegressionPrediction {
    std::vector<double> mean;      ///< Mean proportion, inverse-logit of the mean predictor.
    std::vector<double> precision; ///< Positive shape sum, exponential of the precision predictor.
};

/// @brief Two coefficient blocks and their inverse Fisher information.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> intercept(4, 1, 1.);
/// const auto fit = dm::stats::beta_regression(intercept, intercept, {.2,.4,.6,.8});
/// std::println("{:.1f}", fit.fitted.mean[0]); // 0.5
/// @endcode
struct BetaRegressionResult {
    std::vector<double> mean_coefficients;      ///< Coefficients of the logit mean predictor.
    std::vector<double> precision_coefficients; ///< Coefficients of the log precision predictor.
    linalg::DenseMatrix<double> covariance;     ///< Joint inverse expected information, mean block first. Empty on failure.
    BetaRegressionPrediction fitted;            ///< Predictions at the supplied designs.
    double log_likelihood;                      ///< Weighted log likelihood including the beta normalizer.
    std::size_t iterations;                     ///< Number of accepted scoring updates.
    BetaRegressionStatus status;                ///< Optimization outcome.

    /// @brief Predicts beta means and precisions at new covariates.
    /// @param mean_design Finite predictors with the fitted mean columns.
    /// @param precision_design Finite predictors with the fitted precision columns and the same rows.
    /// @return Mean and precision vectors in row order.
    /// @throws std::invalid_argument Incompatible dimensions or nonfinite design or fitted coefficients.
    /// @throws std::overflow_error A prediction is outside the representable beta parameter domain.
    /// @par Example
    /// @code
    /// const dm::linalg::DenseMatrix<double> intercept(4, 1, 1.);
    /// const auto fit = dm::stats::beta_regression(intercept, intercept, {.2,.4,.6,.8});
    /// std::println("{:.1f}", fit.predict(intercept, intercept).mean[0]); // 0.5
    /// @endcode
    [[nodiscard]] BetaRegressionPrediction predict(const linalg::DenseMatrix<double>& mean_design,
        const linalg::DenseMatrix<double>& precision_design) const;
};

/// @brief Fits a beta response with a logit mean and log precision model, as in R betareg.
///
/// For mean mu and precision phi, the shapes are mu*phi and (1-mu)*phi.
/// Supply intercept columns explicitly. A column of ones in the precision design
/// estimates a common precision across observations.
/// @param mean_design Finite full-rank mean design with one row per observation.
/// @param precision_design Finite full-rank precision design with the same rows.
/// @param response Finite continuous responses strictly between zero and one.
/// @param options Optional case weights and Fisher-scoring controls.
/// @return Fitted coefficients, predictions, weighted likelihood and joint covariance on convergence.
/// @throws std::invalid_argument Incompatible input, invalid options, aliased designs or a perfect logit fit.
/// @throws std::length_error Coefficient matrix exceeds representable capacity.
/// @throws std::overflow_error Fitted parameters exceed the reporting range.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> intercept(4, 1, 1.);
/// const auto fit = dm::stats::beta_regression(intercept, intercept, {.2,.4,.6,.8});
/// std::println("{}", fit.covariance.rows()); // 2
/// @endcode
/// @par References
/// - Ferrari and Cribari-Neto (2004), https://doi.org/10.1080/0266476042000214501
/// - Cribari-Neto and Zeileis (2010), https://doi.org/10.18637/jss.v034.i02
[[nodiscard]] BetaRegressionResult beta_regression(const linalg::DenseMatrix<double>& mean_design,
    const linalg::DenseMatrix<double>& precision_design, const std::vector<double>& response,
    const BetaRegressionOptions& options = {});

} // namespace datamunge::stats
