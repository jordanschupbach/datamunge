#pragma once

/// @file firth_logistic.hpp
/// @brief Bias-reduced binomial logistic regression and penalized profile intervals.

#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/stats/profile_likelihood.hpp>

#include <vector>

namespace datamunge::stats {

/// @brief Count, weight, and inference settings for Firth logistic regression.
/// @par Example
/// @code
/// const dm::stats::FirthLogisticOptions options{.trials = {7, 9}};
/// std::println("{}", options.trials.size()); // 2
/// @endcode
/// @see firth_logistic()
struct FirthLogisticOptions {
    std::vector<double> trials{};  ///< Positive integer trials per row. Empty uses one trial per row.
    std::vector<double> weights{}; ///< Positive finite case weights. Empty uses unit weight.
    std::size_t max_iterations{300}; ///< Iteration limit for the main fit and for each profile fit.
    double tolerance{1e-8};        ///< Tolerance for the adjusted score scaled by marginal information.
    bool profile_intervals{true}; ///< Compute penalized likelihood-ratio intervals for all coefficients.
    double confidence_level{.95}; ///< Confidence level of the pointwise intervals.
};

/// @brief Fitted Firth coefficients and pointwise penalized profile intervals.
/// @par Example
/// @code
/// const auto fit = dm::stats::firth_logistic(dm::linalg::DenseMatrix<double>(4, 1, 1), {0, 0, 0, 0});
/// std::println("{:.1f}", fit.fitted[0]); // 0.1
/// @endcode
/// @see firth_logistic()
struct FirthLogisticResult {
    std::vector<double> coefficients;       ///< In supplied design-column order.
    linalg::DenseMatrix<double> covariance; ///< Inverse ordinary Fisher information at the penalized fit.
    std::vector<double> fitted;             ///< Success probabilities in input row order.
    double penalized_log_likelihood;       ///< Bernoulli log-likelihood kernel plus half log determinant of information.
    std::vector<ProfileLikelihoodInterval> intervals; ///< Penalized profile intervals in coefficient order when requested.

    /// @brief Computes fitted probabilities for new design rows.
    /// @param design Finite rows with the same columns as the training matrix, including any intercept.
    /// @return One probability per row.
    /// @throws std::invalid_argument Wrong column count or nonfinite matrix or coefficient value.
    /// @throws std::overflow_error Nonfinite working linear predictor.
    /// @par Example
    /// @code
    /// const auto fit = dm::stats::firth_logistic(dm::linalg::DenseMatrix<double>(4, 1, 1), {0, 0, 1, 1});
    /// std::println("{:.1f}", fit.predict(dm::linalg::DenseMatrix<double>(1, 1, 1))[0]); // 0.5
    /// @endcode
    [[nodiscard]] std::vector<double> predict(const linalg::DenseMatrix<double>& design) const;
};

/// @brief Fits a binomial logistic model with the Firth bias-reduction penalty.
///
/// Uses the adjusted score and a Fisher-initialized line search to maximize
/// the log likelihood plus half the log determinant of the ordinary Fisher
/// information. The likelihood kernel omits binomial combinatorial constants.
/// The design must have full column rank and may be saturated or separated.
/// Profile fits hold one coefficient fixed and retain the penalty of the full
/// model while refitting all nuisance coefficients. If profiling finds a higher
/// likelihood, the main fit resumes there and the intervals are recomputed.
/// The numerical search is local and targets interval crossings around the
/// best solution it finds.
/// @param design Finite design, with an intercept column supplied by the caller if wanted.
/// @param successes Integer success counts from zero to the number of trials per row.
/// @param options Trial counts, case weights, convergence, and confidence settings.
/// @return Coefficients, fitted probabilities, information-based covariance, and optional profile intervals.
/// @throws std::invalid_argument Invalid design, rank, counts, or numerical options.
/// @throws std::runtime_error A fit loses numerical rank or fails to establish a stable maximum.
/// @throws std::overflow_error A reported quantity exceeds finite double range.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> design(4, 2, {1, 0, 1, 0, 1, 1, 1, 1});
/// const auto fit = dm::stats::firth_logistic(design, {0, 0, 1, 1});
/// std::println("{:.3f}", fit.coefficients[1]); // 3.219
/// @endcode
/// @par References
/// - Firth (1993), https://doi.org/10.1093/biomet/80.1.27
/// - Heinze and Schemper (2002), https://doi.org/10.1002/sim.1047
/// @see profile_likelihood_interval()
[[nodiscard]] FirthLogisticResult firth_logistic(
    const linalg::DenseMatrix<double>& design, const std::vector<double>& successes,
    const FirthLogisticOptions& options = {});

} // namespace datamunge::stats
