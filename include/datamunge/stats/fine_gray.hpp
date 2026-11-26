#pragma once

/// @file fine_gray.hpp
/// @brief Fine–Gray proportional subdistribution hazards regression for competing risks.

#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/stats/multistate.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief Options for fit_fine_gray().
/// @par Example
/// @code
/// const dm::stats::FineGrayOptions o{.cause = 2};
/// std::println("{}", o.confidence); // 0.95
/// @endcode
struct FineGrayOptions {
    int cause{1};                          ///< Cause of interest.
    std::vector<int> censoring_groups{};   ///< Optional group per observation with its own censoring distribution.
    double confidence{0.95};               ///< Coverage of hazard-ratio intervals.
    std::size_t max_iterations{50};        ///< Newton iteration limit.
    double tolerance{1e-10};               ///< Relative log-likelihood change that stops iteration.
};

/// @brief Fitted Fine–Gray model, laid out like `cmprsk::crr`.
/// @par Example
/// @code
/// const dm::stats::FineGrayFit f;
/// std::println("{}", f.converged); // false
/// @endcode
struct FineGrayFit {
    std::vector<double> coefficients;          ///< Log subdistribution hazard ratios.
    std::vector<double> standard_errors;       ///< Robust (sandwich) standard errors, as `crr`'s `var`.
    linalg::DenseMatrix<double> covariance;    ///< Robust covariance accounting for the estimated censoring weights.
    linalg::DenseMatrix<double> information;   ///< Observed information of the weighted partial likelihood.
    std::vector<double> hazard_ratios;         ///< \f$e^{\beta_j}\f$.
    std::vector<double> hazard_ratio_lower;    ///< Lower Wald interval ends.
    std::vector<double> hazard_ratio_upper;    ///< Upper Wald interval ends.
    std::vector<double> p_values;              ///< Two-sided Wald p-values.
    double log_likelihood{};                   ///< Weighted log partial likelihood (Breslow ties).
    double null_log_likelihood{};              ///< Its value at zero.
    std::vector<double> event_times;           ///< Distinct failure times from the cause, increasing.
    std::vector<double> baseline_cumulative_hazard; ///< Cumulative subdistribution hazard at covariates zero.
    std::size_t iterations{};                  ///< Newton iterations.
    bool converged{false};                     ///< True if the tolerance was met.

    /// @brief Predicted cumulative incidence \f$1-\exp\{-\Lambda_0(t)e^{x^\top\beta}\}\f$, as `predict.crr`.
    /// @param time Time.
    /// @param covariates One value per coefficient.
    /// @return Cumulative incidence of the cause by time.
    /// @throws std::invalid_argument For a wrong covariate length.
    /// @par Example
    /// @code
    /// const std::vector<dm::stats::CompetingRiskObservation> obs{{1, 1}, {2, 2}, {3, 1}, {4, 0}, {5, 1}};
    /// const auto f = dm::stats::fit_fine_gray(obs, dm::linalg::DenseMatrix<double>(5, 1, {1, 0, 0, 1, 1}));
    /// std::println("{}", f.predict_incidence(0.5, {1})); // 0
    /// @endcode
    [[nodiscard]] double predict_incidence(double time, const std::vector<double>& covariates) const;
};

/// @brief Fits the Fine–Gray proportional subdistribution hazards model, as `cmprsk::crr`.
///
/// The subdistribution hazard of the cause is modelled as \f$\lambda_0^*(t)e^{x^\top\beta}\f$, which
/// makes each coefficient act directly on the cumulative incidence. Subjects failing from other
/// causes stay in the risk set after their failure with inverse-probability-of-censoring weights
/// \f$\hat G(t-)/\hat G(T_i-)\f$, where \f$\hat G\f$ is the Kaplan–Meier estimate of the censoring
/// distribution (optionally per censoring group). Ties use Breslow's approximation. The robust
/// covariance is the sandwich \f$I^{-1}(\sum_i s_is_i^\top)I^{-1}\f$, where each \f$s_i\f$ adds Fine and
/// Gray's correction for estimating \f$\hat G\f$ to the score residual. It is computed in
/// \f$O(n^2p^2)\f$ time with prefix and suffix sums.
/// @param observations Times and causes (0 censored).
/// @param covariates One row per observation, no intercept.
/// @param options Cause, censoring groups, interval coverage and iteration control.
/// @return The fit with robust and model-based uncertainty and the baseline for prediction.
/// @throws std::invalid_argument For mismatched sizes, invalid data or no failures from the cause.
/// @throws std::runtime_error If the information matrix is singular.
/// @par Example
/// @code
/// const std::vector<dm::stats::CompetingRiskObservation> obs{{1, 1}, {2, 2}, {3, 1}, {4, 0}, {5, 1},
///                                                           {6, 2}, {7, 1}, {8, 0}};
/// const auto f = dm::stats::fit_fine_gray(obs, dm::linalg::DenseMatrix<double>(8, 1, {1, 0, 0, 1, 1, 0, 0, 1}));
/// std::println("{:.3f}", f.coefficients[0]); // 0.410
/// @endcode
/// @par References
/// - J. P. Fine and R. J. Gray (1999). "A proportional hazards model for the subdistribution of a
///   competing risk." JASA 94(446), 496-509. doi:10.1080/01621459.1999.10474144
/// @see gray_test(), cumulative_incidence()
[[nodiscard]] FineGrayFit fit_fine_gray(const std::vector<CompetingRiskObservation>& observations,
                                        const linalg::DenseMatrix<double>& covariates,
                                        const FineGrayOptions& options = {});

} // namespace datamunge::stats
