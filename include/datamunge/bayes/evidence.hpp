#pragma once

/// @file evidence.hpp
/// @brief Marginal likelihood estimators from posterior and proposal evaluations or power posteriors.

#include <cstddef>
#include <vector>

namespace datamunge::bayes {

/// @brief Both density evaluations at one posterior or proposal draw.
/// @par Example
/// @code
/// const dm::bayes::BridgeLogDensities draw{.log_joint = -3, .log_proposal = -1};
/// std::println("{}", draw.log_joint - draw.log_proposal); // -2
/// @endcode
/// @see bridge_sampling()
struct BridgeLogDensities {
    double log_joint;    ///< Log of the unnormalized target, including likelihood and prior constants.
    double log_proposal; ///< Log of the normalized proposal density at the same draw.
};

/// @brief Fixed-point bridge estimate and diagnostics of observed overlap.
/// @par Example
/// @code
/// const auto result = dm::bayes::bridge_sampling({{-3, -1}}, {{-4, -2}});
/// std::println("{} {}", result.log_evidence, result.converged); // -2 true
/// @endcode
/// @see bridge_sampling()
struct BridgeSamplingResult {
    double log_evidence;          ///< Estimated log normalizing constant of the target.
    double overlap;               ///< Mean of fitted bridge overlap estimates from the two samples.
    double posterior_weight_ess;  ///< Kish effective count for posterior bridge contributions.
    double proposal_weight_ess;   ///< Kish effective count for proposal bridge contributions.
    double final_log_change;      ///< Absolute log-evidence change in the final iteration.
    std::size_t iterations;       ///< Fixed-point updates performed.
    bool converged;               ///< True when the final change was at most the requested tolerance.
};

/// @brief Estimates log evidence with the Meng–Wong iterative optimal bridge.
///
/// The proposal must integrate to one and use the same coordinates and base
/// measure as the joint. Counts determine the bridge mixture proportions.
/// The population overlap is the integral of the posterior times the proposal
/// divided by their mixture. Its range is [0, 1]. Its empirical estimate can
/// exceed one. The effective counts describe weight concentration alone.
/// @param posterior Density pairs evaluated at draws from the normalized target.
/// @param proposal Density pairs evaluated at draws from the proposal.
/// @param tolerance Positive finite absolute tolerance on successive log estimates.
/// @param max_iterations Positive limit on fixed-point updates.
/// @return Estimate and diagnostics, including convergence status.
/// @throws std::invalid_argument Empty samples, invalid controls, nonfinite density
/// on its own sample, NaN, or positive infinity. A cross-density may be negative infinity.
/// @throws std::domain_error Either sample has no draws in the other distribution's support.
/// @throws std::overflow_error A reported result exceeds finite double range.
/// @par Example
/// @code
/// const auto result = dm::bayes::bridge_sampling({{1000, 0}, {999, -1}}, {{998, -2}});
/// std::println("{:.1f} {:.1f}", result.log_evidence, result.overlap); // 1000.0 1.0
/// @endcode
/// @par References
/// - Meng and Wong (1996), equation 4.1, https://www3.stat.sinica.edu.tw/statistica/j6n4/j6n43/j6n43.htm
/// @see BridgeLogDensities, BridgeSamplingResult, thermodynamic_integration()
BridgeSamplingResult bridge_sampling(const std::vector<BridgeLogDensities>& posterior,
                                     const std::vector<BridgeLogDensities>& proposal,
                                     double tolerance = 1e-10, std::size_t max_iterations = 1000);

/// @brief Moments of the log likelihood under one power posterior.
/// @par Example
/// @code
/// const dm::bayes::PowerPosteriorRung prior{0, -7, 2};
/// std::println("{}", prior.mean_log_likelihood); // -7
/// @endcode
/// @see thermodynamic_integration()
struct PowerPosteriorRung {
    double inverse_temperature;    ///< Exponent applied to the likelihood in the power posterior.
    double mean_log_likelihood;    ///< Mean original, untempered log likelihood.
    double variance_log_likelihood; ///< Variance of the original log likelihood.
};

/// @brief Plain and variance-corrected evidence integrals.
/// @par Example
/// @code
/// const auto result = dm::bayes::thermodynamic_integration({{0, -3, 2}, {1, -1, 2}});
/// std::println("{}", result.log_evidence); // -2
/// @endcode
/// @see thermodynamic_integration()
struct ThermodynamicIntegrationResult {
    double log_evidence;          ///< Corrected trapezoidal integral.
    double trapezoidal;           ///< Ordinary trapezoidal integral.
    double discretization_error;  ///< Absolute magnitude of the quadrature correction.
};

/// @brief Integrates expected log likelihood from a normalized prior to the posterior.
/// Uses log-likelihood variances as temperature derivatives in the corrected
/// trapezoidal rule. The correction magnitude estimates the discretization error
/// in the ordinary trapezoidal integral.
/// @param ladder At least two rungs with strictly increasing temperatures from
/// exactly zero to one, finite means and finite nonnegative variances.
/// @return Ordinary and corrected integrals and correction magnitude.
/// @throws std::invalid_argument Invalid ladder.
/// @throws std::overflow_error A reported result exceeds finite double range.
/// @par Example
/// @code
/// const auto result = dm::bayes::thermodynamic_integration({{0, 0, 0}, {1, 1, 2}});
/// std::println("{:.3f}", result.log_evidence); // 0.333
/// @endcode
/// @par References
/// - Friel, Hurn and Wyse (2014), section 2.1, https://arxiv.org/abs/1209.3198
/// @see PowerPosteriorRung, ThermodynamicIntegrationResult, bridge_sampling()
ThermodynamicIntegrationResult thermodynamic_integration(const std::vector<PowerPosteriorRung>& ladder);

/// @brief Computes each rung's sample moments before thermodynamic integration.
/// @param inverse_temperatures Strictly increasing ladder from exactly zero to one.
/// @param log_likelihood_draws Original log likelihoods in ladder order, at least two
/// finite draws per rung. The draw counts may differ. Variance uses divisor n - 1.
/// @return Ordinary and variance-corrected integrals and correction magnitude.
/// @throws std::invalid_argument Invalid ladder, mismatched input sizes or invalid draws.
/// @throws std::overflow_error A sample variance or reported result exceeds finite double range.
/// @par Example
/// @code
/// const auto result = dm::bayes::thermodynamic_integration({0, 1}, {{-4, -2}, {-2, 0}});
/// std::println("{}", result.log_evidence); // -2
/// @endcode
/// @see thermodynamic_integration(const std::vector<PowerPosteriorRung>&)
ThermodynamicIntegrationResult thermodynamic_integration(
    const std::vector<double>& inverse_temperatures, const std::vector<std::vector<double>>& log_likelihood_draws);

} // namespace datamunge::bayes
