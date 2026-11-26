#pragma once

/// @file smc.hpp
/// @brief Tempered sequential Monte Carlo sampler with adaptive temperatures and evidence estimation.

#include <datamunge/random/random.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace datamunge::bayes {

/// @brief Log density (up to a constant unless stated) of a parameter vector.
using SmcLogDensity = std::function<double(const std::vector<double>&)>;

/// @brief Draws one parameter vector from the prior.
using SmcPriorSampler = std::function<std::vector<double>(random::SplitMix64&)>;

/// @brief Shape of the random-walk proposal used to move the particles.
enum class SmcProposal {
    full_covariance, ///< Weighted particle covariance; best for correlated posteriors.
    diagonal,        ///< Weighted particle variances only; d instead of d(d+1)/2 adapted quantities.
};

/// @brief Options for sequential_monte_carlo().
/// @par Example
/// @code
/// const dm::bayes::SmcOptions o{.particles = 500};
/// std::println("{}", o.target_cess); // 0.5
/// @endcode
struct SmcOptions {
    std::size_t particles{1000};      ///< Number of particles N.
    double target_cess{0.5};          ///< Conditional ESS fraction kept between temperatures (in (0, 1)).
    double resample_threshold{0.5};   ///< Resample when ESS / N falls below this fraction.
    std::size_t mcmc_steps{5};        ///< Random-walk Metropolis sweeps per temperature.
    double initial_scale{1.0};        ///< Initial multiplier of the 2.38/sqrt(d)-scaled proposal.
    double target_acceptance{0.3};    ///< Acceptance rate targeted when adapting the scale.
    SmcProposal proposal{SmcProposal::full_covariance}; ///< Proposal shape estimated from the particles.
    std::vector<double> temperatures; ///< Fixed increasing schedule ending at 1; empty means adaptive.
    std::size_t max_stages{1000};     ///< Limit on adaptive temperatures.
    std::uint64_t seed{1};            ///< Seed of all randomness.
};

/// @brief Diagnostics of one tempering stage.
/// @par Example
/// @code
/// dm::bayes::SmcStage s{.temperature = 0.25};
/// std::println("{}", s.temperature); // 0.25
/// @endcode
struct SmcStage {
    double temperature{};       ///< Inverse temperature phi reached at this stage.
    double log_increment{};     ///< log of the weighted mean incremental weight.
    double ess{};               ///< ESS of the reweighted particles, before any resampling.
    bool resampled{};           ///< Whether the particles were resampled.
    double acceptance_rate{};   ///< Mean Metropolis acceptance of the move step.
    double scale{};             ///< Proposal scale multiplier used.
};

/// @brief Output of sequential_monte_carlo().
/// @par Example
/// @code
/// dm::bayes::SmcResult r;
/// r.log_weights = {-0.693, -0.693};
/// std::println("{}", r.log_weights.size()); // 2
/// @endcode
struct SmcResult {
    std::vector<std::vector<double>> particles; ///< Final particles, one row each.
    std::vector<double> log_weights;            ///< Normalised log weights of the final particles.
    std::vector<double> log_likelihoods;        ///< Log likelihood of each final particle.
    double log_evidence{};                      ///< Estimate of log integral prior x likelihood.
    double ess{};                               ///< Final ESS.
    std::vector<SmcStage> stages;               ///< One entry per temperature after 0.
};

/// @brief Tempered sequential Monte Carlo sampler (Del Moral, Doucet and Jasra 2006).
///
/// Particles drawn from the prior move through the bridge
/// \f$\pi_\phi(\theta)\propto p(\theta)L(\theta)^\phi\f$ from \f$\phi=0\f$ to 1. At each stage the
/// next \f$\phi\f$ is chosen by bisection so that the conditional ESS
/// \f$(\sum_iW_iw_i)^2/\sum_iW_iw_i^2\f$ of the incremental weights \f$w_i=L(\theta_i)^{\Delta\phi}\f$
/// equals `target_cess` (Zhou, Johansen and Aston 2016), unless a fixed schedule is given. The log
/// evidence accumulates \f$\log\sum_iW_iw_i\f$, an unbiased estimate of each ratio of normalising
/// constants. The particles are resampled systematically when the ESS falls below the threshold, and
/// then moved by `mcmc_steps` random-walk Metropolis sweeps targeting \f$\pi_\phi\f$. The proposal
/// is Gaussian with the weighted particle covariance times \f$(2.38\,s)^2/d\f$, and the scale s is
/// adapted towards the target acceptance between stages. Because the particle cloud explores all
/// modes found under the prior, SMC handles multimodal posteriors that trap single-chain MCMC.
/// Estimating the proposal from the current particles makes the evidence estimate slightly biased,
/// by roughly the number of adapted quantities over N. A diagonal proposal removes most of this bias
/// in high dimensions, at the cost of slower moves when the posterior is strongly correlated.
/// @param log_prior Normalised log prior density (needed for the evidence; -inf outside the support).
/// @param log_likelihood Log likelihood (finite where the prior is positive).
/// @param prior_sampler Draws from the prior.
/// @param options Particle count, adaptivity and seed.
/// @return Weighted particles, log evidence and per-stage diagnostics.
/// @throws std::invalid_argument For bad options, an invalid schedule, or prior draws of differing
/// or zero dimension.
/// @throws std::runtime_error When the likelihood is not finite at a prior draw, or the adaptive
/// schedule exceeds `max_stages`.
/// @par Example
/// Normal prior N(0, 1), one observation y = 1 with unit variance; the evidence is N(1; 0, 2):
/// @code
/// const auto r = dm::bayes::sequential_monte_carlo(
///     [](const std::vector<double>& t) { return -0.5 * t[0] * t[0] - 0.5 * std::log(2 * std::numbers::pi); },
///     [](const std::vector<double>& t) {
///         return -0.5 * (1 - t[0]) * (1 - t[0]) - 0.5 * std::log(2 * std::numbers::pi);
///     },
///     [](dm::random::SplitMix64& g) { return std::vector<double>{g.normal()}; }, {.particles = 4000});
/// const double exact = -0.25 - 0.5 * std::log(4 * std::numbers::pi);
/// std::println("{}", std::abs(r.log_evidence - exact) < 0.05); // true
/// @endcode
SmcResult sequential_monte_carlo(const SmcLogDensity& log_prior, const SmcLogDensity& log_likelihood,
                                 const SmcPriorSampler& prior_sampler, const SmcOptions& options = {});

} // namespace datamunge::bayes
