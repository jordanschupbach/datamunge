#pragma once

/// @file rwm.hpp
/// @brief Random-walk Metropolis with step-size and optional proposal-covariance adaptation.

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::bayes {

/// Settings for RandomWalkMetropolis.
struct RWMOptions {
    /// @brief Number of post-warmup draws.
    std::size_t num_samples{1000};
    /// @brief Number of warmup iterations (adaptation).
    std::size_t num_warmup{1000};
    /// @brief Initial isotropic proposal std dev, adapted via dual averaging during warmup
    ///        (the same scheme HMC/NUTS use for their step size).
    double initial_step_size{1.0};
    /// @brief Target Metropolis acceptance rate for warmup adaptation -- 0.234 is the
    ///        asymptotically optimal rate for a multivariate random-walk proposal in high
    ///        dimensions (Roberts, Gelman & Gilks 1997), a different theoretical optimum than
    ///        HMC/NUTS's 0.8 (trajectory-based proposals) or GibbsSampler's 0.44 (a scalar
    ///        per-coordinate proposal).
    double target_accept_rate{0.234};
    /// @brief Random seed.
    std::uint64_t seed{42};
    /// @brief Learn a proposal covariance during warmup (adaptive Metropolis, Haario et al.
    ///        2001). The first half of warmup is isotropic; at its end the covariance
    ///        \f$\hat\Sigma\f$ of the second quarter's draws, shrunk slightly towards
    ///        \f$10^{-3}I\f$, becomes the proposal shape \f$x'=x+\varepsilon Lz\f$ with
    ///        \f$LL^\top=\hat\Sigma\f$, and the step size is re-adapted from
    ///        \f$\varepsilon=2.38/\sqrt d\f$ for the rest of warmup.
    bool adapt_covariance{false};
};

/// Output of RandomWalkMetropolis::sample().
struct RWMResult {
    /// @brief Post-warmup draws, one vector<double> per sample.
    std::vector<std::vector<double>> samples;
    /// @brief Mean Metropolis acceptance statistic across all iterations (the same quantity
    ///        the step-size adaptation targets).
    double accept_rate{0.0};
    /// @brief Step size used after warmup.
    double final_step_size{0.0};
    /// @brief Learned proposal covariance \f$\hat\Sigma\f$ (before scaling by the step size);
    ///        empty unless RWMOptions::adapt_covariance was set and warmup had at least 8 iterations.
    std::vector<std::vector<double>> proposal_covariance;
};

/// @brief Random-walk Metropolis-Hastings: the classic gradient-free MCMC baseline, proposing
///        x' = x + step_size * N(0, I) and accepting with probability
///        min(1, exp(log_posterior(x') - log_posterior(x))). Complements HMC/NUTS (which need
///        a gradient via optim::DifferentiableFunction) for targets where only a log-density
///        evaluation is available -- a black-box model, a discontinuous/non-differentiable
///        posterior, or a director-subclassed optim::ArbitraryFunction in a language without
///        access to this library's autodiff.
class RandomWalkMetropolis {
public:
    /// @brief Stores the sampler settings.
    /// @param options Sampler settings.
    explicit RandomWalkMetropolis(RWMOptions options = {});

    /// @brief Runs warmup (adapting the step size, and optionally the proposal covariance),
    ///        then num_samples draws.
    /// @param log_posterior Log density up to a constant; only evaluate() is used.
    /// @param initial_params Starting point.
    /// @return Draws, acceptance rate, final step size and learned covariance.
    RWMResult sample(optim::ArbitraryFunction& log_posterior, const std::vector<double>& initial_params) const;

    /// @brief The options this RandomWalkMetropolis was configured with.
    /// @return The stored options.
    [[nodiscard]] const RWMOptions& options() const noexcept { return options_; }

private:
    RWMOptions options_; ///< Sampler settings.
};

} // namespace datamunge::bayes
