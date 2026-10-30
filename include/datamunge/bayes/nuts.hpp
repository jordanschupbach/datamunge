#pragma once

/// @file nuts.hpp
/// @brief The No-U-Turn Sampler, with per-draw diagnostics and multiple chains.

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::bayes {

/// Settings for NUTS.
struct NUTSOptions {
    /// @brief Number of post-warmup draws.
    std::size_t num_samples{1000};
    /// @brief Number of warmup iterations (step-size adaptation).
    std::size_t num_warmup{1000};
    /// @brief Caps the recursive trajectory-doubling at 2^max_tree_depth leapfrog steps per iteration.
    std::size_t max_tree_depth{10};
    /// @brief Leapfrog step size at the start of warmup.
    double initial_step_size{0.1};
    /// @brief Mean acceptance statistic the dual-averaging adaptation aims for.
    double target_accept_rate{0.8};
    /// @brief A trajectory is flagged as diverging once the Hamiltonian drifts by more than
    ///        this from its initial value (Stan's default of 1000).
    double max_delta_error{1000.0};
    /// @brief Random seed (chain k of sample_chains() uses seed + k).
    std::uint64_t seed{42};
};

/// Output of NUTS::sample().
struct NUTSResult {
    /// @brief Post-warmup draws, one vector<double> per sample.
    std::vector<std::vector<double>> samples;
    /// @brief Mean Metropolis acceptance statistic across all iterations (the same quantity
    ///        the step-size adaptation targets).
    double accept_rate{0.0};
    /// @brief Step size used after warmup.
    double final_step_size{0.0};
    /// @brief Number of iterations, warmup included, whose trajectory diverged.
    std::size_t num_divergences{0};
    /// @brief Per post-warmup draw: 1 if that iteration's trajectory diverged, else 0.
    std::vector<int> divergent;
    /// @brief Per post-warmup draw: number of trajectory doublings performed (tree depth).
    std::vector<int> tree_depth;
    /// @brief Per post-warmup draw: number of leapfrog steps (gradient evaluations) used.
    std::vector<int> n_leapfrog;
    /// @brief Per post-warmup draw: the Hamiltonian \f$H=-\log p(q)+\tfrac12|r|^2\f$ at the
    ///        start of the iteration, after momentum resampling (input to energy_bfmi()).
    std::vector<double> energy;
};

/// @brief The No-U-Turn Sampler (Hoffman & Gelman 2014, the efficient slice-sampling
///        variant, Algorithm 3): Hamiltonian Monte Carlo with an automatically chosen
///        trajectory length -- the leapfrog trajectory is grown by repeated doubling until
///        it would start turning back on itself (a "U-turn"), removing the need to hand-tune
///        a fixed number of leapfrog steps. Step size is still adapted via the same
///        dual-averaging scheme as HMC. Uses an identity mass matrix.
class NUTS {
public:
    /// @brief Stores the sampler settings.
    /// @param options Sampler settings.
    explicit NUTS(NUTSOptions options = {});

    /// @brief Runs one chain: warmup with step-size adaptation, then num_samples draws.
    /// @param log_posterior Log density (up to a constant) and its gradient.
    /// @param initial_params Starting point on the unconstrained scale.
    /// @return Draws plus acceptance, step size and per-draw diagnostics.
    NUTSResult sample(optim::DifferentiableFunction& log_posterior, const std::vector<double>& initial_params) const;

    /// @brief Runs @p num_chains independent chains from the same start, chain k using seed
    ///        `options().seed + k`, ready for split_rhat() and effective_sample_size().
    /// @param log_posterior Log density (up to a constant) and its gradient.
    /// @param initial_params Starting point shared by every chain.
    /// @param num_chains Number of chains (at least 1).
    /// @return One NUTSResult per chain.
    /// @throws std::invalid_argument if @p num_chains is 0.
    std::vector<NUTSResult> sample_chains(optim::DifferentiableFunction& log_posterior,
                                          const std::vector<double>& initial_params, std::size_t num_chains) const;

    /// @brief The options this NUTS was configured with.
    /// @return The stored options.
    [[nodiscard]] const NUTSOptions& options() const noexcept { return options_; }

private:
    NUTSOptions options_; ///< Sampler settings.
};

} // namespace datamunge::bayes
