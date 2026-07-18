#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::bayes {

struct NUTSOptions {
    std::size_t num_samples{1000};
    std::size_t num_warmup{1000};
    /// @brief Caps the recursive trajectory-doubling at 2^max_tree_depth leapfrog steps per iteration.
    std::size_t max_tree_depth{10};
    double initial_step_size{0.1};
    double target_accept_rate{0.8};
    /// @brief A trajectory is flagged as diverging once the Hamiltonian drifts by more than
    ///        this from its initial value (Stan's default of 1000).
    double max_delta_error{1000.0};
    std::uint64_t seed{42};
};

struct NUTSResult {
    /// @brief Post-warmup draws, one vector<double> per sample.
    std::vector<std::vector<double>> samples;
    /// @brief Mean Metropolis acceptance statistic across all iterations (the same quantity
    ///        the step-size adaptation targets).
    double accept_rate{0.0};
    double final_step_size{0.0};
    std::size_t num_divergences{0};
};

/// @brief The No-U-Turn Sampler (Hoffman & Gelman 2014, the efficient slice-sampling
///        variant, Algorithm 3): Hamiltonian Monte Carlo with an automatically chosen
///        trajectory length -- the leapfrog trajectory is grown by repeated doubling until
///        it would start turning back on itself (a "U-turn"), removing the need to hand-tune
///        a fixed number of leapfrog steps. Step size is still adapted via the same
///        dual-averaging scheme as HMC. Uses an identity mass matrix.
class NUTS {
public:
    explicit NUTS(NUTSOptions options = {});

    NUTSResult sample(optim::DifferentiableFunction& log_posterior, const std::vector<double>& initial_params) const;

private:
    NUTSOptions options_;
};

} // namespace datamunge::bayes
