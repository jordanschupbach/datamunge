#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::bayes {

struct HMCOptions {
    std::size_t num_samples{1000};
    std::size_t num_warmup{1000};
    /// @brief Number of leapfrog steps per proposal (trajectory length = num_leapfrog_steps * step size).
    std::size_t num_leapfrog_steps{10};
    double initial_step_size{0.1};
    /// @brief Target Metropolis acceptance rate the dual-averaging adaptation aims for during warmup.
    double target_accept_rate{0.8};
    std::uint64_t seed{42};
};

struct HMCResult {
    /// @brief Post-warmup draws, one vector<double> per sample.
    std::vector<std::vector<double>> samples;
    double accept_rate{0.0};
    double final_step_size{0.0};
};

/// @brief Hamiltonian Monte Carlo with a fixed leapfrog trajectory length and Nesterov
///        dual-averaging step-size adaptation during warmup (Hoffman & Gelman 2014), using
///        an identity mass matrix. Samples from the distribution proportional to
///        exp(log_posterior) over unconstrained real parameters.
class HMC {
public:
    explicit HMC(HMCOptions options = {});

    HMCResult sample(optim::DifferentiableFunction& log_posterior, const std::vector<double>& initial_params) const;

private:
    HMCOptions options_;
};

} // namespace datamunge::bayes
