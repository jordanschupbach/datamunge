#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::bayes {

struct RWMOptions {
    std::size_t num_samples{1000};
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
    std::uint64_t seed{42};
};

struct RWMResult {
    /// @brief Post-warmup draws, one vector<double> per sample.
    std::vector<std::vector<double>> samples;
    /// @brief Mean Metropolis acceptance statistic across all iterations (the same quantity
    ///        the step-size adaptation targets).
    double accept_rate{0.0};
    double final_step_size{0.0};
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
    explicit RandomWalkMetropolis(RWMOptions options = {});

    RWMResult sample(optim::ArbitraryFunction& log_posterior, const std::vector<double>& initial_params) const;

private:
    RWMOptions options_;
};

} // namespace datamunge::bayes
