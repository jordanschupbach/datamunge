#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::bayes {

struct GibbsOptions {
    std::size_t num_samples{1000};
    std::size_t num_warmup{1000};
    /// @brief Initial per-coordinate proposal std dev. A single entry is broadcast to every
    ///        coordinate; otherwise its length must match the parameter vector, and each
    ///        coordinate is adapted independently via dual averaging during warmup.
    std::vector<double> initial_step_sizes{1.0};
    /// @brief Target per-coordinate acceptance rate -- 0.44 is the asymptotically optimal
    ///        rate for a SCALAR random-walk proposal (Roberts & Rosenthal 2001), unlike
    ///        RandomWalkMetropolis's 0.234 (the multivariate-proposal optimum).
    double target_accept_rate{0.44};
    std::uint64_t seed{42};
};

struct GibbsResult {
    /// @brief Post-warmup draws, one vector<double> per sample.
    std::vector<std::vector<double>> samples;
    /// @brief Mean acceptance rate per coordinate (across all iterations), in parameter order.
    std::vector<double> accept_rates;
    std::vector<double> final_step_sizes;
};

/// @brief Metropolis-within-Gibbs: updates one coordinate at a time, each via a scalar
///        random-walk Metropolis step evaluated against the SAME joint log-density with every
///        other coordinate held fixed -- mathematically equivalent to a Metropolis step
///        targeting the exact full conditional p(x_i | x_-i), since the joint-density ratio
///        with everything else held fixed IS the full-conditional-density ratio. This is the
///        standard fallback used whenever a full conditional isn't conjugate/directly
///        sampleable (what BUGS/JAGS call "Metropolis-within-Gibbs"), not exact Gibbs sampling
///        via closed-form conditional samplers -- those are necessarily model-specific (they
///        need to know each conditional's actual distributional family) and aren't something
///        a generic engine driven only by a joint log-density can provide. Updating one
///        coordinate at a time (rather than proposing the whole vector jointly, as
///        RandomWalkMetropolis does) can mix substantially better when parameters live on very
///        different scales or are only weakly correlated.
class GibbsSampler {
public:
    explicit GibbsSampler(GibbsOptions options = {});

    GibbsResult sample(optim::ArbitraryFunction& log_posterior, const std::vector<double>& initial_params) const;

private:
    GibbsOptions options_;
};

} // namespace datamunge::bayes
