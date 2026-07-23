#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>
#include <string>

namespace datamunge::optim {

struct EvolutionStrategyOptions {
    /// @brief mu: number of parents.
    std::size_t mu{15};
    /// @brief lambda: number of offspring generated per generation.
    std::size_t offspring_size{100};
    /// @brief One of "comma" (next generation's parents are the best mu of the offspring only --
    ///        the literature-recommended default, since it avoids getting stuck on a
    ///        lucky-but-misleading step size) or "plus" (next generation's parents are the best
    ///        mu of parents union offspring).
    std::string strategy{"comma"};
    /// @brief Initial global self-adapting mutation step size (sigma).
    double initial_step_size{0.5};
    std::size_t max_generations{300};
    /// @brief Stops after the best-ever value has improved by less than this for 20 consecutive generations.
    double tolerance{1e-10};
    std::uint64_t seed{42};
};

/// @brief Classic self-adaptive (mu, lambda) / (mu+lambda) Evolution Strategy with a single
///        global step size mutated log-normally each generation (Rechenberg 1973, Schwefel
///        1981). This is the pre-CMA-ES form -- no covariance matrix adaptation -- and is the
///        foundational lineage CMAES (also in this module) descends from. Derivative-free,
///        box-constrained.
class EvolutionStrategy {
public:
    explicit EvolutionStrategy(EvolutionStrategyOptions options = {});

    /// @brief Minimizes @p function within [@p lower_bound, @p upper_bound], updating
    ///        @p coordinates in place to the best point found, and returns its value. The
    ///        initial @p coordinates (clamped into bounds) seed one parent of the population.
    double optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                     const std::vector<double>& lower_bound, const std::vector<double>& upper_bound) const;

private:
    EvolutionStrategyOptions options_;
};

} // namespace datamunge::optim
