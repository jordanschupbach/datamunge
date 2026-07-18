#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>
#include <string>

namespace datamunge::optim {

struct DEOptions {
    std::size_t population_size{50};
    std::size_t max_generations{500};
    /// @brief F: scales the differential perturbation applied by the mutation strategy.
    double differential_weight{0.8};
    /// @brief CR: probability that a given gene is taken from the mutant vector.
    double crossover_rate{0.9};
    /// @brief One of "rand1", "best1", "current_to_best1", or "rand2" -- Storn & Price's
    ///        DE/x/y/z naming for which base vector(s) the differential perturbation(s) are
    ///        added to.
    std::string mutation_strategy{"rand1"};
    /// @brief One of "binomial" or "exponential".
    std::string crossover_strategy{"binomial"};
    /// @brief Stops after the population best has improved by less than this for 20 consecutive generations.
    double tolerance{1e-10};
    std::uint64_t seed{42};
};

/// @brief Differential evolution on an ArbitraryFunction within a box constraint: each
///        generation, every individual is challenged by a "trial" vector built from a
///        mutated combination of other population members, greedily replacing it only if
///        the trial is at least as good. Derivative-free.
class DifferentialEvolution {
public:
    explicit DifferentialEvolution(DEOptions options = {});

    /// @brief Minimizes @p function within [@p lower_bound, @p upper_bound], updating
    ///        @p coordinates in place to the best point found, and returns its value. The
    ///        initial @p coordinates (clamped into bounds) seed one member of the population.
    double optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                     const std::vector<double>& lower_bound, const std::vector<double>& upper_bound) const;

private:
    DEOptions options_;
};

} // namespace datamunge::optim
