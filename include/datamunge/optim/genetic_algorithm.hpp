#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>
#include <string>

namespace datamunge::optim {

struct GAOptions {
    std::size_t population_size{100};
    std::size_t max_generations{500};
    double crossover_rate{0.8};
    /// @brief Per-gene probability of a Gaussian mutation.
    double mutation_rate{0.1};
    /// @brief Mutation Gaussian standard deviation, as a fraction of each gene's [lower, upper] range.
    double mutation_std_dev{0.1};
    /// @brief One of "tournament", "roulette" (fitness-proportionate), or "rank" (linear ranking).
    std::string selection_strategy{"tournament"};
    std::size_t tournament_size{3};
    /// @brief One of "single_point", "uniform", or "blend" (BLX-alpha).
    std::string crossover_strategy{"blend"};
    /// @brief Alpha parameter for BLX-alpha crossover, used only when crossover_strategy == "blend".
    double blend_alpha{0.5};
    bool elitism{true};
    std::size_t elite_count{2};
    /// @brief Stops after the population best has improved by less than this for 20 consecutive generations.
    double tolerance{1e-10};
    std::uint64_t seed{42};
};

/// @brief A real-valued genetic algorithm on an ArbitraryFunction within a box constraint: a
///        population evolves across generations via selection, crossover, and mutation.
///        Independently configurable selection/crossover strategies and elitism give a
///        combinatorial family of GA variants under one interface. Derivative-free.
class GeneticAlgorithm {
public:
    explicit GeneticAlgorithm(GAOptions options = {});

    /// @brief Minimizes @p function within [@p lower_bound, @p upper_bound], updating
    ///        @p coordinates in place to the best point found, and returns its value. The
    ///        initial @p coordinates (clamped into bounds) seed one member of the population.
    double optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                     const std::vector<double>& lower_bound, const std::vector<double>& upper_bound) const;

private:
    GAOptions options_;
};

} // namespace datamunge::optim
