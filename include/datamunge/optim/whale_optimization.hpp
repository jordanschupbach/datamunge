#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>

namespace datamunge::optim {

struct WhaleOptimizationOptions {
    std::size_t population_size{30};
    /// @brief b: shape constant of the logarithmic bubble-net spiral.
    double spiral_constant{1.0};
    std::size_t max_iterations{500};
    /// @brief Stops after the best-ever value has improved by less than this for 20 consecutive iterations.
    double tolerance{1e-10};
    std::uint64_t seed{42};
};

/// @brief Whale Optimization Algorithm (Mirjalili & Lewis, 2016): a population of whales
///        alternates between encircling prey, random search, and a logarithmic bubble-net
///        spiral to home in on the best point found so far. Derivative-free, box-constrained.
class WhaleOptimization {
public:
    explicit WhaleOptimization(WhaleOptimizationOptions options = {});

    /// @brief Minimizes @p function within [@p lower_bound, @p upper_bound], updating
    ///        @p coordinates in place to the best point found, and returns its value. The
    ///        initial @p coordinates (clamped into bounds) seed one whale of the population.
    double optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                     const std::vector<double>& lower_bound, const std::vector<double>& upper_bound) const;

private:
    WhaleOptimizationOptions options_;
};

} // namespace datamunge::optim
