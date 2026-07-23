#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>

namespace datamunge::optim {

struct FireflyAlgorithmOptions {
    std::size_t population_size{30};
    /// @brief beta0: attractiveness at distance 0.
    double attractiveness_at_zero{1.0};
    /// @brief gamma: controls how fast attractiveness decays with distance.
    double light_absorption{1.0};
    /// @brief alpha: scale of the per-move random walk term.
    double randomization_step{0.2};
    /// @brief alpha is multiplied by this every iteration (geometric decay -- exploration
    ///        shrinks over time).
    double randomization_decay{0.97};
    std::size_t max_iterations{300};
    /// @brief Stops after the best-ever value has improved by less than this for 20
    ///        consecutive iterations.
    double tolerance{1e-10};
    std::uint64_t seed{42};
};

/// @brief The Firefly Algorithm (Yang, 2008/2009): a population of fireflies is pairwise
///        attracted toward brighter (better) neighbors, with attractiveness decaying with
///        squared distance and a shrinking random-walk term for exploration. Derivative-free,
///        O(n^2) per iteration in the population size.
class FireflyAlgorithm {
public:
    explicit FireflyAlgorithm(FireflyAlgorithmOptions options = {});

    /// @brief Minimizes @p function within [@p lower_bound, @p upper_bound], updating
    ///        @p coordinates in place to the best point found, and returns its value. The
    ///        initial @p coordinates (clamped into bounds) seed one member of the population.
    double optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                     const std::vector<double>& lower_bound, const std::vector<double>& upper_bound) const;

private:
    FireflyAlgorithmOptions options_;
};

} // namespace datamunge::optim
