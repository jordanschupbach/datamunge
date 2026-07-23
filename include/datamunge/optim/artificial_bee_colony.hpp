#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>

namespace datamunge::optim {

struct ArtificialBeeColonyOptions {
    /// @brief Number of food sources; also the number of employed bees and the number of
    ///        onlooker bees.
    std::size_t population_size{40};
    /// @brief A food source that fails to improve for this many consecutive trials is
    ///        abandoned and replaced with a fresh random point. Zero selects the standard
    ///        heuristic default of population_size * dimension.
    std::size_t abandonment_limit{0};
    /// @brief One cycle == one employed-bee phase + one onlooker-bee phase (+ scout phase).
    std::size_t max_generations{500};
    /// @brief Stops after the best-ever value has improved by less than this for 20
    ///        consecutive generations.
    double tolerance{1e-10};
    std::uint64_t seed{42};
};

/// @brief Artificial Bee Colony (Karaboga, 2005): a swarm-intelligence method inspired by
///        honey bee foraging. Employed bees exploit known food sources, onlooker bees
///        probabilistically favor the fitter sources, and scout bees replace sources that
///        have stagnated for too long. Derivative-free, box-constrained.
class ArtificialBeeColony {
public:
    explicit ArtificialBeeColony(ArtificialBeeColonyOptions options = {});

    /// @brief Minimizes @p function within [@p lower_bound, @p upper_bound], updating
    ///        @p coordinates in place to the best point found, and returns its value. The
    ///        initial @p coordinates (clamped into bounds) seed one food source.
    double optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                     const std::vector<double>& lower_bound, const std::vector<double>& upper_bound) const;

private:
    ArtificialBeeColonyOptions options_;
};

} // namespace datamunge::optim
