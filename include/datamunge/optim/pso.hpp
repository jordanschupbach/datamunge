#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>
#include <string>

namespace datamunge::optim {

struct PSOOptions {
    std::size_t population_size{40};
    std::size_t max_iterations{1000};
    double inertia_weight{0.7298};
    double cognitive_coefficient{1.49618};
    double social_coefficient{1.49618};
    /// @brief One of "global" (gbest -- every particle is attracted to the single best
    ///        particle found so far) or "ring" (lbest -- each particle is attracted to the
    ///        best particle within a small ring-shaped neighborhood; slower but less prone
    ///        to premature convergence).
    std::string topology{"global"};
    /// @brief Neighbors considered on each side of a particle when topology == "ring".
    std::size_t ring_neighbors{2};
    /// @brief One of "constant" or "linear_decay" (anneals from inertia_weight down to
    ///        final_inertia_weight over the course of the run).
    std::string inertia_strategy{"constant"};
    double final_inertia_weight{0.4};
    /// @brief Stops after the global best has improved by less than this for 20 consecutive iterations.
    double tolerance{1e-10};
    std::uint64_t seed{42};
};

/// @brief Particle swarm optimization on an ArbitraryFunction within a box constraint: a
///        population of particles fly through the search space, each pulled toward its own
///        best-seen position and the best position found by (depending on topology) the
///        whole swarm or a local neighborhood. Derivative-free.
class PSO {
public:
    explicit PSO(PSOOptions options = {});

    /// @brief Minimizes @p function within [@p lower_bound, @p upper_bound], updating
    ///        @p coordinates in place to the best point found, and returns its value. The
    ///        initial @p coordinates (clamped into bounds) seed one particle of the swarm.
    double optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                     const std::vector<double>& lower_bound, const std::vector<double>& upper_bound) const;

private:
    PSOOptions options_;
};

} // namespace datamunge::optim
