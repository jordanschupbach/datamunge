#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>

namespace datamunge::optim {

struct SimulatedAnnealingOptions {
    double initial_temperature{10.0};
    /// @brief Multiplicative cooling factor applied to the temperature after every iteration.
    double cooling_rate{0.98};
    std::size_t max_iterations{10000};
    /// @brief Standard deviation of the Gaussian perturbation proposed at each step.
    double step_std_dev{1.0};
    std::uint64_t seed{42};
};

/// @brief Simulated annealing on an ArbitraryFunction: a derivative-free stochastic method
///        that accepts worsening moves with Metropolis probability exp(-delta/temperature),
///        cooling geometrically, and tracks the best point found. The only optimizer here
///        that requires nothing beyond evaluate() -- suitable for non-differentiable or
///        black-box objectives.
class SimulatedAnnealing {
public:
    explicit SimulatedAnnealing(SimulatedAnnealingOptions options = {});

    double optimize(ArbitraryFunction& function, std::vector<double>& coordinates) const;

private:
    SimulatedAnnealingOptions options_;
};

} // namespace datamunge::optim
