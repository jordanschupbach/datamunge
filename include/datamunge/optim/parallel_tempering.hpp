#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>

namespace datamunge::optim {

struct ParallelTemperingOptions {
    std::size_t num_replicas{10};
    /// @brief T_max, the hottest replica's fixed temperature.
    double initial_temperature{10.0};
    /// @brief T_min, the coldest replica's fixed temperature.
    double final_temperature{0.1};
    /// @brief Base standard deviation of each replica's Gaussian proposal; scaled per-replica by
    ///        sqrt(T_i / T_min) so hotter replicas take proportionally larger exploratory steps.
    double step_std_dev{1.0};
    /// @brief Attempt a replica-exchange swap sweep every this many single-replica update sweeps.
    std::size_t swap_interval{10};
    std::size_t max_sweeps{2000};
    std::uint64_t seed{42};
};

/// @brief Parallel tempering (replica exchange Monte Carlo): the population-based generalization
///        of SimulatedAnnealing. Runs num_replicas Metropolis chains simultaneously, each held at
///        its own fixed temperature on a geometric ladder, and periodically proposes swapping
///        adjacent-temperature replicas' states so hot chains help cold chains escape local
///        minima. Derivative-free, unconstrained, suitable for non-differentiable/black-box
///        objectives -- like SimulatedAnnealing, requires nothing beyond evaluate().
class ParallelTempering {
public:
    explicit ParallelTempering(ParallelTemperingOptions options = {});

    double optimize(ArbitraryFunction& function, std::vector<double>& coordinates) const;

private:
    ParallelTemperingOptions options_;
};

} // namespace datamunge::optim
