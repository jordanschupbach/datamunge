#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>

namespace datamunge::optim {

struct RandomizedBlockCoordinateDescentOptions {
    /// @brief Number of coordinates updated together in each block.
    std::size_t block_size{1};
    double step_size{1.0};
    std::size_t max_iterations{1000};
    double tolerance{1e-8};
    double armijo_c1{1e-4};
    double backtracking_factor{0.5};
    std::size_t max_line_search_trials{30};
    /// @brief Seed for the reproducible random coordinate permutation generated each epoch.
    std::uint64_t seed{42};
};

/// @brief Randomized cyclic block coordinate descent on a DifferentiableFunction.  Every epoch
///        randomly permutes the coordinates, partitions that permutation into blocks, and visits
///        each block exactly once.  A block uses a simultaneous partial-gradient step with an
///        Armijo backtracking line search.
class RandomizedBlockCoordinateDescent {
public:
    explicit RandomizedBlockCoordinateDescent(RandomizedBlockCoordinateDescentOptions options = {});

    double optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const;

private:
    RandomizedBlockCoordinateDescentOptions options_;
};

} // namespace datamunge::optim
