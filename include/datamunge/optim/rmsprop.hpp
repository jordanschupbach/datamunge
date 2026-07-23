#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>

namespace datamunge::optim {

struct RMSPropOptions {
    double step_size{0.001};
    /// @brief Exponential decay of the per-coordinate squared-gradient average.
    double decay_rate{0.9};
    double epsilon{1e-8};
    std::size_t max_iterations{10000};
    double tolerance{1e-8};
};

/// @brief Full-batch RMSProp on a DifferentiableFunction.  RMSProp normalizes every coordinate
///        by a moving average of its recent squared gradients, unlike AdaGrad whose accumulator
///        grows indefinitely.
class RMSProp {
public:
    explicit RMSProp(RMSPropOptions options = {});

    double optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const;

private:
    RMSPropOptions options_;
};

} // namespace datamunge::optim
