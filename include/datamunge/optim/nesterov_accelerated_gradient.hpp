#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>

namespace datamunge::optim {

struct NesterovAcceleratedGradientOptions {
    double step_size{0.01};
    /// @brief Momentum coefficient in [0, 1), typically 0.9.
    double momentum{0.9};
    std::size_t max_iterations{10000};
    double tolerance{1e-8};
};

/// @brief Nesterov accelerated gradient (look-ahead momentum) on a DifferentiableFunction.
class NesterovAcceleratedGradient {
public:
    explicit NesterovAcceleratedGradient(NesterovAcceleratedGradientOptions options = {});

    double optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const;

private:
    NesterovAcceleratedGradientOptions options_;
};

} // namespace datamunge::optim
