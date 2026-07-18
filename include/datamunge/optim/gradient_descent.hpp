#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>

namespace datamunge::optim {

struct GradientDescentOptions {
    double step_size{0.01};
    /// @brief Classical momentum coefficient in [0, 1); 0 disables momentum (plain gradient descent).
    double momentum{0.0};
    std::size_t max_iterations{10000};
    /// @brief Stops when the gradient norm or the objective's per-iteration change drops below this.
    double tolerance{1e-8};
};

/// @brief Gradient descent (optionally with classical momentum) on a DifferentiableFunction.
class GradientDescent {
public:
    explicit GradientDescent(GradientDescentOptions options = {});

    /// @brief Minimizes @p function starting from @p coordinates, updating it in place to the
    ///        best point found, and returns the objective value there.
    double optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const;

private:
    GradientDescentOptions options_;
};

} // namespace datamunge::optim
