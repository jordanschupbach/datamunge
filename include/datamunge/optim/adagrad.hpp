#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>

namespace datamunge::optim {

struct AdaGradOptions {
    double step_size{0.01};
    /// @brief Added inside the square root to keep early and zero-gradient updates finite.
    double epsilon{1e-8};
    std::size_t max_iterations{10000};
    double tolerance{1e-8};
};

/// @brief Full-batch AdaGrad (Duchi, Hazan, Singer 2011) on a DifferentiableFunction.  AdaGrad
///        scales each coordinate's step by the inverse root of its accumulated squared gradient,
///        making it useful when gradient magnitudes differ substantially by coordinate.
class AdaGrad {
public:
    explicit AdaGrad(AdaGradOptions options = {});

    double optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const;

private:
    AdaGradOptions options_;
};

} // namespace datamunge::optim
