#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>

namespace datamunge::optim {

struct AMSGradOptions {
    double step_size{0.001};
    double beta1{0.9};
    double beta2{0.999};
    double epsilon{1e-8};
    std::size_t max_iterations{10000};
    double tolerance{1e-8};
};

/// @brief Full-batch AMSGrad (Reddi et al. 2018) on a DifferentiableFunction.  AMSGrad is Adam
///        with a non-decreasing per-coordinate second-moment estimate, giving it a more stable
///        effective learning-rate schedule.
class AMSGrad {
public:
    explicit AMSGrad(AMSGradOptions options = {});

    double optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const;

private:
    AMSGradOptions options_;
};

} // namespace datamunge::optim
