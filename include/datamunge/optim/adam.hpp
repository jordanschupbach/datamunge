#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>

namespace datamunge::optim {

struct AdamOptions {
    double step_size{0.001};
    double beta1{0.9};
    double beta2{0.999};
    double epsilon{1e-8};
    std::size_t max_iterations{10000};
    double tolerance{1e-8};
};

/// @brief Full-batch Adam (Kingma & Ba 2015) on a DifferentiableFunction.
class Adam {
public:
    explicit Adam(AdamOptions options = {});

    double optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const;

private:
    AdamOptions options_;
};

} // namespace datamunge::optim
