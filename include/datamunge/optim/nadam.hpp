#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>

namespace datamunge::optim {

struct NadamOptions {
    double step_size{0.001};
    double beta1{0.9};
    double beta2{0.999};
    double epsilon{1e-8};
    std::size_t max_iterations{10000};
    double tolerance{1e-8};
};

/// @brief Full-batch Nadam (Dozat 2016) on a DifferentiableFunction: Adam's adaptive second
///        moment combined with a Nesterov-style, bias-corrected first-moment direction.
class Nadam {
public:
    explicit Nadam(NadamOptions options = {});
    double optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const;
private:
    NadamOptions options_;
};

} // namespace datamunge::optim
