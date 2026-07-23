#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>

namespace datamunge::optim {

struct SAGAOptions {
    double step_size{0.01};
    std::size_t max_iterations{10000};
    double tolerance{1e-8};
    std::uint64_t seed{42};
};

/// @brief SAGA (Defazio, Bach, Lacoste-Julien 2014) on a DifferentiableSeparableFunction.
///        SAGA maintains one stored gradient per component and updates from a variance-reduced
///        estimator formed from the sampled gradient, its stored value, and their running mean.
class SAGA {
public:
    explicit SAGA(SAGAOptions options = {});
    double optimize(DifferentiableSeparableFunction& function, std::vector<double>& coordinates) const;
private:
    SAGAOptions options_;
};

} // namespace datamunge::optim
