#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>

namespace datamunge::optim {

struct SVRGOptions {
    double step_size{0.01};
    std::size_t max_epochs{100};
    /// @brief Stochastic updates per epoch; zero uses the number of component functions.
    std::size_t inner_iterations{0};
    double tolerance{1e-8};
    std::uint64_t seed{42};
};

/// @brief Stochastic Variance Reduced Gradient (Johnson & Zhang 2013) on a
///        DifferentiableSeparableFunction.  Each epoch uses a full-gradient snapshot as a
///        control variate, reducing the noise of individual component-gradient updates.
class SVRG {
public:
    explicit SVRG(SVRGOptions options = {});
    double optimize(DifferentiableSeparableFunction& function, std::vector<double>& coordinates) const;
private:
    SVRGOptions options_;
};

} // namespace datamunge::optim
