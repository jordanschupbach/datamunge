#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>

namespace datamunge::optim {

struct NelderMeadOptions {
    /// @brief Per-coordinate displacement used to construct the initial simplex.
    double initial_simplex_scale{0.05};
    double reflection{1.0};
    double expansion{2.0};
    double contraction{0.5};
    double shrink{0.5};
    std::size_t max_iterations{2000};
    /// @brief Stops when both the simplex diameter and its objective spread are below this.
    double tolerance{1e-8};
};

/// @brief Nelder-Mead downhill simplex search for an ArbitraryFunction.  This local,
///        derivative-free method is suitable when gradients are unavailable but repeated
///        objective evaluations are affordable.
class NelderMead {
public:
    explicit NelderMead(NelderMeadOptions options = {});

    double optimize(ArbitraryFunction& function, std::vector<double>& coordinates) const;

private:
    NelderMeadOptions options_;
};

} // namespace datamunge::optim
