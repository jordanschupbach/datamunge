#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>

namespace datamunge::optim {

struct CoordinateDescentOptions {
    /// @brief Initial step considered for each one-dimensional update.
    double step_size{1.0};
    std::size_t max_iterations{1000};
    double tolerance{1e-8};
    /// @brief Armijo sufficient-decrease constant used by the per-coordinate line search.
    double armijo_c1{1e-4};
    double backtracking_factor{0.5};
    std::size_t max_line_search_trials{30};
};

/// @brief Cyclic coordinate descent on a DifferentiableFunction.  Each epoch visits every
///        coordinate in index order and performs an Armijo-backtracked partial-gradient step.
class CoordinateDescent {
public:
    explicit CoordinateDescent(CoordinateDescentOptions options = {});

    double optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const;

private:
    CoordinateDescentOptions options_;
};

} // namespace datamunge::optim
