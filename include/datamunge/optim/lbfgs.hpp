#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>

namespace datamunge::optim {

struct LBFGSOptions {
    std::size_t max_iterations{500};
    /// @brief Stops when the gradient norm drops below this.
    double tolerance{1e-8};
    /// @brief Number of (s, y) curvature pairs retained for the two-loop recursion.
    std::size_t history_size{10};
    /// @brief Armijo sufficient-decrease constant for the line search.
    double armijo_c1{1e-4};
    /// @brief Weak-Wolfe curvature constant for the line search (typical value for L-BFGS).
    double wolfe_c2{0.9};
    std::size_t max_line_search_trials{50};
};

/// @brief Limited-memory BFGS (Nocedal 1980) on a DifferentiableFunction: a quasi-Newton
///        method that approximates the inverse Hessian from a short history of gradient
///        changes, combined with an Armijo backtracking line search.
class LBFGS {
public:
    explicit LBFGS(LBFGSOptions options = {});

    double optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const;

private:
    LBFGSOptions options_;
};

} // namespace datamunge::optim
