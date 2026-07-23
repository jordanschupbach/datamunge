#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>

namespace datamunge::optim {

struct ConjugateGradientOptions {
    std::size_t max_iterations{1000};
    double tolerance{1e-8};
    double armijo_c1{1e-4};
    double backtracking_factor{0.5};
    std::size_t max_line_search_trials{30};
};

/// @brief Nonlinear conjugate gradient with the Polak-Ribiere+ update and an Armijo line search.
class ConjugateGradient {
public:
    explicit ConjugateGradient(ConjugateGradientOptions options = {});

    double optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const;

private:
    ConjugateGradientOptions options_;
};

} // namespace datamunge::optim
