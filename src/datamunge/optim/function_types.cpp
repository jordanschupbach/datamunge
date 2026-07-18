#include <datamunge/optim/function_types.hpp>

namespace datamunge::optim {

double SeparableFunction::evaluate(const std::vector<double>& coordinates) {
    double total = 0.0;
    const std::size_t n = num_functions();
    for (std::size_t i = 0; i < n; ++i) total += evaluate_term(coordinates, i);
    return total;
}

double DifferentiableSeparableFunction::evaluate(const std::vector<double>& coordinates) {
    double total = 0.0;
    const std::size_t n = num_functions();
    for (std::size_t i = 0; i < n; ++i) total += evaluate_term(coordinates, i);
    return total;
}

std::vector<double> DifferentiableSeparableFunction::gradient(const std::vector<double>& coordinates) {
    std::vector<double> grad(coordinates.size(), 0.0);
    const std::size_t n = num_functions();
    for (std::size_t i = 0; i < n; ++i) {
        const auto term_grad = gradient_term(coordinates, i);
        for (std::size_t j = 0; j < grad.size(); ++j) grad[j] += term_grad[j];
    }
    return grad;
}

} // namespace datamunge::optim
