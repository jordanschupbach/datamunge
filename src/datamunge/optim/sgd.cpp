#include <datamunge/optim/sgd.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace datamunge::optim {

SGD::SGD(SGDOptions options) : options_(options) {}

double SGD::optimize(DifferentiableSeparableFunction& function, std::vector<double>& coordinates) const {
    const std::size_t n_terms = function.num_functions();
    if (n_terms == 0) throw std::invalid_argument("SGD::optimize: function has zero terms");

    std::vector<std::size_t> indices(n_terms);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937_64 rng(options_.seed);

    double value = function.evaluate(coordinates);

    for (std::size_t epoch = 0; epoch < options_.max_epochs; ++epoch) {
        if (options_.shuffle) std::shuffle(indices.begin(), indices.end(), rng);

        for (std::size_t start = 0; start < n_terms; start += options_.batch_size) {
            const std::size_t end = std::min(start + options_.batch_size, n_terms);
            std::vector<double> grad(coordinates.size(), 0.0);
            for (std::size_t k = start; k < end; ++k) {
                const auto term_grad = function.gradient_term(coordinates, indices[k]);
                for (std::size_t j = 0; j < grad.size(); ++j) grad[j] += term_grad[j];
            }
            const double batch_n = static_cast<double>(end - start);
            for (std::size_t j = 0; j < coordinates.size(); ++j) coordinates[j] -= options_.step_size * grad[j] / batch_n;
        }

        const double new_value = function.evaluate(coordinates);
        const bool converged = std::abs(new_value - value) < options_.tolerance;
        value = new_value;
        if (converged) break;
    }
    return value;
}

} // namespace datamunge::optim
