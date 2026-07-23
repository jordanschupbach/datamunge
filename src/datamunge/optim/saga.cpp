#include <datamunge/optim/saga.hpp>

#include <cmath>
#include <random>
#include <stdexcept>

namespace datamunge::optim {

SAGA::SAGA(SAGAOptions options) : options_(options) {}

double SAGA::optimize(DifferentiableSeparableFunction& function, std::vector<double>& coordinates) const {
    if (options_.step_size <= 0.0) throw std::invalid_argument("SAGA: step_size must be positive");
    if (options_.tolerance < 0.0) throw std::invalid_argument("SAGA: tolerance must be non-negative");
    const std::size_t terms = function.num_functions();
    if (terms == 0) throw std::invalid_argument("SAGA: function has zero terms");
    const std::size_t dimensions = coordinates.size();
    std::vector<std::vector<double>> gradient_table(terms, std::vector<double>(dimensions));
    std::vector<double> average_gradient(dimensions, 0.0);
    for (std::size_t i = 0; i < terms; ++i) {
        gradient_table[i] = function.gradient_term(coordinates, i);
        for (std::size_t j = 0; j < dimensions; ++j) average_gradient[j] += gradient_table[i][j] / static_cast<double>(terms);
    }
    std::mt19937_64 rng(options_.seed);
    std::uniform_int_distribution<std::size_t> sample(0, terms - 1);
    double value = function.evaluate(coordinates);
    for (std::size_t iteration = 0; iteration < options_.max_iterations; ++iteration) {
        const std::size_t i = sample(rng);
        const auto gradient = function.gradient_term(coordinates, i);
        double estimator_norm_sq = 0.0;
        for (std::size_t j = 0; j < dimensions; ++j) {
            const double estimator = gradient[j] - gradient_table[i][j] + average_gradient[j];
            coordinates[j] -= options_.step_size * estimator;
            average_gradient[j] += (gradient[j] - gradient_table[i][j]) / static_cast<double>(terms);
            gradient_table[i][j] = gradient[j];
            estimator_norm_sq += estimator * estimator;
        }
        if ((iteration + 1) % terms == 0) {
            const double next_value = function.evaluate(coordinates);
            if (std::abs(next_value - value) <= options_.tolerance || std::sqrt(estimator_norm_sq) <= options_.tolerance) {
                value = next_value;
                break;
            }
            value = next_value;
        }
    }
    return function.evaluate(coordinates);
}

} // namespace datamunge::optim
