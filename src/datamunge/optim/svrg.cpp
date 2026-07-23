#include <datamunge/optim/svrg.hpp>

#include <cmath>
#include <random>
#include <stdexcept>

namespace datamunge::optim {

SVRG::SVRG(SVRGOptions options) : options_(options) {}

double SVRG::optimize(DifferentiableSeparableFunction& function, std::vector<double>& coordinates) const {
    if (options_.step_size <= 0.0) throw std::invalid_argument("SVRG: step_size must be positive");
    if (options_.tolerance < 0.0) throw std::invalid_argument("SVRG: tolerance must be non-negative");
    const std::size_t terms = function.num_functions();
    if (terms == 0) throw std::invalid_argument("SVRG: function has zero terms");
    const std::size_t inner = options_.inner_iterations == 0 ? terms : options_.inner_iterations;
    std::mt19937_64 rng(options_.seed);
    std::uniform_int_distribution<std::size_t> sample(0, terms - 1);
    double value = function.evaluate(coordinates);
    for (std::size_t epoch = 0; epoch < options_.max_epochs; ++epoch) {
        const std::vector<double> snapshot = coordinates;
        const auto snapshot_gradient = function.gradient(snapshot);
        for (std::size_t step = 0; step < inner; ++step) {
            const std::size_t i = sample(rng);
            const auto current_gradient = function.gradient_term(coordinates, i);
            const auto snapshot_term_gradient = function.gradient_term(snapshot, i);
            for (std::size_t j = 0; j < coordinates.size(); ++j)
                coordinates[j] -= options_.step_size *
                    (current_gradient[j] - snapshot_term_gradient[j] + snapshot_gradient[j]);
        }
        const double next_value = function.evaluate(coordinates);
        if (std::abs(next_value - value) <= options_.tolerance) {
            value = next_value;
            break;
        }
        value = next_value;
    }
    return value;
}

} // namespace datamunge::optim
