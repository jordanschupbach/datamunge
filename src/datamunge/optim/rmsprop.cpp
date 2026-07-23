#include <datamunge/optim/rmsprop.hpp>

#include <cmath>
#include <stdexcept>

namespace datamunge::optim {

RMSProp::RMSProp(RMSPropOptions options) : options_(options) {}

double RMSProp::optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const {
    if (options_.step_size <= 0.0) throw std::invalid_argument("RMSProp: step_size must be positive");
    if (options_.decay_rate < 0.0 || options_.decay_rate >= 1.0)
        throw std::invalid_argument("RMSProp: decay_rate must be in [0, 1)");
    if (options_.epsilon <= 0.0) throw std::invalid_argument("RMSProp: epsilon must be positive");
    if (options_.tolerance < 0.0) throw std::invalid_argument("RMSProp: tolerance must be non-negative");

    std::vector<double> mean_squares(coordinates.size(), 0.0);
    double value = function.evaluate(coordinates);
    for (std::size_t iteration = 0; iteration < options_.max_iterations; ++iteration) {
        const auto gradient = function.gradient(coordinates);
        double gradient_norm_sq = 0.0;
        for (std::size_t j = 0; j < coordinates.size(); ++j) {
            mean_squares[j] = options_.decay_rate * mean_squares[j] +
                              (1.0 - options_.decay_rate) * gradient[j] * gradient[j];
            coordinates[j] -= options_.step_size * gradient[j] / (std::sqrt(mean_squares[j]) + options_.epsilon);
            gradient_norm_sq += gradient[j] * gradient[j];
        }
        value = function.evaluate(coordinates);
        if (std::sqrt(gradient_norm_sq) <= options_.tolerance) break;
    }
    return value;
}

} // namespace datamunge::optim
