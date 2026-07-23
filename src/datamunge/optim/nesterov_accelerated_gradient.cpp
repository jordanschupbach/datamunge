#include <datamunge/optim/nesterov_accelerated_gradient.hpp>

#include <cmath>
#include <stdexcept>

namespace datamunge::optim {

NesterovAcceleratedGradient::NesterovAcceleratedGradient(NesterovAcceleratedGradientOptions options) : options_(options) {}

double NesterovAcceleratedGradient::optimize(DifferentiableFunction& function,
                                             std::vector<double>& coordinates) const {
    if (options_.step_size <= 0.0) throw std::invalid_argument("NesterovAcceleratedGradient: step_size must be positive");
    if (options_.momentum < 0.0 || options_.momentum >= 1.0)
        throw std::invalid_argument("NesterovAcceleratedGradient: momentum must be in [0, 1)");
    if (options_.tolerance < 0.0)
        throw std::invalid_argument("NesterovAcceleratedGradient: tolerance must be non-negative");

    std::vector<double> previous = coordinates;
    double value = function.evaluate(coordinates);
    for (std::size_t iteration = 0; iteration < options_.max_iterations; ++iteration) {
        std::vector<double> look_ahead(coordinates.size());
        for (std::size_t j = 0; j < coordinates.size(); ++j)
            look_ahead[j] = coordinates[j] + options_.momentum * (coordinates[j] - previous[j]);
        const auto gradient = function.gradient(look_ahead);
        double norm_sq = 0.0;
        std::vector<double> next(look_ahead);
        for (std::size_t j = 0; j < next.size(); ++j) {
            next[j] -= options_.step_size * gradient[j];
            norm_sq += gradient[j] * gradient[j];
        }
        const double next_value = function.evaluate(next);
        previous = coordinates;
        coordinates = std::move(next);
        if (std::sqrt(norm_sq) <= options_.tolerance) {
            value = next_value;
            break;
        }
        value = next_value;
    }
    return value;
}

} // namespace datamunge::optim
