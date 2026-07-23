#include <datamunge/optim/adagrad.hpp>

#include <cmath>
#include <stdexcept>

namespace datamunge::optim {

AdaGrad::AdaGrad(AdaGradOptions options) : options_(options) {}

double AdaGrad::optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const {
    if (options_.step_size <= 0.0) throw std::invalid_argument("AdaGrad: step_size must be positive");
    if (options_.epsilon <= 0.0) throw std::invalid_argument("AdaGrad: epsilon must be positive");
    if (options_.tolerance < 0.0) throw std::invalid_argument("AdaGrad: tolerance must be non-negative");

    std::vector<double> accumulated_squares(coordinates.size(), 0.0);
    double value = function.evaluate(coordinates);
    for (std::size_t iteration = 0; iteration < options_.max_iterations; ++iteration) {
        const auto gradient = function.gradient(coordinates);
        double gradient_norm_sq = 0.0;
        for (std::size_t j = 0; j < coordinates.size(); ++j) {
            accumulated_squares[j] += gradient[j] * gradient[j];
            coordinates[j] -= options_.step_size * gradient[j] /
                              std::sqrt(accumulated_squares[j] + options_.epsilon);
            gradient_norm_sq += gradient[j] * gradient[j];
        }
        value = function.evaluate(coordinates);
        if (std::sqrt(gradient_norm_sq) <= options_.tolerance) break;
    }
    return value;
}

} // namespace datamunge::optim
