#include <datamunge/optim/gradient_descent.hpp>

#include <cmath>

namespace datamunge::optim {

GradientDescent::GradientDescent(GradientDescentOptions options) : options_(options) {}

double GradientDescent::optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const {
    std::vector<double> velocity(coordinates.size(), 0.0);
    double value = function.evaluate(coordinates);

    for (std::size_t iter = 0; iter < options_.max_iterations; ++iter) {
        const auto grad = function.gradient(coordinates);
        double grad_norm_sq = 0.0;
        for (std::size_t j = 0; j < coordinates.size(); ++j) {
            velocity[j] = options_.momentum * velocity[j] - options_.step_size * grad[j];
            coordinates[j] += velocity[j];
            grad_norm_sq += grad[j] * grad[j];
        }
        const double new_value = function.evaluate(coordinates);
        const bool converged =
            std::sqrt(grad_norm_sq) < options_.tolerance || std::abs(new_value - value) < options_.tolerance;
        value = new_value;
        if (converged) break;
    }
    return value;
}

} // namespace datamunge::optim
