#include <datamunge/optim/adadelta.hpp>

#include <cmath>
#include <stdexcept>

namespace datamunge::optim {

AdaDelta::AdaDelta(AdaDeltaOptions options) : options_(options) {}

double AdaDelta::optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const {
    if (options_.decay_rate < 0.0 || options_.decay_rate >= 1.0)
        throw std::invalid_argument("AdaDelta: decay_rate must be in [0, 1)");
    if (options_.epsilon <= 0.0) throw std::invalid_argument("AdaDelta: epsilon must be positive");
    if (options_.tolerance < 0.0) throw std::invalid_argument("AdaDelta: tolerance must be non-negative");
    std::vector<double> gradient_squares(coordinates.size(), 0.0), update_squares(coordinates.size(), 0.0);
    double value = function.evaluate(coordinates);
    for (std::size_t iteration = 0; iteration < options_.max_iterations; ++iteration) {
        const auto gradient = function.gradient(coordinates);
        double norm_sq = 0.0;
        for (std::size_t j = 0; j < coordinates.size(); ++j) {
            gradient_squares[j] = options_.decay_rate * gradient_squares[j] +
                                  (1.0 - options_.decay_rate) * gradient[j] * gradient[j];
            const double update = -std::sqrt(update_squares[j] + options_.epsilon) /
                                  std::sqrt(gradient_squares[j] + options_.epsilon) * gradient[j];
            coordinates[j] += update;
            update_squares[j] = options_.decay_rate * update_squares[j] +
                                (1.0 - options_.decay_rate) * update * update;
            norm_sq += gradient[j] * gradient[j];
        }
        value = function.evaluate(coordinates);
        if (std::sqrt(norm_sq) <= options_.tolerance) break;
    }
    return value;
}

} // namespace datamunge::optim
