#include <datamunge/optim/nadam.hpp>

#include <cmath>
#include <stdexcept>

namespace datamunge::optim {

Nadam::Nadam(NadamOptions options) : options_(options) {}

double Nadam::optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const {
    if (options_.step_size <= 0.0) throw std::invalid_argument("Nadam: step_size must be positive");
    if (options_.beta1 < 0.0 || options_.beta1 >= 1.0) throw std::invalid_argument("Nadam: beta1 must be in [0, 1)");
    if (options_.beta2 < 0.0 || options_.beta2 >= 1.0) throw std::invalid_argument("Nadam: beta2 must be in [0, 1)");
    if (options_.epsilon <= 0.0) throw std::invalid_argument("Nadam: epsilon must be positive");
    if (options_.tolerance < 0.0) throw std::invalid_argument("Nadam: tolerance must be non-negative");
    std::vector<double> first(coordinates.size(), 0.0), second(coordinates.size(), 0.0);
    double value = function.evaluate(coordinates);
    for (std::size_t iteration = 1; iteration <= options_.max_iterations; ++iteration) {
        const auto gradient = function.gradient(coordinates);
        const double first_correction = 1.0 - std::pow(options_.beta1, static_cast<double>(iteration));
        const double second_correction = 1.0 - std::pow(options_.beta2, static_cast<double>(iteration));
        double norm_sq = 0.0;
        for (std::size_t j = 0; j < coordinates.size(); ++j) {
            first[j] = options_.beta1 * first[j] + (1.0 - options_.beta1) * gradient[j];
            second[j] = options_.beta2 * second[j] + (1.0 - options_.beta2) * gradient[j] * gradient[j];
            const double nesterov_first = options_.beta1 * first[j] / first_correction +
                (1.0 - options_.beta1) * gradient[j] / first_correction;
            coordinates[j] -= options_.step_size * nesterov_first /
                              (std::sqrt(second[j] / second_correction) + options_.epsilon);
            norm_sq += gradient[j] * gradient[j];
        }
        value = function.evaluate(coordinates);
        if (std::sqrt(norm_sq) <= options_.tolerance) break;
    }
    return value;
}

} // namespace datamunge::optim
