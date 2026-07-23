#include <datamunge/optim/amsgrad.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace datamunge::optim {

AMSGrad::AMSGrad(AMSGradOptions options) : options_(options) {}

double AMSGrad::optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const {
    if (options_.step_size <= 0.0) throw std::invalid_argument("AMSGrad: step_size must be positive");
    if (options_.beta1 < 0.0 || options_.beta1 >= 1.0) throw std::invalid_argument("AMSGrad: beta1 must be in [0, 1)");
    if (options_.beta2 < 0.0 || options_.beta2 >= 1.0) throw std::invalid_argument("AMSGrad: beta2 must be in [0, 1)");
    if (options_.epsilon <= 0.0) throw std::invalid_argument("AMSGrad: epsilon must be positive");
    if (options_.tolerance < 0.0) throw std::invalid_argument("AMSGrad: tolerance must be non-negative");

    std::vector<double> first_moment(coordinates.size(), 0.0);
    std::vector<double> second_moment(coordinates.size(), 0.0);
    std::vector<double> max_second_moment(coordinates.size(), 0.0);
    double value = function.evaluate(coordinates);
    for (std::size_t iteration = 1; iteration <= options_.max_iterations; ++iteration) {
        const auto gradient = function.gradient(coordinates);
        double gradient_norm_sq = 0.0;
        const double correction1 = 1.0 - std::pow(options_.beta1, static_cast<double>(iteration));
        const double correction2 = 1.0 - std::pow(options_.beta2, static_cast<double>(iteration));
        for (std::size_t j = 0; j < coordinates.size(); ++j) {
            first_moment[j] = options_.beta1 * first_moment[j] + (1.0 - options_.beta1) * gradient[j];
            second_moment[j] = options_.beta2 * second_moment[j] + (1.0 - options_.beta2) * gradient[j] * gradient[j];
            max_second_moment[j] = std::max(max_second_moment[j], second_moment[j]);
            const double corrected_first = first_moment[j] / correction1;
            const double corrected_max_second = max_second_moment[j] / correction2;
            coordinates[j] -= options_.step_size * corrected_first / (std::sqrt(corrected_max_second) + options_.epsilon);
            gradient_norm_sq += gradient[j] * gradient[j];
        }
        value = function.evaluate(coordinates);
        if (std::sqrt(gradient_norm_sq) <= options_.tolerance) break;
    }
    return value;
}

} // namespace datamunge::optim
