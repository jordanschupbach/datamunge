#include <datamunge/optim/coordinate_descent.hpp>

#include <cmath>
#include <stdexcept>

namespace datamunge::optim {
namespace {

void validate(const CoordinateDescentOptions& options) {
    if (options.step_size <= 0.0) throw std::invalid_argument("CoordinateDescent: step_size must be positive");
    if (options.tolerance < 0.0) throw std::invalid_argument("CoordinateDescent: tolerance must be non-negative");
    if (options.armijo_c1 <= 0.0 || options.armijo_c1 >= 1.0)
        throw std::invalid_argument("CoordinateDescent: armijo_c1 must be in (0, 1)");
    if (options.backtracking_factor <= 0.0 || options.backtracking_factor >= 1.0)
        throw std::invalid_argument("CoordinateDescent: backtracking_factor must be in (0, 1)");
}

} // namespace

CoordinateDescent::CoordinateDescent(CoordinateDescentOptions options) : options_(options) {}

double CoordinateDescent::optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const {
    validate(options_);
    double value = function.evaluate(coordinates);
    for (std::size_t epoch = 0; epoch < options_.max_iterations; ++epoch) {
        for (std::size_t j = 0; j < coordinates.size(); ++j) {
            const double partial = function.gradient(coordinates)[j];
            if (std::abs(partial) <= options_.tolerance) continue;
            const double original = coordinates[j];
            double step = options_.step_size;
            bool accepted = false;
            for (std::size_t trial = 0; trial < options_.max_line_search_trials; ++trial) {
                coordinates[j] = original - step * partial;
                const double candidate = function.evaluate(coordinates);
                if (candidate <= value - options_.armijo_c1 * step * partial * partial) {
                    value = candidate;
                    accepted = true;
                    break;
                }
                step *= options_.backtracking_factor;
            }
            if (!accepted) coordinates[j] = original;
        }
        const auto gradient = function.gradient(coordinates);
        double norm_sq = 0.0;
        for (const double component : gradient) norm_sq += component * component;
        if (std::sqrt(norm_sq) <= options_.tolerance) break;
    }
    return value;
}

} // namespace datamunge::optim
