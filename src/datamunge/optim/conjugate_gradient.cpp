#include <datamunge/optim/conjugate_gradient.hpp>

#include <cmath>
#include <stdexcept>

namespace datamunge::optim {
namespace {

double squared_norm(const std::vector<double>& vector) {
    double result = 0.0;
    for (const double value : vector) result += value * value;
    return result;
}

} // namespace

ConjugateGradient::ConjugateGradient(ConjugateGradientOptions options) : options_(options) {}

double ConjugateGradient::optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const {
    if (options_.tolerance < 0.0) throw std::invalid_argument("ConjugateGradient: tolerance must be non-negative");
    if (options_.armijo_c1 <= 0.0 || options_.armijo_c1 >= 1.0)
        throw std::invalid_argument("ConjugateGradient: armijo_c1 must be in (0, 1)");
    if (options_.backtracking_factor <= 0.0 || options_.backtracking_factor >= 1.0)
        throw std::invalid_argument("ConjugateGradient: backtracking_factor must be in (0, 1)");

    double value = function.evaluate(coordinates);
    std::vector<double> gradient = function.gradient(coordinates);
    std::vector<double> direction(gradient.size());
    for (std::size_t j = 0; j < direction.size(); ++j) direction[j] = -gradient[j];

    for (std::size_t iteration = 0; iteration < options_.max_iterations; ++iteration) {
        const double gradient_norm_sq = squared_norm(gradient);
        if (std::sqrt(gradient_norm_sq) <= options_.tolerance) break;
        double slope = 0.0;
        for (std::size_t j = 0; j < direction.size(); ++j) slope += gradient[j] * direction[j];
        if (slope >= 0.0) {
            for (std::size_t j = 0; j < direction.size(); ++j) direction[j] = -gradient[j];
            slope = -gradient_norm_sq;
        }

        std::vector<double> candidate(coordinates.size());
        double candidate_value = value;
        double step = 1.0;
        bool accepted = false;
        for (std::size_t trial = 0; trial < options_.max_line_search_trials; ++trial) {
            for (std::size_t j = 0; j < candidate.size(); ++j) candidate[j] = coordinates[j] + step * direction[j];
            candidate_value = function.evaluate(candidate);
            if (candidate_value <= value + options_.armijo_c1 * step * slope) {
                accepted = true;
                break;
            }
            step *= options_.backtracking_factor;
        }
        if (!accepted) break;

        const auto next_gradient = function.gradient(candidate);
        double numerator = 0.0;
        for (std::size_t j = 0; j < gradient.size(); ++j)
            numerator += next_gradient[j] * (next_gradient[j] - gradient[j]);
        // A periodic restart is standard for nonlinear CG.  It prevents stale conjugate
        // directions from accumulating when the local objective is no longer quadratic.
        double beta = std::max(0.0, numerator / gradient_norm_sq); // Polak-Ribiere+
        if (!direction.empty() && (iteration + 1) % direction.size() == 0) beta = 0.0;
        for (std::size_t j = 0; j < direction.size(); ++j) direction[j] = -next_gradient[j] + beta * direction[j];

        coordinates = std::move(candidate);
        value = candidate_value;
        gradient = next_gradient;
    }
    return value;
}

} // namespace datamunge::optim
