#include <datamunge/optim/randomized_block_coordinate_descent.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace datamunge::optim {
namespace {

void validate(const RandomizedBlockCoordinateDescentOptions& options) {
    if (options.block_size == 0) throw std::invalid_argument("RandomizedBlockCoordinateDescent: block_size must be positive");
    if (options.step_size <= 0.0) throw std::invalid_argument("RandomizedBlockCoordinateDescent: step_size must be positive");
    if (options.tolerance < 0.0) throw std::invalid_argument("RandomizedBlockCoordinateDescent: tolerance must be non-negative");
    if (options.armijo_c1 <= 0.0 || options.armijo_c1 >= 1.0)
        throw std::invalid_argument("RandomizedBlockCoordinateDescent: armijo_c1 must be in (0, 1)");
    if (options.backtracking_factor <= 0.0 || options.backtracking_factor >= 1.0)
        throw std::invalid_argument("RandomizedBlockCoordinateDescent: backtracking_factor must be in (0, 1)");
}

} // namespace

RandomizedBlockCoordinateDescent::RandomizedBlockCoordinateDescent(RandomizedBlockCoordinateDescentOptions options)
    : options_(options) {}

double RandomizedBlockCoordinateDescent::optimize(DifferentiableFunction& function,
                                                   std::vector<double>& coordinates) const {
    validate(options_);
    std::vector<std::size_t> order(coordinates.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937_64 rng(options_.seed);
    double value = function.evaluate(coordinates);

    for (std::size_t epoch = 0; epoch < options_.max_iterations; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);
        for (std::size_t begin = 0; begin < order.size(); begin += options_.block_size) {
            const std::size_t end = std::min(begin + options_.block_size, order.size());
            const auto gradient = function.gradient(coordinates);
            double block_norm_sq = 0.0;
            for (std::size_t k = begin; k < end; ++k) block_norm_sq += gradient[order[k]] * gradient[order[k]];
            if (block_norm_sq <= options_.tolerance * options_.tolerance) continue;

            const std::vector<double> original = coordinates;
            double step = options_.step_size;
            bool accepted = false;
            for (std::size_t trial = 0; trial < options_.max_line_search_trials; ++trial) {
                coordinates = original;
                for (std::size_t k = begin; k < end; ++k) {
                    const std::size_t j = order[k];
                    coordinates[j] -= step * gradient[j];
                }
                const double candidate = function.evaluate(coordinates);
                if (candidate <= value - options_.armijo_c1 * step * block_norm_sq) {
                    value = candidate;
                    accepted = true;
                    break;
                }
                step *= options_.backtracking_factor;
            }
            if (!accepted) coordinates = original;
        }
        const auto gradient = function.gradient(coordinates);
        double norm_sq = 0.0;
        for (const double component : gradient) norm_sq += component * component;
        if (std::sqrt(norm_sq) <= options_.tolerance) break;
    }
    return value;
}

} // namespace datamunge::optim
