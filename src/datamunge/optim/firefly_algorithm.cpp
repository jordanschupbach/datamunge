#include <datamunge/optim/firefly_algorithm.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace datamunge::optim {

namespace {

void validate_bounds(const std::size_t n, const std::vector<double>& lower_bound,
                      const std::vector<double>& upper_bound, const char* op) {
    if (n == 0)
        throw std::invalid_argument(std::string("FireflyAlgorithm::") + op + ": coordinates must not be empty");
    if (lower_bound.size() != n || upper_bound.size() != n)
        throw std::invalid_argument(std::string("FireflyAlgorithm::") + op + ": bound size must match coordinates size");
    for (std::size_t j = 0; j < n; ++j)
        if (!(lower_bound[j] < upper_bound[j]))
            throw std::invalid_argument(std::string("FireflyAlgorithm::") + op +
                                         ": lower_bound must be < upper_bound in every dimension");
}

double squared_distance(const std::vector<double>& a, const std::vector<double>& b) {
    double total = 0.0;
    for (std::size_t d = 0; d < a.size(); ++d) {
        const double diff = a[d] - b[d];
        total += diff * diff;
    }
    return total;
}

} // namespace

FireflyAlgorithm::FireflyAlgorithm(FireflyAlgorithmOptions options) : options_(options) {}

double FireflyAlgorithm::optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                                   const std::vector<double>& lower_bound, const std::vector<double>& upper_bound) const {
    const std::size_t n = coordinates.size();
    validate_bounds(n, lower_bound, upper_bound, "optimize");
    if (options_.population_size < 2)
        throw std::invalid_argument("FireflyAlgorithm::optimize: population_size must be at least 2");
    if (!(options_.attractiveness_at_zero > 0.0))
        throw std::invalid_argument("FireflyAlgorithm::optimize: attractiveness_at_zero must be positive");
    if (!(options_.light_absorption >= 0.0))
        throw std::invalid_argument("FireflyAlgorithm::optimize: light_absorption must be non-negative");
    if (!(options_.randomization_step >= 0.0))
        throw std::invalid_argument("FireflyAlgorithm::optimize: randomization_step must be non-negative");
    if (!(options_.randomization_decay > 0.0 && options_.randomization_decay <= 1.0))
        throw std::invalid_argument("FireflyAlgorithm::optimize: randomization_decay must be in (0, 1]");

    std::mt19937_64 rng(options_.seed);
    std::uniform_real_distribution<double> unif01(0.0, 1.0);

    std::vector<std::vector<double>> positions(options_.population_size, std::vector<double>(n));
    std::vector<double> values(options_.population_size);

    for (auto& position : positions)
        for (std::size_t j = 0; j < n; ++j) position[j] = lower_bound[j] + unif01(rng) * (upper_bound[j] - lower_bound[j]);
    // Seed the caller's initial guess as one firefly.
    for (std::size_t j = 0; j < n; ++j) positions[0][j] = std::clamp(coordinates[j], lower_bound[j], upper_bound[j]);

    for (std::size_t i = 0; i < positions.size(); ++i) values[i] = function.evaluate(positions[i]);

    std::size_t best_idx = 0;
    for (std::size_t i = 1; i < values.size(); ++i)
        if (values[i] < values[best_idx]) best_idx = i;
    std::vector<double> best = positions[best_idx];
    double best_value = values[best_idx];

    double alpha = options_.randomization_step;
    double prev_best = best_value;
    std::size_t stagnant = 0;

    for (std::size_t iter = 0; iter < options_.max_iterations; ++iter) {
        // Snapshot this iteration's positions and values so every pairwise decision this
        // sweep is based on a consistent view, not on positions already mutated mid-sweep.
        const std::vector<std::vector<double>> snapshot_positions = positions;
        const std::vector<double> snapshot_values = values;

        for (std::size_t i = 0; i < positions.size(); ++i) {
            for (std::size_t j = 0; j < positions.size(); ++j) {
                if (i == j) continue;
                if (snapshot_values[j] < snapshot_values[i]) {
                    const double r2 = squared_distance(snapshot_positions[i], snapshot_positions[j]);
                    const double beta = options_.attractiveness_at_zero * std::exp(-options_.light_absorption * r2);
                    for (std::size_t d = 0; d < n; ++d) {
                        positions[i][d] += beta * (snapshot_positions[j][d] - snapshot_positions[i][d]) +
                                            alpha * (unif01(rng) - 0.5);
                        positions[i][d] = std::clamp(positions[i][d], lower_bound[d], upper_bound[d]);
                    }
                }
            }
        }

        for (std::size_t i = 0; i < positions.size(); ++i) {
            values[i] = function.evaluate(positions[i]);
            if (values[i] < best_value) {
                best_value = values[i];
                best = positions[i];
            }
        }

        alpha *= options_.randomization_decay;

        if (std::abs(prev_best - best_value) < options_.tolerance) {
            if (++stagnant > 20) break;
        } else {
            stagnant = 0;
        }
        prev_best = best_value;
    }

    coordinates = best;
    return best_value;
}

} // namespace datamunge::optim
