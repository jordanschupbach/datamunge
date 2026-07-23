#include <datamunge/optim/whale_optimization.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace datamunge::optim {

namespace {

void validate_bounds(const std::size_t n, const std::vector<double>& lower_bound,
                      const std::vector<double>& upper_bound, const char* op) {
    if (n == 0)
        throw std::invalid_argument(std::string("WhaleOptimization::") + op + ": coordinates must not be empty");
    if (lower_bound.size() != n || upper_bound.size() != n)
        throw std::invalid_argument(std::string("WhaleOptimization::") + op + ": bound size must match coordinates size");
    for (std::size_t j = 0; j < n; ++j)
        if (!(lower_bound[j] < upper_bound[j]))
            throw std::invalid_argument(std::string("WhaleOptimization::") + op +
                                         ": lower_bound must be < upper_bound in every dimension");
}

std::size_t best_index(const std::vector<double>& values) {
    std::size_t b = 0;
    for (std::size_t i = 1; i < values.size(); ++i)
        if (values[i] < values[b]) b = i;
    return b;
}

} // namespace

WhaleOptimization::WhaleOptimization(WhaleOptimizationOptions options) : options_(options) {}

double WhaleOptimization::optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                                    const std::vector<double>& lower_bound, const std::vector<double>& upper_bound) const {
    const std::size_t n = coordinates.size();
    validate_bounds(n, lower_bound, upper_bound, "optimize");
    if (options_.population_size < 2)
        throw std::invalid_argument("WhaleOptimization::optimize: population_size must be at least 2");
    if (!(options_.spiral_constant > 0.0))
        throw std::invalid_argument("WhaleOptimization::optimize: spiral_constant must be positive");

    std::mt19937_64 rng(options_.seed);
    std::uniform_real_distribution<double> unif01(0.0, 1.0);
    std::uniform_real_distribution<double> unif_pm1(-1.0, 1.0);
    std::uniform_int_distribution<std::size_t> pop_dist(0, options_.population_size - 1);

    // Initialize the population uniformly within bounds; whale 0 is seeded with the caller's guess.
    std::vector<std::vector<double>> pos(options_.population_size, std::vector<double>(n));
    for (auto& p : pos)
        for (std::size_t j = 0; j < n; ++j) p[j] = lower_bound[j] + unif01(rng) * (upper_bound[j] - lower_bound[j]);
    for (std::size_t j = 0; j < n; ++j) pos[0][j] = std::clamp(coordinates[j], lower_bound[j], upper_bound[j]);

    std::vector<double> val(options_.population_size);
    for (std::size_t i = 0; i < pos.size(); ++i) val[i] = function.evaluate(pos[i]);

    std::size_t init_best = best_index(val);
    std::vector<double> best = pos[init_best];
    double best_value = val[init_best];

    double prev_best = best_value;
    std::size_t stagnant = 0;

    for (std::size_t t = 0; t < options_.max_iterations; ++t) {
        const double a = 2.0 - 2.0 * static_cast<double>(t) / static_cast<double>(options_.max_iterations);
        const std::vector<double>& x_star = pos[best_index(val)];

        std::vector<std::vector<double>> new_pos(options_.population_size, std::vector<double>(n));
        std::vector<double> new_val(options_.population_size);

        for (std::size_t i = 0; i < pos.size(); ++i) {
            const std::vector<double>& x = pos[i];
            std::vector<double> x_new(n);
            const double p = unif01(rng);

            if (p < 0.5) {
                std::vector<double> a_vec(n), c_vec(n);
                for (std::size_t d = 0; d < n; ++d) {
                    const double r1 = unif01(rng), r2 = unif01(rng);
                    a_vec[d] = 2.0 * a * r1 - a;
                    c_vec[d] = 2.0 * r2;
                }
                double norm_a = 0.0;
                for (std::size_t d = 0; d < n; ++d) norm_a += a_vec[d] * a_vec[d];
                norm_a = std::sqrt(norm_a);

                if (norm_a < 1.0) {
                    // Encircling prey.
                    for (std::size_t d = 0; d < n; ++d) {
                        const double dist = std::abs(c_vec[d] * x_star[d] - x[d]);
                        x_new[d] = x_star[d] - a_vec[d] * dist;
                    }
                } else {
                    // Search for prey: move toward a random other whale.
                    const std::vector<double>& x_rand = pos[pop_dist(rng)];
                    for (std::size_t d = 0; d < n; ++d) {
                        const double dist = std::abs(c_vec[d] * x_rand[d] - x[d]);
                        x_new[d] = x_rand[d] - a_vec[d] * dist;
                    }
                }
            } else {
                // Bubble-net attacking: logarithmic spiral toward X_star.
                const double l = unif_pm1(rng);
                for (std::size_t d = 0; d < n; ++d) {
                    const double dist_prime = std::abs(x_star[d] - x[d]);
                    x_new[d] = dist_prime * std::exp(options_.spiral_constant * l) * std::cos(2.0 * M_PI * l) + x_star[d];
                }
            }

            for (std::size_t d = 0; d < n; ++d) x_new[d] = std::clamp(x_new[d], lower_bound[d], upper_bound[d]);
            const double value = function.evaluate(x_new);

            if (value < best_value) {
                best_value = value;
                best = x_new;
            }

            new_pos[i] = std::move(x_new);
            new_val[i] = value;
        }

        pos = std::move(new_pos);
        val = std::move(new_val);

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
