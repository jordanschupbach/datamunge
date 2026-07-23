#include <datamunge/optim/grey_wolf_optimizer.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <stdexcept>

namespace datamunge::optim {

namespace {

struct Wolf {
    std::vector<double> position;
    double value{0.0};
};

void validate_bounds(const std::size_t n, const std::vector<double>& lower_bound,
                      const std::vector<double>& upper_bound, const char* op) {
    if (n == 0)
        throw std::invalid_argument(std::string("GreyWolfOptimizer::") + op + ": coordinates must not be empty");
    if (lower_bound.size() != n || upper_bound.size() != n)
        throw std::invalid_argument(std::string("GreyWolfOptimizer::") + op +
                                     ": bound size must match coordinates size");
    for (std::size_t j = 0; j < n; ++j)
        if (!(lower_bound[j] < upper_bound[j]))
            throw std::invalid_argument(std::string("GreyWolfOptimizer::") + op +
                                         ": lower_bound must be < upper_bound in every dimension");
}

// Indices of the three best (lowest-value) wolves, in order [alpha, beta, delta].
// Simple linear top-3 scan -- population sizes here are small (tens), so this is fine.
std::array<std::size_t, 3> top_three(const std::vector<Wolf>& wolves) {
    std::array<std::size_t, 3> best{0, 1, 2};
    if (wolves[best[0]].value > wolves[best[1]].value) std::swap(best[0], best[1]);
    if (wolves[best[1]].value > wolves[best[2]].value) std::swap(best[1], best[2]);
    if (wolves[best[0]].value > wolves[best[1]].value) std::swap(best[0], best[1]);
    for (std::size_t i = 3; i < wolves.size(); ++i) {
        const double v = wolves[i].value;
        if (v < wolves[best[0]].value) {
            best[2] = best[1];
            best[1] = best[0];
            best[0] = i;
        } else if (v < wolves[best[1]].value) {
            best[2] = best[1];
            best[1] = i;
        } else if (v < wolves[best[2]].value) {
            best[2] = i;
        }
    }
    return best;
}

} // namespace

GreyWolfOptimizer::GreyWolfOptimizer(GreyWolfOptimizerOptions options) : options_(options) {}

double GreyWolfOptimizer::optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                                    const std::vector<double>& lower_bound,
                                    const std::vector<double>& upper_bound) const {
    const std::size_t n = coordinates.size();
    validate_bounds(n, lower_bound, upper_bound, "optimize");
    if (options_.population_size < 4)
        throw std::invalid_argument("GreyWolfOptimizer::optimize: population_size must be at least 4");

    std::mt19937_64 rng(options_.seed);
    std::uniform_real_distribution<double> unif01(0.0, 1.0);

    std::vector<Wolf> wolves(options_.population_size);
    for (auto& w : wolves) {
        w.position.resize(n);
        for (std::size_t j = 0; j < n; ++j) {
            const double range = upper_bound[j] - lower_bound[j];
            w.position[j] = lower_bound[j] + unif01(rng) * range;
        }
    }
    for (std::size_t j = 0; j < n; ++j) wolves[0].position[j] = std::clamp(coordinates[j], lower_bound[j], upper_bound[j]);
    for (auto& w : wolves) w.value = function.evaluate(w.position);

    std::size_t best_idx = 0;
    for (std::size_t i = 1; i < wolves.size(); ++i)
        if (wolves[i].value < wolves[best_idx].value) best_idx = i;
    std::vector<double> best = wolves[best_idx].position;
    double best_value = wolves[best_idx].value;

    double prev_best = best_value;
    std::size_t stagnant = 0;

    const std::size_t max_iterations = options_.max_iterations;
    std::vector<std::vector<double>> new_positions(wolves.size(), std::vector<double>(n));

    for (std::size_t t = 0; t < max_iterations; ++t) {
        const std::array<std::size_t, 3> leaders = top_three(wolves);
        const std::vector<double>& x_alpha = wolves[leaders[0]].position;
        const std::vector<double>& x_beta = wolves[leaders[1]].position;
        const std::vector<double>& x_delta = wolves[leaders[2]].position;

        const double a = 2.0 - 2.0 * static_cast<double>(t) / static_cast<double>(max_iterations);

        for (std::size_t w = 0; w < wolves.size(); ++w) {
            const std::vector<double>& x_w = wolves[w].position;
            for (std::size_t j = 0; j < n; ++j) {
                double pull_sum = 0.0;
                for (const std::vector<double>* leader : {&x_alpha, &x_beta, &x_delta}) {
                    const double r1 = unif01(rng);
                    const double r2 = unif01(rng);
                    const double a_coef = 2.0 * a * r1 - a;
                    const double c_coef = 2.0 * r2;
                    const double d = std::abs(c_coef * (*leader)[j] - x_w[j]);
                    pull_sum += (*leader)[j] - a_coef * d;
                }
                new_positions[w][j] = std::clamp(pull_sum / 3.0, lower_bound[j], upper_bound[j]);
            }
        }

        for (std::size_t w = 0; w < wolves.size(); ++w) {
            wolves[w].position = new_positions[w];
            wolves[w].value = function.evaluate(wolves[w].position);
            if (wolves[w].value < best_value) {
                best_value = wolves[w].value;
                best = wolves[w].position;
            }
        }

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
