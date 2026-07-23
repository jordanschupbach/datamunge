#include <datamunge/optim/cross_entropy_method.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace datamunge::optim {

CrossEntropyMethod::CrossEntropyMethod(CrossEntropyMethodOptions options) : options_(options) {}

double CrossEntropyMethod::optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                                     const std::vector<double>& lower_bound,
                                     const std::vector<double>& upper_bound) const {
    const std::size_t n = coordinates.size();
    if (n == 0) throw std::invalid_argument("CrossEntropyMethod::optimize: coordinates must not be empty");
    if (lower_bound.size() != n || upper_bound.size() != n)
        throw std::invalid_argument("CrossEntropyMethod::optimize: bound size must match coordinates size");
    for (std::size_t j = 0; j < n; ++j)
        if (!(lower_bound[j] < upper_bound[j]))
            throw std::invalid_argument("CrossEntropyMethod::optimize: lower_bound must be < upper_bound in every dimension");
    if (options_.population_size < 4)
        throw std::invalid_argument("CrossEntropyMethod::optimize: population_size must be at least 4");
    if (!(options_.elite_ratio > 0.0) || options_.elite_ratio > 1.0)
        throw std::invalid_argument("CrossEntropyMethod::optimize: elite_ratio must be in (0, 1]");
    if (!(options_.initial_std_dev > 0.0))
        throw std::invalid_argument("CrossEntropyMethod::optimize: initial_std_dev must be positive");
    if (!(options_.smoothing > 0.0) || options_.smoothing > 1.0)
        throw std::invalid_argument("CrossEntropyMethod::optimize: smoothing must be in (0, 1]");

    const std::size_t elite_count = std::max<std::size_t>(
        2, static_cast<std::size_t>(std::ceil(static_cast<double>(options_.population_size) * options_.elite_ratio)));

    std::vector<double> mean(n);
    for (std::size_t j = 0; j < n; ++j) mean[j] = std::clamp(coordinates[j], lower_bound[j], upper_bound[j]);
    std::vector<double> std_dev(n, options_.initial_std_dev);

    std::mt19937_64 rng(options_.seed);
    std::normal_distribution<double> standard_normal(0.0, 1.0);

    std::vector<double> best = mean;
    double best_value = function.evaluate(best);
    double prev_best = best_value;
    std::size_t stagnant = 0;

    struct Candidate {
        std::vector<double> point;
        double value;
    };

    for (std::size_t iteration = 0; iteration < options_.max_iterations; ++iteration) {
        std::vector<Candidate> population;
        population.reserve(options_.population_size);
        for (std::size_t member = 0; member < options_.population_size; ++member) {
            std::vector<double> point(n);
            for (std::size_t j = 0; j < n; ++j) {
                point[j] = mean[j] + std_dev[j] * standard_normal(rng);
                point[j] = std::clamp(point[j], lower_bound[j], upper_bound[j]);
            }
            const double value = function.evaluate(point);
            if (value < best_value) {
                best = point;
                best_value = value;
            }
            population.push_back({std::move(point), value});
        }

        std::sort(population.begin(), population.end(),
                  [](const Candidate& a, const Candidate& b) { return a.value < b.value; });

        std::vector<double> elite_mean(n, 0.0);
        for (std::size_t i = 0; i < elite_count; ++i)
            for (std::size_t j = 0; j < n; ++j) elite_mean[j] += population[i].point[j];
        for (std::size_t j = 0; j < n; ++j) elite_mean[j] /= static_cast<double>(elite_count);

        std::vector<double> elite_std(n, 0.0);
        for (std::size_t i = 0; i < elite_count; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                const double d = population[i].point[j] - elite_mean[j];
                elite_std[j] += d * d;
            }
        for (std::size_t j = 0; j < n; ++j) elite_std[j] = std::sqrt(elite_std[j] / static_cast<double>(elite_count));

        for (std::size_t j = 0; j < n; ++j) {
            mean[j] = options_.smoothing * elite_mean[j] + (1.0 - options_.smoothing) * mean[j];
            double updated_std = options_.smoothing * elite_std[j] + (1.0 - options_.smoothing) * std_dev[j];
            if (!(updated_std > 0.0) || std::isnan(updated_std)) updated_std = 1e-12;
            std_dev[j] = updated_std;
        }

        bool collapsed = true;
        for (std::size_t j = 0; j < n; ++j)
            if (std_dev[j] >= options_.tolerance) {
                collapsed = false;
                break;
            }

        if (std::abs(prev_best - best_value) < options_.tolerance) {
            if (++stagnant > 20) break;
        } else {
            stagnant = 0;
        }
        prev_best = best_value;

        if (collapsed) break;
    }

    coordinates = best;
    return best_value;
}

} // namespace datamunge::optim
