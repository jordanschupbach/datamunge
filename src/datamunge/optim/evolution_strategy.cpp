#include <datamunge/optim/evolution_strategy.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace datamunge::optim {

namespace {

struct Individual {
    std::vector<double> x;
    double sigma{0.0};
    double value{0.0};
};

void validate_bounds(const std::size_t n, const std::vector<double>& lower_bound,
                      const std::vector<double>& upper_bound, const char* op) {
    if (n == 0)
        throw std::invalid_argument(std::string("EvolutionStrategy::") + op + ": coordinates must not be empty");
    if (lower_bound.size() != n || upper_bound.size() != n)
        throw std::invalid_argument(std::string("EvolutionStrategy::") + op +
                                     ": bound size must match coordinates size");
    for (std::size_t j = 0; j < n; ++j)
        if (!(lower_bound[j] < upper_bound[j]))
            throw std::invalid_argument(std::string("EvolutionStrategy::") + op +
                                         ": lower_bound must be < upper_bound in every dimension");
}

} // namespace

EvolutionStrategy::EvolutionStrategy(EvolutionStrategyOptions options) : options_(options) {}

double EvolutionStrategy::optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                                    const std::vector<double>& lower_bound,
                                    const std::vector<double>& upper_bound) const {
    const std::size_t n = coordinates.size();
    validate_bounds(n, lower_bound, upper_bound, "optimize");
    if (options_.mu < 1) throw std::invalid_argument("EvolutionStrategy::optimize: mu must be at least 1");
    if (options_.offspring_size < options_.mu)
        throw std::invalid_argument("EvolutionStrategy::optimize: offspring_size must be at least mu");
    const bool comma = options_.strategy == "comma";
    if (!comma && options_.strategy != "plus")
        throw std::invalid_argument("EvolutionStrategy::optimize: strategy must be 'comma' or 'plus'");
    if (options_.initial_step_size <= 0.0)
        throw std::invalid_argument("EvolutionStrategy::optimize: initial_step_size must be positive");

    // Standard Schwefel self-adaptation learning rate for a single global step size.
    const double tau = 1.0 / std::sqrt(2.0 * std::sqrt(static_cast<double>(n)));

    std::mt19937_64 rng(options_.seed);
    std::uniform_real_distribution<double> unif01(0.0, 1.0);
    std::normal_distribution<double> standard_normal(0.0, 1.0);
    std::uniform_int_distribution<std::size_t> parent_dist(0, options_.mu - 1);

    std::vector<Individual> parents(options_.mu);
    parents[0].x.resize(n);
    for (std::size_t j = 0; j < n; ++j) parents[0].x[j] = std::clamp(coordinates[j], lower_bound[j], upper_bound[j]);
    parents[0].sigma = options_.initial_step_size;
    for (std::size_t i = 1; i < options_.mu; ++i) {
        parents[i].x.resize(n);
        for (std::size_t j = 0; j < n; ++j)
            parents[i].x[j] = lower_bound[j] + unif01(rng) * (upper_bound[j] - lower_bound[j]);
        parents[i].sigma = options_.initial_step_size;
    }
    for (auto& parent : parents) parent.value = function.evaluate(parent.x);

    std::vector<double> best = parents[0].x;
    double best_value = parents[0].value;
    for (const auto& parent : parents)
        if (parent.value < best_value) {
            best_value = parent.value;
            best = parent.x;
        }

    double prev_best = best_value;
    std::size_t stagnant = 0;

    for (std::size_t generation = 0; generation < options_.max_generations; ++generation) {
        std::vector<Individual> offspring(options_.offspring_size);
        for (auto& child : offspring) {
            const std::size_t p = parent_dist(rng);
            const double z = standard_normal(rng);
            child.sigma = parents[p].sigma * std::exp(tau * z);
            child.x.resize(n);
            for (std::size_t j = 0; j < n; ++j) {
                const double value = parents[p].x[j] + child.sigma * standard_normal(rng);
                child.x[j] = std::clamp(value, lower_bound[j], upper_bound[j]);
            }
            child.value = function.evaluate(child.x);
            if (child.value < best_value) {
                best_value = child.value;
                best = child.x;
            }
        }

        if (comma) {
            std::sort(offspring.begin(), offspring.end(),
                      [](const Individual& a, const Individual& b) { return a.value < b.value; });
            for (std::size_t i = 0; i < options_.mu; ++i) parents[i] = std::move(offspring[i]);
        } else {
            std::vector<Individual> pool;
            pool.reserve(options_.mu + options_.offspring_size);
            for (auto& parent : parents) pool.push_back(std::move(parent));
            for (auto& child : offspring) pool.push_back(std::move(child));
            std::sort(pool.begin(), pool.end(),
                      [](const Individual& a, const Individual& b) { return a.value < b.value; });
            for (std::size_t i = 0; i < options_.mu; ++i) parents[i] = std::move(pool[i]);
        }

        // Note: with "comma" selection, the current parent generation is NOT guaranteed to
        // contain the best point ever found -- comma-selection can and does discard a better
        // parent in favor of exploring, which is exactly why best-ever tracking (above) matters
        // here more than for most other methods in this module.
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
