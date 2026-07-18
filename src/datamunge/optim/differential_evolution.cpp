#include <datamunge/optim/differential_evolution.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace datamunge::optim {

namespace {

enum class Mutation { Rand1, Best1, CurrentToBest1, Rand2 };

void validate_bounds(const std::size_t n, const std::vector<double>& lower_bound,
                      const std::vector<double>& upper_bound, const char* op) {
    if (n == 0) throw std::invalid_argument(std::string("DifferentialEvolution::") + op + ": coordinates must not be empty");
    if (lower_bound.size() != n || upper_bound.size() != n)
        throw std::invalid_argument(std::string("DifferentialEvolution::") + op + ": bound size must match coordinates size");
    for (std::size_t j = 0; j < n; ++j)
        if (!(lower_bound[j] < upper_bound[j]))
            throw std::invalid_argument(std::string("DifferentialEvolution::") + op +
                                         ": lower_bound must be < upper_bound in every dimension");
}

std::vector<std::size_t> random_distinct_indices(std::mt19937_64& rng, const std::size_t population_size,
                                                   const std::size_t exclude, const std::size_t count) {
    std::uniform_int_distribution<std::size_t> dist(0, population_size - 1);
    std::vector<std::size_t> result;
    result.reserve(count);
    while (result.size() < count) {
        const std::size_t r = dist(rng);
        if (r == exclude) continue;
        if (std::find(result.begin(), result.end(), r) != result.end()) continue;
        result.push_back(r);
    }
    return result;
}

} // namespace

DifferentialEvolution::DifferentialEvolution(DEOptions options) : options_(options) {}

double DifferentialEvolution::optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                                        const std::vector<double>& lower_bound,
                                        const std::vector<double>& upper_bound) const {
    const std::size_t n = coordinates.size();
    validate_bounds(n, lower_bound, upper_bound, "optimize");

    Mutation mutation;
    if (options_.mutation_strategy == "rand1") mutation = Mutation::Rand1;
    else if (options_.mutation_strategy == "best1") mutation = Mutation::Best1;
    else if (options_.mutation_strategy == "current_to_best1") mutation = Mutation::CurrentToBest1;
    else if (options_.mutation_strategy == "rand2") mutation = Mutation::Rand2;
    else
        throw std::invalid_argument("DifferentialEvolution::optimize: unknown mutation_strategy '" +
                                     options_.mutation_strategy + "'");

    const bool exponential = options_.crossover_strategy == "exponential";
    if (!exponential && options_.crossover_strategy != "binomial")
        throw std::invalid_argument("DifferentialEvolution::optimize: crossover_strategy must be 'binomial' or 'exponential'");

    const std::size_t needed_distinct =
        (mutation == Mutation::Rand2) ? 5 : (mutation == Mutation::CurrentToBest1 ? 2 : 3);
    if (options_.population_size < needed_distinct + 1)
        throw std::invalid_argument(
            "DifferentialEvolution::optimize: population_size too small for the chosen mutation_strategy");

    std::mt19937_64 rng(options_.seed);
    std::uniform_real_distribution<double> unif01(0.0, 1.0);
    std::uniform_int_distribution<std::size_t> dim_dist(0, n - 1);

    std::vector<std::vector<double>> pop(options_.population_size, std::vector<double>(n));
    std::vector<double> values(options_.population_size);
    for (auto& individual : pop)
        for (std::size_t j = 0; j < n; ++j) individual[j] = lower_bound[j] + unif01(rng) * (upper_bound[j] - lower_bound[j]);
    for (std::size_t j = 0; j < n; ++j) pop[0][j] = std::clamp(coordinates[j], lower_bound[j], upper_bound[j]);
    for (std::size_t i = 0; i < pop.size(); ++i) values[i] = function.evaluate(pop[i]);

    std::size_t best_idx = 0;
    for (std::size_t i = 1; i < pop.size(); ++i)
        if (values[i] < values[best_idx]) best_idx = i;

    double prev_best = values[best_idx];
    std::size_t stagnant = 0;

    for (std::size_t gen = 0; gen < options_.max_generations; ++gen) {
        for (std::size_t i = 0; i < pop.size(); ++i) {
            std::vector<double> mutant(n);
            switch (mutation) {
                case Mutation::Rand1: {
                    const auto idx = random_distinct_indices(rng, pop.size(), i, 3);
                    for (std::size_t j = 0; j < n; ++j)
                        mutant[j] = pop[idx[0]][j] + options_.differential_weight * (pop[idx[1]][j] - pop[idx[2]][j]);
                    break;
                }
                case Mutation::Best1: {
                    const auto idx = random_distinct_indices(rng, pop.size(), i, 2);
                    for (std::size_t j = 0; j < n; ++j)
                        mutant[j] = pop[best_idx][j] + options_.differential_weight * (pop[idx[0]][j] - pop[idx[1]][j]);
                    break;
                }
                case Mutation::CurrentToBest1: {
                    const auto idx = random_distinct_indices(rng, pop.size(), i, 2);
                    for (std::size_t j = 0; j < n; ++j)
                        mutant[j] = pop[i][j] + options_.differential_weight * (pop[best_idx][j] - pop[i][j]) +
                                    options_.differential_weight * (pop[idx[0]][j] - pop[idx[1]][j]);
                    break;
                }
                case Mutation::Rand2: {
                    const auto idx = random_distinct_indices(rng, pop.size(), i, 5);
                    for (std::size_t j = 0; j < n; ++j)
                        mutant[j] = pop[idx[0]][j] + options_.differential_weight * (pop[idx[1]][j] - pop[idx[2]][j]) +
                                    options_.differential_weight * (pop[idx[3]][j] - pop[idx[4]][j]);
                    break;
                }
            }
            for (std::size_t j = 0; j < n; ++j) mutant[j] = std::clamp(mutant[j], lower_bound[j], upper_bound[j]);

            std::vector<double> trial = pop[i];
            if (exponential) {
                std::size_t j = dim_dist(rng);
                std::size_t count = 0;
                do {
                    trial[j] = mutant[j];
                    j = (j + 1) % n;
                    ++count;
                } while (unif01(rng) < options_.crossover_rate && count < n);
            } else {
                const std::size_t j_rand = dim_dist(rng);
                for (std::size_t j = 0; j < n; ++j)
                    if (j == j_rand || unif01(rng) < options_.crossover_rate) trial[j] = mutant[j];
            }

            const double trial_value = function.evaluate(trial);
            if (trial_value <= values[i]) {
                pop[i] = trial;
                values[i] = trial_value;
                if (trial_value < values[best_idx]) best_idx = i;
            }
        }

        if (std::abs(prev_best - values[best_idx]) < options_.tolerance) {
            if (++stagnant > 20) break;
        } else {
            stagnant = 0;
        }
        prev_best = values[best_idx];
    }

    coordinates = pop[best_idx];
    return values[best_idx];
}

} // namespace datamunge::optim
