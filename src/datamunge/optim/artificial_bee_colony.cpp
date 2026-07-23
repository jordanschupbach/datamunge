#include <datamunge/optim/artificial_bee_colony.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

namespace datamunge::optim {

namespace {

void validate_bounds(const std::size_t n, const std::vector<double>& lower_bound,
                      const std::vector<double>& upper_bound, const char* op) {
    if (n == 0)
        throw std::invalid_argument(std::string("ArtificialBeeColony::") + op + ": coordinates must not be empty");
    if (lower_bound.size() != n || upper_bound.size() != n)
        throw std::invalid_argument(std::string("ArtificialBeeColony::") + op +
                                     ": bound size must match coordinates size");
    for (std::size_t j = 0; j < n; ++j)
        if (!(lower_bound[j] < upper_bound[j]))
            throw std::invalid_argument(std::string("ArtificialBeeColony::") + op +
                                         ": lower_bound must be < upper_bound in every dimension");
}

} // namespace

ArtificialBeeColony::ArtificialBeeColony(ArtificialBeeColonyOptions options) : options_(options) {}

double ArtificialBeeColony::optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                                      const std::vector<double>& lower_bound,
                                      const std::vector<double>& upper_bound) const {
    const std::size_t n = coordinates.size();
    validate_bounds(n, lower_bound, upper_bound, "optimize");
    if (options_.population_size < 2)
        throw std::invalid_argument("ArtificialBeeColony::optimize: population_size must be at least 2");

    const std::size_t pop_size = options_.population_size;
    const std::size_t limit = options_.abandonment_limit == 0 ? pop_size * n : options_.abandonment_limit;

    std::mt19937_64 rng(options_.seed);
    std::uniform_real_distribution<double> unif01(0.0, 1.0);
    std::uniform_real_distribution<double> phi_dist(-1.0, 1.0);
    std::uniform_int_distribution<std::size_t> dim_dist(0, n - 1);
    std::uniform_int_distribution<std::size_t> pop_dist(0, pop_size - 1);

    // Initialize food sources uniformly at random within bounds; seed one with the caller's guess.
    std::vector<std::vector<double>> pop(pop_size, std::vector<double>(n));
    for (auto& source : pop)
        for (std::size_t j = 0; j < n; ++j) source[j] = lower_bound[j] + unif01(rng) * (upper_bound[j] - lower_bound[j]);
    for (std::size_t j = 0; j < n; ++j) pop[0][j] = std::clamp(coordinates[j], lower_bound[j], upper_bound[j]);

    std::vector<double> values(pop_size);
    for (std::size_t i = 0; i < pop_size; ++i) values[i] = function.evaluate(pop[i]);

    std::vector<std::size_t> trial(pop_size, 0);

    std::size_t best_idx = 0;
    for (std::size_t i = 1; i < pop_size; ++i)
        if (values[i] < values[best_idx]) best_idx = i;
    std::vector<double> best = pop[best_idx];
    double best_value = values[best_idx];

    // A random population index other than i.
    auto random_other = [&](const std::size_t i) -> std::size_t {
        std::size_t k = pop_dist(rng);
        while (k == i) k = pop_dist(rng);
        return k;
    };

    // Perturbs food source i along one random dimension toward a random other food source,
    // evaluates the candidate, and greedily replaces x_i (resetting its trial counter) on
    // improvement, else increments the trial counter. Shared by the employed and onlooker
    // phases, which generate candidates identically.
    auto explore = [&](const std::size_t i) {
        const std::size_t j = dim_dist(rng);
        const std::size_t k = random_other(i);
        const double phi = phi_dist(rng);
        std::vector<double> v = pop[i];
        v[j] = pop[i][j] + phi * (pop[i][j] - pop[k][j]);
        v[j] = std::clamp(v[j], lower_bound[j], upper_bound[j]);
        const double fv = function.evaluate(v);
        if (fv < values[i]) {
            pop[i] = std::move(v);
            values[i] = fv;
            trial[i] = 0;
        } else {
            ++trial[i];
        }
        if (values[i] < best_value) {
            best_value = values[i];
            best = pop[i];
        }
    };

    double prev_best = best_value;
    std::size_t stagnant = 0;

    for (std::size_t gen = 0; gen < options_.max_generations; ++gen) {
        // 1. Employed bee phase: every food source is exploited once.
        for (std::size_t i = 0; i < pop_size; ++i) explore(i);

        // 2. Selection "fitness" for minimization: lower objective value => higher fitness.
        //    std::discrete_distribution normalizes weights internally, so constructing it
        //    directly from the fitness values is equivalent to roulette-wheel sampling over
        //    the explicit probabilities p_i = fit_i / sum_j(fit_j).
        std::vector<double> fitness(pop_size);
        for (std::size_t i = 0; i < pop_size; ++i)
            fitness[i] = values[i] >= 0.0 ? 1.0 / (1.0 + values[i]) : 1.0 + std::abs(values[i]);
        std::discrete_distribution<std::size_t> roulette(fitness.begin(), fitness.end());

        // 3. Onlooker bee phase: population_size onlookers, each choosing a source by roulette.
        for (std::size_t b = 0; b < pop_size; ++b) explore(roulette(rng));

        // 4. Scout bee phase: replace any food source stagnant for too long.
        for (std::size_t i = 0; i < pop_size; ++i) {
            if (trial[i] > limit) {
                for (std::size_t j = 0; j < n; ++j)
                    pop[i][j] = lower_bound[j] + unif01(rng) * (upper_bound[j] - lower_bound[j]);
                values[i] = function.evaluate(pop[i]);
                trial[i] = 0;
                if (values[i] < best_value) {
                    best_value = values[i];
                    best = pop[i];
                }
            }
        }

        // 6. Stopping: no improvement in the best-ever value for 20 consecutive generations.
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
