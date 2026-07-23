#include <datamunge/optim/harmony_search.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace datamunge::optim {

namespace {

void validate_bounds(const std::size_t n, const std::vector<double>& lower_bound,
                      const std::vector<double>& upper_bound, const char* op) {
    if (n == 0) throw std::invalid_argument(std::string("HarmonySearch::") + op + ": coordinates must not be empty");
    if (lower_bound.size() != n || upper_bound.size() != n)
        throw std::invalid_argument(std::string("HarmonySearch::") + op + ": bound size must match coordinates size");
    for (std::size_t j = 0; j < n; ++j)
        if (!(lower_bound[j] < upper_bound[j]))
            throw std::invalid_argument(std::string("HarmonySearch::") + op +
                                         ": lower_bound must be < upper_bound in every dimension");
}

std::size_t worst_index(const std::vector<double>& values) {
    std::size_t w = 0;
    for (std::size_t i = 1; i < values.size(); ++i)
        if (values[i] > values[w]) w = i;
    return w;
}

std::size_t best_index(const std::vector<double>& values) {
    std::size_t b = 0;
    for (std::size_t i = 1; i < values.size(); ++i)
        if (values[i] < values[b]) b = i;
    return b;
}

} // namespace

HarmonySearch::HarmonySearch(HarmonySearchOptions options) : options_(options) {}

double HarmonySearch::optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                                const std::vector<double>& lower_bound, const std::vector<double>& upper_bound) const {
    const std::size_t n = coordinates.size();
    validate_bounds(n, lower_bound, upper_bound, "optimize");
    if (options_.population_size < 2)
        throw std::invalid_argument("HarmonySearch::optimize: population_size must be at least 2");
    if (!(options_.memory_consideration_rate > 0.0 && options_.memory_consideration_rate < 1.0))
        throw std::invalid_argument("HarmonySearch::optimize: memory_consideration_rate must be in (0, 1)");
    if (!(options_.pitch_adjustment_rate >= 0.0 && options_.pitch_adjustment_rate <= 1.0))
        throw std::invalid_argument("HarmonySearch::optimize: pitch_adjustment_rate must be in [0, 1]");
    if (!(options_.bandwidth_fraction > 0.0))
        throw std::invalid_argument("HarmonySearch::optimize: bandwidth_fraction must be positive");

    std::mt19937_64 rng(options_.seed);
    std::uniform_real_distribution<double> unif01(0.0, 1.0);
    std::uniform_real_distribution<double> unif_pm1(-1.0, 1.0);
    std::uniform_int_distribution<std::size_t> pop_dist(0, options_.population_size - 1);

    // Initialize the harmony memory: coordinates (clamped) seeds member 0, the rest are uniform
    // random within bounds.
    std::vector<std::vector<double>> hm(options_.population_size, std::vector<double>(n));
    for (auto& member : hm)
        for (std::size_t j = 0; j < n; ++j) member[j] = lower_bound[j] + unif01(rng) * (upper_bound[j] - lower_bound[j]);
    for (std::size_t j = 0; j < n; ++j) hm[0][j] = std::clamp(coordinates[j], lower_bound[j], upper_bound[j]);

    std::vector<double> values(hm.size());
    for (std::size_t i = 0; i < hm.size(); ++i) values[i] = function.evaluate(hm[i]);

    std::vector<double> bw(n);
    for (std::size_t j = 0; j < n; ++j) bw[j] = options_.bandwidth_fraction * (upper_bound[j] - lower_bound[j]);

    std::size_t best_idx = best_index(values);
    std::vector<double> best = hm[best_idx];
    double best_value = values[best_idx];

    double prev_best = best_value;
    std::size_t stagnant = 0;
    // HS improvises only one candidate per iteration (vs. a whole generation for the other
    // population methods here), so a comparable amount of "work" takes roughly population_size
    // times as many iterations -- scale the stagnation patience window accordingly.
    const std::size_t patience = 20 * options_.population_size;

    std::vector<double> x_new(n);
    for (std::size_t iter = 0; iter < options_.max_iterations; ++iter) {
        for (std::size_t d = 0; d < n; ++d) {
            const double r1 = unif01(rng);
            if (r1 < options_.memory_consideration_rate) {
                const std::size_t m = pop_dist(rng);
                x_new[d] = hm[m][d];
                const double r2 = unif01(rng);
                if (r2 < options_.pitch_adjustment_rate) {
                    x_new[d] += unif_pm1(rng) * bw[d];
                    x_new[d] = std::clamp(x_new[d], lower_bound[d], upper_bound[d]);
                }
            } else {
                x_new[d] = lower_bound[d] + unif01(rng) * (upper_bound[d] - lower_bound[d]);
            }
        }

        const double new_value = function.evaluate(x_new);
        const std::size_t w = worst_index(values);
        if (new_value < values[w]) {
            hm[w] = x_new;
            values[w] = new_value;
        }

        if (new_value < best_value) {
            best_value = new_value;
            best = x_new;
        }

        if (std::abs(prev_best - best_value) < options_.tolerance) {
            if (++stagnant > patience) break;
        } else {
            stagnant = 0;
        }
        prev_best = best_value;
    }

    coordinates = best;
    return best_value;
}

} // namespace datamunge::optim
