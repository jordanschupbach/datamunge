#include <datamunge/optim/genetic_algorithm.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace datamunge::optim {

namespace {

enum class Selection { Tournament, Roulette, Rank };
enum class Crossover { SinglePoint, Uniform, Blend };

void validate_bounds(const std::size_t n, const std::vector<double>& lower_bound,
                      const std::vector<double>& upper_bound, const char* op) {
    if (n == 0) throw std::invalid_argument(std::string("GeneticAlgorithm::") + op + ": coordinates must not be empty");
    if (lower_bound.size() != n || upper_bound.size() != n)
        throw std::invalid_argument(std::string("GeneticAlgorithm::") + op + ": bound size must match coordinates size");
    for (std::size_t j = 0; j < n; ++j)
        if (!(lower_bound[j] < upper_bound[j]))
            throw std::invalid_argument(std::string("GeneticAlgorithm::") + op +
                                         ": lower_bound must be < upper_bound in every dimension");
}

std::size_t best_index(const std::vector<double>& values) {
    std::size_t b = 0;
    for (std::size_t i = 1; i < values.size(); ++i)
        if (values[i] < values[b]) b = i;
    return b;
}

} // namespace

GeneticAlgorithm::GeneticAlgorithm(GAOptions options) : options_(options) {}

double GeneticAlgorithm::optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                                   const std::vector<double>& lower_bound, const std::vector<double>& upper_bound) const {
    const std::size_t n = coordinates.size();
    validate_bounds(n, lower_bound, upper_bound, "optimize");
    if (options_.population_size < 2) throw std::invalid_argument("GeneticAlgorithm::optimize: population_size must be at least 2");

    Selection selection;
    if (options_.selection_strategy == "tournament") selection = Selection::Tournament;
    else if (options_.selection_strategy == "roulette") selection = Selection::Roulette;
    else if (options_.selection_strategy == "rank") selection = Selection::Rank;
    else
        throw std::invalid_argument("GeneticAlgorithm::optimize: unknown selection_strategy '" +
                                     options_.selection_strategy + "'");

    Crossover crossover;
    if (options_.crossover_strategy == "single_point") crossover = Crossover::SinglePoint;
    else if (options_.crossover_strategy == "uniform") crossover = Crossover::Uniform;
    else if (options_.crossover_strategy == "blend") crossover = Crossover::Blend;
    else
        throw std::invalid_argument("GeneticAlgorithm::optimize: unknown crossover_strategy '" +
                                     options_.crossover_strategy + "'");

    std::mt19937_64 rng(options_.seed);
    std::uniform_real_distribution<double> unif01(0.0, 1.0);
    std::uniform_int_distribution<std::size_t> pop_dist(0, options_.population_size - 1);

    std::vector<std::vector<double>> pop(options_.population_size, std::vector<double>(n));
    for (auto& individual : pop)
        for (std::size_t j = 0; j < n; ++j) individual[j] = lower_bound[j] + unif01(rng) * (upper_bound[j] - lower_bound[j]);
    for (std::size_t j = 0; j < n; ++j) pop[0][j] = std::clamp(coordinates[j], lower_bound[j], upper_bound[j]);

    std::vector<double> values(pop.size());
    for (std::size_t i = 0; i < pop.size(); ++i) values[i] = function.evaluate(pop[i]);

    double prev_best = values[best_index(values)];
    std::size_t stagnant = 0;

    for (std::size_t gen = 0; gen < options_.max_generations; ++gen) {
        std::vector<double> selection_weights;
        if (selection == Selection::Roulette) {
            const double max_val = *std::max_element(values.begin(), values.end());
            selection_weights.resize(pop.size());
            for (std::size_t i = 0; i < pop.size(); ++i) selection_weights[i] = (max_val - values[i]) + 1e-9;
        } else if (selection == Selection::Rank) {
            std::vector<std::size_t> order(pop.size());
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](const std::size_t a, const std::size_t b) { return values[a] < values[b]; });
            selection_weights.resize(pop.size());
            for (std::size_t rank = 0; rank < pop.size(); ++rank)
                selection_weights[order[rank]] = static_cast<double>(pop.size() - rank);
        }
        std::discrete_distribution<std::size_t> weighted_dist(selection_weights.begin(), selection_weights.end());

        auto select_parent = [&]() -> std::size_t {
            if (selection == Selection::Tournament) {
                std::size_t best = pop_dist(rng);
                for (std::size_t k = 1; k < options_.tournament_size; ++k) {
                    const std::size_t cand = pop_dist(rng);
                    if (values[cand] < values[best]) best = cand;
                }
                return best;
            }
            return weighted_dist(rng);
        };

        std::vector<std::vector<double>> next_pop;
        std::vector<double> next_values;
        next_pop.reserve(pop.size());
        next_values.reserve(pop.size());

        if (options_.elitism) {
            std::vector<std::size_t> order(pop.size());
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](const std::size_t a, const std::size_t b) { return values[a] < values[b]; });
            for (std::size_t k = 0; k < std::min(options_.elite_count, pop.size()); ++k) {
                next_pop.push_back(pop[order[k]]);
                next_values.push_back(values[order[k]]);
            }
        }

        while (next_pop.size() < pop.size()) {
            const std::size_t p1 = select_parent();
            const std::size_t p2 = select_parent();
            std::vector<double> child1 = pop[p1];
            std::vector<double> child2 = pop[p2];

            if (unif01(rng) < options_.crossover_rate) {
                switch (crossover) {
                    case Crossover::SinglePoint: {
                        const std::size_t cut = n > 1 ? std::uniform_int_distribution<std::size_t>(1, n - 1)(rng) : 1;
                        for (std::size_t j = cut; j < n; ++j) std::swap(child1[j], child2[j]);
                        break;
                    }
                    case Crossover::Uniform: {
                        for (std::size_t j = 0; j < n; ++j)
                            if (unif01(rng) < 0.5) std::swap(child1[j], child2[j]);
                        break;
                    }
                    case Crossover::Blend: {
                        for (std::size_t j = 0; j < n; ++j) {
                            const double lo = std::min(pop[p1][j], pop[p2][j]);
                            const double hi = std::max(pop[p1][j], pop[p2][j]);
                            const double d = hi - lo;
                            const double range_lo = lo - options_.blend_alpha * d;
                            const double range_hi = hi + options_.blend_alpha * d;
                            child1[j] = range_lo + unif01(rng) * (range_hi - range_lo);
                            child2[j] = range_lo + unif01(rng) * (range_hi - range_lo);
                        }
                        break;
                    }
                }
            }

            for (std::vector<double>* child : {&child1, &child2}) {
                for (std::size_t j = 0; j < n; ++j) {
                    if (unif01(rng) < options_.mutation_rate) {
                        std::normal_distribution<double> gauss(0.0, options_.mutation_std_dev * (upper_bound[j] - lower_bound[j]));
                        (*child)[j] += gauss(rng);
                    }
                    (*child)[j] = std::clamp((*child)[j], lower_bound[j], upper_bound[j]);
                }
            }

            next_pop.push_back(child1);
            next_values.push_back(function.evaluate(child1));
            if (next_pop.size() < pop.size()) {
                next_pop.push_back(child2);
                next_values.push_back(function.evaluate(child2));
            }
        }

        pop = std::move(next_pop);
        values = std::move(next_values);

        const double current_best = values[best_index(values)];
        if (std::abs(prev_best - current_best) < options_.tolerance) {
            if (++stagnant > 20) break;
        } else {
            stagnant = 0;
        }
        prev_best = current_best;
    }

    const std::size_t final_best = best_index(values);
    coordinates = pop[final_best];
    return values[final_best];
}

} // namespace datamunge::optim
