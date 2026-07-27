#include <datamunge/optim/nsga2.hpp>

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

namespace datamunge::optim {

namespace {

struct Individual {
    std::vector<double> x;
    std::vector<double> f;
    int rank{0};
    double crowding{0.0};
};

void validate_bounds(const std::vector<double>& lower, const std::vector<double>& upper) {
    const std::size_t n = lower.size();
    if (n == 0) throw std::invalid_argument("NSGA2::optimize: bounds must not be empty");
    if (upper.size() != n) throw std::invalid_argument("NSGA2::optimize: lower/upper bound sizes must match");
    for (std::size_t j = 0; j < n; ++j)
        if (!(lower[j] < upper[j]))
            throw std::invalid_argument("NSGA2::optimize: lower_bound must be < upper_bound in every dimension");
}

// Partition a population into Pareto fronts (Deb et al. 2002, fast non-dominated sort, O(M N^2))
// and write each individual's front index into `rank`. Returns the fronts as index lists.
std::vector<std::vector<std::size_t>> fast_non_dominated_sort(std::vector<Individual>& pop) {
    const std::size_t N = pop.size();
    std::vector<std::vector<std::size_t>> dominated(N);   // solutions each p dominates
    std::vector<int> dom_count(N, 0);                     // how many dominate p
    std::vector<std::vector<std::size_t>> fronts(1);

    for (std::size_t p = 0; p < N; ++p) {
        for (std::size_t q = 0; q < N; ++q) {
            if (p == q) continue;
            if (dominates(pop[p].f, pop[q].f)) dominated[p].push_back(q);
            else if (dominates(pop[q].f, pop[p].f)) ++dom_count[p];
        }
        if (dom_count[p] == 0) {
            pop[p].rank = 0;
            fronts[0].push_back(p);
        }
    }

    std::size_t fi = 0;
    while (!fronts[fi].empty()) {
        std::vector<std::size_t> next;
        for (const std::size_t p : fronts[fi]) {
            for (const std::size_t q : dominated[p]) {
                if (--dom_count[q] == 0) {
                    pop[q].rank = static_cast<int>(fi + 1);
                    next.push_back(q);
                }
            }
        }
        ++fi;
        fronts.push_back(next);
    }
    fronts.pop_back(); // last pushed front is empty
    return fronts;
}

// Crowding distance within one front: the sum, over objectives, of the normalized gap between
// each solution's two objective-space neighbors; boundary solutions get an infinite distance so
// the extremes of the front are always preserved.
void assign_crowding_distance(std::vector<Individual>& pop, const std::vector<std::size_t>& front) {
    const std::size_t sz = front.size();
    for (const std::size_t i : front) pop[i].crowding = 0.0;
    if (sz == 0) return;
    const std::size_t m = pop[front[0]].f.size();
    const double inf = std::numeric_limits<double>::infinity();

    for (std::size_t k = 0; k < m; ++k) {
        std::vector<std::size_t> order(front);
        std::sort(order.begin(), order.end(),
                  [&](std::size_t a, std::size_t b) { return pop[a].f[k] < pop[b].f[k]; });
        pop[order.front()].crowding = inf;
        pop[order.back()].crowding = inf;
        const double range = pop[order.back()].f[k] - pop[order.front()].f[k];
        if (range <= 0.0) continue;
        for (std::size_t idx = 1; idx + 1 < sz; ++idx) {
            if (pop[order[idx]].crowding == inf) continue;
            pop[order[idx]].crowding += (pop[order[idx + 1]].f[k] - pop[order[idx - 1]].f[k]) / range;
        }
    }
}

// Crowded-comparison operator: lower Pareto rank wins; ties broken by larger crowding distance.
bool crowded_less(const Individual& a, const Individual& b) {
    if (a.rank != b.rank) return a.rank < b.rank;
    return a.crowding > b.crowding;
}

} // namespace

NSGA2::NSGA2(NSGA2Options options) : options_(options) {}

std::vector<ParetoPoint> NSGA2::optimize(MultiObjectiveFunction& function, const std::vector<double>& lower_bound,
                                         const std::vector<double>& upper_bound) const {
    validate_bounds(lower_bound, upper_bound);
    const std::size_t n = lower_bound.size();
    std::size_t N = std::max<std::size_t>(options_.population_size, 2);
    if (N % 2 != 0) ++N; // even population for exact pairing
    const double mut_rate = options_.mutation_rate < 0.0 ? 1.0 / static_cast<double>(n) : options_.mutation_rate;

    std::mt19937_64 rng(options_.seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    auto random_point = [&]() {
        std::vector<double> x(n);
        for (std::size_t j = 0; j < n; ++j) x[j] = lower_bound[j] + u(rng) * (upper_bound[j] - lower_bound[j]);
        return x;
    };

    std::vector<Individual> pop(N);
    for (auto& ind : pop) {
        ind.x = random_point();
        ind.f = function.evaluate(ind.x);
    }
    {
        auto fronts = fast_non_dominated_sort(pop);
        for (const auto& fr : fronts) assign_crowding_distance(pop, fr);
    }

    auto tournament = [&](const std::vector<Individual>& p) -> const Individual& {
        std::uniform_int_distribution<std::size_t> pick(0, p.size() - 1);
        const std::size_t a = pick(rng), b = pick(rng);
        return crowded_less(p[a], p[b]) ? p[a] : p[b];
    };

    for (std::size_t gen = 0; gen < options_.max_generations; ++gen) {
        // ---- Create offspring via crowded binary tournament + SBX + polynomial mutation ----
        std::vector<Individual> offspring;
        offspring.reserve(N);
        while (offspring.size() < N) {
            const Individual& parent1 = tournament(pop);
            const Individual& parent2 = tournament(pop);
            std::vector<double> c1, c2;
            detail::sbx_crossover(parent1.x, parent2.x, c1, c2, lower_bound, upper_bound, options_.crossover_rate,
                                  options_.eta_crossover, rng);
            detail::polynomial_mutation(c1, lower_bound, upper_bound, mut_rate, options_.eta_mutation, rng);
            detail::polynomial_mutation(c2, lower_bound, upper_bound, mut_rate, options_.eta_mutation, rng);
            Individual child1{std::move(c1), {}, 0, 0.0};
            child1.f = function.evaluate(child1.x);
            offspring.push_back(std::move(child1));
            if (offspring.size() < N) {
                Individual child2{std::move(c2), {}, 0, 0.0};
                child2.f = function.evaluate(child2.x);
                offspring.push_back(std::move(child2));
            }
        }

        // ---- Elitist survival: rank the combined 2N pool, fill N front by front ----
        std::vector<Individual> combined;
        combined.reserve(2 * N);
        combined.insert(combined.end(), pop.begin(), pop.end());
        combined.insert(combined.end(), std::make_move_iterator(offspring.begin()),
                        std::make_move_iterator(offspring.end()));

        auto fronts = fast_non_dominated_sort(combined);
        std::vector<Individual> next;
        next.reserve(N);
        for (const auto& fr : fronts) {
            assign_crowding_distance(combined, fr);
            if (next.size() + fr.size() <= N) {
                for (const std::size_t i : fr) next.push_back(combined[i]);
            } else {
                // Partially fill from this front, taking the most crowding-distant first.
                std::vector<std::size_t> order(fr);
                std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
                    return combined[a].crowding > combined[b].crowding;
                });
                for (std::size_t k = 0; next.size() < N; ++k) next.push_back(combined[order[k]]);
                break;
            }
            if (next.size() == N) break;
        }
        pop = std::move(next);
    }

    std::vector<ParetoPoint> points;
    points.reserve(pop.size());
    for (auto& ind : pop) points.push_back(ParetoPoint{std::move(ind.x), std::move(ind.f)});
    return non_dominated_front(points);
}

} // namespace datamunge::optim
