#include <datamunge/optim/spea2.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace datamunge::optim {

namespace {

struct Individual {
    std::vector<double> x;
    std::vector<double> f;
    double fitness{0.0};
};

void validate_bounds(const std::vector<double>& lower, const std::vector<double>& upper) {
    const std::size_t n = lower.size();
    if (n == 0) throw std::invalid_argument("SPEA2::optimize: bounds must not be empty");
    if (upper.size() != n) throw std::invalid_argument("SPEA2::optimize: lower/upper bound sizes must match");
    for (std::size_t j = 0; j < n; ++j)
        if (!(lower[j] < upper[j]))
            throw std::invalid_argument("SPEA2::optimize: lower_bound must be < upper_bound in every dimension");
}

double objective_distance(const Individual& a, const Individual& b) {
    double s = 0.0;
    for (std::size_t k = 0; k < a.f.size(); ++k) {
        const double d = a.f[k] - b.f[k];
        s += d * d;
    }
    return std::sqrt(s);
}

// SPEA2 fitness: F(i) = raw(i) + density(i). raw(i) is the sum of the strengths (dominated
// counts) of every solution dominating i (0 iff i is non-dominated); density(i) =
// 1 / (dist to the k-th nearest neighbor + 2), k = floor(sqrt(|pool|)).
void assign_fitness(std::vector<Individual>& pool) {
    const std::size_t P = pool.size();
    std::vector<int> strength(P, 0);
    std::vector<std::vector<bool>> dom(P, std::vector<bool>(P, false));
    for (std::size_t i = 0; i < P; ++i)
        for (std::size_t j = 0; j < P; ++j)
            if (i != j && dominates(pool[i].f, pool[j].f)) {
                dom[i][j] = true;
                ++strength[i];
            }

    for (std::size_t i = 0; i < P; ++i) {
        double raw = 0.0;
        for (std::size_t j = 0; j < P; ++j)
            if (dom[j][i]) raw += strength[j];

        std::vector<double> dists;
        dists.reserve(P - 1);
        for (std::size_t j = 0; j < P; ++j)
            if (i != j) dists.push_back(objective_distance(pool[i], pool[j]));
        std::sort(dists.begin(), dists.end());
        const std::size_t k = static_cast<std::size_t>(std::sqrt(static_cast<double>(P)));
        const double kth = dists.empty() ? 0.0 : dists[std::min(k, dists.size() - 1)];
        pool[i].fitness = raw + 1.0 / (kth + 2.0);
    }
}

// Environmental selection: keep every non-dominated solution (fitness < 1); then fill from the
// best dominated ones, or truncate the most crowded ones, to reach exactly `cap`.
std::vector<Individual> environmental_selection(std::vector<Individual> pool, std::size_t cap) {
    std::vector<Individual> archive;
    std::vector<Individual> dominated;
    for (auto& ind : pool) {
        if (ind.fitness < 1.0) archive.push_back(std::move(ind));
        else dominated.push_back(std::move(ind));
    }

    if (archive.size() < cap) {
        std::sort(dominated.begin(), dominated.end(),
                  [](const Individual& a, const Individual& b) { return a.fitness < b.fitness; });
        for (std::size_t i = 0; i < dominated.size() && archive.size() < cap; ++i)
            archive.push_back(std::move(dominated[i]));
        return archive;
    }

    // Truncation: repeatedly remove the individual whose distance to its nearest neighbor is
    // smallest (ties broken by the next-nearest, and so on).
    while (archive.size() > cap) {
        const std::size_t A = archive.size();
        std::vector<std::vector<double>> sorted_d(A);
        for (std::size_t i = 0; i < A; ++i) {
            sorted_d[i].reserve(A - 1);
            for (std::size_t j = 0; j < A; ++j)
                if (i != j) sorted_d[i].push_back(objective_distance(archive[i], archive[j]));
            std::sort(sorted_d[i].begin(), sorted_d[i].end());
        }
        std::size_t victim = 0;
        for (std::size_t i = 1; i < A; ++i) {
            // Lexicographic comparison of the sorted nearest-neighbor distance lists.
            bool i_more_crowded = false, decided = false;
            for (std::size_t t = 0; t < sorted_d[i].size(); ++t) {
                if (sorted_d[i][t] < sorted_d[victim][t]) { i_more_crowded = true; decided = true; break; }
                if (sorted_d[i][t] > sorted_d[victim][t]) { decided = true; break; }
            }
            if (decided && i_more_crowded) victim = i;
        }
        archive.erase(archive.begin() + static_cast<std::ptrdiff_t>(victim));
    }
    return archive;
}

} // namespace

SPEA2::SPEA2(SPEA2Options options) : options_(options) {}

std::vector<ParetoPoint> SPEA2::optimize(MultiObjectiveFunction& function, const std::vector<double>& lower_bound,
                                         const std::vector<double>& upper_bound) const {
    validate_bounds(lower_bound, upper_bound);
    const std::size_t n = lower_bound.size();
    const std::size_t N = std::max<std::size_t>(options_.population_size, 2);
    const std::size_t Nbar = options_.archive_size == 0 ? N : options_.archive_size;
    const double mut_rate = options_.mutation_rate < 0.0 ? 1.0 / static_cast<double>(n) : options_.mutation_rate;

    std::mt19937_64 rng(options_.seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    auto random_point = [&]() {
        std::vector<double> x(n);
        for (std::size_t j = 0; j < n; ++j) x[j] = lower_bound[j] + u(rng) * (upper_bound[j] - lower_bound[j]);
        return x;
    };
    auto evaluate = [&](std::vector<double> x) {
        Individual ind;
        ind.f = function.evaluate(x);
        ind.x = std::move(x);
        return ind;
    };

    std::vector<Individual> population;
    population.reserve(N);
    for (std::size_t i = 0; i < N; ++i) population.push_back(evaluate(random_point()));
    std::vector<Individual> archive;

    for (std::size_t gen = 0; gen < options_.max_generations; ++gen) {
        std::vector<Individual> pool;
        pool.reserve(population.size() + archive.size());
        pool.insert(pool.end(), population.begin(), population.end());
        pool.insert(pool.end(), archive.begin(), archive.end());
        assign_fitness(pool);
        archive = environmental_selection(std::move(pool), Nbar);

        if (gen + 1 == options_.max_generations) break;

        // Mating selection: binary tournament on the archive (lower fitness wins), then vary.
        auto tournament = [&]() -> const Individual& {
            std::uniform_int_distribution<std::size_t> pick(0, archive.size() - 1);
            const std::size_t a = pick(rng), b = pick(rng);
            return archive[a].fitness <= archive[b].fitness ? archive[a] : archive[b];
        };
        std::vector<Individual> next;
        next.reserve(N);
        while (next.size() < N) {
            const Individual& p1 = tournament();
            const Individual& p2 = tournament();
            std::vector<double> c1, c2;
            detail::sbx_crossover(p1.x, p2.x, c1, c2, lower_bound, upper_bound, options_.crossover_rate,
                                  options_.eta_crossover, rng);
            detail::polynomial_mutation(c1, lower_bound, upper_bound, mut_rate, options_.eta_mutation, rng);
            next.push_back(evaluate(std::move(c1)));
            if (next.size() < N) {
                detail::polynomial_mutation(c2, lower_bound, upper_bound, mut_rate, options_.eta_mutation, rng);
                next.push_back(evaluate(std::move(c2)));
            }
        }
        population = std::move(next);
    }

    std::vector<ParetoPoint> points;
    points.reserve(archive.size());
    for (auto& ind : archive) points.push_back(ParetoPoint{std::move(ind.x), std::move(ind.f)});
    return non_dominated_front(points);
}

} // namespace datamunge::optim
