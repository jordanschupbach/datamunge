#include <datamunge/optim/moead.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace datamunge::optim {

namespace {

void validate_bounds(const std::vector<double>& lower, const std::vector<double>& upper) {
    const std::size_t n = lower.size();
    if (n == 0) throw std::invalid_argument("MOEAD::optimize: bounds must not be empty");
    if (upper.size() != n) throw std::invalid_argument("MOEAD::optimize: lower/upper bound sizes must match");
    for (std::size_t j = 0; j < n; ++j)
        if (!(lower[j] < upper[j]))
            throw std::invalid_argument("MOEAD::optimize: lower_bound must be < upper_bound in every dimension");
}

// Number of Das-Dennis lattice points for m objectives at H divisions: C(H + m - 1, m - 1).
std::size_t lattice_count(std::size_t H, std::size_t m) {
    // Compute the binomial coefficient with a small loop (values here are tiny).
    const std::size_t k = m - 1;
    double c = 1.0;
    for (std::size_t i = 0; i < k; ++i) c = c * static_cast<double>(H + m - 1 - i) / static_cast<double>(i + 1);
    return static_cast<std::size_t>(std::llround(c));
}

// Recursively enumerate all m-tuples of nonnegative integers summing to H, as weight vectors
// (each divided by H). This is the Das & Dennis (1998) structured simplex-lattice design.
void das_dennis(std::size_t left, std::size_t depth, std::size_t m, std::size_t H, std::vector<std::size_t>& partial,
                std::vector<std::vector<double>>& out) {
    if (depth == m - 1) {
        partial[depth] = left;
        std::vector<double> w(m);
        for (std::size_t i = 0; i < m; ++i) w[i] = static_cast<double>(partial[i]) / static_cast<double>(H);
        out.push_back(std::move(w));
        return;
    }
    for (std::size_t i = 0; i <= left; ++i) {
        partial[depth] = i;
        das_dennis(left - i, depth + 1, m, H, partial, out);
    }
}

std::vector<std::vector<double>> make_weights(std::size_t requested, std::size_t m) {
    // Choose the largest H whose lattice size does not exceed the requested population.
    std::size_t H = 1;
    while (lattice_count(H + 1, m) <= std::max<std::size_t>(requested, m)) ++H;
    std::vector<std::vector<double>> weights;
    std::vector<std::size_t> partial(m, 0);
    das_dennis(H, 0, m, H, partial, weights);
    return weights;
}

} // namespace

MOEAD::MOEAD(MOEADOptions options) : options_(options) {}

std::vector<ParetoPoint> MOEAD::optimize(MultiObjectiveFunction& function, const std::vector<double>& lower_bound,
                                         const std::vector<double>& upper_bound) const {
    validate_bounds(lower_bound, upper_bound);
    const bool weighted_sum = options_.decomposition == "weighted_sum";
    if (!weighted_sum && options_.decomposition != "tchebycheff")
        throw std::invalid_argument("MOEAD::optimize: decomposition must be 'tchebycheff' or 'weighted_sum'");

    const std::size_t n = lower_bound.size();
    const double mut_rate = options_.mutation_rate < 0.0 ? 1.0 / static_cast<double>(n) : options_.mutation_rate;

    std::mt19937_64 rng(options_.seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    auto random_point = [&]() {
        std::vector<double> x(n);
        for (std::size_t j = 0; j < n; ++j) x[j] = lower_bound[j] + u(rng) * (upper_bound[j] - lower_bound[j]);
        return x;
    };

    // ---- Decision variables: weight vectors, neighborhoods, initial population, ideal point ----
    std::vector<double> f0 = function.evaluate(random_point());
    const std::size_t m = f0.size();
    const std::vector<std::vector<double>> weights = make_weights(options_.population_size, m);
    const std::size_t Npop = weights.size();
    const std::size_t T = std::min(std::max<std::size_t>(options_.neighborhood_size, 2), Npop);

    // Neighborhoods: the T nearest weight vectors (Euclidean) of each subproblem.
    std::vector<std::vector<std::size_t>> neighbors(Npop);
    for (std::size_t i = 0; i < Npop; ++i) {
        std::vector<std::size_t> order(Npop);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            double da = 0.0, db = 0.0;
            for (std::size_t k = 0; k < m; ++k) {
                da += (weights[i][k] - weights[a][k]) * (weights[i][k] - weights[a][k]);
                db += (weights[i][k] - weights[b][k]) * (weights[i][k] - weights[b][k]);
            }
            return da < db;
        });
        neighbors[i].assign(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(T));
    }

    std::vector<std::vector<double>> X(Npop), F(Npop);
    std::vector<double> ideal(m, std::numeric_limits<double>::infinity());
    for (std::size_t i = 0; i < Npop; ++i) {
        X[i] = random_point();
        F[i] = function.evaluate(X[i]);
        for (std::size_t k = 0; k < m; ++k) ideal[k] = std::min(ideal[k], F[i][k]);
    }

    auto scalarize = [&](const std::vector<double>& f, const std::vector<double>& w) {
        if (weighted_sum) {
            double s = 0.0;
            for (std::size_t k = 0; k < m; ++k) s += w[k] * (f[k] - ideal[k]);
            return s;
        }
        double g = 0.0; // Tchebycheff: max_k w_k * |f_k - z*_k|, with a weight floor for extremes.
        for (std::size_t k = 0; k < m; ++k) g = std::max(g, std::max(w[k], 1e-6) * std::fabs(f[k] - ideal[k]));
        return g;
    };

    for (std::size_t gen = 0; gen < options_.max_generations; ++gen) {
        for (std::size_t i = 0; i < Npop; ++i) {
            // Reproduction: two distinct parents drawn from subproblem i's neighborhood.
            std::uniform_int_distribution<std::size_t> pick(0, T - 1);
            const std::size_t a = neighbors[i][pick(rng)];
            std::size_t b = neighbors[i][pick(rng)];
            std::vector<double> c1, c2;
            detail::sbx_crossover(X[a], X[b], c1, c2, lower_bound, upper_bound, options_.crossover_rate,
                                  options_.eta_crossover, rng);
            detail::polynomial_mutation(c1, lower_bound, upper_bound, mut_rate, options_.eta_mutation, rng);
            const std::vector<double> fy = function.evaluate(c1);

            // Update the ideal point, then let the child replace any neighbor it improves.
            for (std::size_t k = 0; k < m; ++k) ideal[k] = std::min(ideal[k], fy[k]);
            for (const std::size_t j : neighbors[i]) {
                if (scalarize(fy, weights[j]) <= scalarize(F[j], weights[j])) {
                    X[j] = c1;
                    F[j] = fy;
                }
            }
        }
    }

    std::vector<ParetoPoint> points;
    points.reserve(Npop);
    for (std::size_t i = 0; i < Npop; ++i) points.push_back(ParetoPoint{X[i], F[i]});
    return non_dominated_front(points);
}

} // namespace datamunge::optim
