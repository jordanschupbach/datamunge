#include <datamunge/optim/cuckoo_search.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace datamunge::optim {

namespace {

void validate_bounds(const std::size_t n, const std::vector<double>& lower_bound,
                      const std::vector<double>& upper_bound, const char* op) {
    if (n == 0) throw std::invalid_argument(std::string("CuckooSearch::") + op + ": coordinates must not be empty");
    if (lower_bound.size() != n || upper_bound.size() != n)
        throw std::invalid_argument(std::string("CuckooSearch::") + op + ": bound size must match coordinates size");
    for (std::size_t j = 0; j < n; ++j)
        if (!(lower_bound[j] < upper_bound[j]))
            throw std::invalid_argument(std::string("CuckooSearch::") + op + ": lower_bound must be < upper_bound in every dimension");
}

// Draws one n-dimensional Levy-distributed step vector with stability index beta, via
// Mantegna's algorithm. sigma_u depends only on beta (compute once outside the hot loop).
std::vector<double> levy_step(std::size_t n, double beta, double sigma_u, std::mt19937_64& rng) {
    std::normal_distribution<double> u_dist(0.0, sigma_u);
    std::normal_distribution<double> v_dist(0.0, 1.0);
    std::vector<double> step(n);
    for (std::size_t d = 0; d < n; ++d) {
        const double u = u_dist(rng);
        const double v = v_dist(rng);
        step[d] = u / std::pow(std::abs(v), 1.0 / beta);
    }
    return step;
}

} // namespace

CuckooSearch::CuckooSearch(CuckooSearchOptions options) : options_(options) {}

double CuckooSearch::optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                               const std::vector<double>& lower_bound, const std::vector<double>& upper_bound) const {
    const std::size_t n = coordinates.size();
    validate_bounds(n, lower_bound, upper_bound, "optimize");
    if (options_.population_size < 2)
        throw std::invalid_argument("CuckooSearch::optimize: population_size must be at least 2");
    if (options_.discovery_rate < 0.0 || options_.discovery_rate > 1.0)
        throw std::invalid_argument("CuckooSearch::optimize: discovery_rate must be in [0, 1]");
    if (options_.levy_beta <= 0.0 || options_.levy_beta > 2.0)
        throw std::invalid_argument("CuckooSearch::optimize: levy_beta must be in (0, 2]");
    if (options_.step_scale <= 0.0)
        throw std::invalid_argument("CuckooSearch::optimize: step_scale must be positive");

    std::mt19937_64 rng(options_.seed);
    std::uniform_real_distribution<double> unif01(0.0, 1.0);
    std::uniform_int_distribution<std::size_t> pick_nest(0, options_.population_size - 1);

    const double beta = options_.levy_beta;
    const double sigma_u = std::pow(
        std::tgamma(1.0 + beta) * std::sin(M_PI * beta / 2.0) /
            (std::tgamma((1.0 + beta) / 2.0) * beta * std::pow(2.0, (beta - 1.0) / 2.0)),
        1.0 / beta);

    // Initialize nests uniformly at random within bounds; seed nest 0 with the caller's guess.
    std::vector<std::vector<double>> nests(options_.population_size, std::vector<double>(n));
    std::vector<double> values(options_.population_size);
    for (std::size_t i = 0; i < options_.population_size; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const double range = upper_bound[j] - lower_bound[j];
            nests[i][j] = lower_bound[j] + unif01(rng) * range;
        }
    }
    for (std::size_t j = 0; j < n; ++j) nests[0][j] = std::clamp(coordinates[j], lower_bound[j], upper_bound[j]);
    for (std::size_t i = 0; i < options_.population_size; ++i) values[i] = function.evaluate(nests[i]);

    std::size_t best_idx = 0;
    for (std::size_t i = 1; i < options_.population_size; ++i)
        if (values[i] < values[best_idx]) best_idx = i;
    std::vector<double> best = nests[best_idx];
    double best_value = values[best_idx];

    double prev_best = best_value;
    std::size_t stagnant = 0;

    for (std::size_t iter = 0; iter < options_.max_iterations; ++iter) {
        // 1. Levy flight phase. X_best is the current best nest at the start of this phase.
        const std::vector<double> x_best = best;
        for (std::size_t i = 0; i < options_.population_size; ++i) {
            const std::vector<double> step = levy_step(n, beta, sigma_u, rng);
            std::vector<double> candidate(n);
            for (std::size_t d = 0; d < n; ++d) {
                candidate[d] = nests[i][d] + options_.step_scale * step[d] * (nests[i][d] - x_best[d]);
                candidate[d] = std::clamp(candidate[d], lower_bound[d], upper_bound[d]);
            }
            const double candidate_value = function.evaluate(candidate);
            if (candidate_value < best_value) {
                best_value = candidate_value;
                best = candidate;
            }

            const std::size_t j = pick_nest(rng);
            if (candidate_value < values[j]) {
                nests[j] = candidate;
                values[j] = candidate_value;
            }
        }

        // 2. Discovery/abandonment phase.
        for (std::size_t i = 0; i < options_.population_size; ++i) {
            if (unif01(rng) >= options_.discovery_rate) continue;
            std::size_t j = pick_nest(rng);
            std::size_t k = pick_nest(rng);
            while (k == j) k = pick_nest(rng);
            for (std::size_t d = 0; d < n; ++d) {
                double v = nests[i][d] + unif01(rng) * (nests[j][d] - nests[k][d]);
                nests[i][d] = std::clamp(v, lower_bound[d], upper_bound[d]);
            }
            values[i] = function.evaluate(nests[i]);
            if (values[i] < best_value) {
                best_value = values[i];
                best = nests[i];
            }
        }

        // 3 & 4. best-ever tracking already done above; check stagnation for stopping.
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
