#include <datamunge/optim/acor.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace datamunge::optim {

namespace {

struct Ant {
    std::vector<double> position;
    double value{0.0};
};

void validate_bounds(const std::size_t n, const std::vector<double>& lower_bound,
                      const std::vector<double>& upper_bound, const char* op) {
    if (n == 0) throw std::invalid_argument(std::string("ACOR::") + op + ": coordinates must not be empty");
    if (lower_bound.size() != n || upper_bound.size() != n)
        throw std::invalid_argument(std::string("ACOR::") + op + ": bound size must match coordinates size");
    for (std::size_t j = 0; j < n; ++j)
        if (!(lower_bound[j] < upper_bound[j]))
            throw std::invalid_argument(std::string("ACOR::") + op + ": lower_bound must be < upper_bound in every dimension");
}

// Roulette-wheel selection: draw u ~ Uniform(0,1) and walk the cumulative sum of
// probabilities until it exceeds u. `cumulative` must be non-decreasing and end at ~1.0.
std::size_t roulette_select(const std::vector<double>& cumulative, double u) {
    for (std::size_t i = 0; i < cumulative.size(); ++i)
        if (u <= cumulative[i]) return i;
    return cumulative.size() - 1;
}

} // namespace

ACOR::ACOR(ACOROptions options) : options_(options) {}

double ACOR::optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                       const std::vector<double>& lower_bound, const std::vector<double>& upper_bound) const {
    const std::size_t n = coordinates.size();
    validate_bounds(n, lower_bound, upper_bound, "optimize");
    if (options_.archive_size < 2) throw std::invalid_argument("ACOR::optimize: archive_size must be at least 2");
    if (options_.samples_per_iteration < 1)
        throw std::invalid_argument("ACOR::optimize: samples_per_iteration must be at least 1");
    if (!(options_.locality > 0.0)) throw std::invalid_argument("ACOR::optimize: locality must be positive");
    if (!(options_.convergence_speed > 0.0))
        throw std::invalid_argument("ACOR::optimize: convergence_speed must be positive");

    const std::size_t k = options_.archive_size;
    const std::size_t m = options_.samples_per_iteration;
    const double q = options_.locality;
    const double xi = options_.convergence_speed;

    std::mt19937_64 rng(options_.seed);
    std::uniform_real_distribution<double> unif01(0.0, 1.0);

    // Initialize the archive.
    std::vector<Ant> archive(k);
    for (std::size_t i = 0; i < k; ++i) {
        archive[i].position.resize(n);
        for (std::size_t j = 0; j < n; ++j)
            archive[i].position[j] = lower_bound[j] + unif01(rng) * (upper_bound[j] - lower_bound[j]);
    }
    for (std::size_t j = 0; j < n; ++j) archive[0].position[j] = std::clamp(coordinates[j], lower_bound[j], upper_bound[j]);
    for (auto& a : archive) a.value = function.evaluate(a.position);

    auto by_value = [](const Ant& a, const Ant& b) { return a.value < b.value; };
    std::sort(archive.begin(), archive.end(), by_value);

    std::vector<double> best = archive[0].position;
    double best_value = archive[0].value;
    for (const auto& a : archive)
        if (a.value < best_value) { best_value = a.value; best = a.position; }

    double prev_best = best_value;
    std::size_t stagnant = 0;

    // A small fallback spread used only in the (validation-excluded) k == 1 edge case.
    double fallback_range = 0.0;
    for (std::size_t j = 0; j < n; ++j) fallback_range = std::max(fallback_range, upper_bound[j] - lower_bound[j]);

    std::vector<double> weights(k);
    std::vector<double> cumulative(k);

    for (std::size_t iter = 0; iter < options_.max_iterations; ++iter) {
        // Archive is already sorted best-first from the previous iteration's truncation
        // (or initialization). Compute Gaussian-kernel selection weights by rank.
        const double denom = q * static_cast<double>(k) * std::sqrt(2.0 * M_PI);
        double weight_sum = 0.0;
        for (std::size_t i = 0; i < k; ++i) {
            const double rank = static_cast<double>(i); // rank i+1 in 1-indexed terms -> (i+1-1) = i
            weights[i] = std::exp(-(rank * rank) / (2.0 * q * q * static_cast<double>(k) * static_cast<double>(k))) / denom;
            weight_sum += weights[i];
        }
        double running = 0.0;
        for (std::size_t i = 0; i < k; ++i) {
            running += weights[i] / weight_sum;
            cumulative[i] = running;
        }
        cumulative[k - 1] = 1.0; // guard against floating-point drift

        std::vector<Ant> ants(m);
        for (std::size_t a = 0; a < m; ++a) {
            ants[a].position.resize(n);
            for (std::size_t d = 0; d < n; ++d) {
                const std::size_t l = roulette_select(cumulative, unif01(rng));

                double sigma;
                if (k > 1) {
                    double sum_abs_dist = 0.0;
                    for (std::size_t e = 0; e < k; ++e)
                        if (e != l) sum_abs_dist += std::abs(archive[e].position[d] - archive[l].position[d]);
                    sigma = (xi / static_cast<double>(k - 1)) * sum_abs_dist;
                } else {
                    sigma = 1e-6 * fallback_range;
                }
                if (sigma <= 0.0) sigma = 1e-12 * std::max(fallback_range, 1.0);

                std::normal_distribution<double> gauss(archive[l].position[d], sigma);
                ants[a].position[d] = std::clamp(gauss(rng), lower_bound[d], upper_bound[d]);
            }
            ants[a].value = function.evaluate(ants[a].position);
            if (ants[a].value < best_value) { best_value = ants[a].value; best = ants[a].position; }
        }

        std::vector<Ant> merged;
        merged.reserve(k + m);
        merged.insert(merged.end(), archive.begin(), archive.end());
        merged.insert(merged.end(), ants.begin(), ants.end());
        std::sort(merged.begin(), merged.end(), by_value);
        archive.assign(merged.begin(), merged.begin() + static_cast<std::ptrdiff_t>(k));

        if (archive[0].value < best_value) { best_value = archive[0].value; best = archive[0].position; }

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
