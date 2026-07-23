#include <datamunge/optim/estimation_of_distribution.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace datamunge::optim {

namespace {

void validate_bounds(const std::size_t n, const std::vector<double>& lower_bound,
                      const std::vector<double>& upper_bound, const char* op) {
    if (n == 0) throw std::invalid_argument(std::string("EstimationOfDistribution::") + op + ": coordinates must not be empty");
    if (lower_bound.size() != n || upper_bound.size() != n)
        throw std::invalid_argument(std::string("EstimationOfDistribution::") + op + ": bound size must match coordinates size");
    for (std::size_t j = 0; j < n; ++j)
        if (!(lower_bound[j] < upper_bound[j]))
            throw std::invalid_argument(std::string("EstimationOfDistribution::") + op +
                                         ": lower_bound must be < upper_bound in every dimension");
}

// Cholesky-Banachiewicz factorization of a symmetric positive-(semi)definite matrix
// (n*n, row-major) into a lower-triangular L (n*n, row-major) with L*L^T == covariance.
// Non-positive diagonal terms (numerical drift) are floored to a small positive value rather
// than producing NaN.
std::vector<double> cholesky(const std::vector<double>& covariance, const std::size_t n) {
    std::vector<double> l(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double sum = covariance[i * n + j];
            for (std::size_t k = 0; k < j; ++k) sum -= l[i * n + k] * l[j * n + k];
            if (j < i) {
                l[i * n + j] = sum / l[j * n + j];
            } else {
                l[i * n + i] = std::sqrt(std::max(sum, 1e-300));
            }
        }
    }
    return l;
}

} // namespace

EstimationOfDistribution::EstimationOfDistribution(EstimationOfDistributionOptions options) : options_(options) {}

double EstimationOfDistribution::optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                                           const std::vector<double>& lower_bound,
                                           const std::vector<double>& upper_bound) const {
    const std::size_t n = coordinates.size();
    validate_bounds(n, lower_bound, upper_bound, "optimize");
    if (options_.population_size < 4)
        throw std::invalid_argument("EstimationOfDistribution::optimize: population_size must be at least 4");
    if (!(options_.selection_ratio > 0.0) || options_.selection_ratio > 1.0)
        throw std::invalid_argument("EstimationOfDistribution::optimize: selection_ratio must be in (0, 1]");
    if (!(options_.initial_std_dev > 0.0))
        throw std::invalid_argument("EstimationOfDistribution::optimize: initial_std_dev must be positive");
    if (options_.covariance_regularization < 0.0)
        throw std::invalid_argument("EstimationOfDistribution::optimize: covariance_regularization must be non-negative");

    const std::size_t mu = std::max<std::size_t>(
        2, static_cast<std::size_t>(std::ceil(static_cast<double>(options_.population_size) * options_.selection_ratio)));

    std::mt19937_64 rng(options_.seed);
    std::normal_distribution<double> standard_normal(0.0, 1.0);

    std::vector<double> mean(n);
    for (std::size_t j = 0; j < n; ++j) mean[j] = std::clamp(coordinates[j], lower_bound[j], upper_bound[j]);
    std::vector<double> covariance(n * n, 0.0);
    for (std::size_t j = 0; j < n; ++j) covariance[j * n + j] = options_.initial_std_dev * options_.initial_std_dev;

    std::vector<double> best = mean;
    double best_value = function.evaluate(best);
    double prev_best_value = best_value;
    std::size_t stagnant = 0;

    struct Candidate { std::vector<double> point; double value; };

    for (std::size_t generation = 0; generation < options_.max_generations; ++generation) {
        const std::vector<double> l = cholesky(covariance, n);

        std::vector<Candidate> population;
        population.reserve(options_.population_size);
        for (std::size_t member = 0; member < options_.population_size; ++member) {
            std::vector<double> z(n);
            for (double& component : z) component = standard_normal(rng);
            std::vector<double> point(n);
            for (std::size_t row = 0; row < n; ++row) {
                double displacement = 0.0;
                for (std::size_t k = 0; k <= row; ++k) displacement += l[row * n + k] * z[k];
                point[row] = std::clamp(mean[row] + displacement, lower_bound[row], upper_bound[row]);
            }
            const double value = function.evaluate(point);
            if (value < best_value) {
                best = point;
                best_value = value;
            }
            population.push_back({std::move(point), value});
        }

        std::sort(population.begin(), population.end(),
                  [](const Candidate& a, const Candidate& b) { return a.value < b.value; });

        std::vector<double> mean_new(n, 0.0);
        for (std::size_t i = 0; i < mu; ++i)
            for (std::size_t j = 0; j < n; ++j) mean_new[j] += population[i].point[j];
        for (double& v : mean_new) v /= static_cast<double>(mu);

        std::vector<double> covariance_new(n * n, 0.0);
        for (std::size_t i = 0; i < mu; ++i) {
            for (std::size_t d1 = 0; d1 < n; ++d1) {
                const double diff1 = population[i].point[d1] - mean_new[d1];
                for (std::size_t d2 = 0; d2 < n; ++d2) {
                    const double diff2 = population[i].point[d2] - mean_new[d2];
                    covariance_new[d1 * n + d2] += diff1 * diff2;
                }
            }
        }
        for (double& v : covariance_new) v /= static_cast<double>(mu);
        for (std::size_t d = 0; d < n; ++d) covariance_new[d * n + d] += options_.covariance_regularization;

        mean = std::move(mean_new);
        covariance = std::move(covariance_new);

        if (std::abs(prev_best_value - best_value) < options_.tolerance) {
            if (++stagnant > 20) break;
        } else {
            stagnant = 0;
        }
        prev_best_value = best_value;

        bool collapsed = true;
        for (std::size_t d = 0; d < n; ++d) {
            if (covariance[d * n + d] >= options_.tolerance * options_.tolerance) {
                collapsed = false;
                break;
            }
        }
        if (collapsed) break;
    }

    coordinates = best;
    return best_value;
}

} // namespace datamunge::optim
