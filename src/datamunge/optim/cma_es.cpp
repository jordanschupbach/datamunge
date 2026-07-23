#include <datamunge/optim/cma_es.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace datamunge::optim {
namespace {

struct EigenSystem {
    std::vector<double> values;
    std::vector<double> vectors; // Columns are eigenvectors.
};

EigenSystem symmetric_eigensystem(std::vector<double> matrix, const std::size_t n) {
    EigenSystem result{std::vector<double>(n), std::vector<double>(n * n, 0.0)};
    for (std::size_t i = 0; i < n; ++i) result.vectors[i * n + i] = 1.0;
    const std::size_t max_sweeps = 20 * n * n + 1;
    for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep) {
        std::size_t p = 0, q = 0;
        double largest = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                const double candidate = std::abs(matrix[i * n + j]);
                if (candidate > largest) {
                    largest = candidate;
                    p = i;
                    q = j;
                }
            }
        }
        if (largest <= 1e-14) break;
        const double app = matrix[p * n + p];
        const double aqq = matrix[q * n + q];
        const double apq = matrix[p * n + q];
        const double angle = 0.5 * std::atan2(2.0 * apq, aqq - app);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        for (std::size_t k = 0; k < n; ++k) {
            const double mkp = matrix[k * n + p];
            const double mkq = matrix[k * n + q];
            matrix[k * n + p] = cosine * mkp - sine * mkq;
            matrix[k * n + q] = sine * mkp + cosine * mkq;
        }
        for (std::size_t k = 0; k < n; ++k) {
            const double mpk = matrix[p * n + k];
            const double mqk = matrix[q * n + k];
            matrix[p * n + k] = cosine * mpk - sine * mqk;
            matrix[q * n + k] = sine * mpk + cosine * mqk;
        }
        matrix[p * n + q] = matrix[q * n + p] = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            const double vkp = result.vectors[k * n + p];
            const double vkq = result.vectors[k * n + q];
            result.vectors[k * n + p] = cosine * vkp - sine * vkq;
            result.vectors[k * n + q] = sine * vkp + cosine * vkq;
        }
    }
    for (std::size_t i = 0; i < n; ++i) result.values[i] = std::max(matrix[i * n + i], 1e-30);
    return result;
}

double norm(const std::vector<double>& values) {
    double squared = 0.0;
    for (const double value : values) squared += value * value;
    return std::sqrt(squared);
}

} // namespace

CMAES::CMAES(CMAESOptions options) : options_(options) {}

double CMAES::optimize(ArbitraryFunction& function, std::vector<double>& coordinates) const {
    if (options_.initial_step_size <= 0.0) throw std::invalid_argument("CMAES: initial_step_size must be positive");
    if (options_.tolerance < 0.0) throw std::invalid_argument("CMAES: tolerance must be non-negative");
    const std::size_t n = coordinates.size();
    if (n == 0) return function.evaluate(coordinates);

    const std::size_t lambda = options_.population_size == 0
        ? 4 + static_cast<std::size_t>(std::floor(3.0 * std::log(static_cast<double>(n))))
        : options_.population_size;
    if (lambda < 2) throw std::invalid_argument("CMAES: population_size must be at least 2");
    const std::size_t mu = lambda / 2;
    std::vector<double> weights(mu);
    double weight_sum = 0.0;
    for (std::size_t i = 0; i < mu; ++i) {
        weights[i] = std::log(static_cast<double>(mu) + 0.5) - std::log(static_cast<double>(i + 1));
        weight_sum += weights[i];
    }
    double weight_sq_sum = 0.0;
    for (double& weight : weights) {
        weight /= weight_sum;
        weight_sq_sum += weight * weight;
    }
    const double mu_eff = 1.0 / weight_sq_sum;
    const double n_d = static_cast<double>(n);
    const double c_sigma = (mu_eff + 2.0) / (n_d + mu_eff + 5.0);
    const double d_sigma = 1.0 + 2.0 * std::max(0.0, std::sqrt((mu_eff - 1.0) / (n_d + 1.0)) - 1.0) + c_sigma;
    const double c_c = (4.0 + mu_eff / n_d) / (n_d + 4.0 + 2.0 * mu_eff / n_d);
    const double c1 = 2.0 / (std::pow(n_d + 1.3, 2.0) + mu_eff);
    const double c_mu = std::min(1.0 - c1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / (std::pow(n_d + 2.0, 2.0) + mu_eff));
    const double expected_norm = std::sqrt(n_d) * (1.0 - 1.0 / (4.0 * n_d) + 1.0 / (21.0 * n_d * n_d));

    std::mt19937_64 rng(options_.seed);
    std::normal_distribution<double> standard_normal(0.0, 1.0);
    std::vector<double> mean = coordinates;
    std::vector<double> covariance(n * n, 0.0), path_sigma(n, 0.0), path_covariance(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) covariance[i * n + i] = 1.0;
    double sigma = options_.initial_step_size;
    std::vector<double> best = mean;
    double best_value = function.evaluate(best);

    struct Candidate { std::vector<double> point; std::vector<double> step; double value; };
    for (std::size_t generation = 0; generation < options_.max_generations; ++generation) {
        const EigenSystem eigen = symmetric_eigensystem(covariance, n);
        std::vector<double> axes(n);
        for (std::size_t i = 0; i < n; ++i) axes[i] = std::sqrt(eigen.values[i]);
        std::vector<Candidate> population;
        population.reserve(lambda);
        for (std::size_t member = 0; member < lambda; ++member) {
            std::vector<double> normal(n), step(n, 0.0), point(n);
            for (double& component : normal) component = standard_normal(rng);
            for (std::size_t row = 0; row < n; ++row) {
                for (std::size_t column = 0; column < n; ++column)
                    step[row] += eigen.vectors[row * n + column] * axes[column] * normal[column];
                point[row] = mean[row] + sigma * step[row];
            }
            const double value = function.evaluate(point);
            if (value < best_value) {
                best = point;
                best_value = value;
            }
            population.push_back({std::move(point), std::move(step), value});
        }
        std::sort(population.begin(), population.end(), [](const Candidate& a, const Candidate& b) {
            return a.value < b.value;
        });
        std::vector<double> weighted_step(n, 0.0);
        std::fill(mean.begin(), mean.end(), 0.0);
        for (std::size_t i = 0; i < mu; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                mean[j] += weights[i] * population[i].point[j];
                weighted_step[j] += weights[i] * population[i].step[j];
            }
        }
        std::vector<double> inverse_sqrt_step(n, 0.0);
        for (std::size_t column = 0; column < n; ++column) {
            double projection = 0.0;
            for (std::size_t row = 0; row < n; ++row) projection += eigen.vectors[row * n + column] * weighted_step[row];
            projection /= axes[column];
            for (std::size_t row = 0; row < n; ++row) inverse_sqrt_step[row] += eigen.vectors[row * n + column] * projection;
        }
        const double path_factor = std::sqrt(c_sigma * (2.0 - c_sigma) * mu_eff);
        for (std::size_t j = 0; j < n; ++j)
            path_sigma[j] = (1.0 - c_sigma) * path_sigma[j] + path_factor * inverse_sqrt_step[j];
        const double normalized_path = norm(path_sigma) /
            std::sqrt(1.0 - std::pow(1.0 - c_sigma, 2.0 * static_cast<double>(generation + 1)));
        const bool h_sigma = normalized_path < (1.4 + 2.0 / (n_d + 1.0)) * expected_norm;
        const double covariance_path_factor = h_sigma ? std::sqrt(c_c * (2.0 - c_c) * mu_eff) : 0.0;
        for (std::size_t j = 0; j < n; ++j)
            path_covariance[j] = (1.0 - c_c) * path_covariance[j] + covariance_path_factor * weighted_step[j];

        const double old_covariance_factor = 1.0 - c1 - c_mu + (h_sigma ? 0.0 : c1 * c_c * (2.0 - c_c));
        std::vector<double> next_covariance(n * n, 0.0);
        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t column = 0; column < n; ++column) {
                double value = old_covariance_factor * covariance[row * n + column] +
                               c1 * path_covariance[row] * path_covariance[column];
                for (std::size_t i = 0; i < mu; ++i)
                    value += c_mu * weights[i] * population[i].step[row] * population[i].step[column];
                next_covariance[row * n + column] = value;
            }
        }
        covariance = std::move(next_covariance);
        sigma *= std::exp((c_sigma / d_sigma) * (norm(path_sigma) / expected_norm - 1.0));
        const double largest_axis = *std::max_element(axes.begin(), axes.end());
        if (sigma * largest_axis <= options_.tolerance) break;
    }
    coordinates = best;
    return best_value;
}

} // namespace datamunge::optim
