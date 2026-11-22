/// @file cma_es.cpp
/// @brief CMAES native update, trial and stopping implementation.
#include <datamunge/optim/cma_es.hpp>
#include <datamunge/optim/cma_es_variants.hpp>

#include "boxed_search_detail.hpp"
#include "symmetric_eigen_detail.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace datamunge::optim {
/// @brief Private cma es arithmetic, validation and local state.
namespace {

/// @brief Compute the ordinary Euclidean evolution-path norm.
/// @param values Path components.
/// @return Square root of the double sum of squares.
double norm(const std::vector<double>& values) {
    double squared = 0.0;
    for (const double value : values) squared += value * value;
    return std::sqrt(squared);
}

/// @brief Run one dense or diagonal CMA search, optionally projected to the unit box.
/// @param function Borrowed objective operating on the chosen coordinate representation.
/// @param coordinates Start overwritten with the best local point.
/// @param options Native CMA controls.
/// @param diagonal Whether covariance is represented by its diagonal alone.
/// @param boxed Whether proposals are projected into the unit box.
/// @param budget Maximum calls in this local run, including the start.
/// @return Loss at the local best point.
double run_cma(ArbitraryFunction& function, std::vector<double>& coordinates,
               const CMAESOptions& options, bool diagonal, bool boxed, std::size_t budget) {
    const std::size_t n = coordinates.size();
    if (n == 0) return function.evaluate(coordinates);

    const std::size_t lambda = options.population_size == 0
        ? 4 + static_cast<std::size_t>(std::floor(3.0 * std::log(static_cast<double>(n))))
        : options.population_size;
    if (lambda < 2) throw std::invalid_argument("CMAES: population_size must be at least 2");
    if (boxed && budget == 1) return function.evaluate(coordinates);
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
    const double learning = diagonal ? (n_d + 1.5) / 3 : 1;
    const double c1 = learning * 2.0 / (std::pow(n_d + 1.3, 2.0) + mu_eff);
    const double c_mu = std::min(1.0 - c1, learning * 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / (std::pow(n_d + 2.0, 2.0) + mu_eff));
    const double expected_norm = std::sqrt(n_d) * (1.0 - 1.0 / (4.0 * n_d) + 1.0 / (21.0 * n_d * n_d));

    std::mt19937_64 rng(options.seed);
    std::normal_distribution<double> standard_normal(0.0, 1.0);
    std::vector<double> mean = coordinates;
    std::vector<double> covariance(diagonal ? n : n * n, 0.0), path_sigma(n, 0.0), path_covariance(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) covariance[diagonal ? i : i * n + i] = 1.0;
    double sigma = options.initial_step_size;
    std::vector<double> best = mean;
    double best_value = function.evaluate(best);
    std::size_t calls = 1;

    /// @brief One locally sampled CMA offspring and its objective value.
    struct Candidate {
      std::vector<double> point; ///< Sampled coordinates.
      std::vector<double> step; ///< Covariance-transformed step before sigma scaling.
      double value; ///< Objective at point.
    };
    for (std::size_t generation = 0; generation < options.max_generations && calls < budget; ++generation) {
        const auto eigen = diagonal ? detail::EigenSystem{covariance, {}}
                                    : detail::symmetric_eigensystem(covariance, n, 1e-30);
        std::vector<double> axes(n);
        for (std::size_t i = 0; i < n; ++i) axes[i] = std::sqrt(std::max(1e-30, eigen.values[i]));
        const auto count = std::min(lambda, budget - calls);
        std::vector<Candidate> population;
        population.reserve(count);
        for (std::size_t member = 0; member < count; ++member) {
            std::vector<double> normal(n), step(n, 0.0), point(n);
            for (double& component : normal) component = standard_normal(rng);
            for (std::size_t row = 0; row < n; ++row) {
                if (diagonal) {
                    step[row] = axes[row] * normal[row];
                } else {
                    for (std::size_t column = 0; column < n; ++column)
                        step[row] += eigen.vectors[row * n + column] * axes[column] * normal[column];
                }
                point[row] = mean[row] + sigma * step[row];
                if (boxed) {
                    point[row] = std::clamp(point[row], 0.0, 1.0);
                    step[row] = (point[row] - mean[row]) / sigma;
                }
            }
            const double value = function.evaluate(point);
            ++calls;
            if (value < best_value) {
                best = point;
                best_value = value;
            }
            population.push_back({std::move(point), std::move(step), value});
        }
        if (count < lambda) break;
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
        if (diagonal) {
            for (std::size_t j = 0; j < n; ++j) inverse_sqrt_step[j] = weighted_step[j] / axes[j];
        } else {
            for (std::size_t column = 0; column < n; ++column) {
                double projection = 0.0;
                for (std::size_t row = 0; row < n; ++row) projection += eigen.vectors[row * n + column] * weighted_step[row];
                projection /= axes[column];
                for (std::size_t row = 0; row < n; ++row) inverse_sqrt_step[row] += eigen.vectors[row * n + column] * projection;
            }
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
        std::vector<double> next_covariance(covariance.size(), 0.0);
        for (std::size_t row = 0; row < n; ++row) {
            const auto first_column = diagonal ? row : 0;
            const auto last_column = diagonal ? row + 1 : n;
            for (std::size_t column = first_column; column < last_column; ++column) {
                const auto index = diagonal ? row : row * n + column;
                double value = old_covariance_factor * covariance[index] +
                               c1 * path_covariance[row] * path_covariance[column];
                for (std::size_t i = 0; i < mu; ++i)
                    value += c_mu * weights[i] * population[i].step[row] * population[i].step[column];
                next_covariance[index] = value;
            }
        }
        covariance = std::move(next_covariance);
        sigma *= std::exp((c_sigma / d_sigma) * (norm(path_sigma) / expected_norm - 1.0));
        const double largest_axis = *std::max_element(axes.begin(), axes.end());
        if (!diagonal && sigma * largest_axis <= options.tolerance) break;
        if (boxed) sigma = std::clamp(sigma, 1e-12, 1.0);
    }
    coordinates = best;
    return best_value;
}

} // namespace

CMAES::CMAES(CMAESOptions options) : options_(options) {}

double CMAES::optimize(ArbitraryFunction& function, std::vector<double>& coordinates) const {
    if (options_.initial_step_size <= 0.0) throw std::invalid_argument("CMAES: initial_step_size must be positive");
    if (options_.tolerance < 0.0) throw std::invalid_argument("CMAES: tolerance must be non-negative");
    return run_cma(function, coordinates, options_, false, false, std::numeric_limits<std::size_t>::max());
}

CMASearch::CMASearch(CMASearchOptions options) : options_(options) {}

double CMASearch::optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                           const std::vector<double>& lower, const std::vector<double>& upper) const {
    const auto n = coordinates.size();
    detail::validate_box(n, lower, upper, options_.evaluations);
    const bool diagonal = options_.method == CMASearchMethod::separable;
    const bool bipop = options_.method == CMASearchMethod::bipop;
    if ((!diagonal && !bipop && options_.method != CMASearchMethod::ipop) ||
        options_.population == 1 || !(options_.initial_scale > 0 && options_.initial_scale <= 1) ||
        (!diagonal && (options_.generations_per_restart == 0 ||
                      !(options_.tolerance >= 0 && std::isfinite(options_.tolerance))))) {
        throw std::invalid_argument("Invalid CMA search controls");
    }
    if (!diagonal && n > (std::numeric_limits<std::size_t>::max() - 1) / 20 / n) {
        throw std::length_error("Dense covariance size overflows");
    }

    auto start = coordinates;
    auto center = detail::normalize_start(start, lower, upper);
    detail::CountingObjective observe(function);
    /// @brief Map the unit box to original coordinates and count all optimizer callbacks.
    struct NormalizedObjective : ArbitraryFunction {
        detail::CountingObjective& observe; ///< Borrowed recorder.
        const std::vector<double>& lower; ///< Lower bounds borrowed for this run.
        const std::vector<double>& upper; ///< Upper bounds borrowed for this run.
        std::vector<double> point; ///< Mapped scratch coordinates.

        /// @brief Borrow the objective and bounds, allocating coordinate scratch.
        /// @param callback Counting original-space objective.
        /// @param lo Lower bounds.
        /// @param hi Upper bounds.
        NormalizedObjective(detail::CountingObjective& callback, const std::vector<double>& lo,
                            const std::vector<double>& hi) : observe(callback), lower(lo), upper(hi), point(lo.size()) {}

        /// @brief Score through the mapping from unit to original coordinates.
        /// @param unit Candidate in normalized coordinates.
        /// @return Loss already recorded in original coordinates.
        double evaluate(const std::vector<double>& unit) override {
            for (std::size_t j = 0; j < unit.size(); ++j) point[j] = std::lerp(lower[j], upper[j], unit[j]);
            return observe.evaluate(point);
        }
    } normalized(observe, lower, upper);

    const auto base = options_.population == 0 ? 4 + static_cast<std::size_t>(3 * std::log(static_cast<double>(n)))
                                               : options_.population;
    auto next_large = base;
    std::size_t large_work = 0, small_work = 0;
    std::mt19937_64 rng(options_.seed);
    std::uniform_real_distribution<double> uniform;
    while (observe.calls < options_.evaluations) {
        const auto before = observe.calls;
        auto remaining = options_.evaluations - before;
        auto population = next_large;
        double scale = std::max(1e-12, options_.initial_scale);
        const bool small = bipop && small_work < large_work;
        if (small) {
            const double u = uniform(rng);
            const auto upper_ratio = std::max(1.0, (0.5 * next_large) / base);
            const auto small_population = base * std::pow(upper_ratio, u * u);
            population = static_cast<std::size_t>(std::min(small_population,
                                                           static_cast<double>(remaining)));
            scale = std::max(1e-12, scale * std::pow(10.0, -2 * uniform(rng)));
            remaining = std::min(remaining, large_work - small_work);
        } else if (!diagonal) {
            next_large = next_large <= options_.evaluations / 2 ? 2 * next_large
                                                               : std::max(next_large, options_.evaluations);
        }
        const CMAESOptions controls{
            std::max(std::size_t{2}, std::min(population, remaining)), scale,
            diagonal ? remaining : options_.generations_per_restart,
            options_.tolerance, rng()
        };
        run_cma(normalized, center, controls, diagonal, true, remaining);
        if (small) small_work += observe.calls - before;
        else large_work += observe.calls - before;
        if (observe.calls < options_.evaluations) {
            for (auto& value : center) value = uniform(rng);
        }
    }
    coordinates = std::move(observe.best);
    return observe.best_value;
}

} // namespace datamunge::optim
