#include <datamunge/optim/parallel_tempering.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace datamunge::optim {

ParallelTempering::ParallelTempering(ParallelTemperingOptions options) : options_(options) {}

double ParallelTempering::optimize(ArbitraryFunction& function, std::vector<double>& coordinates) const {
    if (options_.num_replicas < 2)
        throw std::invalid_argument("ParallelTempering: num_replicas must be at least 2");
    if (options_.final_temperature <= 0.0)
        throw std::invalid_argument("ParallelTempering: final_temperature must be positive");
    if (options_.initial_temperature <= options_.final_temperature)
        throw std::invalid_argument("ParallelTempering: initial_temperature must exceed final_temperature");
    if (options_.step_std_dev <= 0.0)
        throw std::invalid_argument("ParallelTempering: step_std_dev must be positive");
    if (options_.swap_interval < 1)
        throw std::invalid_argument("ParallelTempering: swap_interval must be at least 1");

    std::mt19937_64 rng(options_.seed);
    std::normal_distribution<double> perturb(0.0, 1.0);
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);

    const std::size_t num_replicas = options_.num_replicas;
    const double t_min = options_.final_temperature;
    const double t_max = options_.initial_temperature;

    // Geometrically spaced temperature ladder: T_0 = t_max (hottest), T_{n-1} = t_min (coldest).
    // This order is fixed for the whole run -- only replica STATES move between slots on a swap.
    std::vector<double> temperature(num_replicas);
    for (std::size_t i = 0; i < num_replicas; ++i) {
        const double exponent = static_cast<double>(i) / static_cast<double>(num_replicas - 1);
        temperature[i] = t_max * std::pow(t_min / t_max, exponent);
    }

    // All replicas start at the caller's initial point.
    std::vector<std::vector<double>> current(num_replicas, coordinates);
    std::vector<double> current_value(num_replicas);
    for (std::size_t i = 0; i < num_replicas; ++i) current_value[i] = function.evaluate(current[i]);

    std::vector<double> best = coordinates;
    double best_value = current_value[0];
    for (std::size_t i = 1; i < num_replicas; ++i) {
        if (current_value[i] < best_value) {
            best = current[i];
            best_value = current_value[i];
        }
    }

    for (std::size_t sweep = 0; sweep < options_.max_sweeps; ++sweep) {
        // 1. Independent single-replica Metropolis update, at each replica's own fixed
        //    temperature, mirroring SimulatedAnnealing's proposal/accept step exactly.
        for (std::size_t i = 0; i < num_replicas; ++i) {
            const double t_i = temperature[i];
            const double step = options_.step_std_dev * std::sqrt(t_i / t_min);

            std::vector<double> candidate = current[i];
            for (double& c : candidate) c += step * perturb(rng);

            const double candidate_value = function.evaluate(candidate);
            const double delta = candidate_value - current_value[i];
            if (delta < 0.0 || uniform01(rng) < std::exp(-delta / std::max(t_i, 1e-12))) {
                current[i] = std::move(candidate);
                current_value[i] = candidate_value;
                if (current_value[i] < best_value) {
                    best = current[i];
                    best_value = current_value[i];
                }
            }
        }

        // 2. Every swap_interval sweeps, attempt a replica-exchange swap for every adjacent pair.
        if ((sweep + 1) % options_.swap_interval == 0) {
            for (std::size_t i = 0; i + 1 < num_replicas; ++i) {
                const double t_i = temperature[i];
                const double t_next = temperature[i + 1];
                const double log_accept =
                    (1.0 / t_i - 1.0 / t_next) * (current_value[i] - current_value[i + 1]);
                const double accept_prob = std::min(1.0, std::exp(log_accept));
                if (uniform01(rng) < accept_prob) {
                    std::swap(current[i], current[i + 1]);
                    std::swap(current_value[i], current_value[i + 1]);
                }
            }
        }
    }

    coordinates = best;
    return best_value;
}

} // namespace datamunge::optim
