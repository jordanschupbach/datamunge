#include <datamunge/optim/pso.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace datamunge::optim {

namespace {

struct Particle {
    std::vector<double> position;
    std::vector<double> velocity;
    std::vector<double> best_position;
    double best_value{0.0};
    double value{0.0};
};

void validate_bounds(const std::size_t n, const std::vector<double>& lower_bound,
                      const std::vector<double>& upper_bound, const char* op) {
    if (n == 0) throw std::invalid_argument(std::string("PSO::") + op + ": coordinates must not be empty");
    if (lower_bound.size() != n || upper_bound.size() != n)
        throw std::invalid_argument(std::string("PSO::") + op + ": bound size must match coordinates size");
    for (std::size_t j = 0; j < n; ++j)
        if (!(lower_bound[j] < upper_bound[j]))
            throw std::invalid_argument(std::string("PSO::") + op + ": lower_bound must be < upper_bound in every dimension");
}

} // namespace

PSO::PSO(PSOOptions options) : options_(options) {}

double PSO::optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                      const std::vector<double>& lower_bound, const std::vector<double>& upper_bound) const {
    const std::size_t n = coordinates.size();
    validate_bounds(n, lower_bound, upper_bound, "optimize");
    if (options_.population_size < 2)
        throw std::invalid_argument("PSO::optimize: population_size must be at least 2");

    const bool ring = options_.topology == "ring";
    if (!ring && options_.topology != "global")
        throw std::invalid_argument("PSO::optimize: topology must be 'global' or 'ring'");
    const bool linear_decay = options_.inertia_strategy == "linear_decay";
    if (!linear_decay && options_.inertia_strategy != "constant")
        throw std::invalid_argument("PSO::optimize: inertia_strategy must be 'constant' or 'linear_decay'");

    std::mt19937_64 rng(options_.seed);
    std::uniform_real_distribution<double> unif01(0.0, 1.0);

    std::vector<Particle> swarm(options_.population_size);
    for (auto& p : swarm) {
        p.position.resize(n);
        p.velocity.resize(n);
        for (std::size_t j = 0; j < n; ++j) {
            const double range = upper_bound[j] - lower_bound[j];
            p.position[j] = lower_bound[j] + unif01(rng) * range;
            p.velocity[j] = (unif01(rng) - 0.5) * range * 0.1;
        }
        p.value = function.evaluate(p.position);
        p.best_position = p.position;
        p.best_value = p.value;
    }
    // Seed the caller's initial guess as one particle.
    for (std::size_t j = 0; j < n; ++j) swarm[0].position[j] = std::clamp(coordinates[j], lower_bound[j], upper_bound[j]);
    swarm[0].value = function.evaluate(swarm[0].position);
    swarm[0].best_position = swarm[0].position;
    swarm[0].best_value = swarm[0].value;

    std::size_t global_best_idx = 0;
    for (std::size_t i = 1; i < swarm.size(); ++i)
        if (swarm[i].best_value < swarm[global_best_idx].best_value) global_best_idx = i;
    double global_best_value = swarm[global_best_idx].best_value;
    std::vector<double> global_best_position = swarm[global_best_idx].best_position;

    double prev_best = global_best_value;
    std::size_t stagnant = 0;

    for (std::size_t iter = 0; iter < options_.max_iterations; ++iter) {
        const double w = linear_decay
            ? options_.inertia_weight + (options_.final_inertia_weight - options_.inertia_weight) *
                  (static_cast<double>(iter) /
                   static_cast<double>(options_.max_iterations > 1 ? options_.max_iterations - 1 : 1))
            : options_.inertia_weight;

        for (std::size_t i = 0; i < swarm.size(); ++i) {
            const std::vector<double>* attractor = &global_best_position;
            std::vector<double> ring_best;
            if (ring) {
                std::size_t best_neighbor = i;
                for (std::size_t d = 1; d <= options_.ring_neighbors; ++d) {
                    const std::size_t left = (i + swarm.size() - (d % swarm.size())) % swarm.size();
                    const std::size_t right = (i + d) % swarm.size();
                    if (swarm[left].best_value < swarm[best_neighbor].best_value) best_neighbor = left;
                    if (swarm[right].best_value < swarm[best_neighbor].best_value) best_neighbor = right;
                }
                ring_best = swarm[best_neighbor].best_position;
                attractor = &ring_best;
            }

            Particle& p = swarm[i];
            for (std::size_t j = 0; j < n; ++j) {
                const double r1 = unif01(rng), r2 = unif01(rng);
                p.velocity[j] = w * p.velocity[j] + options_.cognitive_coefficient * r1 * (p.best_position[j] - p.position[j]) +
                                 options_.social_coefficient * r2 * ((*attractor)[j] - p.position[j]);
                p.position[j] += p.velocity[j];
                if (p.position[j] < lower_bound[j]) {
                    p.position[j] = lower_bound[j];
                    p.velocity[j] = 0.0;
                } else if (p.position[j] > upper_bound[j]) {
                    p.position[j] = upper_bound[j];
                    p.velocity[j] = 0.0;
                }
            }
            p.value = function.evaluate(p.position);
            if (p.value < p.best_value) {
                p.best_value = p.value;
                p.best_position = p.position;
                if (p.value < global_best_value) {
                    global_best_value = p.value;
                    global_best_position = p.position;
                }
            }
        }

        if (std::abs(prev_best - global_best_value) < options_.tolerance) {
            if (++stagnant > 20) break;
        } else {
            stagnant = 0;
        }
        prev_best = global_best_value;
    }

    coordinates = global_best_position;
    return global_best_value;
}

} // namespace datamunge::optim
