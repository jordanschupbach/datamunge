#include <datamunge/optim/simulated_annealing.hpp>

#include <algorithm>
#include <cmath>
#include <random>

namespace datamunge::optim {

SimulatedAnnealing::SimulatedAnnealing(SimulatedAnnealingOptions options) : options_(options) {}

double SimulatedAnnealing::optimize(ArbitraryFunction& function, std::vector<double>& coordinates) const {
    std::mt19937_64 rng(options_.seed);
    std::normal_distribution<double> perturb(0.0, 1.0);
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);

    std::vector<double> current = coordinates;
    double current_value = function.evaluate(current);
    std::vector<double> best = current;
    double best_value = current_value;

    double temperature = options_.initial_temperature;
    for (std::size_t iter = 0; iter < options_.max_iterations; ++iter) {
        std::vector<double> candidate = current;
        for (double& c : candidate) c += options_.step_std_dev * perturb(rng);

        const double candidate_value = function.evaluate(candidate);
        const double delta = candidate_value - current_value;
        if (delta < 0.0 || uniform01(rng) < std::exp(-delta / std::max(temperature, 1e-12))) {
            current = std::move(candidate);
            current_value = candidate_value;
            if (current_value < best_value) {
                best = current;
                best_value = current_value;
            }
        }
        temperature *= options_.cooling_rate;
    }

    coordinates = best;
    return best_value;
}

} // namespace datamunge::optim
