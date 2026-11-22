/// @file
/// @brief Classical Gaussian evolutionary programming under an evaluation budget.

#include <datamunge/optim/evolutionary_programming.hpp>

#include "boxed_search_detail.hpp"

#include <random>
#include <utility>

namespace datamunge::optim {
/// @brief Internal evaluated candidate for survivor tournaments.
namespace {

/// @brief Current state and heritable per-axis strategy parameters.
struct Individual {
  std::vector<double> coordinates; ///< Unit-box coordinates.
  std::vector<double> scales; ///< Per-axis mutation scales.
  double loss{}; ///< Observed objective at coordinates.
  std::size_t wins{}; ///< Current tournament wins.
};

} // namespace

EvolutionaryProgramming::EvolutionaryProgramming(EvolutionaryProgrammingOptions options) : options_(options) {}

double EvolutionaryProgramming::optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                                          const std::vector<double>& lower,
                                          const std::vector<double>& upper) const {
  const auto n = coordinates.size();
  detail::validate_box(n, lower, upper, options_.evaluations);
  if (options_.population < 2 || options_.opponents == 0 ||
      !(options_.initial_scale > 0 && options_.initial_scale <= 1)) {
    throw std::invalid_argument("Invalid evolutionary programming options");
  }
  auto point = coordinates;
  auto start = detail::normalize_start(point, lower, upper);
  detail::CountingObjective observe(function);
  /// Map one candidate to the original box and record its objective.
  const auto evaluate = [&](const std::vector<double>& unit) {
    for (std::size_t j = 0; j < unit.size(); ++j) point[j] = std::lerp(lower[j], upper[j], unit[j]);
    return observe.evaluate(point);
  };

  std::mt19937_64 rng(options_.seed);
  std::uniform_real_distribution<double> uniform;
  std::normal_distribution<double> gaussian;
  const std::vector<double> scales(n, options_.initial_scale);
  std::vector<Individual> parents;
  const auto size = std::min(options_.population, options_.evaluations);
  parents.reserve(size);
  parents.push_back({std::move(start), scales, observe.evaluate(point), 0});
  for (std::size_t i = 1; i < size; ++i) {
    std::vector<double> trial(n);
    for (auto& value : trial) value = uniform(rng);
    const auto loss = evaluate(trial);
    parents.push_back({std::move(trial), scales, loss, 0});
  }

  const auto common_rate = 1.0 / std::sqrt(2.0 * n);
  const auto local_rate = 1.0 / std::sqrt(2.0 * std::sqrt(static_cast<double>(n)));
  while (observe.calls < options_.evaluations) {
    const auto count = std::min(size, options_.evaluations - observe.calls);
    auto combined = parents;
    for (std::size_t i = 0; i < count; ++i) {
      auto child = parents[i];
      const auto shared = common_rate * gaussian(rng);
      for (std::size_t j = 0; j < n; ++j) {
        if (lower[j] == upper[j]) continue;
        child.scales[j] = std::clamp(child.scales[j] * std::exp(shared + local_rate * gaussian(rng)), 1e-12, 1.0);
        child.coordinates[j] = std::clamp(child.coordinates[j] + child.scales[j] * gaussian(rng), 0.0, 1.0);
      }
      child.loss = evaluate(child.coordinates);
      combined.push_back(std::move(child));
    }
    if (observe.calls == options_.evaluations) break;

    std::uniform_int_distribution<std::size_t> opponent(0, combined.size() - 2);
    for (std::size_t i = 0; i < combined.size(); ++i) {
      combined[i].wins = 0;
      for (std::size_t j = 0; j < options_.opponents; ++j) {
        auto index = opponent(rng);
        if (index >= i) ++index;
        if (combined[i].loss <= combined[index].loss) ++combined[i].wins;
      }
    }
    std::stable_sort(combined.begin(), combined.end(), [](const Individual& a, const Individual& b) {
      return a.wins == b.wins ? a.loss < b.loss : a.wins > b.wins;
    });
    combined.resize(size);
    parents = std::move(combined);
  }

  coordinates = std::move(observe.best);
  return observe.best_value;
}

} // namespace datamunge::optim
