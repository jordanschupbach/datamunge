/// @file
/// @brief Bounded rolling energy rings and Metropolis archive jumps.

#include <datamunge/optim/equi_energy_search.hpp>

#include "boxed_search_detail.hpp"

#include <deque>
#include <random>
#include <utility>

namespace datamunge::optim {
/// @brief One scored state reused by ring jumps.
namespace {

/// @brief Coordinate and cached objective pair.
struct EnergyState {
  std::vector<double> position; ///< Normalized position.
  double value; ///< Loss previously evaluated at this position.
};

} // namespace

EquiEnergySearch::EquiEnergySearch(EquiEnergyOptions options) : options_(std::move(options)) {}

EquiEnergyResult EquiEnergySearch::run(ArbitraryFunction& function, const std::vector<double>& start,
                                      const std::vector<double>& lower, const std::vector<double>& upper) const {
  detail::validate_box(start.size(), lower, upper, options_.evaluations);
  const auto& ladder = options_.temperatures;
  if (ladder.size() < 2 || options_.history_capacity == 0 ||
      !(options_.ring_width > 0 && std::isfinite(options_.ring_width) &&
        options_.jump_probability >= 0 && options_.jump_probability <= 1 &&
        options_.proposal_scale > 0 && options_.proposal_scale <= 1)) {
    throw std::invalid_argument("Invalid equi-energy options");
  }
  double previous = 0;
  for (const auto temperature : ladder) {
    if (!(std::isfinite(temperature) && temperature > previous)) {
      throw std::invalid_argument("Equi-energy temperatures must be positive and strictly increasing");
    }
    previous = temperature;
  }
  auto point = start;
  auto unit = detail::normalize_start(point, lower, upper);
  detail::CountingObjective observe(function);
  std::mt19937_64 rng(options_.seed);
  std::uniform_real_distribution<double> uniform;
  std::normal_distribution<double> normal;
  /// Score after mapping the normalized coordinates into the caller's box.
  const auto evaluate = [&](std::vector<double> normalized) {
    for (std::size_t j = 0; j < normalized.size(); ++j) {
      point[j] = std::lerp(lower[j], upper[j], normalized[j]);
    }
    const auto loss = observe.evaluate(point);
    return EnergyState{std::move(normalized), loss};
  };
  /// Map an objective value to an origin-aligned energy ring.
  const auto ring = [&](double loss) { return std::floor(static_cast<long double>(loss) / options_.ring_width); };

  std::vector<EnergyState> chains;
  const auto count = std::min(ladder.size(), options_.evaluations);
  chains.push_back({std::move(unit), observe.evaluate(point)});
  while (chains.size() < count) {
    std::vector<double> position(start.size());
    for (std::size_t j = 0; j < position.size(); ++j) {
      if (lower[j] != upper[j]) position[j] = uniform(rng);
    }
    chains.push_back(evaluate(std::move(position)));
  }
  std::vector<std::deque<EnergyState>> archive(count);
  for (std::size_t i = 0; i < count; ++i) archive[i].push_back(chains[i]);
  std::size_t jumps = 0, accepted = 0;
  while (observe.calls < options_.evaluations) {
    for (std::size_t i = count; i > 0 && observe.calls < options_.evaluations;) {
      --i;
      auto& chain = chains[i];
      auto trial = chain.position;
      for (std::size_t j = 0; j < trial.size(); ++j) {
        if (lower[j] != upper[j]) trial[j] = std::clamp(trial[j] + options_.proposal_scale * normal(rng), 0.0, 1.0);
      }
      auto candidate = evaluate(std::move(trial));
      if (candidate.value <= chain.value ||
          uniform(rng) < std::exp((static_cast<long double>(chain.value) - candidate.value) / ladder[i])) {
        chain = std::move(candidate);
      }
      if (i + 1 < count && uniform(rng) < options_.jump_probability) {
        std::vector<const EnergyState*> eligible;
        const auto band = ring(chain.value);
        for (const auto& warm : archive[i+1]) {
          if (ring(warm.value) == band) eligible.push_back(&warm);
        }
        if (!eligible.empty()) {
          ++jumps;
          const auto& guide = *eligible[std::uniform_int_distribution<std::size_t>(0, eligible.size() - 1)(rng)];
          const auto beta_difference = (1 - static_cast<long double>(ladder[i]) / ladder[i+1]) / ladder[i];
          if (guide.value <= chain.value ||
              uniform(rng) < std::exp(beta_difference * (static_cast<long double>(chain.value) - guide.value))) {
            chain = guide;
            ++accepted;
          }
        }
      }
      if (archive[i].size() == options_.history_capacity) archive[i].pop_front();
      archive[i].push_back(chain);
    }
  }
  return {std::move(observe.best), observe.best_value, observe.calls, jumps, accepted};
}

double EquiEnergySearch::optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                                 const std::vector<double>& lower, const std::vector<double>& upper) const {
  auto result = run(function, coordinates, lower, upper);
  coordinates = std::move(result.coordinates);
  return result.value;
}

} // namespace datamunge::optim
