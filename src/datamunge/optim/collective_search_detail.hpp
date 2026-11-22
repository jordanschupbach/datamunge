#pragma once

/// @file
/// @brief Shared population and objective accounting for collective search kernels.

#include <datamunge/optim/collective_search.hpp>
#include "boxed_search_detail.hpp"

#include <random>
#include <utility>

namespace datamunge::optim::detail {

/// @brief One scored member of a unit-box population.
struct Member {
  std::vector<double> point; ///< Position in normalized coordinates.
  double value; ///< Finite loss previously observed at point.
};

/// @brief Prefer the lower-loss member; stable sorts break ties by arrival.
/// @param a First evaluated member.
/// @param b Second evaluated member.
/// @return Whether a has strictly lower loss.
inline bool better(const Member& a, const Member& b) { return a.value < b.value; }

/// @brief Scoring, clipping, random generators and total work of a population search.
struct Work {
  detail::CountingObjective observe; ///< Borrowed objective and historical incumbent.
  const std::vector<double>& lower; ///< Lower bounds.
  const std::vector<double>& upper; ///< Upper bounds.
  const CollectiveSearchOptions& options; ///< Validated run settings.
  std::mt19937_64 rng; ///< Local random engine.
  std::uniform_real_distribution<double> uniform; ///< Unit uniform draws.
  std::normal_distribution<double> normal; ///< Standard normal draws.
  std::vector<double> mapped; ///< Scratch coordinates in the original domain.
  std::vector<double> leader; ///< Historical best in normalized coordinates.

  /// @brief Whether another scored trial fits the budget.
  /// @return True before all objective calls have been used.
  bool available() const { return observe.calls < options.evaluations; }

  /// @brief Fraction of total budget already used.
  /// @return Objective calls divided by budget.
  double progress() const { return static_cast<double>(observe.calls) / options.evaluations; }

  /// @brief Clip, map and score one member, updating the historical best.
  /// @param trial Owned normalized candidate.
  /// @return Clipped normalized candidate and observed loss.
  Member score(std::vector<double> trial) {
    for (std::size_t j = 0; j < trial.size(); ++j) {
      trial[j] = lower[j] == upper[j] ? 0 : std::clamp(trial[j], 0.0, 1.0);
      mapped[j] = std::lerp(lower[j], upper[j], trial[j]);
    }
    const auto previous = observe.best_value;
    const auto loss = observe.evaluate(mapped);
    if (loss < previous) leader = trial;
    return {std::move(trial), loss};
  }

  /// @brief Generate an unscored independent uniform point.
  /// @return Unit-box coordinates.
  std::vector<double> random_point() {
    std::vector<double> x(mapped.size());
    for (auto& value : x) value = uniform(rng);
    return x;
  }
};

/// @brief Run tumble-and-swim moves with optional reproduction and dispersal.
/// @param work Shared objective-call budget and local random state.
/// @param population Evaluated cohort in the unit box.
/// @param extended Enable the additional phases of the extended method.
void chemotaxis(Work& work, std::vector<Member>& population, bool extended);

/// @brief Run prey, follow and swarm proposals using neighborhood visibility.
/// @param work Shared objective-call budget and local random state.
/// @param population Evaluated cohort in the unit box.
void fish(Work& work, std::vector<Member>& population);

/// @brief Advance fitness-weighted mass interactions and damped velocity.
/// @param work Shared objective-call budget and local random state.
/// @param population Evaluated cohort in the unit box.
void gravity(Work& work, std::vector<Member>& population);

/// @brief Update induced, food-directed and diffusion motion.
/// @param work Shared objective-call budget and local random state.
/// @param population Evaluated cohort in the unit box.
void krill(Work& work, std::vector<Member>& population);

/// @brief Update search particles using electric-force and velocity terms.
/// @param work Shared objective-call budget and local random state.
/// @param population Evaluated cohort in the unit box.
void charged(Work& work, std::vector<Member>& population);

/// @brief Apply memetic leaps, global fallback and shuffling.
/// @param work Shared objective-call budget and local random state.
/// @param population Evaluated cohort in the unit box.
void frogs(Work& work, std::vector<Member>& population);

/// @brief Assimilate and redistribute colonies under empire competition.
/// @param work Shared objective-call budget and local random state.
/// @param population Evaluated cohort in the unit box.
void empires(Work& work, std::vector<Member>& population);

/// @brief Run affinity cloning or a suppression-based artificial immune network.
/// @param work Shared objective-call budget and local random state.
/// @param population Evaluated cohort in the unit box.
/// @param extended Enable the additional phases of the extended method.
void clones(Work& work, std::vector<Member>& population, bool extended);

/// @brief Apply diffusion and two rank-dependent fractal update phases.
/// @param work Shared objective-call budget and local random state.
/// @param population Evaluated cohort in the unit box.
void fractals(Work& work, std::vector<Member>& population);

} // namespace datamunge::optim::detail
