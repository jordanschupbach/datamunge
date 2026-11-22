#pragma once

/// @file
/// @brief Budgeted boxed swarm and nature-inspired population methods.

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>

namespace datamunge::optim {

/// @brief Population search mechanism.
/// @par Example
/// @code
/// const auto method = dm::optim::CollectiveSearchMethod::teaching_learning;
/// @endcode
/// @see CollectiveSearch
enum class CollectiveSearchMethod {
  teaching_learning,  ///< Improve by teacher/mean displacement and learner interaction.
  salp_swarm,         ///< Food-guided leader and averaging chain followers.
  flower_pollination, ///< Lévy attraction or local differential pollination.
  invasive_weed,      ///< Rank-dependent seed production and shrinking dispersal.
  bat,                ///< Frequency-driven velocities, loudness and pulse switching.
  biogeography,       ///< Fitness-ranked immigration and emigration with mutation.
  bacterial_chemotaxis, ///< Random tumbles and directional swims.
  bacterial_foraging, ///< Chemotaxis, swarm interaction, reproduction and dispersal.
  artificial_fish,    ///< Visual preying, crowd-aware following and swarming.
  gravitational,      ///< Fitness-dependent masses and stochastic attraction.
  krill_herd,         ///< Induced motion, foraging and random diffusion.
  shuffled_frog,      ///< Local memeplex leaps with periodic rank shuffling.
  imperialist,        ///< Colony assimilation, revolution and empire competition.
  clonal_selection,   ///< Clone the strongest members and mutate inversely to affinity.
  artificial_immune_network, ///< Clonal maturation and suppression of nearby cells.
  stochastic_fractal, ///< Gaussian diffusion and rank-dependent population updates.
  charged_system      ///< Position-dependent electrostatic forces and adaptive motion.
};

/// @brief Budget, seed and method-specific controls for collective search.
/// @par Example
/// @code
/// dm::optim::CollectiveSearchOptions options;
/// options.method = dm::optim::CollectiveSearchMethod::flower_pollination;
/// @endcode
/// @see CollectiveSearch
struct CollectiveSearchOptions {
  std::size_t evaluations{1000}; ///< Positive score-call budget, including initialization.
  std::uint64_t seed{42}; ///< Seed of the local engine.
  CollectiveSearchMethod method{CollectiveSearchMethod::teaching_learning}; ///< Distinct proposal mechanism.
  std::size_t population{24}; ///< Initial size or weed capacity, at least four.
  double step_scale{0.2}; ///< Normalized local motion scale for methods that use one, in (0,1].
  double global_probability{0.8}; ///< Flower probability of the global rule in [0,1].
  double mutation_probability{0.03}; ///< Biogeography maximum random replacement probability in [0,1].
};

/// @brief Minimize boxed losses using biology-inspired or teaching–learning updates.
/// @par Example
/// @code
/// const dm::optim::CollectiveSearch search;
/// @endcode
/// @see CollectiveSearchOptions
class CollectiveSearch {
public:
  /// @brief Copy collective-search settings.
  /// @param options Population, method and score-call controls.
  /// @par Example
  /// @code
  /// const dm::optim::CollectiveSearch search({128, 7});
  /// @endcode
  /// @see optimize()
  explicit CollectiveSearch(CollectiveSearchOptions options = {});

  /// @brief Return the earliest best trial evaluated by the configured population dynamics.
  ///
  /// Uses a clamped start and uniform initial population, with smaller initial
  /// cohorts for invasive weed. All proposals use unit-box coordinates and
  /// are clipped. Every call counts, including partial phases, random seeds
  /// and repeated trials. Salp positions may move uphill. Bat accepts improving
  /// proposals through the loudness gate. Biogeography preserves one elite
  /// and selects migration partners by rank. The incumbent tracks every
  /// scored point. Seeded replay holds within a build and exceptions leave
  /// coordinates unchanged.
  /// @param function Borrowed finite-return objective.
  /// @param coordinates Finite start, replaced by the overall best trial.
  /// @param lower Finite box lower bounds.
  /// @param upper Finite inclusive upper bounds at least as large as lower.
  /// @return Loss observed at the retained point.
  /// @throws std::invalid_argument Invalid box, start, budget, method or active options.
  /// @throws std::domain_error The objective returns a nonfinite value.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2}, {1});
  /// std::vector<double> x{0};
  /// const auto loss = dm::optim::CollectiveSearch({8, 7}).optimize(f, x, {1}, {1});
  /// std::print("{}\n", loss); // 0
  /// @endcode
  /// @par References
  /// - Rao, Savsani and Vakharia (2012). https://doi.org/10.1016/j.ins.2011.08.006
  /// - Mirjalili et al. (2017). https://doi.org/10.1016/j.advengsoft.2017.07.002
  /// - Yang (2012). https://arxiv.org/abs/1312.5673
  /// - Mehrabian and Lucas (2006). https://doi.org/10.1016/j.ecoinf.2006.07.003
  /// - Yang (2010), A New Metaheuristic Bat-Inspired Algorithm.
  /// - Simon (2008). https://doi.org/10.1109/TEVC.2008.919004
  /// @see PSO, trace_optimize()
  double optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                  const std::vector<double>& lower, const std::vector<double>& upper) const;

private:
  CollectiveSearchOptions options_; ///< Owned method controls.
};

} // namespace datamunge::optim
