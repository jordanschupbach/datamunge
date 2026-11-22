#pragma once

/// @file
/// @brief Evaluation-budgeted archive and differential search on Pareto fronts.

#include <datamunge/optim/multi_objective.hpp>

#include <cstdint>

namespace datamunge::optim {

/// @brief Search and survivor rule for Pareto optimization.
/// @par Example
/// @code
/// const auto method = dm::optim::ParetoSearchMethod::paes;
/// @endcode
/// @see pareto_search()
enum class ParetoSearchMethod {
  paes,    ///< One-parent mutation with adaptive-grid archive density.
  pesa2,   ///< Adaptive-grid region selection with real-valued variation.
  mopso,   ///< Multi-objective particle swarm with archive guides.
  gde3,    ///< Generalized differential evolution with dominance and crowding.
  nsga3,   ///< Nondominated sorting and reference-direction niching.
  ibea,    ///< Additive epsilon-indicator fitness and iterative removal.
  hype,    ///< Monte Carlo weighted hypervolume estimates.
  sms_emoa ///< Steady-state evolution with exact two-objective hypervolume contributions.
};

/// @brief Budget, seed, population and variation controls for Pareto search.
/// @par Example
/// @code
/// dm::optim::ParetoSearchOptions options;
/// options.method = dm::optim::ParetoSearchMethod::mopso;
/// @endcode
/// @see pareto_search()
struct ParetoSearchOptions {
  std::size_t evaluations{1000}; ///< Positive objective-call cap, including initialization.
  std::uint64_t seed{42}; ///< Seed of the local engine.
  ParetoSearchMethod method{ParetoSearchMethod::paes}; ///< Search rule.
  std::size_t population{24}; ///< Initial population for PESA-II, MOPSO or GDE3.
  std::size_t capacity{50}; ///< Positive adaptive-grid archive capacity.
  std::size_t grid_divisions{8}; ///< Number of adaptive bins on each objective axis, at least two.
  double mutation_rate{0}; ///< Per-axis mutation probability in [0,1]; zero uses inverse dimension.
  double differential_weight{0.6}; ///< GDE3 scale in (0,2].
  double crossover_rate{0.9}; ///< Binomial or SBX crossover probability in [0,1].
  std::size_t reference_divisions{4}; ///< NSGA-III positive default simplex lattice subdivision count.
  std::vector<std::vector<double>> reference_directions{}; ///< Optional nonnegative NSGA-III directions.
  double indicator_scale{0.05}; ///< Positive finite fitness scale for IBEA.
  std::size_t hypervolume_samples{1000}; ///< Positive Monte Carlo sample count for HypE.
};

/// @brief Owned approximation to a Pareto front and its scoring count.
/// @par Example
/// @code
/// const dm::optim::ParetoSearchResult result{{{{0}, {0, 1}}}, 1};
/// @endcode
/// @see pareto_search()
struct ParetoSearchResult {
  std::vector<ParetoPoint> front; ///< Mutually nondominated points with unique objective vectors.
  std::size_t evaluations; ///< Actual objective callbacks, including duplicate trials.
};

/// @brief Approximate the Pareto front of a vector-valued objective over a finite box.
///
/// All objective components are minimized. PAES uses a single random start.
/// PESA-II selects parents through a tournament of occupied grid regions;
/// MOPSO samples guides with weights inverse to region occupancy.
/// Adaptive grids are bounded relative to their members and remove a point
/// from a most crowded region on overflow. GDE3 uses synchronous differential
/// proposals, keeps both mutually nondominating points for survival, then
/// truncates by the shared nondominated rank and crowding procedure.
/// NSGA-III associates normalized objectives with reference directions.
/// IBEA ranks by additive epsilon indicators. HypE estimates normalized
/// hypervolume with a reference of 1.1 in every objective. SMS-EMOA uses
/// exact contributions with the same reference in two objectives.
/// Variation takes place in the unit box. Equal bounds fix a coordinate.
/// Every method uses the entire budget including partial initialization and
/// generations. Replay within a build requires deterministic callbacks.
/// Callback exceptions propagate.
/// @param function Borrowed objective returning a fixed vector of at least two finite values.
/// @param lower Finite nonempty lower box bounds.
/// @param upper Finite inclusive upper bounds at least as large as lower.
/// @param options Evolution, archive and budget controls.
/// @return Final archive or population front, with actual work count.
/// @throws std::invalid_argument Invalid box, objective shape or active controls.
/// @throws std::domain_error Objective returns a nonfinite component.
/// @throws std::length_error The ranker rejects an oversized cohort or the default lattice exceeds 4096 directions.
/// @par Example
/// @code
/// struct Tradeoff : dm::optim::MultiObjectiveFunction {
///   std::vector<double> evaluate(const std::vector<double>& x) override { return {x[0], 1-x[0]}; }
/// } f;
/// const auto result = dm::optim::pareto_search(f, {0}, {1}, {24});
/// std::print("{}\n", result.evaluations); // 24
/// @endcode
/// @par References
/// - Knowles and Corne (2000). https://doi.org/10.1162/106365600568167
/// - Corne et al. (2001). PESA-II: Region-based selection in evolutionary multiobjective optimization.
/// - Coello Coello, Toscano Pulido and Salazar Lechuga (2004). https://doi.org/10.1109/TEVC.2004.826067
/// - Kukkonen and Lampinen (2005). https://doi.org/10.1109/CEC.2005.1554717
/// - Deb and Jain (2014). https://doi.org/10.1109/TEVC.2013.2281535
/// - Zitzler and Künzli (2004). https://doi.org/10.1007/978-3-540-30217-9_84
/// - Bader and Zitzler (2011). https://doi.org/10.1162/EVCO_a_00009
/// - Beume, Naujoks and Emmerich (2007). https://doi.org/10.1016/j.ejor.2006.08.008
/// @see MultiObjectiveFunction, NSGA2
ParetoSearchResult pareto_search(MultiObjectiveFunction& function,
                                const std::vector<double>& lower, const std::vector<double>& upper,
                                const ParetoSearchOptions& options = {});

} // namespace datamunge::optim
