#pragma once

/// @file
/// @brief Fixed-candidate simulation allocation and sequential racing.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace datamunge::optim {

/// @brief Finite-candidate simulation selection policy.
/// @par Example
/// @code
/// const auto method = dm::optim::SimulationSearchMethod::kim_nelson;
/// @endcode
/// @see simulation_search()
enum class SimulationSearchMethod {
  kim_nelson,       ///< Fully sequential indifference-zone screening using paired pilot variances.
  ocba,             ///< Allocate additional replications by estimated optimal allocation ratios.
  confidence_race,  ///< Screen with simultaneous time-uniform Hoeffding bounds.
  friedman_race     ///< Repeated block rank testing with adjusted post-hoc screening.
};

/// @brief One simulation replication for a candidate index and a supplied local seed.
using SimulationObjective = std::function<double(std::size_t, std::uint64_t)>;

/// @brief Budget and statistical settings for ranking and selection.
/// @par Example
/// @code
/// dm::optim::SimulationSearchOptions options;
/// options.method = dm::optim::SimulationSearchMethod::confidence_race;
/// @endcode
/// @see simulation_search()
struct SimulationSearchOptions {
  std::size_t evaluations{1000}; ///< Positive replication budget, including the pilot.
  std::uint64_t seed{42}; ///< Seed of the local replication seed generator.
  SimulationSearchMethod method{SimulationSearchMethod::ocba}; ///< Allocation or screening policy.
  std::size_t pilot{5}; ///< Desired initial complete blocks, at least two.
  double alpha{0.05}; ///< Error level in (0,0.5) for screening methods.
  double indifference{0.1}; ///< Finite positive indifference-zone width for Kim–Nelson.
  double lower_loss{0}; ///< Known finite lower outcome bound for confidence racing.
  double upper_loss{1}; ///< Known finite upper outcome bound for confidence racing.
};

/// @brief Replication count, estimated mean and whether a candidate survived screening.
/// @par Example
/// @code
/// const dm::optim::SimulationEstimate estimate{2, 0.5, true};
/// @endcode
/// @see SimulationSearchResult
struct SimulationEstimate {
  std::size_t evaluations{}; ///< Observed replications of this candidate.
  double mean{}; ///< Sample mean, NaN when no evaluation was made.
  bool active{true}; ///< Candidate remains eligible for selection.
};

/// @brief Owned ranking-and-selection result with exact work counts.
/// @par Example
/// @code
/// const dm::optim::SimulationSearchResult run;
/// std::print("{}\n", run.evaluations); // 0
/// @endcode
/// @see simulation_search()
struct SimulationSearchResult {
  std::vector<SimulationEstimate> estimates; ///< One estimate per candidate, in callback index order.
  std::optional<std::size_t> selected; ///< Best sampled surviving candidate index.
  std::size_t evaluations{}; ///< Total completed replication calls.
  bool screened_to_one{}; ///< A screening method stopped with a single eligible candidate.
};

/// @brief Select a candidate with small expected loss using budgeted repeated simulations.
///
/// Screening methods share a seed across a block to allow paired scenarios,
/// eliminating candidates only at complete blocks. Kim–Nelson uses variance
/// of paired differences estimated from the pilot; its error guarantee
/// requires independent normal blocks and the indifference-zone conditions.
/// Confidence racing requires independent bounded draws across blocks.
/// Friedman racing uses average tied ranks, the approximate Friedman test,
/// and a Bonferroni-adjusted post-hoc comparison with the leading mean rank.
/// OCBA uses independent seeds per observation and spends the full budget.
/// A partial final block counts every sampled outcome and performs no screen.
/// The selected candidate minimizes the sample mean among sampled survivors,
/// with index ties. Callback exceptions propagate and seeded replay holds
/// within a build for deterministic simulations on the supplied seeds.
/// @param simulation Borrowed finite-return simulator invoked with candidate index and seed.
/// @param candidates Positive number of candidate indices from zero.
/// @param options Work cap, allocation policy and statistical controls.
/// @return Owned estimates and chosen sampled survivor.
/// @throws std::invalid_argument Invalid controls or empty simulator.
/// @throws std::domain_error Nonfinite outcome or loss outside the given confidence-race bounds.
/// @par Example
/// @code
/// const auto run = dm::optim::simulation_search(
///     [](std::size_t i, std::uint64_t) { return double(i); }, 3, {32, 7});
/// std::print("{}\n", *run.selected); // 0
/// @endcode
/// @par References
/// - Kim and Nelson (2001). https://doi.org/10.1145/502109.502111
/// - Chen, Lin, Yücesan and Chick (2000), Simulation Budget Allocation for Further
///   Enhancing the Efficiency of Ordinal Optimization.
/// - Birattari et al. (2002). https://www.metaheuristics.org/downloads/BirStuPaqVar2002_gecco.pdf
/// @see SimulationSearchOptions
SimulationSearchResult simulation_search(const SimulationObjective& simulation, std::size_t candidates,
                                          const SimulationSearchOptions& options = {});

} // namespace datamunge::optim
