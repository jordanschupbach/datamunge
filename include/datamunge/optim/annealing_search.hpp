#pragma once

/// @file
/// @brief Budgeted continuous variants of annealing and stochastic tunneling.

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>

namespace datamunge::optim {

/// @brief Visiting and acceptance rules for boxed annealing.
/// @par Example
/// @code
/// const auto method = dm::optim::AnnealingMethod::tsallis;
/// @endcode
/// @see AnnealingSearch
enum class AnnealingMethod {
  reheating, ///< Geometric cooling with stagnation-based reheating and optional restarts.
  tunneling, ///< Metropolis acceptance in a transformed loss landscape.
  tsallis,   ///< Generalized heavy-tailed visiting distribution and q-acceptance.
  vfsa       ///< Very fast coordinate cooling with sensitivity-based reannealing.
};

/// @brief Objective-call budget and controls for selected annealing rule.
/// @par Example
/// @code
/// dm::optim::AnnealingSearchOptions options;
/// options.method = dm::optim::AnnealingMethod::vfsa;
/// @endcode
/// @see AnnealingSearch
struct AnnealingSearchOptions {
  std::size_t evaluations{1000}; ///< Positive objective-call count including starts and probes.
  std::uint64_t seed{42}; ///< Seed of the local engine.
  AnnealingMethod method{AnnealingMethod::reheating}; ///< Visiting and acceptance rule.
  double temperature{1}; ///< Positive finite initial acceptance temperature, in loss units.
  double proposal_scale{0.2}; ///< Initial normalized Gaussian, q-Gaussian or VFSA proposal scale in (0,1].
  double cooling{0.98}; ///< Geometric multiplier in (0,1] for reheating and tunneling.
  std::size_t patience{64}; ///< Consecutive non-record trials before reheating, at least one.
  std::size_t restart_every{3}; ///< Reheats per uniform restart; zero disables restarting.
  double tunneling_strength{1}; ///< Positive finite scale of the nonlinear loss transformation.
  double visiting_q{2.3}; ///< Tsallis visiting index in (1,3).
  double acceptance_q{-5}; ///< Finite Tsallis acceptance index less than one.
  double vfsa_cooling{0.4}; ///< Positive finite exponent coefficient in very-fast cooling.
  std::size_t reanneal_every{32}; ///< VFSA proposals between sensitivity probes; zero disables.
  double probe_distance{0.001}; ///< Normalized finite-difference probe size in (0,1].
};

/// @brief Run boxed annealing with explicit work accounting and seeded replay.
/// @par Example
/// @code
/// const dm::optim::AnnealingSearch search;
/// @endcode
/// @see AnnealingSearchOptions, SimulatedAnnealing
class AnnealingSearch {
public:
  /// @brief Store annealing controls by value.
  /// @param options Budget, seed, proposal and acceptance settings.
  /// @par Example
  /// @code
  /// const dm::optim::AnnealingSearch search({256, 7});
  /// @endcode
  /// @see optimize()
  explicit AnnealingSearch(AnnealingSearchOptions options = {});

  /// @brief Return the earliest best state scored during annealing.
  ///
  /// Starts from clamped coordinates and projects normalized trials to the
  /// box. Reheating restores the initial temperature on stagnation and
  /// optionally resets the current point to a budgeted uniform proposal.
  /// Tunneling applies expm1 to the loss above the running record and accepts
  /// using differences of the transformed losses. Tsallis uses an isotropic
  /// multivariate q-Gaussian proposal and a power-law cooling schedule.
  /// VFSA uses individually cooled coordinate temperatures with
  /// sensitivity probes around the best point and temperature rescaling.
  /// Probe improvements affect the returned incumbent. Duplicate trials
  /// and partially completed probe sweeps count. Callback exceptions
  /// propagate without updating coordinates, and replay holds within a build.
  /// @param function Borrowed objective returning finite values.
  /// @param coordinates Finite starting point, replaced by the incumbent.
  /// @param lower Finite inclusive lower bounds.
  /// @param upper Finite inclusive upper bounds at least as large as lower.
  /// @return Loss already evaluated at the output coordinates.
  /// @throws std::invalid_argument Invalid shape, bounds, start or options.
  /// @throws std::domain_error The objective returns a nonfinite value.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2}, {1});
  /// std::vector<double> x{0};
  /// const auto loss = dm::optim::AnnealingSearch({8, 42}).optimize(f, x, {1}, {1});
  /// std::print("{}\n", loss); // 0
  /// @endcode
  /// @par References
  /// - Kirkpatrick, Gelatt and Vecchi (1983). https://doi.org/10.1126/science.220.4598.671
  /// - Wenzel and Hamacher (1999). https://doi.org/10.1103/PhysRevLett.82.3003
  /// - Tsallis and Stariolo (1996). https://doi.org/10.1016/S0378-4371(96)00271-3
  /// - Ingber (1989). https://doi.org/10.1016/0895-7177(89)90202-1
  /// @see SimulatedAnnealing, trace_optimize()
  double optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                  const std::vector<double>& lower, const std::vector<double>& upper) const;

private:
  AnnealingSearchOptions options_; ///< Owned run controls.
};

} // namespace datamunge::optim
