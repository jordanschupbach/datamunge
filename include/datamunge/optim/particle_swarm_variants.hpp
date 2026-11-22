#pragma once

/// @file
/// @brief Evaluation-budgeted continuous and binary particle swarm variants.

#include <datamunge/optim/discrete_distribution_search.hpp>
#include <datamunge/optim/function_types.hpp>

namespace datamunge::optim {

/// @brief Continuous particle update rule.
/// @par Example
/// @code
/// const auto rule = dm::optim::ParticleSwarmRule::fully_informed;
/// @endcode
/// @see ParticleSwarmSearch
enum class ParticleSwarmRule {
  bare_bones,             ///< Gaussian draw between personal and global best.
  fully_informed,         ///< Velocity attraction to all personal bests in a ring neighborhood.
  comprehensive_learning ///< Select and refresh a tournament exemplar for each coordinate.
};

/// @brief Evaluation budget, neighborhood and motion controls.
/// @par Example
/// @code
/// dm::optim::ParticleSwarmSearchOptions options;
/// options.rule = dm::optim::ParticleSwarmRule::comprehensive_learning;
/// @endcode
/// @see ParticleSwarmSearch
struct ParticleSwarmSearchOptions {
  std::size_t evaluations{1000}; ///< Positive objective-call count including initialization.
  std::uint64_t seed{42}; ///< Seed of the local engine.
  ParticleSwarmRule rule{ParticleSwarmRule::bare_bones}; ///< Continuous update rule.
  std::size_t population{20}; ///< Number of particles, at least three.
  double inertia{0.7298}; ///< Velocity multiplier in [0, 1].
  double attraction{1.49618}; ///< Finite positive velocity acceleration coefficient.
  double velocity_limit{0.5}; ///< Positive maximum velocity as a fraction of box width, at most one.
  std::size_t ring_radius{2}; ///< Positive number of neighbors on each side for FIPS.
  std::size_t refresh_after{7}; ///< Positive stalled iterations before refreshing CLPSO exemplars.
};

/// @brief Run bare-bones, fully informed or comprehensive learning particle swarm search.
/// @par Example
/// @code
/// const dm::optim::ParticleSwarmSearch search;
/// @endcode
/// @see ParticleSwarmSearchOptions, PSO
class ParticleSwarmSearch {
public:
  /// @brief Store the swarm controls by value.
  /// @param options Search budget, seed and motion controls.
  /// @par Example
  /// @code
  /// const dm::optim::ParticleSwarmSearch search({128, 7});
  /// @endcode
  /// @see optimize()
  explicit ParticleSwarmSearch(ParticleSwarmSearchOptions options = {});

  /// @brief Minimize a finite objective using the selected swarm update rule.
  ///
  /// Starts with a clamped supplied point and uniform particles. Each sweep
  /// uses the previous sweep's personal and global bests. Bare-bones sampling
  /// draws a normal point with mean at the midpoint of the personal and global
  /// best and deviation given by their absolute separation. Particle velocities
  /// for FIPS and CLPSO are limited in normalized coordinates, while positions
  /// are clipped to the box. CLPSO draws two distinct peers for each tournament
  /// and uses at least one peer when refreshing. At most a partial sweep is
  /// evaluated at the budget limit. Repeated points count. Local seeded replay
  /// holds within a build. Callback exceptions leave coordinates unchanged.
  /// @param function Borrowed finite-return objective.
  /// @param coordinates Finite starting point, replaced by the earliest best trial.
  /// @param lower Finite lower bounds.
  /// @param upper Finite upper bounds at least as large as lower, including fixed axes.
  /// @return Loss already evaluated at the returned point.
  /// @throws std::invalid_argument Invalid shape, box, start or options.
  /// @throws std::domain_error Objective returns a nonfinite value.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2}, {3});
  /// std::vector<double> x{1};
  /// const auto loss = dm::optim::ParticleSwarmSearch({8, 42}).optimize(f, x, {3}, {3});
  /// std::print("{}\n", loss); // 0
  /// @endcode
  /// @par References
  /// - Kennedy (2003). Bare Bones Particle Swarms. https://doi.org/10.1109/SIS.2003.1202251
  /// - Mendes, Kennedy and Neves (2004). https://doi.org/10.1109/TEVC.2004.826074
  /// - Liang, Qin, Suganthan and Baskar (2006). https://doi.org/10.1109/TEVC.2005.857610
  /// @see PSO, ParticleSwarmRule
  double optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                  const std::vector<double>& lower, const std::vector<double>& upper) const;

private:
  ParticleSwarmSearchOptions options_; ///< Owned configuration.
};

/// @brief Evaluation budget and logit update coefficients for binary PSO.
/// @par Example
/// @code
/// const dm::optim::BinarySwarmOptions options{256, 7, 12};
/// @endcode
/// @see binary_particle_swarm()
struct BinarySwarmOptions {
  std::size_t evaluations{1000}; ///< Positive objective-call count including initialization.
  std::uint64_t seed{42}; ///< Seed of the local engine.
  std::size_t population{20}; ///< At least two particles.
  double inertia{0.7298}; ///< Logit carryover in [0, 1].
  double cognitive{1.49618}; ///< Finite nonnegative attraction to the personal record.
  double social{1.49618}; ///< Finite nonnegative attraction to the swarm record.
  double velocity_limit{6}; ///< Positive finite absolute limit on the logits.
};

/// @brief Minimize a binary loss with sigmoid velocity and Bernoulli position updates.
///
/// The first sampled swarm uses independent fair bits and zero velocities.
/// Each bit's logit changes based on inertia and the difference between its
/// current value and the previous personal and global records. New bits are
/// independent given the updated logits. The budget includes every point,
/// including repeats and partial initialization or sweeps. Callback exceptions
/// propagate. Seeded replay holds within a build for deterministic callbacks.
/// @param objective Borrowed objective on feasible vectors of zero and one.
/// @param dimension Positive number of bits.
/// @param options Budget, seed, size and update coefficients.
/// @return Best evaluated point, its loss and actual work count.
/// @throws std::invalid_argument Empty callback, zero dimension or invalid controls.
/// @throws std::domain_error Objective returns nonfinite loss.
/// @par Example
/// @code
/// const auto result = dm::optim::binary_particle_swarm(
///     [](const auto& x) { return double(x[0]); }, 1, {64, 42});
/// std::print("{}\n", result.value); // 0
/// @endcode
/// @par References
/// - Kennedy and Eberhart (1997). https://doi.org/10.1109/ICSMC.1997.637339
/// @see ParticleSwarmSearch, discrete_distribution_search()
DiscreteDistributionResult binary_particle_swarm(const CategoricalObjective& objective,
                                                  std::size_t dimension,
                                                  const BinarySwarmOptions& options = {});

} // namespace datamunge::optim
