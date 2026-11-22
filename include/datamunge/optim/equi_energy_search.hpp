#pragma once

/// @file
/// @brief Equi-energy-inspired bounded optimization using archived warmer states.

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>

namespace datamunge::optim {

/// @brief Work cap, temperature ladder and energy-history controls.
/// @par Example
/// @code
/// const dm::optim::EquiEnergyOptions options{128, 7, {0.2, 1.0, 5.0}};
/// @endcode
/// @see EquiEnergySearch
struct EquiEnergyOptions {
  std::size_t evaluations{1000}; ///< Positive cap including all chains and duplicated trials.
  std::uint64_t seed{42}; ///< Local mt19937_64 seed.
  std::vector<double> temperatures{0.2, 1, 5}; ///< At least two strictly increasing positive temperatures.
  double ring_width{1}; ///< Positive finite energy interval width in objective units.
  double jump_probability{0.25}; ///< Probability of looking for a ring jump after a local proposal.
  double proposal_scale{0.15}; ///< Gaussian standard deviation in units of box width, in (0,1].
  std::size_t history_capacity{128}; ///< Positive FIFO capacity per temperature.
};

/// @brief Best observed point and work counts for equi-energy optimization.
/// @par Example
/// @code
/// const dm::optim::EquiEnergyResult result{{0}, 0, 1, 0, 0};
/// @endcode
/// @see EquiEnergySearch
struct EquiEnergyResult {
  std::vector<double> coordinates; ///< Best evaluated point.
  double value; ///< Objective loss already evaluated at the retained point.
  std::size_t evaluations; ///< Actual local scoring work.
  std::size_t jumps; ///< Proposed jumps with a nonempty matching energy ring.
  std::size_t accepted_jumps; ///< Accepted archived-state transitions.
};

/// @brief Search a box by local Metropolis proposals and equi-energy jumps.
/// @par Example
/// @code
/// const dm::optim::EquiEnergySearch search;
/// @endcode
/// @see EquiEnergyOptions
class EquiEnergySearch {
public:
  /// @brief Store the controls by value.
  /// @param options Budget, temperatures and proposal settings.
  /// @par Example
  /// @code
  /// const dm::optim::EquiEnergySearch search({128, 7});
  /// @endcode
  /// @see run()
  explicit EquiEnergySearch(EquiEnergyOptions options = {});

  /// @brief Execute a finite-history optimization variant of equi-energy sampling.
  ///
  /// The cold chain starts from the clamped input and hotter chains from
  /// uniform points. Every active chain makes a budgeted Gaussian Metropolis
  /// proposal in each hot-to-cold sweep. It can then jump to an already scored
  /// state from the next hotter chain in the same energy interval, using an
  /// exchange-ratio Metropolis test. This optimization variant uses bounded
  /// rolling histories and returns the best scored trial. Local duplicates
  /// count and cached jumps use no additional calls. Seeded replay holds
  /// within a build. Callback exceptions propagate.
  /// @param function Borrowed finite-return objective.
  /// @param start Nonempty finite initial coordinates for the cold chain.
  /// @param lower Finite lower box bounds.
  /// @param upper Finite inclusive upper bounds at least as large as lower.
  /// @return Best observed point and scoring and jump counts.
  /// @throws std::invalid_argument Invalid box, start or controls.
  /// @throws std::domain_error The objective returns a nonfinite loss.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2}, {1});
  /// const auto run = dm::optim::EquiEnergySearch({8, 7}).run(f, {1}, {1}, {1});
  /// std::print("{}\n", run.evaluations); // 8
  /// @endcode
  /// @par References
  /// - Kou, Zhou and Wong (2006). https://doi.org/10.1214/009053606000000515
  /// @see optimize(), ParallelTempering
  EquiEnergyResult run(ArbitraryFunction& function, const std::vector<double>& start,
                       const std::vector<double>& lower, const std::vector<double>& upper) const;

  /// @brief Run equi-energy optimization and update the input on success.
  /// @param function Borrowed finite-return objective.
  /// @param coordinates Finite start, replaced by the best point.
  /// @param lower Finite lower box bounds.
  /// @param upper Finite inclusive upper box bounds at least as large as lower.
  /// @return Observed loss at the returned point.
  /// @throws std::invalid_argument Invalid box, start or controls.
  /// @throws std::domain_error The objective returns a nonfinite loss.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2}, {1});
  /// std::vector<double> x{0};
  /// const auto value = dm::optim::EquiEnergySearch({8, 7}).optimize(f, x, {1}, {1});
  /// std::print("{}\n", value); // 0
  /// @endcode
  /// @see run()
  double optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                  const std::vector<double>& lower, const std::vector<double>& upper) const;

private:
  EquiEnergyOptions options_; ///< Owned run controls.
};

} // namespace datamunge::optim
