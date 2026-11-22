#pragma once

/// @file
/// @brief Boxed evolutionary programming with self-adaptive mutations and survival tournaments.

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>

namespace datamunge::optim {

/// @brief Budget, population and adaptive mutation settings.
/// @par Example
/// @code
/// const dm::optim::EvolutionaryProgrammingOptions options{128, 7, 16, 6, 0.2};
/// @endcode
/// @see EvolutionaryProgramming
struct EvolutionaryProgrammingOptions {
  std::size_t evaluations{1000}; ///< Positive objective-call budget, including initialization.
  std::uint64_t seed{42}; ///< Local mt19937_64 seed.
  std::size_t population{20}; ///< Initial population count, at least two.
  std::size_t opponents{10}; ///< Positive tournament comparisons per candidate.
  double initial_scale{0.2}; ///< Initial per-axis Gaussian deviation, in normalized units and in (0,1].
};

/// @brief Evolve mutations and retain survivors through stochastic comparisons.
/// @par Example
/// @code
/// const dm::optim::EvolutionaryProgramming search;
/// @endcode
/// @see EvolutionaryProgrammingOptions
class EvolutionaryProgramming {
public:
  /// @brief Copy optimizer controls.
  /// @param options Budget, seed and population and mutation settings.
  /// @par Example
  /// @code
  /// const dm::optim::EvolutionaryProgramming search({128, 7});
  /// @endcode
  /// @see optimize()
  explicit EvolutionaryProgramming(EvolutionaryProgrammingOptions options = {});

  /// @brief Minimize the objective with Gaussian mutation and parent–offspring tournaments.
  ///
  /// Initializes one clamped start and uniformly sampled additional parents.
  /// Each parent creates one offspring. Mutations use log-normally adapted
  /// per-axis scales with a shared and an independent normal component.
  /// Each parent and offspring plays a fixed number of randomly chosen
  /// opponents from the combined population. Survivors have the most wins,
  /// with lower loss breaking ties. The returned incumbent is the earliest
  /// best observed trial. Partial generations and repeated scores count.
  /// Replay holds within a build and callback failures leave coordinates unchanged.
  /// @param function Borrowed objective returning finite loss.
  /// @param coordinates Finite start, replaced by the best scored point.
  /// @param lower Finite box lower bounds.
  /// @param upper Finite box upper bounds at least as large as lower.
  /// @return Loss at the retained point, already evaluated.
  /// @throws std::invalid_argument Invalid start, box, budget or options.
  /// @throws std::domain_error A callback returns nonfinite loss.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2}, {1});
  /// std::vector<double> x{0};
  /// const auto loss = dm::optim::EvolutionaryProgramming({12, 7}).optimize(f, x, {1}, {1});
  /// std::print("{}\n", loss); // 0
  /// @endcode
  /// @par References
  /// - Fogel (1992), Evolving artificial intelligence.
  /// - Yao, Liu and Lin (1999), Evolutionary programming made faster.
  ///   https://doi.org/10.1109/4235.771163
  /// @see EvolutionStrategy, trace_optimize()
  double optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                  const std::vector<double>& lower, const std::vector<double>& upper) const;

private:
  EvolutionaryProgrammingOptions options_; ///< Owned budget and variation controls.
};

} // namespace datamunge::optim
