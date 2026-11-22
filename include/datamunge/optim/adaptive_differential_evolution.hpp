#pragma once

/// @file
/// @brief Modern differential evolution variants with exact objective-call budgets.

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>

namespace datamunge::optim {

/// @brief Adaptation strategy for differential mutation and crossover.
/// @par Example
/// @code
/// const auto method = dm::optim::AdaptiveDEMethod::lshade;
/// @endcode
/// @see AdaptiveDifferentialEvolution
enum class AdaptiveDEMethod {
  jde,   ///< Inherited control parameters independently refreshed with probability 0.1.
  sade,  ///< Adapt choice among four strategies and their crossover centers.
  jade,  ///< Current-to-pbest mutation with an archive and smoothed success statistics.
  shade, ///< Multiple success-history memories and gain-weighted updates.
  lshade ///< Success-history DE with linear population size reduction.
};

/// @brief Budget, population and active adaptation controls.
/// @par Example
/// @code
/// dm::optim::AdaptiveDEOptions options;
/// options.method = dm::optim::AdaptiveDEMethod::jade;
/// @endcode
/// @see AdaptiveDifferentialEvolution
struct AdaptiveDEOptions {
  std::size_t evaluations{1000}; ///< Positive objective-call count, including initialization and repeats.
  std::uint64_t seed{42}; ///< Seed of the local engine.
  AdaptiveDEMethod method{AdaptiveDEMethod::jde}; ///< Adaptation policy.
  std::size_t population{24}; ///< Number of initial members, at least six.
  double pbest_fraction{0.2}; ///< Elite fraction for JADE and L-SHADE, in (0, 1].
  double jade_learning{0.1}; ///< JADE smoothing weight for current successes, in (0, 1].
  std::size_t learning_period{20}; ///< Positive success window for SaDE, in generations.
  std::size_t history_size{6}; ///< Positive number of SHADE and L-SHADE memory slots.
  std::size_t minimum_population{4}; ///< L-SHADE endpoint, at least four and at most population.
  bool archive{true}; ///< Preserve replaced parents for current-to-pbest mutations.
};

/// @brief Minimize a boxed objective using adaptive differential evolution.
/// @par Example
/// @code
/// const dm::optim::AdaptiveDifferentialEvolution search;
/// @endcode
/// @see AdaptiveDEOptions, DifferentialEvolution
class AdaptiveDifferentialEvolution {
public:
  /// @brief Store evolution controls by value.
  /// @param options Budget, variant and evolution controls.
  /// @par Example
  /// @code
  /// const dm::optim::AdaptiveDifferentialEvolution search({128, 7});
  /// @endcode
  /// @see optimize()
  explicit AdaptiveDifferentialEvolution(AdaptiveDEOptions options = {});

  /// @brief Return the lowest loss observed by the configured adaptive DE variant.
  ///
  /// Initializes with a clamped start and uniform points, then performs
  /// synchronous mutation, crossover and non-worsening replacement in normalized
  /// coordinates. jDE uses rand/1 with binomial crossover. SaDE uses rolling
  /// strategy outcomes and successful crossover medians. JADE, SHADE and
  /// L-SHADE use current-to-pbest with midpoint correction between a violated
  /// bound and the target's position. SHADE varies the elite fraction by trial.
  /// L-SHADE culls the worst members according to progress through the budget.
  /// Partial initialization and generations use every available call. The
  /// result is the earliest best trial. Callbacks propagate exceptions with
  /// coordinates unchanged. Seeded replay holds within a build.
  /// @param function Borrowed objective returning finite losses.
  /// @param coordinates Finite start replaced by the best point on return.
  /// @param lower Finite inclusive lower bounds.
  /// @param upper Finite inclusive upper bounds at least as large as lower.
  /// @return Already evaluated loss of the output coordinates.
  /// @throws std::invalid_argument Invalid box, start or active controls.
  /// @throws std::domain_error Objective returns nonfinite loss.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2}, {1});
  /// std::vector<double> x{0};
  /// const auto loss = dm::optim::AdaptiveDifferentialEvolution({12, 42}).optimize(f, x, {1}, {1});
  /// std::print("{}\n", loss); // 0
  /// @endcode
  /// @par References
  /// - Brest et al. (2006). https://doi.org/10.1109/TEVC.2006.872133
  /// - Qin, Huang and Suganthan (2009). https://doi.org/10.1109/TEVC.2008.927706
  /// - Zhang and Sanderson (2009). https://doi.org/10.1109/TEVC.2009.2014613
  /// - Tanabe and Fukunaga (2013). https://doi.org/10.1109/CEC.2013.6557555
  /// - Tanabe and Fukunaga (2014). https://doi.org/10.1109/CEC.2014.6900380
  /// @see DifferentialEvolution, trace_optimize()
  double optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                  const std::vector<double>& lower, const std::vector<double>& upper) const;

private:
  AdaptiveDEOptions options_; ///< Owned run controls.
};

} // namespace datamunge::optim
