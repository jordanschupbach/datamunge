#pragma once

/// @file
/// @brief Boxed stochastic approximation using paired objective probes.

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>

namespace datamunge::optim {

/// @brief Perturbation and gradient estimator for stochastic approximation.
/// @par Example
/// @code
/// const auto method = dm::optim::StochasticApproximationMethod::rdsa;
/// @endcode
/// @see StochasticApproximation
enum class StochasticApproximationMethod {
  spsa, ///< Independent signed simultaneous perturbations.
  rdsa  ///< Uniform spherical directions and a random-direction derivative.
};

/// @brief Gain schedule and budget for paired derivative-free trials.
/// @par Example
/// @code
/// dm::optim::StochasticApproximationOptions options;
/// options.evaluations = 501;
/// @endcode
/// @see StochasticApproximation
struct StochasticApproximationOptions {
  std::size_t evaluations{1000}; ///< Positive objective-call budget including the initial trial.
  std::uint64_t seed{42}; ///< Local mt19937_64 seed.
  StochasticApproximationMethod method{StochasticApproximationMethod::spsa}; ///< Gradient estimator.
  double gain{0.1}; ///< Positive finite numerator of the descent step.
  double perturbation{0.1}; ///< Initial normalized probe radius, in (0, 1].
  double stability{10}; ///< Finite nonnegative shift in the descent gain schedule.
  double gain_decay{0.602}; ///< Finite nonnegative exponent for descent gain decay.
  double perturbation_decay{0.101}; ///< Finite nonnegative exponent for probe radius decay.
};

/// @brief Fit a stochastic descent direction using objective evaluations only.
/// @par Example
/// @code
/// const dm::optim::StochasticApproximation search;
/// @endcode
/// @see StochasticApproximationOptions
class StochasticApproximation {
public:
  /// @brief Store the approximation controls by value.
  /// @param options Budget, perturbation rule and gain schedule.
  /// @par Example
  /// @code
  /// const dm::optim::StochasticApproximation search({501, 7});
  /// @endcode
  /// @see optimize()
  explicit StochasticApproximation(StochasticApproximationOptions options = {});

  /// @brief Return the earliest best point scored by paired-probe stochastic descent.
  ///
  /// Scores the clamped start, then draws two opposite perturbations per
  /// update in normalized box coordinates. Boundary probes are clipped;
  /// estimators use the realized displacements, with zero motion on fixed
  /// axes. The paired loss difference is divided by the per-axis span for
  /// SPSA or projected back along the probe separation with the number of
  /// free axes as a multiplier for RDSA. On iteration k starting at one,
  /// the two gains are gain/(stability+k)^gain_decay and
  /// perturbation/k^perturbation_decay. If one call remains, scores the
  /// current center. Repeated and coincident probes still count.
  /// Replay holds within a build. Callback exceptions leave coordinates unchanged.
  /// @param function Borrowed finite-return objective.
  /// @param coordinates Finite start point, replaced by the best observed coordinates.
  /// @param lower Finite lower box bounds.
  /// @param upper Finite upper box bounds at least as large as lower.
  /// @return Observed loss at the retained point.
  /// @throws std::invalid_argument Invalid box, start, budget or controls.
  /// @throws std::domain_error The objective returns a nonfinite loss.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2}, {1});
  /// std::vector<double> x{1};
  /// auto value = dm::optim::StochasticApproximation({9, 7}).optimize(f, x, {1}, {1});
  /// std::print("{}\n", value); // 0
  /// @endcode
  /// @par References
  /// - Spall (1992). https://www.jhuapl.edu/spsa/pdf-spsa/spall_tac92.pdf
  /// - Fabian (1967), Stochastic Approximation of Minima with Improved Asymptotic Speed.
  /// @see StochasticLocalSearch, trace_optimize()
  double optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                  const std::vector<double>& lower, const std::vector<double>& upper) const;

private:
  StochasticApproximationOptions options_; ///< Owned objective-call and gain controls.
};

} // namespace datamunge::optim
