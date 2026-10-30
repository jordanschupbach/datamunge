#pragma once
/// @file simulated_annealing.hpp
/// @brief SimulatedAnnealing controls and native optimization contract.
#include <datamunge/optim/function_types.hpp>
#include <cstddef>
#include <cstdint>
namespace datamunge::optim {
/// @brief Configuration for SimulatedAnnealing; fields retain their documented native defaults.
/// Supply finite values in the supported ranges. Controls are copied at construction;
/// validation, stopping and loss scaling belong to optimize().
/// @par Example
/// @code
/// dm::optim::SimulatedAnnealingOptions o;
/// std::print("{}\n",o.max_iterations); // 10000
/// @endcode
/// @see SimulatedAnnealing
struct SimulatedAnnealingOptions {
  double initial_temperature{10.0}; ///< Positive starting Metropolis temperature. Default 10.0.
  double cooling_rate{0.98}; ///< Geometric temperature multiplier. Default 0.98.
  std::size_t max_iterations{10000}; ///< Maximum native iterations/epochs as described by optimize(); zero performs no updates. Default 10000.
  double step_std_dev{1.0}; ///< Gaussian proposal standard deviation. Default 1.0.
  std::uint64_t seed{42}; ///< Local mt19937_64 seed; same-build reproducibility, not guaranteed independent streams. Default 42.
};
/// @brief SimulatedAnnealing on the ArbitraryFunction callback interface.
/// Gaussian random-walk proposals with geometrically decaying temperature and Metropolis acceptance of worse values. Returns the best visited point, not the last accepted state. A finite cooling schedule does not certify a global optimum; O(p) storage.
/// The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed;
/// coordinates mutate in place and can retain a partial state if a callback throws.
/// Calls use local optimizer state and propagate callback exceptions; sharing a mutable
/// objective across threads requires external synchronisation. Normal return alone
/// is not a uniform convergence certificate.
/// @par Example
/// @code
/// const dm::optim::SimulatedAnnealing optimizer;
/// std::print("{}\n",optimizer.options().max_iterations); // 10000
/// @endcode
/// @par References
/// - S. Kirkpatrick, C. D. Gelatt and M. P. Vecchi (1983), Optimization by Simulated Annealing, https://doi.org/10.1126/science.220.4598.671. Finite geometric cooling here supplies no global guarantee.
/// @see SimulatedAnnealingOptions, trace_optimize()
class SimulatedAnnealing {
 public:
  /// @brief Store a copy of the controls without running the objective.
  /// @param options Method-specific controls; defaults are listed by SimulatedAnnealingOptions.
  /// @par Example
  /// @code
  /// const dm::optim::SimulatedAnnealing optimizer;
  /// std::print("{}\n",optimizer.options().max_iterations); // 10000
  /// @endcode
  /// @see optimize()
  explicit SimulatedAnnealing(SimulatedAnnealingOptions options = {});
  /// @brief Run SimulatedAnnealing from the supplied point using the configured finite budget.
  /// Gaussian random-walk proposals with geometrically decaying temperature and Metropolis acceptance of worse values. Returns the best visited point, not the last accepted state. A finite cooling schedule does not certify a global optimum; O(p) storage.
  /// @param function Borrowed ArbitraryFunction with matching finite output shapes.
  /// @param coordinates Owned-by-caller finite start; overwritten with the native returned point.
  /// @return Objective at the native returned coordinates; not a stop-status code.
  /// @throws std::exception Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2,8},{1,-2});
  /// dm::optim::SimulatedAnnealingOptions controls;controls.max_iterations=5;
  /// std::vector<double> x{0,0};
  /// const auto value=dm::optim::SimulatedAnnealing(controls).optimize(f,x);
  /// std::print("{}\n",std::isfinite(value)); // true
  /// @endcode
  /// @see trace_optimize(), SimulatedAnnealingOptions
  double optimize(ArbitraryFunction& function, std::vector<double>& coordinates) const;
  /// @brief Borrow this optimizer's stored configuration.
  /// @return Const option reference, valid during this object's lifetime; no mutation.
  [[nodiscard]] const SimulatedAnnealingOptions& options() const noexcept {return options_;}
 private:
  SimulatedAnnealingOptions options_; ///< Owned method configuration, copied at construction.
};
} // namespace datamunge::optim
