#pragma once
/// @file cma_es.hpp
/// @brief CMAES controls and native optimization contract.
#include <datamunge/optim/function_types.hpp>
#include <cstddef>
#include <cstdint>
namespace datamunge::optim {
/// @brief Configuration for CMAES; fields retain their documented native defaults.
/// Supply finite values in the supported ranges. Controls are copied at construction;
/// validation, stopping and loss scaling belong to optimize().
/// @par Example
/// @code
/// dm::optim::CMAESOptions o;
/// std::print("{}\n",o.max_generations); // 500
/// @endcode
/// @see CMAES
struct CMAESOptions {
  std::size_t population_size{0}; ///< Population size; CMA-ES zero selects its dimension-based default. Default 0.
  double initial_step_size{0.5}; ///< Positive initial CMA Gaussian search SD. Default 0.5.
  std::size_t max_generations{500}; ///< Maximum population generations. Default 500.
  double tolerance{1e-8}; ///< Stopping threshold; its actual gradient/change/displacement/scale meaning is method-specific. Default 1e-8.
  std::uint64_t seed{42}; ///< Local mt19937_64 seed; same-build reproducibility, not guaranteed independent streams. Default 42.
};
/// @brief CMAES on the ArbitraryFunction callback interface.
/// Unbounded covariance-adaptation evolution strategy with weighted elites, evolution paths and global step-size adaptation. Default offspring count is 4+floor(3*log(p)). Covariance eigensolve is dense with numerical eigenvalue floors. Stops on search-scale/axis tolerance or generation budget; stochastic results depend on seed and population, not a global certificate.
/// The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed;
/// coordinates mutate in place and can retain a partial state if a callback throws.
/// Calls use local optimizer state and propagate callback exceptions; sharing a mutable
/// objective across threads requires external synchronisation. Normal return alone
/// is not a uniform convergence certificate.
/// @par Example
/// @code
/// const dm::optim::CMAES optimizer;
/// std::print("{}\n",optimizer.options().max_generations); // 500
/// @endcode
/// @par References
/// - N. Hansen and A. Ostermeier (2001), Completely Derandomized Self-Adaptation in Evolution Strategies, https://doi.org/10.1162/106365601750190398
/// @see CMAESOptions, trace_optimize()
class CMAES {
 public:
  /// @brief Store a copy of the controls without running the objective.
  /// @param options Method-specific controls; defaults are listed by CMAESOptions.
  /// @par Example
  /// @code
  /// const dm::optim::CMAES optimizer;
  /// std::print("{}\n",optimizer.options().max_generations); // 500
  /// @endcode
  /// @see optimize()
  explicit CMAES(CMAESOptions options = {});
  /// @brief Run CMAES from the supplied point using the configured finite budget.
  /// Unbounded covariance-adaptation evolution strategy with weighted elites, evolution paths and global step-size adaptation. Default offspring count is 4+floor(3*log(p)). Covariance eigensolve is dense with numerical eigenvalue floors. Stops on search-scale/axis tolerance or generation budget; stochastic results depend on seed and population, not a global certificate.
  /// @param function Borrowed ArbitraryFunction with matching finite output shapes.
  /// @param coordinates Owned-by-caller finite start; overwritten with the native returned point.
  /// @return Objective at the native returned coordinates; not a stop-status code.
  /// @throws std::invalid_argument CMAES: initial_step_size must be positive; CMAES: tolerance must be non-negative; CMAES: population_size must be at least 2.
  /// @throws std::exception Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2,8},{1,-2});
  /// dm::optim::CMAESOptions controls;controls.max_generations=5;
  /// std::vector<double> x{0,0};
  /// const auto value=dm::optim::CMAES(controls).optimize(f,x);
  /// std::print("{}\n",std::isfinite(value)); // true
  /// @endcode
  /// @see trace_optimize(), CMAESOptions
  double optimize(ArbitraryFunction& function, std::vector<double>& coordinates) const;
  /// @brief Borrow this optimizer's stored configuration.
  /// @return Const option reference, valid during this object's lifetime; no mutation.
  [[nodiscard]] const CMAESOptions& options() const noexcept {return options_;}
 private:
  CMAESOptions options_; ///< Owned method configuration, copied at construction.
};
} // namespace datamunge::optim
