#pragma once
/// @file sgd.hpp
/// @brief SGD controls and native optimization contract.
#include <datamunge/optim/function_types.hpp>
#include <cstddef>
#include <cstdint>
namespace datamunge::optim {
/// @brief Configuration for SGD; fields retain their documented native defaults.
/// Supply finite values in the supported ranges. Controls are copied at construction;
/// validation, stopping and loss scaling belong to optimize().
/// @par Example
/// @code
/// dm::optim::SGDOptions o;
/// std::print("{}\n",o.max_epochs); // 100
/// @endcode
/// @see SGD
struct SGDOptions {
  double step_size{0.01}; ///< Fixed learning-rate/initial search step; use the objective scaling stated by optimize(). Default 0.01.
  std::size_t max_epochs{100}; ///< Maximum passes/outer epochs; zero performs no epoch updates. Default 100.
  std::size_t batch_size{1}; ///< Positive number of terms per averaged mini-batch. Default 1.
  double tolerance{1e-8}; ///< Stopping threshold; its actual gradient/change/displacement/scale meaning is method-specific. Default 1e-8.
  bool shuffle{true}; ///< Reshuffle all component indices before each epoch. Default true.
  std::uint64_t seed{42}; ///< Local mt19937_64 seed; same-build reproducibility, not guaranteed independent streams. Default 42.
};
/// @brief SGD on the DifferentiableSeparableFunction callback interface.
/// Uses the mean gradient of each mini-batch, reshuffling component indices each epoch when enabled. Full objective remains the sum of components. Stops on absolute full-objective change per epoch. Nonempty components and positive batch_size are required; seeds reproduce the same build, not a cross-library shuffle transcript.
/// The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed;
/// coordinates mutate in place and can retain a partial state if a callback throws.
/// Calls use local optimizer state and propagate callback exceptions; sharing a mutable
/// objective across threads requires external synchronisation. Normal return alone
/// is not a uniform convergence certificate.
/// @par Example
/// @code
/// const dm::optim::SGD optimizer;
/// std::print("{}\n",optimizer.options().max_epochs); // 100
/// @endcode
/// @par References
/// - H. Robbins and S. Monro (1951), A Stochastic Approximation Method, https://doi.org/10.1214/aoms/1177729586. The fixed-rate reshuffling variant here differs from their decreasing-step stochastic approximation.
/// @see SGDOptions, trace_optimize()
class SGD {
 public:
  /// @brief Store a copy of the controls without running the objective.
  /// @param options Method-specific controls; defaults are listed by SGDOptions.
  /// @par Example
  /// @code
  /// const dm::optim::SGD optimizer;
  /// std::print("{}\n",optimizer.options().max_epochs); // 100
  /// @endcode
  /// @see optimize()
  explicit SGD(SGDOptions options = {});
  /// @brief Run SGD from the supplied point using the configured finite budget.
  /// Uses the mean gradient of each mini-batch, reshuffling component indices each epoch when enabled. Full objective remains the sum of components. Stops on absolute full-objective change per epoch. Nonempty components and positive batch_size are required; seeds reproduce the same build, not a cross-library shuffle transcript.
  /// @param function Borrowed DifferentiableSeparableFunction with matching finite output shapes.
  /// @param coordinates Owned-by-caller finite start; overwritten with the native returned point.
  /// @return Objective at the native returned coordinates; not a stop-status code.
  /// @throws std::invalid_argument SGD: batch_size must be positive; SGD::optimize: function has zero terms.
  /// @throws std::exception Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs.
  /// @par Example
  /// @code
  /// dm::optim::LeastSquaresObjective f({{1},{1}},{1,3});
  /// dm::optim::SGDOptions controls;controls.max_epochs=5;
  /// std::vector<double> x{0};
  /// const auto value=dm::optim::SGD(controls).optimize(f,x);
  /// std::print("{}\n",std::isfinite(value)); // true
  /// @endcode
  /// @see trace_optimize(), SGDOptions
  double optimize(DifferentiableSeparableFunction& function, std::vector<double>& coordinates) const;
  /// @brief Borrow this optimizer's stored configuration.
  /// @return Const option reference, valid during this object's lifetime; no mutation.
  [[nodiscard]] const SGDOptions& options() const noexcept {return options_;}
 private:
  SGDOptions options_; ///< Owned method configuration, copied at construction.
};
} // namespace datamunge::optim
