#pragma once

/// @file
/// @brief Separable CMA-ES and population-size restart schedules.

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>

namespace datamunge::optim {

/// @brief Covariance form and restart policy for a boxed CMA search.
/// @par Example
/// @code
/// const auto method = dm::optim::CMASearchMethod::bipop;
/// @endcode
/// @see CMASearch
enum class CMASearchMethod {
  separable, ///< One search with diagonal covariance and scaled learning rates.
  ipop,      ///< Restart dense CMA-ES with successively doubled populations.
  bipop      ///< Alternate dense increasing and small-population restart regimes by work spent.
};

/// @brief Hard evaluation budget and CMA-ES restart controls.
/// @par Example
/// @code
/// dm::optim::CMASearchOptions options;
/// options.method = dm::optim::CMASearchMethod::ipop;
/// @endcode
/// @see CMASearch
struct CMASearchOptions {
  std::size_t evaluations{1000}; ///< Positive count of objective calls, including all restart starts.
  std::uint64_t seed{42}; ///< Seed of the local generator for restart seeds and centers.
  CMASearchMethod method{CMASearchMethod::separable}; ///< Covariance/restart variant.
  std::size_t population{0}; ///< At least two, or zero for the dimension-based default.
  double initial_scale{0.25}; ///< Initial standard deviation in box-width units, in (0,1].
  double tolerance{1e-10}; ///< Nonnegative finite scale threshold for restarting dense search.
  std::size_t generations_per_restart{30}; ///< Maximum complete generations in a dense restart.
};

/// @brief Minimize a finite objective with boxed separable, IPOP or BIPOP CMA-ES.
/// @par Example
/// @code
/// const dm::optim::CMASearch search;
/// @endcode
/// @see CMASearchOptions, CMAES
class CMASearch {
public:
  /// @brief Copy the CMA search controls.
  /// @param options Budget, seed and covariance/restart controls.
  /// @par Example
  /// @code
  /// const dm::optim::CMASearch search({200, 7});
  /// @endcode
  /// @see optimize()
  explicit CMASearch(CMASearchOptions options = {});

  /// @brief Run CMA search under a global evaluation budget and return its earliest best trial.
  ///
  /// Proposal and step adaptation take place in the unit box. Clipped
  /// offspring use their realized displacement for covariance adaptation.
  /// Separable CMA-ES keeps only a diagonal covariance and runs until the
  /// budget. Dense variants restart at a new uniform center after convergence
  /// or the generation cap. IPOP doubles population sizes. BIPOP interleaves
  /// increasing populations with smaller random populations and step sizes,
  /// selecting the small regime when it has spent fewer objective calls.
  /// Partial populations are evaluated without another covariance update.
  /// Duplicate points count, seeded replay holds within a build, and
  /// callback exceptions leave the input coordinates unchanged.
  /// @param function Borrowed objective returning finite losses.
  /// @param coordinates Finite start replaced by the best scored coordinates on success.
  /// @param lower Finite inclusive lower bounds.
  /// @param upper Finite inclusive upper bounds at least as large as lower.
  /// @return Loss observed at the returned point.
  /// @throws std::invalid_argument Invalid box, start or active controls.
  /// @throws std::length_error Dense covariance dimensions are too large.
  /// @throws std::domain_error Objective returns nonfinite loss.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2}, {1});
  /// std::vector<double> x{1};
  /// const auto loss = dm::optim::CMASearch({12, 42}).optimize(f, x, {1}, {1});
  /// std::print("{}\n", loss); // 0
  /// @endcode
  /// @par References
  /// - Ros and Hansen (2008), A Simple Modification in CMA-ES Achieving Linear Time and Space Complexity.
  /// - Auger and Hansen (2005). https://www.cmap.polytechnique.fr/~nikolaus.hansen/cec2005ipopcmaes.pdf
  /// - Hansen (2009), Benchmarking a BI-population CMA-ES. https://cma-es.github.io/index.html
  /// @see CMAES, trace_optimize()
  double optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                  const std::vector<double>& lower, const std::vector<double>& upper) const;

private:
  CMASearchOptions options_; ///< Owned budget and covariance/restart controls.
};

} // namespace datamunge::optim
