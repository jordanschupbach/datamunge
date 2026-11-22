#pragma once

/// @file
/// @brief Box-constrained stochastic local search and randomized local minimization.

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>

namespace datamunge::optim {

/// @brief Proposal and scale update for an improving-only local search.
/// @par Example
/// @code
/// const auto method = dm::optim::StochasticLocalMethod::one_plus_one;
/// @endcode
/// @see StochasticLocalSearch
enum class StochasticLocalMethod {
  hill_climbing,  ///< Fixed Gaussian scale.
  luus_jaakola,   ///< Uniform proposals in a contracting rectangle.
  one_plus_one    ///< Gaussian mutation with one-fifth success-rule adaptation.
};

/// @brief Controls shared by boxed improving-only stochastic searches.
/// @par Example
/// @code
/// dm::optim::StochasticLocalOptions options;
/// options.method = dm::optim::StochasticLocalMethod::luus_jaakola;
/// @endcode
/// @see StochasticLocalSearch
struct StochasticLocalOptions {
  std::size_t evaluations{1000}; ///< Positive budget, including the starting point and duplicates.
  std::uint64_t seed{42}; ///< Seed for the local engine.
  StochasticLocalMethod method{StochasticLocalMethod::hill_climbing}; ///< Proposal rule.
  double initial_scale{0.2}; ///< Gaussian standard deviation or uniform half-width in (0, 1].
  std::size_t update_interval{10}; ///< Positive trial count between region or scale updates.
  double scale_factor{0.85}; ///< Contraction factor in (0, 1) for the adaptive methods.
};

/// @brief Improve a clamped start using budgeted stochastic proposals.
/// @par Example
/// @code
/// const dm::optim::StochasticLocalSearch search;
/// @endcode
/// @see StochasticLocalOptions
class StochasticLocalSearch {
public:
  /// @brief Store local search controls.
  /// @param options Proposal, budget, seed and scale settings.
  /// @par Example
  /// @code
  /// const dm::optim::StochasticLocalSearch search({128, 42});
  /// @endcode
  /// @see optimize()
  explicit StochasticLocalSearch(StochasticLocalOptions options = {});

  /// @brief Return the earliest best observed trial in the box.
  ///
  /// All scales are relative to each box width. Gaussian proposals are clipped;
  /// Luus–Jaakola samples uniformly from a contracting rectangle intersected
  /// with the box. The one-fifth rule increases scale above 20% successes,
  /// decreases it below 20%, and caps the scale at one. Strict improvement
  /// defines success. Every trial uses one objective call, including duplicates.
  /// Replay holds within a build and exceptions leave coordinates unchanged.
  /// @param function Borrowed objective returning finite losses.
  /// @param coordinates Finite starting coordinates, replaced by the incumbent.
  /// @param lower Finite lower bounds, one per axis.
  /// @param upper Finite inclusive upper bounds at least as large as lower.
  /// @return Loss already evaluated at the returned coordinates.
  /// @throws std::invalid_argument Invalid box, start, budget, method or scale controls.
  /// @throws std::domain_error Objective returns a nonfinite value.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2}, {1});
  /// std::vector<double> x{1};
  /// const auto loss = dm::optim::StochasticLocalSearch({8, 42}).optimize(f, x, {0}, {2});
  /// std::print("{}\n", loss); // 0
  /// @endcode
  /// @par References
  /// - Luus and Jaakola (1973). https://doi.org/10.1002/aic.690190413
  /// - Rechenberg (1973). Evolutionsstrategie. https://katalog.slub-dresden.de/id/0-024852090
  /// @see SolisWets, trace_optimize()
  double optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                  const std::vector<double>& lower, const std::vector<double>& upper) const;

private:
  StochasticLocalOptions options_; ///< Owned solver controls.
};

/// @brief Global proposal and acceptance rule when combining local solves.
/// @par Example
/// @code
/// const auto method = dm::optim::HybridSearchMethod::basin_hopping;
/// @endcode
/// @see HybridRandomSearch
enum class HybridSearchMethod {
  multistart,     ///< Start independent local solves from uniformly distributed seeds.
  basin_hopping, ///< Perturb the current local minimizer and use Metropolis basin acceptance.
  mlsl           ///< Filter low-ranked uniform samples using a shrinking single-linkage radius.
};

/// @brief Controls the budget, local minimizer and hybrid strategy.
/// @par Example
/// @code
/// dm::optim::HybridSearchOptions options;
/// options.method = dm::optim::HybridSearchMethod::mlsl;
/// @endcode
/// @see HybridRandomSearch
struct HybridSearchOptions {
  std::size_t evaluations{1000}; ///< Positive total evaluation count, including local work.
  std::uint64_t seed{42}; ///< Seed for global proposals and derived local seeds.
  HybridSearchMethod method{HybridSearchMethod::multistart}; ///< Restart, hopping or clustering rule.
  std::size_t local_evaluations{64}; ///< Positive cap for each Solis–Wets minimization.
  double local_step{0.15}; ///< Initial Solis–Wets scale in (0, 1].
  double hop_scale{0.2}; ///< Basin-hopping Gaussian scale in normalized box units, in (0, 1].
  double temperature{1}; ///< Positive finite basin acceptance temperature in loss units.
  std::size_t sample_batch{16}; ///< Positive number of new uniform samples per MLSL level.
  double sample_fraction{0.25}; ///< Fraction of retained MLSL samples eligible for descent, in (0, 1].
  double linkage_scale{1}; ///< Positive finite multiplier of the MLSL critical radius.
};

/// @brief Combine global proposals with budgeted Solis–Wets local minimizations.
/// @par Example
/// @code
/// const dm::optim::HybridRandomSearch search;
/// @endcode
/// @see HybridSearchOptions, SolisWets
class HybridRandomSearch {
public:
  /// @brief Store hybrid search controls.
  /// @param options Global and local control settings.
  /// @par Example
  /// @code
  /// const dm::optim::HybridRandomSearch search({128, 42});
  /// @endcode
  /// @see optimize()
  explicit HybridRandomSearch(HybridSearchOptions options = {});

  /// @brief Minimize through independent restarts, basin hopping or multilevel clustering.
  ///
  /// Multistart and basin hopping locally minimize the clamped initial point
  /// first. MLSL first scores that point, then accumulates uniform sample batches.
  /// Its radius in normalized coordinates decays as (log(N)/N)^(1/d), where
  /// d counts non-fixed axes. It descends from eligible samples with no
  /// better-ranked nearby sample and no nearby previously found local optimum.
  /// Each start or trial counts once per actual callback, including duplicates.
  /// A small remaining budget truncates local work. Exactly the total budget
  /// is consumed. Replay holds within a build; exceptions leave coordinates unchanged.
  /// @param function Borrowed objective returning finite losses.
  /// @param coordinates Finite starting point, replaced by the earliest best observed point.
  /// @param lower Finite lower bounds, one per axis.
  /// @param upper Finite inclusive upper bounds at least as large as lower.
  /// @return Loss already evaluated at the returned coordinates.
  /// @throws std::invalid_argument Invalid box, start, method or active controls.
  /// @throws std::domain_error Objective returns a nonfinite loss.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2}, {1});
  /// std::vector<double> x{1};
  /// const auto loss = dm::optim::HybridRandomSearch({8, 42}).optimize(f, x, {1}, {1});
  /// std::print("{}\n", loss); // 0
  /// @endcode
  /// @par References
  /// - Wales and Doye (1997). https://doi.org/10.1021/jp970984n
  /// - Rinnooy Kan and Timmer (1987). https://doi.org/10.1007/BF02592071
  /// @see SolisWets, trace_optimize()
  double optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                  const std::vector<double>& lower, const std::vector<double>& upper) const;

private:
  HybridSearchOptions options_; ///< Owned global and local settings.
};

} // namespace datamunge::optim
