#pragma once

/// @file
/// @brief Randomized design search with an explicit objective-call budget.

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>

namespace datamunge::optim {

/// @brief Sampling design used for the entire evaluation budget.
/// @par Example
/// @code
/// const auto design = dm::optim::RandomDesign::latin_hypercube;
/// @endcode
/// @see RandomSearch
enum class RandomDesign {
  uniform,          ///< Independent uniform draws.
  latin_hypercube,  ///< Independently permuted, uniformly jittered strata.
  sobol,            ///< Sobol sequence with binary digital shifts; at most 16 axes.
  halton            ///< Prime-base Halton sequence with uniform shifts modulo one.
};

/// @brief Budget, seed and sampling design for randomized search.
/// @par Example
/// @code
/// dm::optim::RandomSearchOptions options{128, 42};
/// @endcode
/// @see RandomSearch
struct RandomSearchOptions {
  std::size_t evaluations{1000}; ///< Positive objective-call budget, including duplicate points.
  std::uint64_t seed{42}; ///< Seed for the local mt19937_64 engine.
  RandomDesign design{RandomDesign::uniform}; ///< Sampling rule.
};

/// @brief Minimize an objective by testing a randomized design in a box.
/// @par Example
/// @code
/// const dm::optim::RandomSearch search;
/// @endcode
/// @see RandomSearchOptions
class RandomSearch {
public:
  /// @brief Store search controls by value.
  /// @param options Evaluation budget, seed and design, defaulting to 1000, 42 and uniform.
  /// @par Example
  /// @code
  /// const dm::optim::RandomSearch search({128, 42});
  /// @endcode
  /// @see optimize()
  explicit RandomSearch(RandomSearchOptions options = {});

  /// @brief Evaluate the design and return the loss of its earliest minimizer.
  ///
  /// Consumes the positive objective-call budget exactly, counting duplicate
  /// points. Equal bounds fix an axis. Latin hypercube storage is O(budget × dimension)
  /// and strata depend on the full budget. Other designs use O(dimension) storage.
  /// Sobol supports at most 16 axes and 2^53 points.
  /// Seeded replay is reproducible within a build. Callback exceptions propagate
  /// and leave coordinates unchanged. Optimizer state is local to the call.
  /// @param function Borrowed objective returning finite values.
  /// @param coordinates Nonempty vector whose length specifies the dimension.
  ///        Replaced by the best evaluated point on success.
  /// @param lower Finite lower bounds, one per axis.
  /// @param upper Finite upper bounds at least as large as the corresponding lower bounds.
  /// @return Objective value already evaluated at the returned point.
  /// @throws std::invalid_argument The budget, dimension, design or bounds are invalid.
  /// @throws std::domain_error The objective returns a nonfinite value.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2}, {1});
  /// std::vector<double> x{0};
  /// const auto value = dm::optim::RandomSearch({8, 42}).optimize(f, x, {1}, {1});
  /// std::print("{}\n", value); // 0
  /// @endcode
  /// @par References
  /// - Solis and Wets (1981), Minimization by Random Search Techniques.
  ///   https://doi.org/10.1287/moor.6.1.19
  /// - McKay, Beckman and Conover (1979). https://doi.org/10.1080/00401706.1979.10489755
  /// - Halton (1960). https://doi.org/10.1007/BF01386213
  /// - Joe and Kuo, Notes on generating Sobol sequences.
  ///   https://web.maths.unsw.edu.au/~fkuo/sobol/joe-kuo-notes.pdf
  /// @see trace_optimize(), RandomSearchOptions
  double optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                  const std::vector<double>& lower, const std::vector<double>& upper) const;

private:
  RandomSearchOptions options_; ///< Owned search configuration.
};

/// @brief Budget and scale for Solis–Wets adaptive search.
/// @par Example
/// @code
/// const dm::optim::SolisWetsOptions options{128, 42, 0.1};
/// @endcode
/// @see SolisWets
struct SolisWetsOptions {
  std::size_t evaluations{1000}; ///< Positive objective-call budget, including the start.
  std::uint64_t seed{42}; ///< Seed for the local mt19937_64 engine.
  double initial_step{0.2}; ///< Gaussian deviation in box-width units, in (0, 1].
};

/// @brief Minimize a boxed objective with biased, paired Gaussian trials.
/// @par Example
/// @code
/// const dm::optim::SolisWets search;
/// @endcode
/// @see SolisWetsOptions
class SolisWets {
public:
  /// @brief Store adaptive search controls by value.
  /// @param options Budget, seed and initial step, defaulting to 1000, 42 and 0.2.
  /// @par Example
  /// @code
  /// const dm::optim::SolisWets search({128, 42, 0.1});
  /// @endcode
  /// @see optimize()
  explicit SolisWets(SolisWetsOptions options = {});

  /// @brief Return the best point observed by box-projected Solis–Wets search.
  ///
  /// Scores the clamped start. Trials use a normalized perturbation and its
  /// opposite, with the second trial skipped on improvement or budget exhaustion.
  /// Bias follows the signed accepted displacement and halves on a failed pair.
  /// The scale doubles after five successes and halves after three failures,
  /// with an upper limit of one. Duplicate points consume budget. Seeded replay
  /// holds within a build. Callback exceptions propagate with coordinates unchanged.
  /// @param function Borrowed objective returning finite values.
  /// @param coordinates Finite starting point, replaced on success by the incumbent.
  /// @param lower Finite lower bounds, one per axis.
  /// @param upper Finite upper bounds at least as large as each lower bound.
  /// @return Objective value already evaluated at the returned point.
  /// @throws std::invalid_argument Invalid dimensions, bounds, budget, start or step.
  /// @throws std::domain_error The objective returns a nonfinite value.
  /// @par Example
  /// @code
  /// dm::optim::QuadraticObjective f({2}, {1});
  /// std::vector<double> x{0};
  /// const auto value = dm::optim::SolisWets({8, 42, 0.2}).optimize(f, x, {1}, {1});
  /// std::print("{}\n", value); // 0
  /// @endcode
  /// @par References
  /// - Solis and Wets (1981), Minimization by Random Search Techniques.
  ///   https://doi.org/10.1287/moor.6.1.19
  /// @see trace_optimize(), RandomSearch
  double optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                  const std::vector<double>& lower, const std::vector<double>& upper) const;

private:
  SolisWetsOptions options_; ///< Owned search configuration.
};

} // namespace datamunge::optim
