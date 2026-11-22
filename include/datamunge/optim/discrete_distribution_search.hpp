#pragma once

/// @file
/// @brief Estimation-of-distribution algorithms for categorical and binary search.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace datamunge::optim {

/// @brief One category index per coordinate in a Cartesian product.
using CategoricalPoint = std::vector<std::size_t>;

/// @brief Borrowed-call minimization objective for a complete categorical point.
using CategoricalObjective = std::function<double(const CategoricalPoint&)>;

/// @brief Probability model and update for categorical search.
/// @par Example
/// @code
/// const auto method = dm::optim::DiscreteDistributionMethod::compact_ga;
/// @endcode
/// @see discrete_distribution_search()
enum class DiscreteDistributionMethod {
  cross_entropy, ///< Smooth per-axis categorical frequencies toward selected samples.
  umda,          ///< Refit independent binary marginals by selected sample frequencies.
  pbil,          ///< Move binary probabilities toward the best batch sample and mutate them.
  compact_ga,    ///< Pairwise winner-loser updates to independent binary probabilities.
  mimic,         ///< Refit a binary dependency chain by conditional entropy.
  boa            ///< Fit a sparse binary Bayesian network by greedy BIC edge addition.
};

/// @brief Budget, model and update parameters for discrete distribution search.
/// @par Example
/// @code
/// dm::optim::DiscreteDistributionOptions options;
/// options.method = dm::optim::DiscreteDistributionMethod::mimic;
/// @endcode
/// @see discrete_distribution_search()
struct DiscreteDistributionOptions {
  std::size_t evaluations{1000}; ///< Positive objective-call budget, including duplicates.
  std::uint64_t seed{42}; ///< Local mt19937_64 seed.
  DiscreteDistributionMethod method{DiscreteDistributionMethod::cross_entropy}; ///< Model and update rule.
  std::size_t population{32}; ///< Positive sampled batch size; compact GA always uses pairs.
  double elite_fraction{0.25}; ///< Fraction selected by CEM, UMDA and MIMIC, in (0, 1].
  double learning_rate{0.3}; ///< CEM and PBIL blend toward selected values, in (0, 1].
  double mutation_probability{0.02}; ///< PBIL chance of nudging a coordinate toward a random bit.
  double mutation_strength{0.05}; ///< PBIL blend toward the random bit, in [0, 1].
  std::size_t virtual_population{32}; ///< Compact GA inverse update step, at least two.
  std::size_t maximum_parents{2}; ///< BOA parent limit in [1, 3].
  double exploration{0.02}; ///< Uniform mixture probability for each sampled coordinate, in [0, 1].
};

/// @brief An owned categorical incumbent and its exact work count.
/// @par Example
/// @code
/// const dm::optim::DiscreteDistributionResult result{{1, 0}, 0, 1};
/// @endcode
/// @see discrete_distribution_search()
struct DiscreteDistributionResult {
  CategoricalPoint point; ///< Earliest point with the lowest observed value.
  double value; ///< Objective value at the retained point.
  std::size_t evaluations; ///< Actual objective calls, including duplicate samples.
};

/// @brief Sample feasible category assignments and adapt their probability model.
///
/// Labels on coordinate j are the integers from zero through cardinalities[j]-1.
/// CEM supports different positive cardinalities, including fixed axes. Other
/// methods require binary coordinates. MIMIC fits a chain by picking a root of
/// minimum entropy and successive coordinates minimizing conditional entropy
/// given the current endpoint. Unobserved conditioning events use the fitted
/// marginal. BOA greedily adds acyclic edges using the BIC score and samples
/// smoothed Boolean conditionals in topological order. PBIL uses the best point of each batch. Compact GA compares pairs,
/// moving probabilities by 1/virtual_population where winner and loser differ;
/// an incomplete final pair is scored without an update. A final partial batch
/// counts against the same positive budget. Exceptions from callbacks propagate.
/// Seeded replay holds within a build for deterministic objectives.
/// @param objective Borrowed finite-return loss on each complete point.
/// @param cardinalities Positive category counts for a nonempty point.
/// @param options Budget and model controls (default categorical cross-entropy).
/// @return Best scored assignment and exact objective-call count.
/// @throws std::invalid_argument Invalid controls, empty objective or incompatible shape.
/// @throws std::domain_error Objective result is nonfinite.
/// @par Example
/// @code
/// const auto result = dm::optim::discrete_distribution_search(
///     [](const auto& point) { return double(point[0]); }, {1}, {8});
/// std::print("{}\n", result.value); // 0
/// @endcode
/// @par References
/// - Rubinstein (1999). https://doi.org/10.1023/A:1010091220143
/// - Baluja (1994). Population-Based Incremental Learning. CMU-CS-94-163.
/// - Harik, Lobo and Goldberg (1999). https://doi.org/10.1109/4235.797971
/// - De Bonet, Isbell and Viola (1997). MIMIC: Finding Optima by Estimating Probability Densities.
///   https://proceedings.neurips.cc/paper/1996/hash/4c22bd444899d3b6047a10b20a2f26db-Abstract.html
/// - Pelikan, Goldberg and Cantú-Paz (1999), BOA: The Bayesian Optimization Algorithm.
/// @see DiscreteDistributionOptions
DiscreteDistributionResult discrete_distribution_search(
    const CategoricalObjective& objective, const std::vector<std::size_t>& cardinalities,
    const DiscreteDistributionOptions& options = {});

} // namespace datamunge::optim
