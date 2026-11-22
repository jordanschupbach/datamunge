#pragma once

/// @file
/// @brief Binary linkage-tree and gene-pool optimal mixing.

#include <datamunge/optim/discrete_distribution_search.hpp>

namespace datamunge::optim {

/// @brief Linkage-model donor selection mode.
/// @par Example
/// @code
/// const auto method = dm::optim::LinkageMixingMethod::gene_pool;
/// @endcode
/// @see linkage_search()
enum class LinkageMixingMethod {
  parent_pair, ///< Use one paired donor across linkage-tree subsets.
  gene_pool    ///< Draw a fresh donor per subset and use forced elite mixing on stagnation.
};

/// @brief Population, bit shape and budget options for linkage mixing.
/// @par Example
/// @code
/// const dm::optim::LinkageSearchOptions options{128, 7, 24, dm::optim::LinkageMixingMethod::gene_pool};
/// @endcode
/// @see linkage_search()
struct LinkageSearchOptions {
  std::size_t evaluations{1000}; ///< Positive budget, including initial points and distinct trial evaluations.
  std::uint64_t seed{42}; ///< Seed of the local engine.
  std::size_t population{24}; ///< Population size, at least two.
  LinkageMixingMethod method{LinkageMixingMethod::gene_pool}; ///< Donor sampling and forced-improvement mode.
};

/// @brief Optimize bit vectors with a learned linkage tree.
///
/// Conditional blocks are formed by average-linkage clustering on empirical
/// mutual information. Each randomized pass tries donor bits for one subset
/// at a time and retains non-worsening trials. Parent-pair mixing uses a
/// fixed donor per receiver. Gene-pool mixing chooses donors per subset,
/// draws around the population and uses the elite to force improvement
/// when a receiver stagnates. Identical transfers require no scoring.
/// If a sweep proposes no changes, a fresh uniform chromosome provides
/// diversification. Every actual score call counts, with a partial final
/// sweep allowed. Callback exceptions propagate and same-build replay uses
/// the configured local seed.
/// @param objective Borrowed finite objective accepting bit vectors.
/// @param dimension Positive number of bit coordinates.
/// @param options Evaluation budget, seed, population size and mixing mode.
/// @return Earliest best scored vector and exact evaluation count.
/// @throws std::invalid_argument Empty objective, zero dimension or invalid options.
/// @throws std::domain_error Objective returns a nonfinite loss.
/// @throws std::length_error Linkage matrix size overflows.
/// @par Example
/// @code
/// const auto run = dm::optim::linkage_search(
///     [](const auto& x) { return double(x[0]); }, 1, {32, 7});
/// std::print("{}\n", run.value); // 0
/// @endcode
/// @par References
/// - Thierens (2010). https://doi.org/10.1007/978-3-642-15844-5_27
/// - Thierens and Bosman (2011). https://ir.cwi.nl/pub/18916
/// @see discrete_genetic_search(), discrete_distribution_search()
DiscreteDistributionResult linkage_search(const CategoricalObjective& objective, std::size_t dimension,
                                           const LinkageSearchOptions& options = {});

} // namespace datamunge::optim
