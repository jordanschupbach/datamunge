#pragma once
/// @file table_symmetry.hpp
/// @brief Homogeneity of stratified odds ratios and symmetry of matched categorical tables.

#include <array>
#include <cstddef>
#include <vector>
#include <datamunge/stats/hypothesis_test_result.hpp>

namespace datamunge::stats {

/// @brief Tests the homogeneity of odds ratios across independent 2 by 2 strata.
///
/// Fits each stratum at the pooled Mantel–Haenszel odds ratio and its
/// observed margins. Strata with fixed tables given their margins are
/// omitted. Optional Tarone adjustment removes the pooled score component.
/// The reference is asymptotic chi-square.
/// @param strata Row-major counts {a,b,c,d}, with at least two informative strata.
/// @param tarone Apply the Tarone adjustment when true, the default.
/// @return Statistic, degrees of freedom in parameter1, p-value and common odds ratio in estimate1.
/// @throws std::invalid_argument Fewer than two informative strata or a boundary pooled odds ratio.
/// @par Example
/// @code
/// const auto test = dm::stats::breslow_day_test({{8, 2, 4, 6}, {5, 3, 2, 6}});
/// std::print("{}\n", test.parameter1); // 1
/// @endcode
/// @par References
/// - N. E. Breslow and N. E. Day (1980), https://pubmed.ncbi.nlm.nih.gov/7216345/
/// - R. E. Tarone (1985), https://doi.org/10.1093/biomet/72.1.91
/// @see mantel_haenszel_test()
HypothesisTestResult breslow_day_test(const std::vector<std::array<std::size_t, 4>>& strata, bool tarone = true);

/// @brief Tests equality of the two marginal distributions in a matched categorical table.
///
/// The covariance uses off-diagonal pair totals. Each connected
/// component of the observed transitions contributes its size minus
/// one degrees of freedom. With no transitions the result has zero
/// degrees of freedom and p-value one.
/// @param table Nonempty square count table with at least two categories in the same order on both axes.
/// @return Stuart–Maxwell statistic, effective degrees of freedom in parameter1, and asymptotic p-value.
/// @throws std::invalid_argument Invalid table dimensions or no observations.
/// @par Example
/// @code
/// const auto fit = dm::stats::stuart_maxwell_test({{9, 6}, {2, 8}});
/// std::print("{}\n", fit.statistic); // 2
/// @endcode
/// @par References
/// - A. Stuart (1955), https://doi.org/10.1093/biomet/42.3-4.412
/// @see bowker_test()
HypothesisTestResult stuart_maxwell_test(const std::vector<std::vector<std::size_t>>& table);

/// @brief Tests whether the off-diagonal cells of a matched categorical table are symmetric.
///
/// Uses one degree of freedom per observed nonempty off-diagonal pair.
/// With no transitions the result has zero degrees of freedom and
/// p-value one. For two categories this is uncorrected McNemar.
/// @param table Nonempty square count table with at least two categories in the same order on both axes.
/// @return Bowker statistic, effective degrees of freedom in parameter1, and asymptotic p-value.
/// @throws std::invalid_argument Invalid table dimensions or no observations.
/// @par Example
/// @code
/// const auto fit = dm::stats::bowker_test({{8, 6}, {2, 7}});
/// std::print("{}\n", fit.statistic); // 2
/// @endcode
/// @par References
/// - A. H. Bowker (1948), https://doi.org/10.1080/01621459.1948.10483284
/// @see stuart_maxwell_test()
HypothesisTestResult bowker_test(const std::vector<std::vector<std::size_t>>& table);

} // namespace datamunge::stats
