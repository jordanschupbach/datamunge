#pragma once

/// @file chi_squared_test.hpp
/// @brief Classical chi squared test contracts, assumptions and numerical results.

#include <cstddef>
#include <datamunge/stats/hypothesis_test_result.hpp>
#include <vector>

namespace datamunge::stats {

/// @brief Pearson chi-square test of specified category probabilities, like R chisq.test().
///
/// Uses sum((observed-expected)^2/expected) and chi-square(k-1); the probabilities are fixed, not
/// estimated here. Expected counts must be sufficiently large for the approximation. Nonnegative
/// fractional counts are accepted algebraically but do not automatically define a sampling model.
/// O(k) work/storage, inputs unchanged, no shared state or omission.
/// @param observed At least two finite nonnegative counts with positive finite total.
/// @param expected_probabilities Empty (default) means uniform; otherwise positive finite
/// probabilities matching counts and summing to one within 1e-10.
/// @return statistic=X-squared, parameter1=k-1 and approximate upper-tail p_value; no CI.
/// @throws std::invalid_argument Invalid counts/probabilities, sizes or zero/nonfinite total.
/// @par Example
/// @code
/// const auto q=dm::stats::chi_squared_goodness_of_fit({10,10});
/// std::print("{:.0f}\n",q.statistic); // 0
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/chisq.test.html
/// @see HypothesisTestResult
HypothesisTestResult chi_squared_goodness_of_fit(
    const std::vector<double>& observed,
    const std::vector<double>& expected_probabilities = {});

/// @brief Pearson chi-square test for a contingency table, like R chisq.test().
///
/// Expected counts are row_sum*column_sum/total; df=(rows-1)(columns-1). The asymptotic reference
/// assumes independent multinomial/product-multinomial observations and adequately populated
/// expected cells. O(rows*columns) work and O(rows+columns) space; inputs unchanged, no shared
/// state.
/// @param table Finite nonnegative row-major counts with strictly positive margins; integer counts
/// are the usual sampling model.
/// @param nrows At least two rows.
/// @param ncols At least two columns; table length must equal nrows*ncols.
/// @param correct True (default) subtracts min(.5,abs(O-E)) from each absolute residual only for
/// 2x2 tables.
/// @return statistic=X-squared, parameter1=df, approximate upper-tail p_value; no CI.
/// @throws std::invalid_argument Invalid shape, counts/margins or nonfinite total.
/// @par Example
/// @code
/// const auto q=dm::stats::chi_squared_test_independence({10,10,10,10},2,2);
/// std::print("{:.0f}\n",q.statistic); // 0
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/chisq.test.html
/// @see HypothesisTestResult
HypothesisTestResult chi_squared_test_independence(
    const std::vector<double>& table,
    std::size_t                nrows,
    std::size_t                ncols,
    bool                       correct = true);

} // namespace datamunge::stats
