#pragma once

/// @file fisher_exact_test.hpp
/// @brief Classical fisher exact test contracts, assumptions and numerical results.

#include <cstddef>
#include <datamunge/stats/hypothesis_test_result.hpp>

namespace datamunge::stats {

/// @brief Conditional Fisher exact test of a 2x2 table with fixed margins.
///
/// Conditions on row/column totals and tests odds ratio one. Two-sided p-values sum probabilities
/// no larger than the observed table (relative ordering tolerance 1e-7); they are not generally
/// twice the smaller tail. Direct tail summation avoids 1-CDF cancellation. Empty margins give
/// p=1. O(s) work/O(1) auxiliary storage for feasible support s; inputs unchanged, no shared
/// state.
/// @param a Top-left nonnegative integer count.
/// @param b Top-right count.
/// @param c Bottom-left count.
/// @param d Bottom-right count; total must fit size_t.
/// @param alternative TwoSided (default), Less or Greater for the odds ratio relative to one.
/// @return statistic=a, p_value and estimate1=sample odds ratio a*d/(b*c), not R conditional MLE;
/// denominator zero gives infinity for positive numerator or NaN for 0/0. No confidence interval.
/// @throws std::invalid_argument Count-total overflow or invalid alternative.
/// @par Example
/// @code
/// const auto f=dm::stats::fisher_exact_test_2x2(0,3,3,0);
/// std::print("{:.1f}\n",f.p_value); // 0.1
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/fisher.test.html
/// @see HypothesisTestResult
HypothesisTestResult fisher_exact_test_2x2(
    std::size_t a,
    std::size_t b,
    std::size_t c,
    std::size_t d,
    Alternative alternative = Alternative::TwoSided);

} // namespace datamunge::stats
