#pragma once

/// @file normality_test.hpp
/// @brief Classical normality test contracts, assumptions and numerical results.

#include <datamunge/stats/hypothesis_test_result.hpp>
#include <vector>

namespace datamunge::stats {

/// @brief Shapiro-Francia composite normality test, like R nortest::sf.test().
///
/// Squared correlation of sorted observations with Blom normal scores, using Royston (1993) log-
/// normal p-value approximation. Tests some normal distribution; a large p-value does not prove
/// normality. Rounding/ties affect calibration. O(n log n) work/O(n) storage; input unchanged, no
/// omission/shared state.
/// @param x Between 5 and 5000 finite, nonconstant observations.
/// @return statistic=estimate1=W-prime and approximate p_value; no CI.
/// @throws std::invalid_argument Invalid sample size, nonfinite input or constant sample.
/// @par Example
/// @code
/// const auto s=dm::stats::shapiro_francia_test({1,2,3,4,5});
/// std::print("{}\n",s.has_conf_int); // false
/// @endcode
/// @par References
/// - https://search.r-project.org/CRAN/refmans/nortest/html/sf.test.html
/// @see HypothesisTestResult
HypothesisTestResult shapiro_francia_test(const std::vector<double>& x);

} // namespace datamunge::stats
