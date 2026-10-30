#pragma once

/// @file variance_test.hpp
/// @brief Classical variance test contracts, assumptions and numerical results.

#include <datamunge/stats/hypothesis_test_result.hpp>
#include <vector>

namespace datamunge::stats {

/// @brief F-test and interval for a ratio of variances, like R var.test().
///
/// Uses sample variance(x)/sample variance(y). Independent normal populations give the
/// F(df_x,df_y) reference; it is sensitive to nonnormality and is not a gatekeeper for choosing
/// Welch versus pooled t tests. O(n_x+n_y) work/O(1) auxiliary storage; inputs unchanged, no
/// omission/shared state.
/// @param x At least two finite first-sample values with positive finite sample variance.
/// @param y At least two finite independent second-sample values with positive finite sample
/// variance.
/// @param alternative TwoSided (default), Less or Greater for population variance ratio relative to
/// one.
/// @param conf_level Finite confidence level in (0,1), default .95.
/// @return statistic=estimate1=variance ratio, parameter1/2=sample sizes minus one, p_value and
/// ratio confidence bounds; one-sided interval ends at zero or infinity.
/// @throws std::invalid_argument Too few/nonfinite observations, nonpositive/nonfinite variances
/// or invalid controls.
/// @par Example
/// @code
/// const auto f=dm::stats::f_test_variance({1,2,3},{2,4,6});
/// std::print("{:.2f}\n",f.estimate1); // 0.25
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/var.test.html
/// @see HypothesisTestResult
HypothesisTestResult f_test_variance(
    const std::vector<double>& x,
    const std::vector<double>& y,
    Alternative                alternative = Alternative::TwoSided,
    double                     conf_level  = 0.95);

} // namespace datamunge::stats
