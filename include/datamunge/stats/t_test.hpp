#pragma once

/// @file t_test.hpp
/// @brief Classical t test contracts, assumptions and numerical results.

#include <datamunge/stats/hypothesis_test_result.hpp>
#include <vector>

namespace datamunge::stats {

/// @brief One-sample Student t-test of a population mean, like R t.test().
///
/// Independent normal observations give t=(mean-mu)/SE with n-1 df. The CI estimates the
/// population mean, not mean-mu. Inputs are copied only for paired differences; all routines
/// leave inputs unchanged and use no shared mutable state. O(n) work; no null/nonfinite omission.
/// @param x At least two finite observations with positive representable sample variance.
/// @param mu Finite null mean, default zero, in observation units.
/// @param alternative TwoSided (default), Less or Greater, relative to mu.
/// @param conf_level Finite confidence level strictly in (0,1), default .95.
/// @return statistic=t, parameter1=df, estimate1=sample mean, p_value and mean confidence bounds;
/// remaining scalars retain defaults. One-sided intervals have one infinite endpoint.
/// @throws std::invalid_argument Too few/nonfinite values, invalid mu/alternative/confidence or
/// zero/nonfinite standard error.
/// @par Example
/// @code
/// const auto t=dm::stats::t_test_one_sample({1,2,3,4});
/// std::print("{:.0f}\n",t.parameter1); // 3
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/t.test.html
/// @see HypothesisTestResult
HypothesisTestResult t_test_one_sample(
    const std::vector<double>& x,
    double                     mu          = 0.0,
    Alternative                alternative = Alternative::TwoSided,
    double                     conf_level  = 0.95);

/// @brief Independent two-sample Welch or pooled Student mean test, like R t.test().
///
/// Welch uses unequal sample variances and Satterthwaite approximate df; normality alone does not
/// make its reference exact. Pooled Student requires equal population variances and independent
/// normal observations for its exact reference. CI estimates mean(x)-mean(y). O(n_x+n_y) work,
/// inputs unchanged, no shared mutable state or omission.
/// @param x At least two finite observations in the first group.
/// @param y At least two finite independent observations in the second group.
/// @param equal_variance False (default) uses Welch; true uses pooled variance.
/// @param alternative TwoSided (default), Less or Greater for mean(x)-mean(y).
/// @param conf_level Finite confidence level in (0,1), default .95.
/// @return statistic=t, parameter1=df, estimate1/2=group means, p_value and confidence bounds for
/// their difference. One-sided bounds can be infinite.
/// @throws std::invalid_argument Invalid sample sizes/values, alternative/confidence,
/// unrepresentable mean difference or zero/nonfinite combined standard error.
/// @par Example
/// @code
/// const auto t=dm::stats::t_test_two_sample({1,2,3},{4,5,6});
/// std::print("{:.0f}\n",t.estimate1); // 2
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/t.test.html
/// @see HypothesisTestResult
HypothesisTestResult t_test_two_sample(
    const std::vector<double>& x,
    const std::vector<double>& y,
    bool                       equal_variance = false,
    Alternative                alternative    = Alternative::TwoSided,
    double                     conf_level     = 0.95);

/// @brief Paired Student t-test on aligned differences, like R t.test(paired=TRUE).
///
/// Runs the one-sample test on x[i]-y[i]. Differences, rather than the two marginal samples, need
/// independent normal sampling for the exact t reference. Row order defines pairing; no values
/// are omitted. O(n) work/storage; inputs unchanged, no shared mutable state.
/// @param x At least two finite first measurements.
/// @param y Equal-length finite second measurements paired by index.
/// @param alternative TwoSided (default), Less or Greater for the population mean difference.
/// @param conf_level Finite confidence level in (0,1), default .95.
/// @return One-sample difference result with method Paired t-test; estimate1 is mean difference,
/// CI targets the same difference.
/// @throws std::invalid_argument Length mismatch, nonfinite data/differences, invalid controls or
/// zero/nonfinite difference standard error.
/// @par Example
/// @code
/// const auto t=dm::stats::t_test_paired({2,4,7},{1,2,4});
/// std::print("{:.0f}\n",t.parameter1); // 2
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/t.test.html
/// @see HypothesisTestResult
HypothesisTestResult t_test_paired(
    const std::vector<double>& x,
    const std::vector<double>& y,
    Alternative                alternative = Alternative::TwoSided,
    double                     conf_level  = 0.95);

} // namespace datamunge::stats
