#pragma once

/// @file ks_test.hpp
/// @brief Kolmogorov-Smirnov goodness-of-fit tests (one-sample normal or uniform, two-sample).

#include <datamunge/stats/hypothesis_test_result.hpp>
#include <vector>

namespace datamunge::stats {

/// @brief One-sample KS test against a fully specified continuous normal CDF.
///
/// Two-sided p-values use the Stephens corrected asymptotic Kolmogorov reference; one-sided
/// values use exp(-2*n*D^2). Mean and sd must be fixed independently of this sample: fitting them
/// here invalidates the stated calibration. This is not a Lilliefors test. Ties are accepted, but
/// continuous-null calibration is affected by rounding. O(n log n) work/O(n) storage, input
/// unchanged, no omission/shared state.
/// @param x At least two finite observations.
/// @param mean Finite prespecified normal mean, default zero.
/// @param sd Positive finite prespecified normal SD, default one.
/// @param alternative TwoSided (default); Greater tests empirical CDF above reference, Less below
/// (opposite to a location alternative).
/// @return statistic=D or D+/D-, parameter1=n and approximate p_value; no CI.
/// @throws std::invalid_argument Too few/nonfinite values, invalid mean/sd or alternative.
/// @par Example
/// @code
/// const auto k=dm::stats::ks_test_one_sample_normal({-1,0,1});
/// std::print("{:.0f}\n",k.parameter1); // 3
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/ks.test.html
/// @see HypothesisTestResult
HypothesisTestResult ks_test_one_sample_normal(
    const std::vector<double>& x,
    double                     mean        = 0.0,
    double                     sd          = 1.0,
    Alternative                alternative = Alternative::TwoSided);

/// @brief One-sample KS test against a specified continuous uniform CDF.
///
/// Same asymptotic p-value and tie caveats as ks_test_one_sample_normal(); limits must be
/// prespecified rather than estimated from x. O(n log n) work/O(n) storage; no mutation, omission
/// or shared state.
/// @param x At least two finite observations.
/// @param lower Finite prespecified lower bound, default zero.
/// @param upper Finite upper bound greater than lower, default one.
/// @param alternative TwoSided (default), Greater for empirical CDF above reference or Less below.
/// @return statistic=D or D+/D-, parameter1=n, approximate p_value; no CI.
/// @throws std::invalid_argument Invalid sample, limits or alternative.
/// @par Example
/// @code
/// const auto k=dm::stats::ks_test_one_sample_uniform({.2,.4,.6,.8});
/// std::print("{:.1f}\n",k.statistic); // 0.2
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/ks.test.html
/// @see HypothesisTestResult
HypothesisTestResult ks_test_one_sample_uniform(
    const std::vector<double>& x,
    double                     lower       = 0.0,
    double                     upper       = 1.0,
    Alternative                alternative = Alternative::TwoSided);

/// @brief Two-sample KS comparison of empirical CDFs with asymptotic p-values.
///
/// Uses effective n=n_x*n_y/(n_x+n_y) in the asymptotic reference. This is not an exact finite-
/// sample calculation; ties/rounding and very small groups affect calibration. O(N log N)
/// work/O(N) storage; inputs unchanged, no omission/shared state.
/// @param x Nonempty finite first sample.
/// @param y Nonempty finite independent second sample.
/// @param alternative TwoSided (default); Greater tests F_x above F_y, Less below, so Greater
/// corresponds to stochastically smaller x.
/// @return statistic=D or D+/D-, parameter1=n_x, parameter2=n_y and approximate p_value; no CI.
/// @throws std::invalid_argument Empty/nonfinite samples or invalid alternative.
/// @par Example
/// @code
/// const auto k=dm::stats::ks_test_two_sample({1,2,3},{1,2,3});
/// std::print("{:.0f}\n",k.statistic); // 0
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/ks.test.html
/// @see HypothesisTestResult
HypothesisTestResult ks_test_two_sample(
    const std::vector<double>& x,
    const std::vector<double>& y,
    Alternative                alternative = Alternative::TwoSided);

} // namespace datamunge::stats
