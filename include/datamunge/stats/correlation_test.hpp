#pragma once

/// @file correlation_test.hpp
/// @brief Correlation tests (Pearson, Spearman) and Kendall's rank correlation.

#include <datamunge/stats/hypothesis_test_result.hpp>
#include <vector>

namespace datamunge::stats {

/// @brief Pearson product-moment correlation test with Fisher-z confidence bounds.
///
/// Uses t=r*sqrt((n-2)/(1-r^2)). The exact zero-correlation null reference requires independent
/// bivariate-normal pairs; Fisher-z intervals are approximate. At n=3 no confidence interval is
/// reported; n>=4 gets one. Perfect correlations use infinite-statistic limits. O(n) work, no
/// input mutation/omission or shared state.
/// @param x At least three finite nonconstant first measurements.
/// @param y Same-length finite nonconstant second measurements, paired by index.
/// @param alternative TwoSided (default), Less or Greater for population correlation.
/// @param conf_level Finite confidence level in (0,1), default .95.
/// @return statistic=t, parameter1=n-2, estimate1=r and p_value; has_conf_int only for n>=4.
/// @throws std::invalid_argument Invalid lengths, nonfinite/constant samples or invalid controls.
/// @par Example
/// @code
/// const auto r=dm::stats::pearson_correlation_test({1,2,3,4},{2,4,6,8});
/// std::print("{:.1f}\n",r.estimate1); // 1.0
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/cor.test.html
/// @see HypothesisTestResult
HypothesisTestResult pearson_correlation_test(
    const std::vector<double>& x,
    const std::vector<double>& y,
    Alternative                alternative = Alternative::TwoSided,
    double                     conf_level  = 0.95);

/// @brief Spearman correlation with average ranks and a t-approximation p-value.
///
/// Computes Pearson correlation on ranks. The reference is always t(n-2), not a small-sample
/// exact permutation distribution; ties receive average ranks. Tests zero rank correlation under
/// the approximation, not arbitrary independence. O(n log n) work/O(n) storage, no
/// mutation/omission or shared state.
/// @param x At least three finite nonconstant first measurements.
/// @param y Same-length finite nonconstant paired second measurements.
/// @param alternative TwoSided (default), Less or Greater for population rank correlation.
/// @return statistic=rank-correlation t, parameter1=n-2, estimate1=rho, approximate p_value; no
/// CI.
/// @throws std::invalid_argument Invalid lengths, nonfinite/constant samples or alternative.
/// @par Example
/// @code
/// const auto r=dm::stats::spearman_correlation_test({1,2,3,4},{10,20,30,40});
/// std::print("{:.1f}\n",r.estimate1); // 1.0
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/cor.test.html
/// @see HypothesisTestResult
HypothesisTestResult spearman_correlation_test(
    const std::vector<double>& x,
    const std::vector<double>& y,
    Alternative                alternative = Alternative::TwoSided);

/// @brief Owned pair counts and coefficients from kendall_tau().
/// @par Example
/// See kendall_tau().
/// @see kendall_tau()
struct KendallTau {
  long long concordant = 0;   ///< Pairs ordered the same way by x and y.
  long long discordant = 0;   ///< Pairs ordered oppositely by x and y.
  long long ties_x     = 0;   ///< Pairs tied in x only.
  long long ties_y     = 0;   ///< Pairs tied in y only.
  long long ties_xy    = 0;   ///< Pairs tied in both.
  double    tau_a      = 0.0; ///< \f$(C-D)/\binom n2\f$.
  double    tau_b =
      0.0; ///< \f$(C-D)/\sqrt{(n_0-n_x)(n_0-n_y)}\f$, with \f$n_x,n_y\f$ the pairs tied in x, y.
};

/// @brief Kendall rank coefficients by Knight inversion counting.
///
/// Sorts paired values and counts concordant, discordant and tied pairs in O(n log n) work/O(n)
/// storage. Inputs are unchanged and calls share no mutable state. This is a coefficient
/// calculation, not a hypothesis test; tau_b is zero when its denominator vanishes.
/// @param x At least two finite first coordinates.
/// @param y Equal-length finite second coordinates, paired by index.
/// @return Owned pair counts, tau_a and tie-adjusted tau_b.
/// @throws std::invalid_argument Lengths differ, fewer than two values, or nonfinite
/// observations.
/// @par Example
/// @code
/// const auto k=dm::stats::kendall_tau({1,2,3,4},{1,2,3,4});
/// std::print("{}\n",k.concordant); // 6
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/cor.html
/// @see HypothesisTestResult
KendallTau kendall_tau(const std::vector<double>& x, const std::vector<double>& y);

} // namespace datamunge::stats
