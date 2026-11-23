#pragma once
/// @file delong.hpp
/// @brief Nonparametric inference for the areas under paired ROC curves.

#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/stats/hypothesis_test_result.hpp>
#include <vector>

namespace datamunge::stats {

/// @brief Full ROC areas and their DeLong covariance for scores on shared observations.
///
/// The positive class has label one. Higher scores rank ahead of lower scores.
/// Cross-class score ties count as half a win. Rows and columns of the
/// covariance matrix follow the input curve order.
/// @par Example
/// @code
/// const auto fit = dm::stats::delong_auc_covariance({0, 0, 1, 1}, {{0, 1, 1, 2}});
/// std::print("{}\n", fit.areas[0]); // 0.875
/// @endcode
/// @par References
/// - E. R. DeLong, D. M. DeLong and D. L. Clarke-Pearson (1988), https://doi.org/10.2307/2531595
/// @see delong_auc_covariance(), delong_auc_test()
struct DeLongAucCovariance {
  std::vector<double> areas;               ///< Estimated full AUC for each score vector.
  linalg::DenseMatrix<double> covariance;   ///< Covariance of the AUC estimates; diagonal gives their variances.
};

/// @brief Estimates one or more full ROC areas and their paired DeLong covariance.
///
/// Requires independent observations within each outcome group and the same
/// observations in the same order for every score vector. The covariance
/// combines sample covariances of the case and control placements. The
/// placements use one sort per score vector and a linear pass through ties.
/// @param truth Binary labels in observation order, with at least two observations per class.
/// @param scores Nonempty collection containing one finite score per observation for each curve.
/// @return Areas and an equally ordered square covariance matrix.
/// @throws std::invalid_argument Invalid labels, class counts, score lengths, or nonfinite scores.
/// @throws std::length_error The covariance matrix exceeds the container's maximum size.
/// @par Example
/// @code
/// const auto fit = dm::stats::delong_auc_covariance({0, 0, 1, 1}, {{0, 1, 1, 2}});
/// std::print("{}\n", fit.covariance(0, 0)); // 0.03125
/// @endcode
/// @par References
/// - E. R. DeLong, D. M. DeLong and D. L. Clarke-Pearson (1988), https://doi.org/10.2307/2531595
/// @see DeLongAucCovariance, delong_auc_test()
DeLongAucCovariance delong_auc_covariance(const std::vector<int>& truth,
                                         const std::vector<std::vector<double>>& scores);

/// @brief Compares two full ROC areas with a paired, two-sided DeLong z test.
///
/// The normal-approximation interval estimates the first area minus the
/// second, clipped to the feasible range [-1, 1]. With zero contrast
/// variance the interval collapses to the observed difference. The test
/// returns z zero and p one for equal areas, or signed infinite z and p
/// zero for unequal areas, using the limits of the normal approximation.
/// @param truth Binary labels in observation order, with at least two observations per class.
/// @param first Finite scores for the first curve, in observation order.
/// @param second Finite scores for the second curve, in the same order.
/// @param conf_level Two-sided confidence level strictly between zero and one, default 0.95.
/// @return Signed z, two-sided p-value, both areas in estimate1 and estimate2, and a confidence interval for their difference.
/// @throws std::invalid_argument Invalid labels, class counts, score arguments, or confidence level.
/// @par Example
/// @code
/// const auto test = dm::stats::delong_auc_test({0, 0, 1, 1}, {0, 1, 1, 2}, {0, 1, 1, 2});
/// std::print("{}\n", test.p_value); // 1
/// @endcode
/// @par References
/// - E. R. DeLong, D. M. DeLong and D. L. Clarke-Pearson (1988), https://doi.org/10.2307/2531595
/// @see delong_auc_covariance(), HypothesisTestResult
HypothesisTestResult delong_auc_test(const std::vector<int>& truth,
                                     const std::vector<double>& first,
                                     const std::vector<double>& second,
                                     double conf_level = 0.95);

} // namespace datamunge::stats
