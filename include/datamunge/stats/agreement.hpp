#pragma once
/// @file agreement.hpp
/// @brief Agreement of categorical ratings and paired quantitative measurements.

#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::stats {

/// @brief Disagreement weights for ordered categories in Cohen's kappa.
/// @par Example
/// @code
/// const auto fit = dm::stats::cohen_kappa({{4, 0}, {0, 5}}, dm::stats::KappaWeights::Linear);
/// std::print("{}\n", fit.kappa); // 1
/// @endcode
/// @see cohen_kappa()
enum class KappaWeights {
  Unweighted, ///< All unequal ratings receive the same penalty.
  Linear,     ///< Penalty proportional to the distance between ordered categories.
  Quadratic   ///< Penalty proportional to squared distance between ordered categories.
};

/// @brief Chance-corrected agreement and its observed and expected components.
/// @par Example
/// @code
/// std::print("{:.2f}\n", dm::stats::cohen_kappa({{3, 1}, {1, 3}}).observed_agreement); // 0.75
/// @endcode
/// @see cohen_kappa(), fleiss_kappa()
struct KappaAgreement {
  double kappa;               ///< Chance-corrected agreement coefficient.
  double observed_agreement;  ///< Observed agreement on the selected weight scale.
  double expected_agreement;  ///< Agreement on the same scale from the empirical marginals.
};

/// @brief Computes Cohen's unweighted or ordinally weighted kappa for two raters.
/// @param table Square paired rating count table, with the same category order for both raters.
/// @param weights Disagreement weighting, default Unweighted.
/// @return Kappa and the observed and expected agreement.
/// @throws std::invalid_argument Invalid table, weight mode or zero expected disagreement.
/// @par Example
/// @code
/// std::print("{:.2f}\n", dm::stats::cohen_kappa({{3, 1}, {1, 3}}).kappa); // 0.50
/// @endcode
/// @par References
/// - J. Cohen (1960), https://doi.org/10.1177/001316446002000104
/// - J. Cohen (1968), https://doi.org/10.1037/h0026256
/// @see fleiss_kappa()
KappaAgreement cohen_kappa(const std::vector<std::vector<std::size_t>>& table,
                           KappaWeights weights = KappaWeights::Unweighted);

/// @brief Computes Fleiss' kappa from item-level counts for equally sized rater panels.
/// @param ratings One row per item, one column per category, and the same total of at least two ratings per row.
/// @return Kappa and the observed and expected pairwise agreement.
/// @throws std::invalid_argument Invalid count table or zero expected disagreement.
/// @par Example
/// @code
/// std::print("{}\n", dm::stats::fleiss_kappa({{3, 0}, {0, 3}}).kappa); // 1
/// @endcode
/// @par References
/// - J. L. Fleiss (1971), https://doi.org/10.1037/h0031619
/// @see cohen_kappa()
KappaAgreement fleiss_kappa(const std::vector<std::vector<std::size_t>>& ratings);

/// @brief Bias, agreement limits, and individual confidence intervals for paired differences.
/// @par Example
/// @code
/// const auto result = dm::stats::bland_altman({2, 3, 4}, {1, 2, 3});
/// std::print("{}\n", result.bias); // 1
/// @endcode
/// @see bland_altman()
struct BlandAltmanAgreement {
  double bias;                              ///< Mean of first minus second.
  double difference_sd;                     ///< Sample standard deviation of the paired differences.
  std::pair<double, double> limits;          ///< Lower and upper normal-theory limits of agreement.
  std::pair<double, double> bias_interval;   ///< Student t confidence interval for the mean difference.
  std::pair<double, double> lower_interval;  ///< Approximate confidence interval for the lower limit.
  std::pair<double, double> upper_interval;  ///< Approximate confidence interval for the upper limit.
};

/// @brief Computes Bland–Altman normal-theory limits and uncertainty for each limit.
///
/// Uses independent, approximately normally distributed paired differences.
/// The limits are bias plus or minus a normal quantile times sample SD.
/// Each limit's interval uses a delta-method standard error and Student t
/// critical value. The confidence intervals apply individually.
/// @param first First measurements, at least two finite values.
/// @param second Second measurements at the corresponding rows.
/// @param population_coverage Middle fraction of the difference distribution, default .95.
/// @param confidence_level Confidence level for each reported interval, default .95.
/// @return Bias, SD, limits and three confidence intervals in measurement units.
/// @throws std::invalid_argument Invalid pairs or probability.
/// @throws std::overflow_error A reported value is outside finite double range.
/// @par Example
/// @code
/// const auto result = dm::stats::bland_altman({2, 3, 4}, {1, 2, 3});
/// std::print("{} {}\n", result.limits.first, result.limits.second); // 1 1
/// @endcode
/// @par References
/// - J. M. Bland and D. G. Altman (1986), https://pubmed.ncbi.nlm.nih.gov/2868172/
/// @see lin_concordance()
BlandAltmanAgreement bland_altman(const std::vector<double>& first, const std::vector<double>& second,
                                  double population_coverage = .95, double confidence_level = .95);

/// @brief Lin's concordance correlation and its large-sample confidence interval.
/// @par Example
/// @code
/// const auto fit = dm::stats::lin_concordance({1, 2, 3}, {1, 2, 3});
/// std::print("{}\n", fit.concordance); // 1
/// @endcode
/// @see lin_concordance()
struct LinConcordance {
  double concordance;                    ///< Twice the covariance divided by the sum of variances and squared mean gap.
  double standard_error;                 ///< Estimated influence-function standard error on the coefficient scale.
  std::pair<double, double> interval;     ///< Fisher-transformed delta-method interval in [-1,1].
};

/// @brief Estimates Lin's concordance for paired measurements on the same scale.
///
/// Uses empirical moments with denominator n. The confidence interval
/// uses an empirical influence-function variance and a Fisher transform.
/// Perfect concordance or discordance gives a point interval.
/// @param first First measurements, at least three finite values.
/// @param second Second measurements at the corresponding rows.
/// @param confidence_level Confidence level in (0,1), default .95.
/// @return Concordance, its standard error and the confidence interval.
/// @throws std::invalid_argument Invalid pairs, confidence level or zero concordance denominator.
/// @par Example
/// @code
/// const auto fit = dm::stats::lin_concordance({1, 2, 3}, {3, 2, 1});
/// std::print("{}\n", fit.concordance); // -1
/// @endcode
/// @par References
/// - L. I. Lin (1989), https://doi.org/10.2307/2532051
/// @see bland_altman()
LinConcordance lin_concordance(const std::vector<double>& first, const std::vector<double>& second,
                              double confidence_level = .95);

} // namespace datamunge::stats
