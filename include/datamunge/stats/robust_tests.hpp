#pragma once

/// @file robust_tests.hpp
/// @brief Independent-group inference using trimmed means and probability of superiority.

#include <datamunge/stats/hypothesis_test_result.hpp>
#include <optional>
#include <utility>
#include <vector>

namespace datamunge::stats {

/// @brief Compares independent trimmed means with Yuen's unequal-variance t test.
///
/// Each end loses floor(trim * sample size) observations. Winsorized
/// variances and the retained counts determine the standard error and
/// Satterthwaite degrees of freedom. Zero trim gives Welch's two-sample test.
/// @param x First finite sample with at least two retained observations.
/// @param y Second finite sample with at least two retained observations.
/// @param trim Fraction removed from each tail, in [0, 0.5), default 0.2.
/// @param alternative Tail for the signed difference x minus y, default two-sided.
/// @param conf_level Confidence level in (0,1), default 0.95.
/// @return t statistic, degrees of freedom in parameter1, group trimmed means
/// in estimate1 and estimate2, and a confidence interval for their difference.
/// @throws std::invalid_argument Invalid input, zero standard error or unrepresentable sample variance.
/// @par Example
/// @code
/// const auto fit = dm::stats::yuen_test({1, 2, 3, 4, 100}, {0, 1, 2, 3, 4});
/// std::print("{:.1f}\n", fit.estimate1 - fit.estimate2); // 1.0
/// @endcode
/// @par References
/// - K. K. Yuen (1974), https://doi.org/10.1093/biomet/61.1.165
/// @see trimmed_summary(), t_test_two_sample()
HypothesisTestResult yuen_test(const std::vector<double>& x, const std::vector<double>& y,
                              double trim = .2, Alternative alternative = Alternative::TwoSided,
                              double conf_level = .95);

/// @brief Tests whether the independent-group probability of superiority is one half.
///
/// The effect is P(X > Y) + P(X = Y)/2. Pooled and within-group midranks
/// give separate variance estimates and a Student t reference. The
/// approximation is intended for groups of at least ten observations.
/// @param x First sample with at least two finite observations.
/// @param y Second sample with at least two finite observations.
/// @param alternative Tail for superiority of x over y, default two-sided.
/// @return Signed statistic, degrees of freedom in parameter1, probability
/// of superiority in estimate1 and a p-value.
/// @throws std::invalid_argument Invalid input or zero rank-placement variance.
/// @par Example
/// @code
/// const auto fit = dm::stats::brunner_munzel_test({1, 2, 3}, {0, 1, 2});
/// std::print("{:.3f}\n", fit.estimate1); // 0.778
/// @endcode
/// @par References
/// - E. Brunner and U. Munzel (2000), https://doi.org/10.1002/(SICI)1521-4036(200001)42:1%3C17::AID-BIMJ17%3E3.0.CO;2-U
/// @see cliffs_delta(), mann_whitney_test()
HypothesisTestResult brunner_munzel_test(const std::vector<double>& x, const std::vector<double>& y,
                                        Alternative alternative = Alternative::TwoSided);

/// @brief Signed independent-group dominance with its rank-based sampling uncertainty.
/// @par Example
/// @code
/// const auto fit = dm::stats::cliffs_delta({1, 2, 3}, {0, 1, 2});
/// std::print("{:.3f}\n", fit.estimate); // 0.556
/// @endcode
/// @see cliffs_delta()
struct CliffsDeltaResult {
  double estimate;       ///< P(X > Y) minus P(X < Y), between -1 and 1.
  double standard_error; ///< Standard error from the two rank-placement variances.
  std::optional<std::pair<double, double>> confidence_interval; ///< Bounds if rank-placement variance is positive.
};

/// @brief Estimates Cliff's delta and a tie-aware confidence interval.
///
/// Delta is twice the Mann–Whitney probability of superiority minus one.
/// The interval rescales the Student interval from the Brunner–Munzel
/// rank-placement variance and clips it to the effect's [-1,1] support.
/// When rank-placement variance is zero, the point estimate is still defined
/// and the interval is absent.
/// @param x First sample with at least two finite observations.
/// @param y Second sample with at least two finite observations.
/// @param conf_level Confidence level in (0,1), default 0.95.
/// @return Signed dominance, standard error and optional two-sided interval.
/// @throws std::invalid_argument Invalid inputs or confidence level.
/// @par Example
/// @code
/// const auto fit = dm::stats::cliffs_delta({1, 2, 3}, {0, 1, 2});
/// std::print("{}", fit.confidence_interval.has_value()); // true
/// @endcode
/// @par References
/// - N. Cliff (1993), https://doi.org/10.1037/0033-2909.114.3.494
/// - E. Brunner and U. Munzel (2000), https://doi.org/10.1002/(SICI)1521-4036(200001)42:1%3C17::AID-BIMJ17%3E3.0.CO;2-U
/// @see brunner_munzel_test(), mann_whitney_test()
CliffsDeltaResult cliffs_delta(const std::vector<double>& x, const std::vector<double>& y,
                               double conf_level = .95);

} // namespace datamunge::stats
