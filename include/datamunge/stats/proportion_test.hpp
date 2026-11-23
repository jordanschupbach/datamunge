#pragma once

/// @file proportion_test.hpp
/// @brief Classical proportion test contracts, assumptions and numerical results.

#include <cstddef>
#include <optional>
#include <vector>
#include <datamunge/stats/hypothesis_test_result.hpp>

namespace datamunge::stats {

/// @brief One-proportion score test with an uncorrected Wilson confidence interval.
///
/// Tests a fixed Bernoulli success probability using the signed null-standard-error z statistic;
/// statistic stores z^2 but the chosen tail uses its sign. The optional continuity correction
/// affects the test only: its Wilson interval is uncorrected and need not invert that test.
/// Approximation requires sufficiently populated success/failure counts. O(1) work/storage, no
/// shared state.
/// @param successes Integer successes in [0,n].
/// @param n Positive independent-trial count.
/// @param p Finite null probability strictly in (0,1), default .5.
/// @param alternative TwoSided (default), Less or Greater for true probability relative to p.
/// @param correct True (default) applies the capped half-count correction to z.
/// @param conf_level Finite confidence level in (0,1), default .95.
/// @return statistic=z^2, parameter1=1, estimate1=successes/n, p_value and uncorrected Wilson
/// confidence bounds.
/// @throws std::invalid_argument Invalid counts, null probability, alternative or confidence
/// level.
/// @par Example
/// @code
/// const auto p=dm::stats::proportion_test_one_sample(45,100);
/// std::print("{:.2f}\n",p.estimate1); // 0.45
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/prop.test.html
/// @see HypothesisTestResult
HypothesisTestResult proportion_test_one_sample(
    std::size_t successes,
    std::size_t n,
    double      p           = 0.5,
    Alternative alternative = Alternative::TwoSided,
    bool        correct     = true,
    double      conf_level  = 0.95);

/// @brief Pooled score test for two proportions, with an unpooled difference interval.
///
/// Tests equality using the pooled null SE and optionally a capped continuity correction. The
/// difference CI uses an unpooled Wald SE and an uncapped half-count width correction when
/// requested; it is approximate and is not a Wilson/Newcombe interval. All-zero/all-one pooled
/// samples return z=0,p=1. O(1) work/storage, no shared state.
/// @param successes1 First success count in [0,n1].
/// @param n1 Positive first independent-trial count.
/// @param successes2 Second success count in [0,n2].
/// @param n2 Positive independent second-trial count.
/// @param alternative TwoSided (default), Less or Greater for p1-p2.
/// @param correct True (default) corrects the score test and adds a half-count CI width.
/// @param conf_level Finite confidence level in (0,1), default .95.
/// @return statistic=z^2, parameter1=1, estimates=sample proportions, p_value and confidence
/// bounds for p1-p2, clamped to [-1,1].
/// @throws std::invalid_argument Invalid counts, alternative or confidence level.
/// @par Example
/// @code
/// const auto p=dm::stats::proportion_test_two_sample(2,10,3,10);
/// std::print("{:.1f}\n",p.estimate1); // 0.2
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/prop.test.html
/// @see HypothesisTestResult
HypothesisTestResult proportion_test_two_sample(
    std::size_t successes1,
    std::size_t n1,
    std::size_t successes2,
    std::size_t n2,
    Alternative alternative = Alternative::TwoSided,
    bool        correct     = true,
    double      conf_level  = 0.95);

/// @brief Exact binomial test with Clopper-Pearson confidence bounds, like R binom.test().
///
/// One-sided p-values are inclusive tails. Two-sided p-values sum PMFs no larger than the
/// observed PMF, using relative comparison tolerance 1e-7. The equal-tailed Clopper-Pearson
/// interval has at-least-nominal coverage but need not invert the probability-ordered two-sided
/// test. Null endpoints zero/one are supported. Two-sided enumeration costs O(n), intervals use
/// bounded beta-CDF bisection; O(1) auxiliary storage, no shared state.
/// @param successes Integer observed successes in [0,n].
/// @param n Positive independent Bernoulli trial count.
/// @param p Finite null probability in [0,1], default .5.
/// @param alternative TwoSided (default), Less or Greater for true success probability.
/// @param conf_level Finite confidence level in (0,1), default .95.
/// @return statistic=successes, parameter1=n, estimate1=successes/n, p_value and Clopper-Pearson
/// interval; one-sided intervals end at zero/one.
/// @throws std::invalid_argument Invalid counts, probability, alternative or confidence level.
/// @par Example
/// @code
/// const auto b=dm::stats::binomial_test(9,10);
/// std::print("{:.6f}\n",b.p_value); // 0.021484
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/binom.test.html
/// @see HypothesisTestResult
HypothesisTestResult binomial_test(
    std::size_t successes,
    std::size_t n,
    double      p           = 0.5,
    Alternative alternative = Alternative::TwoSided,
    double      conf_level  = 0.95);

/// @brief Confidence bounds for a difference of two independent binomial proportions.
/// @par Example
/// @code
/// const auto ci = dm::stats::newcombe_difference_interval(3, 10, 1, 10);
/// std::print("{:.1f}\n", ci.difference); // 0.2
/// @endcode
/// @see newcombe_difference_interval()
struct ProportionDifferenceInterval {
  double difference; ///< Observed first proportion minus second proportion.
  double lower;      ///< Lower Newcombe confidence bound.
  double upper;      ///< Upper Newcombe confidence bound.
};

/// @brief Combines separate Wilson score intervals to estimate a difference in proportions.
///
/// Implements Newcombe's uncorrected method 10 for two independent binomial samples.
/// @param successes1 Successes in the first sample, from zero to n1.
/// @param n1 Positive size of the first sample.
/// @param successes2 Successes in the second sample, from zero to n2.
/// @param n2 Positive size of the second sample.
/// @param confidence_level Two-sided nominal confidence level in (0,1), default .95.
/// @return Difference and bounds on the probability scale.
/// @throws std::invalid_argument Invalid counts or confidence level.
/// @par Example
/// @code
/// const auto ci = dm::stats::newcombe_difference_interval(0, 10, 0, 10);
/// std::print("{:.4f} {:.4f}\n", ci.lower, ci.upper); // -0.2775 0.2775
/// @endcode
/// @par References
/// - Newcombe (1998), Interval estimation for the difference between independent proportions,
///   https://pubmed.ncbi.nlm.nih.gov/9595617/.
/// @see proportion_test_two_sample()
ProportionDifferenceInterval newcombe_difference_interval(
    std::size_t successes1, std::size_t n1, std::size_t successes2, std::size_t n2,
    double confidence_level = .95);

/// @brief Confidence bounds on the ratio of two independent binomial risks.
/// @par Example
/// @code
/// const auto ci = dm::stats::risk_ratio_score_interval(10, 20, 5, 20);
/// std::print("{}\n", *ci.estimate); // 2
/// @endcode
/// @par References
/// - P. A. R. Koopman (1984), https://doi.org/10.2307/2531405
/// @see risk_ratio_score_interval()
struct RiskRatioInterval {
  std::optional<double> estimate; ///< First observed proportion divided by the second, absent for two zero counts.
  double lower;                   ///< Lower score bound, zero when the first count is zero.
  double upper;                   ///< Upper score bound, positive infinity when the second count is zero.
};

/// @brief Inverts the uncorrected Koopman two-binomial score test for a risk ratio.
///
/// At each candidate ratio, the binomial likelihood is maximized subject
/// to that ratio. The fitted risks determine the score variance. This is a
/// two-sided, asymptotic interval. The estimate is positive infinity when
/// only the denominator risk is zero. When both counts are zero, the
/// estimate is absent and the interval is [0, +infinity].
/// @param successes1 Successes in the first sample, from zero to n1.
/// @param n1 Positive first sample size.
/// @param successes2 Successes in the second sample, from zero to n2.
/// @param n2 Positive second sample size.
/// @param confidence_level Two-sided nominal confidence level in (0,1), default .95.
/// @return Risk ratio estimate and lower and upper score bounds.
/// @throws std::invalid_argument Invalid counts or confidence level.
/// @par Example
/// @code
/// const auto ci = dm::stats::risk_ratio_score_interval(0, 20, 4, 20);
/// std::print("{} {:.3f}\n", ci.lower, ci.upper); // 0 0.870
/// @endcode
/// @par References
/// - P. A. R. Koopman (1984), https://doi.org/10.2307/2531405
/// @see newcombe_difference_interval(), RiskRatioInterval
RiskRatioInterval risk_ratio_score_interval(
    std::size_t successes1, std::size_t n1, std::size_t successes2, std::size_t n2,
    double confidence_level = .95);

/// @brief Exact Poisson rate or rate-ratio interval and its observed estimate.
/// @par Example
/// @code
/// const auto ci = dm::stats::poisson_rate_interval(5, 2);
/// std::print("{:.1f}\n", *ci.estimate); // 2.5
/// @endcode
/// @see poisson_rate_interval(), poisson_rate_ratio_interval()
struct PoissonRateInterval {
  std::optional<double> estimate; ///< Observed rate or ratio, absent for a ratio of two zero rates.
  double lower;                   ///< Lower exact confidence bound.
  double upper;                   ///< Upper exact confidence bound, possibly positive infinity.
};

/// @brief Inverts Poisson count tails for a Garwood interval on one event rate.
/// @param events Number of observed events, including zero.
/// @param exposure Positive finite exposure in caller-selected units.
/// @param confidence_level Two-sided nominal coverage in (0,1), default .95.
/// @return Observed rate and interval in events per unit of exposure.
/// @throws std::invalid_argument Invalid exposure or confidence level.
/// @par Example
/// @code
/// const auto ci = dm::stats::poisson_rate_interval(0, 5);
/// std::print("{:.3f}\n", ci.upper); // 0.738
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/poisson.test.html
/// @see poisson_rate_ratio_interval()
PoissonRateInterval poisson_rate_interval(std::size_t events, double exposure,
                                         double confidence_level = .95);

/// @brief Computes an exact conditional interval for a ratio of two independent Poisson rates.
///
/// Conditional on the combined count, the first count is binomial. Transforms
/// its Clopper–Pearson bounds to a rate ratio using the two exposures.
/// @param events1 Number of events from the first exposure.
/// @param exposure1 Positive finite first exposure.
/// @param events2 Number of events from the second exposure.
/// @param exposure2 Positive finite second exposure, in the same units.
/// @param confidence_level Two-sided nominal coverage in (0,1), default .95.
/// @return First rate divided by the second and its interval. Zero events in both
/// exposures give an absent estimate and the interval [0,+infinity].
/// @throws std::invalid_argument Invalid exposure or confidence level.
/// @par Example
/// @code
/// const auto ci = dm::stats::poisson_rate_ratio_interval(10, 20, 5, 20);
/// std::print("{:.0f}\n", *ci.estimate); // 2
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/poisson.test.html
/// @see poisson_rate_interval()
PoissonRateInterval poisson_rate_ratio_interval(std::size_t events1, double exposure1,
                                               std::size_t events2, double exposure2,
                                               double confidence_level = .95);

/// @brief Cochran–Armitage score test for a linear trend in independent binomial proportions.
///
/// A positive statistic indicates that success is associated with larger
/// category scores. The standard normal null reference is asymptotic.
/// @param successes One success count per category, at most the corresponding trial count.
/// @param trials Positive trial counts for at least two categories.
/// @param scores Finite numeric category scores, or empty to use 1, 2, ... in input order.
/// @param alternative Tail of the signed statistic, two-sided by default.
/// @return Signed z statistic and approximate p-value.
/// @throws std::invalid_argument Invalid input, constant scores, or a constant outcome.
/// @par Example
/// @code
/// const auto fit = dm::stats::cochran_armitage_test({2, 5, 8}, {10, 10, 10});
/// std::print("{:.3f}\n", fit.statistic); // 2.683
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/prop.trend.test.html
/// @see proportion_test_two_sample()
HypothesisTestResult cochran_armitage_test(const std::vector<std::size_t>& successes,
                                           const std::vector<std::size_t>& trials,
                                           const std::vector<double>& scores = {},
                                           Alternative alternative = Alternative::TwoSided);

} // namespace datamunge::stats
