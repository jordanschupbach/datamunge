#pragma once

/// @file proportion_test.hpp
/// @brief Classical proportion test contracts, assumptions and numerical results.

#include <cstddef>
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

} // namespace datamunge::stats
