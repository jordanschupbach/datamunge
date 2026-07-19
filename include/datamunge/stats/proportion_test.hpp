#pragma once

#include <datamunge/stats/hypothesis_test_result.hpp>

#include <cstddef>

namespace datamunge::stats {

/// @brief One-sample test of the proportion @p successes / @p n against a null value @p p
///        (default 0.5), via the chi-squared/normal approximation with a continuity
///        correction (matching R's prop.test()). estimate1 is the sample proportion; the
///        confidence interval uses the Wilson score interval (without continuity
///        correction, a close approximation to R's corrected version).
HypothesisTestResult proportion_test_one_sample(std::size_t successes, std::size_t n, double p = 0.5,
                                                 Alternative alternative = Alternative::TwoSided, bool correct = true,
                                                 double conf_level = 0.95);

/// @brief Two-sample test of whether two groups have the same success proportion, via the
///        pooled chi-squared/normal approximation with a continuity correction (matching
///        R's prop.test()). estimate1/estimate2 are the two sample proportions; the
///        confidence interval is for their difference (estimate1 - estimate2).
HypothesisTestResult proportion_test_two_sample(std::size_t successes1, std::size_t n1, std::size_t successes2,
                                                 std::size_t n2, Alternative alternative = Alternative::TwoSided,
                                                 bool correct = true, double conf_level = 0.95);

/// @brief Exact binomial test of whether @p successes out of @p n trials is consistent with
///        success probability @p p (default 0.5), summing exact binomial probabilities
///        (matching R's binom.test()). estimate1 is the sample proportion; the confidence
///        interval is the exact Clopper-Pearson interval.
HypothesisTestResult binomial_test(std::size_t successes, std::size_t n, double p = 0.5,
                                    Alternative alternative = Alternative::TwoSided, double conf_level = 0.95);

} // namespace datamunge::stats
