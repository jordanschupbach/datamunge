#pragma once

#include <datamunge/stats/hypothesis_test_result.hpp>

#include <vector>

namespace datamunge::stats {

/// @brief One-sample Kolmogorov-Smirnov test of whether @p x is drawn from a
///        Normal(@p mean, @p sd) distribution, comparing the empirical CDF of @p x against
///        that normal CDF. Uses the asymptotic Kolmogorov distribution for the two-sided
///        p-value (Stephens' 1970 correction to the sample size) and the classic Smirnov
///        one-sided asymptotic formula for "less"/"greater"; this will differ slightly from
///        software that computes the exact finite-sample null distribution for small,
///        tie-free samples (e.g. R's ks.test() default), but agrees closely once n is more
///        than a few dozen.
HypothesisTestResult ks_test_one_sample_normal(const std::vector<double>& x, double mean = 0.0, double sd = 1.0,
                                                Alternative alternative = Alternative::TwoSided);

/// @brief Two-sample Kolmogorov-Smirnov test of whether @p x and @p y are drawn from the
///        same continuous distribution, comparing their empirical CDFs. Same asymptotic
///        p-value approach as ks_test_one_sample_normal.
HypothesisTestResult ks_test_two_sample(const std::vector<double>& x, const std::vector<double>& y,
                                         Alternative alternative = Alternative::TwoSided);

} // namespace datamunge::stats
