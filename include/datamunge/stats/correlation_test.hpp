#pragma once

#include <datamunge/stats/hypothesis_test_result.hpp>

#include <vector>

namespace datamunge::stats {

/// @brief Pearson product-moment correlation test between @p x and @p y. estimate1 is the
///        correlation coefficient r; the confidence interval is computed via the Fisher
///        z-transform.
HypothesisTestResult pearson_correlation_test(const std::vector<double>& x, const std::vector<double>& y,
                                               Alternative alternative = Alternative::TwoSided,
                                               double conf_level = 0.95);

/// @brief Spearman's rank correlation test between @p x and @p y: the Pearson correlation
///        of their ranks, with a p-value from the same t-approximation used for Pearson's
///        test (matching R's asymptotic method, used whenever ties are present). estimate1
///        is rho; no confidence interval is reported (has_conf_int stays false), matching R.
HypothesisTestResult spearman_correlation_test(const std::vector<double>& x, const std::vector<double>& y,
                                                Alternative alternative = Alternative::TwoSided);

} // namespace datamunge::stats
