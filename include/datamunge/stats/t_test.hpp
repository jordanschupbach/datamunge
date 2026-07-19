#pragma once

#include <datamunge/stats/hypothesis_test_result.hpp>

#include <vector>

namespace datamunge::stats {

/// @brief One-sample t-test of whether the mean of @p x differs from @p mu.
HypothesisTestResult t_test_one_sample(const std::vector<double>& x, double mu = 0.0,
                                        Alternative alternative = Alternative::TwoSided, double conf_level = 0.95);

/// @brief Two-sample t-test of whether @p x and @p y have the same mean. Uses Welch's
///        unequal-variance approximation (Satterthwaite degrees of freedom) by default,
///        matching R's t.test() default; set @p equal_variance to use the classic pooled-
///        variance Student's t-test instead.
HypothesisTestResult t_test_two_sample(const std::vector<double>& x, const std::vector<double>& y,
                                        bool equal_variance = false,
                                        Alternative alternative = Alternative::TwoSided, double conf_level = 0.95);

/// @brief Paired t-test: a one-sample t-test on the elementwise differences x[i] - y[i].
///        Requires x and y to have the same length.
HypothesisTestResult t_test_paired(const std::vector<double>& x, const std::vector<double>& y,
                                    Alternative alternative = Alternative::TwoSided, double conf_level = 0.95);

} // namespace datamunge::stats
