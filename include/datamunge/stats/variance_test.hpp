#pragma once

#include <datamunge/stats/hypothesis_test_result.hpp>

#include <vector>

namespace datamunge::stats {

/// @brief F-test comparing the variances of @p x and @p y (H0: equal variances). estimate1
///        is the ratio var(x) / var(y); statistic is the same F ratio; parameter1/2 are the
///        two groups' degrees of freedom.
HypothesisTestResult f_test_variance(const std::vector<double>& x, const std::vector<double>& y,
                                      Alternative alternative = Alternative::TwoSided, double conf_level = 0.95);

} // namespace datamunge::stats
