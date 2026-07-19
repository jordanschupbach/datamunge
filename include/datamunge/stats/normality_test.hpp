#pragma once

#include <datamunge/stats/hypothesis_test_result.hpp>

#include <vector>

namespace datamunge::stats {

/// @brief Shapiro-Francia test for normality: the squared correlation between the sorted
///        sample and the expected normal order statistics (Blom's approximation), with
///        Royston's (1993) log-normal p-value approximation. A simpler, closely-related
///        cousin of the (more commonly cited but more involved) Shapiro-Wilk test, valid
///        for 5 <= n <= 5000; estimate1 is the W' statistic (near 1 for normal-looking
///        data, well below 1 for non-normal data). A small p-value is evidence against
///        normality.
HypothesisTestResult shapiro_francia_test(const std::vector<double>& x);

} // namespace datamunge::stats
