#pragma once

#include <datamunge/stats/hypothesis_test_result.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::stats {

/// @brief Chi-squared goodness-of-fit test: whether the counts in @p observed match the
///        given @p expected_probabilities (which must sum to 1). If @p expected_probabilities
///        is empty, a uniform distribution across categories is assumed, matching R's
///        chisq.test() default.
HypothesisTestResult chi_squared_goodness_of_fit(const std::vector<double>& observed,
                                                  const std::vector<double>& expected_probabilities = {});

/// @brief Chi-squared test of independence on a two-way contingency table given as a
///        row-major flattened vector of counts (length @p nrows * @p ncols). Applies Yates'
///        continuity correction when the table is 2x2 and @p correct is true (the default,
///        matching R's chisq.test()).
HypothesisTestResult chi_squared_test_independence(const std::vector<double>& table, std::size_t nrows,
                                                     std::size_t ncols, bool correct = true);

} // namespace datamunge::stats
