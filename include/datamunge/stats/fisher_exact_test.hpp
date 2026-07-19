#pragma once

#include <datamunge/stats/hypothesis_test_result.hpp>

#include <cstddef>

namespace datamunge::stats {

/// @brief Fisher's exact test on a 2x2 contingency table
///
///            | col1 | col2
///        row1|  a   |  b
///        row2|  c   |  d
///
///        testing independence of the row and column classifications by summing exact
///        hypergeometric probabilities (matching R's fisher.test() p-values). estimate1 is
///        the sample odds ratio (a*d)/(b*c), not the conditional MLE R reports by default;
///        no confidence interval is computed (has_conf_int stays false). statistic holds the
///        observed count `a`.
HypothesisTestResult fisher_exact_test_2x2(std::size_t a, std::size_t b, std::size_t c, std::size_t d,
                                            Alternative alternative = Alternative::TwoSided);

} // namespace datamunge::stats
