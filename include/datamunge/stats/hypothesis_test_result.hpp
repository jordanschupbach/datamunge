#pragma once
/// @file hypothesis_test_result.hpp
/// @brief Shared alternatives and htest-style result storage for classical inference.
#include <string>
namespace datamunge::stats {
/// @brief Prespecified tail of a test's null reference, like R's alternative argument.
/// Location/proportion alternatives refer to the signed effect; KS alternatives refer
/// to empirical CDF ordering and consequently reverse the stochastic-location direction.
/// @par Example
/// @code
/// const auto a=dm::stats::Alternative::Less;
/// const auto t=dm::stats::t_test_one_sample({1,2,3,4},0,a);
/// std::print("{}\n",t.alternative==a); // true
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/t.test.html
/// @see HypothesisTestResult
enum class Alternative {
  TwoSided, ///< Both directions, using each test's stated two-sided ordering.
  Less,     ///< Lower signed-effect tail (KS: empirical CDF below reference).
  Greater   ///< Upper signed-effect tail (KS: empirical CDF above reference).
};
/// @brief Owned hypothesis-test result, analogous to R's htest list.
/// Each test documents which fields it populates and what its interval estimates.
/// Unused numeric fields retain zero; this default is storage, not a claim of zero effect.
/// Read bounds only when has_conf_int is true; one-sided intervals may be unbounded.
/// p_value is a null-model tail probability, not a posterior probability of the null.
/// Objects own their method string and contain no borrowed references or shared state.
/// @par Example
/// @code
/// const auto t=dm::stats::t_test_one_sample({1,2,3,4});
/// std::print("{}\n",t.has_conf_int); // true
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/t.test.html
/// @see Alternative, hypothesis_test_table()
struct HypothesisTestResult {
  double statistic{0.0};  ///< Test-specific statistic (t, U, D, F, H, count or chi-square).
  double parameter1{0.0}; ///< Primary df or count parameter; consult the test contract.
  double parameter2{0.0}; ///< Secondary df/count parameter where used; otherwise zero.
  double p_value{0.0};    ///< Null-reference p-value, not adjusted for multiple testing.
  double estimate1{0.0};  ///< First test-specific estimate, such as a mean, ratio or correlation.
  double estimate2{0.0};  ///< Second estimate when applicable, such as a second group mean.
  double conf_int_lower{0.0}; ///< Lower confidence bound, meaningful only when has_conf_int.
  double conf_int_upper{0.0}; ///< Upper confidence bound, meaningful only when has_conf_int.
  bool   has_conf_int{false}; ///< Whether confidence bounds were computed for this result.
  Alternative alternative{Alternative::TwoSided}; ///< Tail requested by the caller.
  std::string method; ///< Owned human-readable method, including relevant approximation labels.
};
} // namespace datamunge::stats
