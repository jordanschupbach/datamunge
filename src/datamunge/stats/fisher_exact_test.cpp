/// @file fisher_exact_test.cpp
/// @brief Conditional hypergeometric Fisher tests with direct tail summation.
#include <algorithm>
#include <datamunge/stats/detail/test_common.hpp>
#include <datamunge/stats/fisher_exact_test.hpp>
#include <limits>

namespace datamunge::stats {
HypothesisTestResult fisher_exact_test_2x2(
    std::size_t a,
    std::size_t b,
    std::size_t c,
    std::size_t d,
    Alternative alternative) {
  detail::check_alternative(alternative);
  std::size_t total = 0;
  for (auto count : {a, b, c, d}) {
    if (count > std::numeric_limits<std::size_t>::max() - total)
      throw std::invalid_argument("fisher_exact_test_2x2: count total overflow");
    total += count;
  }
  const std::size_t row1 = a + b, col1 = a + c,
                    lo = row1 > total - col1 ? row1 - (total - col1) : 0, hi = std::min(row1, col1);
  /// @brief Conditional log probability with fixed margins.
  const auto log_pmf = [&](std::size_t k) {
    return detail::test_log_choose(col1, k) + detail::test_log_choose(total - col1, row1 - k) -
           detail::test_log_choose(total, row1);
  };
  const long double threshold = log_pmf(a) + std::log1p(1e-7L);
  long double       sum       = 0;
  for (std::size_t k = lo;; ++k) {
    const auto lp = log_pmf(k);
    if ((alternative == Alternative::Less && k <= a) ||
        (alternative == Alternative::Greater && k >= a) ||
        (alternative == Alternative::TwoSided && lp <= threshold))
      sum += std::exp(lp);
    if (k == hi)
      break;
  }
  HypothesisTestResult result;
  result.statistic              = a;
  result.alternative            = alternative;
  result.method                 = "Fisher's Exact Test for Count Data";
  const long double numerator   = static_cast<long double>(a) * d,
                    denominator = static_cast<long double>(b) * c;
  result.estimate1              = denominator > 0 ? static_cast<double>(numerator / denominator)
                                  : numerator > 0 ? std::numeric_limits<double>::infinity()
                                                  : std::numeric_limits<double>::quiet_NaN();
  result.p_value                = std::clamp(static_cast<double>(sum), 0.0, 1.0);
  return result;
}
} // namespace datamunge::stats
