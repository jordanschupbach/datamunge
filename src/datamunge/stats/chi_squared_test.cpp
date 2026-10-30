/// @file chi_squared_test.cpp
/// @brief Classical chi squared test statistics, domain checks and stable probabilities.
#include <cmath>
#include <datamunge/random/distributions.hpp>
#include <datamunge/stats/chi_squared_test.hpp>
#include <datamunge/stats/detail/test_common.hpp>
#include <stdexcept>

namespace datamunge::stats {

HypothesisTestResult chi_squared_goodness_of_fit(
    const std::vector<double>& observed,
    const std::vector<double>& expected_probabilities) {
  if (observed.size() < 2)
    throw std::invalid_argument("chi_squared_goodness_of_fit: need at least 2 categories");

  detail::check_sample(observed, 2);
  for (double v : observed)
    if (v < 0)
      throw std::invalid_argument("chi_squared: counts must be nonnegative");
  std::vector<double> probabilities = expected_probabilities;
  if (probabilities.empty())
    probabilities.assign(observed.size(), 1.0 / static_cast<double>(observed.size()));
  if (probabilities.size() != observed.size())
    throw std::invalid_argument(
        "chi_squared_goodness_of_fit: expected_probabilities size mismatch");

  double probability_sum = 0;
  for (double p : probabilities) {
    if (!std::isfinite(p) || p <= 0)
      throw std::invalid_argument(
          "chi_squared: expected probabilities must be positive and finite");
    probability_sum += p;
  }
  if (std::abs(probability_sum - 1) > 1e-10)
    throw std::invalid_argument("chi_squared: probabilities must sum to one");
  double total = 0.0;
  for (const double v : observed)
    total += v;

  if (!(total > 0) || !std::isfinite(total))
    throw std::invalid_argument("chi_squared: positive finite total required");
  double statistic = 0.0;
  for (std::size_t i = 0; i < observed.size(); ++i) {
    const double expected = total * probabilities[i];
    const double diff     = observed[i] - expected;
    statistic += diff * diff / expected;
  }
  const double df = static_cast<double>(observed.size() - 1);

  HypothesisTestResult result;
  result.statistic  = statistic;
  result.parameter1 = df;
  result.p_value    = detail::chi_squared_test_survival(statistic, df);
  result.method     = "Chi-squared test for given probabilities";
  return result;
}

HypothesisTestResult chi_squared_test_independence(
    const std::vector<double>& table,
    std::size_t                nrows,
    std::size_t                ncols,
    bool                       correct) {
  if (nrows < 2 || ncols < 2)
    throw std::invalid_argument("chi_squared_test_independence: need at least a 2x2 table");
  if (nrows > table.size() / ncols || table.size() != nrows * ncols)
    throw std::invalid_argument(
        "chi_squared_test_independence: table size must equal nrows * ncols");

  detail::check_sample(table, 4);
  for (double v : table)
    if (v < 0)
      throw std::invalid_argument("chi_squared: counts must be nonnegative");
  std::vector<double> row_sums(nrows, 0.0), col_sums(ncols, 0.0);
  double              total = 0.0;
  for (std::size_t i = 0; i < nrows; ++i) {
    for (std::size_t j = 0; j < ncols; ++j) {
      const double v = table[i * ncols + j];
      row_sums[i] += v;
      col_sums[j] += v;
      total += v;
    }
  }

  if (!std::isfinite(total))
    throw std::invalid_argument("chi_squared: total overflow");
  for (double v : row_sums)
    if (!(v > 0))
      throw std::invalid_argument("chi_squared: positive margins required");
  for (double v : col_sums)
    if (!(v > 0))
      throw std::invalid_argument("chi_squared: positive margins required");
  const bool apply_yates = correct && nrows == 2 && ncols == 2;
  double     statistic   = 0.0;
  for (std::size_t i = 0; i < nrows; ++i) {
    for (std::size_t j = 0; j < ncols; ++j) {
      const double expected = (row_sums[i] / total) * col_sums[j];
      double       diff     = std::fabs(table[i * ncols + j] - expected);
      if (apply_yates)
        diff = std::max(0.0, diff - 0.5);
      statistic += diff * diff / expected;
    }
  }
  const double df = static_cast<double>((nrows - 1) * (ncols - 1));

  HypothesisTestResult result;
  result.statistic  = statistic;
  result.parameter1 = df;
  result.p_value    = detail::chi_squared_test_survival(statistic, df);
  result.method     = apply_yates ? "Pearson's Chi-squared test with Yates' continuity correction"
                                  : "Pearson's Chi-squared test";
  return result;
}

} // namespace datamunge::stats
