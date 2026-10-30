/// @file normality_test.cpp
/// @brief Classical normality test statistics, domain checks and stable probabilities.
#include <algorithm>
#include <cmath>
#include <datamunge/random/distributions.hpp>
#include <datamunge/stats/detail/test_common.hpp>
#include <datamunge/stats/normality_test.hpp>
#include <stdexcept>

namespace datamunge::stats {

HypothesisTestResult shapiro_francia_test(const std::vector<double>& x) {
  detail::check_sample(x, 5);
  if (x.size() > 5000)
    throw std::invalid_argument("shapiro_francia_test: at most 5000 observations");
  const auto n = static_cast<double>(x.size());
  if (x.size() < 5)
    throw std::invalid_argument("shapiro_francia_test: need at least 5 observations");

  std::vector<double> sorted = x;
  std::sort(sorted.begin(), sorted.end());

  std::vector<double> scores(sorted.size());
  for (std::size_t i = 0; i < sorted.size(); ++i)
    scores[i] = random::normal_quantile((static_cast<double>(i + 1) - 0.375) / (n + 0.25));

  long double mean_x = 0.0, mean_m = 0.0;
  for (std::size_t i = 0; i < sorted.size(); ++i) {
    mean_x += static_cast<long double>(sorted[i]) / n;
    mean_m += static_cast<long double>(scores[i]) / n;
  }

  long double cov = 0.0, var_x = 0.0, var_m = 0.0;
  for (std::size_t i = 0; i < sorted.size(); ++i) {
    const long double dx = sorted[i] - mean_x;
    const long double dm = scores[i] - mean_m;
    cov += dx * dm;
    var_x += dx * dx;
    var_m += dm * dm;
  }
  if (!(var_x > 0))
    throw std::invalid_argument("shapiro_francia_test: sample is constant");
  const double r =
      std::clamp(static_cast<double>(cov / std::sqrt(var_x) / std::sqrt(var_m)), -1.0, 1.0);
  const double w = r * r;

  const double u     = std::log(n);
  const double v     = std::log(u);
  const double mu    = -1.2725 + 1.0521 * (v - u);
  const double sigma = 1.0308 - 0.26758 * (v + 2.0 / u);
  const double z     = (std::log(1.0 - w) - mu) / sigma;

  HypothesisTestResult result;
  result.statistic = w;
  result.estimate1 = w;
  result.p_value   = detail::normal_test_p(z, Alternative::Greater);
  result.method    = "Shapiro-Francia normality test";
  return result;
}

} // namespace datamunge::stats
