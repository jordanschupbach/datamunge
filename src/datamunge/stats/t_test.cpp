/// @file t_test.cpp
/// @brief Classical t test statistics, domain checks and stable probabilities.
#include <cmath>
#include <datamunge/random/distributions.hpp>
#include <datamunge/stats/detail/test_common.hpp>
#include <datamunge/stats/t_test.hpp>
#include <limits>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Sample moments, Student tail evaluation and interval assembly.
namespace {

/// @brief Owned sample moments and count used by t statistics.
struct MeanVar {
  double      mean;     ///< Arithmetic sample mean.
  double      variance; ///< Sample variance with n-1 denominator.
  std::size_t n;        ///< Sample count.
};

/// @brief Computes checked sample moments.
/// @param x At least two finite observations.
/// @return Mean, sample variance and count.
MeanVar mean_variance(const std::vector<double>& x) {
  if (x.size() < 2)
    throw std::invalid_argument("t_test: need at least 2 observations");
  const auto moments = detail::test_moments(x);
  return {static_cast<double>(moments.mean), static_cast<double>(moments.variance), x.size()};
}

/// @brief Computes a cancellation-free Student tail.
/// @param t Signed statistic.
/// @param df Positive degrees of freedom.
/// @param alternative Requested tail.
/// @return Null probability.
double t_p_value(double t, double df, Alternative alternative) {
  return detail::student_test_p(t, df, alternative);
}

/// @brief Writes Student confidence bounds into an owned result.
/// @param result Result to mutate.
/// @param estimate Population-parameter estimate.
/// @param se Positive standard error.
/// @param df Positive degrees of freedom.
/// @param conf_level Confidence probability.
/// @param alternative Requested interval direction.
void fill_ci(
    HypothesisTestResult& result,
    double                estimate,
    double                se,
    double                df,
    double                conf_level,
    Alternative           alternative) {
  result.has_conf_int = true;
  const double inf    = std::numeric_limits<double>::infinity();
  if (alternative == Alternative::TwoSided) {
    const double crit     = random::student_t_quantile(1.0 - 0.5 * (1.0 - conf_level), df);
    result.conf_int_lower = estimate - crit * se;
    result.conf_int_upper = estimate + crit * se;
  } else if (alternative == Alternative::Less) {
    const double crit     = random::student_t_quantile(conf_level, df);
    result.conf_int_lower = -inf;
    result.conf_int_upper = estimate + crit * se;
  } else {
    const double crit     = random::student_t_quantile(conf_level, df);
    result.conf_int_lower = estimate - crit * se;
    result.conf_int_upper = inf;
  }
}

} // namespace

HypothesisTestResult t_test_one_sample(
    const std::vector<double>& x,
    double                     mu,
    Alternative                alternative,
    double                     conf_level) {
  detail::check_alternative(alternative);
  detail::check_confidence(conf_level);
  if (!std::isfinite(mu))
    throw std::invalid_argument("t_test: mu must be finite");
  const auto   mv = mean_variance(x);
  const double se = std::sqrt(mv.variance / static_cast<double>(mv.n));
  if (!(se > 0) || !std::isfinite(se))
    throw std::invalid_argument("t_test: standard error must be positive and finite");
  const double df = static_cast<double>(mv.n - 1);
  const double t  = (mv.mean - mu) / se;

  HypothesisTestResult result;
  result.statistic   = t;
  result.parameter1  = df;
  result.p_value     = t_p_value(t, df, alternative);
  result.estimate1   = mv.mean;
  result.alternative = alternative;
  result.method      = "One Sample t-test";
  fill_ci(result, mv.mean, se, df, conf_level, alternative);
  return result;
}

HypothesisTestResult t_test_two_sample(
    const std::vector<double>& x,
    const std::vector<double>& y,
    bool                       equal_variance,
    Alternative                alternative,
    double                     conf_level) {
  detail::check_alternative(alternative);
  detail::check_confidence(conf_level);
  const auto   mx = mean_variance(x);
  const auto   my = mean_variance(y);
  const double n1 = static_cast<double>(mx.n);
  const double n2 = static_cast<double>(my.n);

  double se, df;
  if (equal_variance) {
    const double pooled_var =
        ((n1 - 1.0) * mx.variance + (n2 - 1.0) * my.variance) / (n1 + n2 - 2.0);
    se = std::sqrt(pooled_var * (1.0 / n1 + 1.0 / n2));
    df = n1 + n2 - 2.0;
  } else {
    const double v1 = mx.variance / n1;
    const double v2 = my.variance / n2;
    se              = std::sqrt(v1 + v2);
    const double w1 = v1 / (v1 + v2), w2 = v2 / (v1 + v2);
    df = 1 / (w1 * w1 / (n1 - 1) + w2 * w2 / (n2 - 1));
  }
  if (!(se > 0) || !std::isfinite(se) || !std::isfinite(df))
    throw std::invalid_argument("t_test: standard error must be positive and finite");
  const double estimate = mx.mean - my.mean;
  if (!std::isfinite(estimate))
    throw std::invalid_argument("t_test: mean difference overflow");
  const double t = estimate / se;

  HypothesisTestResult result;
  result.statistic   = t;
  result.parameter1  = df;
  result.p_value     = t_p_value(t, df, alternative);
  result.estimate1   = mx.mean;
  result.estimate2   = my.mean;
  result.alternative = alternative;
  result.method      = equal_variance ? "Two Sample t-test" : "Welch Two Sample t-test";
  fill_ci(result, estimate, se, df, conf_level, alternative);
  return result;
}

HypothesisTestResult t_test_paired(
    const std::vector<double>& x,
    const std::vector<double>& y,
    Alternative                alternative,
    double                     conf_level) {
  if (x.size() != y.size())
    throw std::invalid_argument("t_test_paired: x and y must have the same length");
  detail::check_sample(x, 2);
  detail::check_sample(y, 2);
  std::vector<double> diff(x.size());
  for (std::size_t i = 0; i < x.size(); ++i)
    diff[i] = x[i] - y[i];
  auto result   = t_test_one_sample(diff, 0.0, alternative, conf_level);
  result.method = "Paired t-test";
  return result;
}

} // namespace datamunge::stats
