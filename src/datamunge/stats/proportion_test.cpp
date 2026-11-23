/// @file proportion_test.cpp
/// @brief Classical proportion test statistics, domain checks and stable probabilities.
#include <algorithm>
#include <cmath>
#include <datamunge/random/distributions.hpp>
#include <datamunge/stats/detail/test_common.hpp>
#include <datamunge/stats/proportion_test.hpp>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Internal sign, tail and interval calculations.
namespace {

/// @brief Returns -1, zero or one according to the finite value sign.
/// @param v Signed value.
/// @return Sign of v.
double sign(double v) {
  return (v > 0.0) - (v < 0.0);
}

/// @brief Evaluates the selected normal tail without cancellation.
/// @param z Signed normal statistic.
/// @param alternative Requested tail.
/// @return Probability in [0,1].
double z_p_value(double z, Alternative alternative) {
  return detail::normal_test_p(z, alternative);
}

/// @brief Uncorrected Wilson score interval for a binomial proportion.
/// @param x Success count.
/// @param n Positive trial count.
/// @param z Positive normal critical value.
/// @param lower Receives lower bound.
/// @param upper Receives upper bound.
void wilson_interval(double x, double n, double z, double& lower, double& upper) {
  const double p_hat  = x / n;
  const double denom  = 1.0 + z * z / n;
  const double center = (p_hat + z * z / (2.0 * n)) / denom;
  const double half   = z * std::sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n)) / denom;
  lower               = std::clamp(center - half, 0.0, 1.0);
  upper               = std::clamp(center + half, 0.0, 1.0);
}

/// @brief Inverts a beta CDF with 200 bisection steps.
/// @param target Probability in (0,1).
/// @param a Positive first beta shape.
/// @param b Positive second beta shape.
/// @return Quantile in [0,1].
double beta_quantile(double target, double a, double b) {
  double lo = 0.0, hi = 1.0;
  for (int i = 0; i < 200; ++i) {
    const double mid = 0.5 * (lo + hi);
    if (random::detail::regularized_incomplete_beta(a, b, mid) < target)
      lo = mid;
    else
      hi = mid;
  }
  return 0.5 * (lo + hi);
}

/// @brief Checks the positive finite exposure required for a Poisson rate.
/// @param exposure Exposure in caller-selected units.
/// @throws std::invalid_argument Nonpositive or nonfinite exposure.
void check_exposure(double exposure) {
  if (!(exposure > 0) || !std::isfinite(exposure))
    throw std::invalid_argument("Poisson rate interval: exposure must be positive and finite");
}

/// @brief Computes a beta-odds quantile with direct probability evaluation on either side of one half.
/// @param tail Lower-tail probability strictly between zero and one half.
/// @param a First beta shape, positive.
/// @param b Second beta shape, positive.
/// @return Lower-tail odds quantile.
long double beta_lower_odds(double tail, double a, double b) {
  // Beta symmetry preserves the distance to one when the odds are large.
  if (random::detail::regularized_incomplete_beta(a, b, .5) >= tail) {
    const long double p = beta_quantile(tail, a, b);
    return p / (1 - p);
  }
  double lo = 0, hi = .5;
  const auto log_beta = random::detail::log_beta(a, b);
  for (int i = 0; i < 200; ++i) {
    const auto mid = .5 * (lo + hi);
    // P(Beta(a,b) <= 1-mid), with its complement preserved for small mid.
    const auto log_density = a * std::log1p(-mid) + b * std::log(mid) - log_beta;
    // The direct lower-beta fraction is stable until the shape-dependent switch.
    const auto probability = mid > (b + 1) / (a + b + 2)
        ? std::exp(log_density) * random::detail::incomplete_beta_cf(a, b, 1 - mid) / a
        : 1 - random::detail::regularized_incomplete_beta(b, a, mid);
    if (probability > tail)
      lo = mid;
    else
      hi = mid;
  }
  const long double complement = .5 * (lo + hi);
  return (1 - complement) / complement;
}

/// @brief Widened binomial counts that preserve small failure counts at high risks.
struct RiskSample {
  long double successes; ///< Observed success count.
  long double failures;  ///< Observed failure count.
  long double size;      ///< Strictly positive sample size.
};

/// @brief Computes the signed constrained score for a risk ratio at most one.
/// @param first Numerator sample.
/// @param second Denominator sample.
/// @param log_ratio Finite nonpositive log ratio.
/// @return Standardized difference with the common-ratio null variance.
long double risk_ratio_z(const RiskSample& first, const RiskSample& second, long double log_ratio) {
  const auto ratio = std::exp(log_ratio);
  const auto gap = -std::expm1(log_ratio);
  const auto successes = first.successes + second.successes;
  const auto a = ratio * (first.size + second.size);
  const auto u = gap * (successes + second.failures) - ratio * (first.failures + second.failures);
  const auto d = gap * second.failures;
  // The constrained risk solves a quadratic. Evaluate the smaller root
  // for the risk and the larger root of its complement independently.
  const auto discriminant = std::sqrt(u * u + 4 * a * d);
  const auto risk2 = 2 * successes / (a + successes + d + discriminant);
  const auto failure2 = u < 0 ? (discriminant - u) / (2 * a)
                              : d == 0 ? 0 : (2 * d) / (discriminant + u);
  const auto risk1 = ratio * risk2;
  const auto failure1 = gap + ratio * failure2;
  const auto variance = risk1 * failure1 / first.size +
                        ratio * ratio * risk2 * failure2 / second.size;
  const auto residual = first.successes > first.failures && second.successes > second.failures
      ? gap + std::fma(ratio, second.failures / second.size, -first.failures / first.size)
      : std::fma(-ratio, second.successes / second.size, first.successes / first.size);
  return variance == 0 ? 0 : residual / std::sqrt(variance);
}

/// @brief Inverts the upper-tail score to find the lower positive ratio bound.
/// @param first Numerator sample, with at least one success.
/// @param second Denominator sample, possibly with no successes.
/// @param accepted_log_ratio Accepted log-ratio value, at or below the observed log ratio.
/// @param critical Positive standard normal quantile.
/// @return Logarithm of the lower risk-ratio bound.
long double lower_risk_ratio(const RiskSample& first, const RiskSample& second,
                             long double accepted_log_ratio, long double critical) {
  // At exp(-512) the null variance is negligible even for the widest
  // supported counts, and the positive numerator count rules out the ratio.
  auto lo = -512.L, hi = accepted_log_ratio;
  for (int i = 0; i < 160; ++i) {
    const auto mid = std::midpoint(lo, hi);
    if (mid == lo || mid == hi)
      break;
    const auto z = mid <= 0 ? risk_ratio_z(first, second, mid) : -risk_ratio_z(second, first, -mid);
    if (z > critical)
      lo = mid;
    else
      hi = mid;
  }
  return std::midpoint(lo, hi);
}

} // namespace

HypothesisTestResult proportion_test_one_sample(
    std::size_t successes,
    std::size_t n,
    double      p,
    Alternative alternative,
    bool        correct,
    double      conf_level) {
  detail::check_alternative(alternative);
  detail::check_confidence(conf_level);
  if (!std::isfinite(p) || p < 0 || p > 1)
    throw std::invalid_argument("proportion test: invalid null probability");
  if (n == 0)
    throw std::invalid_argument("proportion_test_one_sample: n must be > 0");
  if (successes > n)
    throw std::invalid_argument("proportion_test_one_sample: successes must be <= n");

  const double x     = static_cast<double>(successes);
  const double nd    = static_cast<double>(n);
  const double p_hat = x / nd;
  const double diff  = p_hat - p;
  const double yates = correct ? std::min(0.5 / nd, std::fabs(diff)) : 0.0;
  if (p == 0 || p == 1)
    throw std::invalid_argument(
        "proportion_test: require 0 < null p < 1; use binomial_test for endpoints");
  const double se = std::sqrt(p * (1.0 - p) / nd);
  const double z  = (diff - sign(diff) * yates) / se;

  HypothesisTestResult result;
  result.statistic   = z * z;
  result.parameter1  = 1.0;
  result.p_value     = z_p_value(z, alternative);
  result.estimate1   = p_hat;
  result.alternative = alternative;
  result.method      = correct ? "1-sample proportions test with continuity correction"
                               : "1-sample proportions test";

  const double alpha  = 1.0 - conf_level;
  result.has_conf_int = true;
  if (alternative == Alternative::TwoSided) {
    wilson_interval(
        x,
        nd,
        random::normal_quantile(1.0 - 0.5 * alpha),
        result.conf_int_lower,
        result.conf_int_upper);
  } else if (alternative == Alternative::Less) {
    wilson_interval(
        x, nd, random::normal_quantile(conf_level), result.conf_int_lower, result.conf_int_upper);
    result.conf_int_lower = 0.0;
  } else {
    wilson_interval(
        x, nd, random::normal_quantile(conf_level), result.conf_int_lower, result.conf_int_upper);
    result.conf_int_upper = 1.0;
  }
  return result;
}

HypothesisTestResult proportion_test_two_sample(
    std::size_t successes1,
    std::size_t n1,
    std::size_t successes2,
    std::size_t n2,
    Alternative alternative,
    bool        correct,
    double      conf_level) {
  detail::check_alternative(alternative);
  detail::check_confidence(conf_level);
  if (n1 == 0 || n2 == 0)
    throw std::invalid_argument("proportion_test_two_sample: n1 and n2 must be > 0");
  if (successes1 > n1 || successes2 > n2)
    throw std::invalid_argument("proportion_test_two_sample: successes must be <= n");

  const double x1 = static_cast<double>(successes1), n1d = static_cast<double>(n1);
  const double x2 = static_cast<double>(successes2), n2d = static_cast<double>(n2);
  const double p1 = x1 / n1d, p2 = x2 / n2d;
  const double p_pool = (x1 + x2) / (n1d + n2d);

  const double diff      = p1 - p2;
  const double inv_n_sum = 1.0 / n1d + 1.0 / n2d;
  const double yates     = correct ? std::min(0.5 * inv_n_sum, std::fabs(diff)) : 0.0;
  const double se_pooled = std::sqrt(p_pool * (1.0 - p_pool) * inv_n_sum);
  const double z         = se_pooled > 0 ? (diff - sign(diff) * yates) / se_pooled : 0;

  HypothesisTestResult result;
  result.statistic   = z * z;
  result.parameter1  = 1.0;
  result.p_value     = z_p_value(z, alternative);
  result.estimate1   = p1;
  result.estimate2   = p2;
  result.alternative = alternative;
  result.method = correct ? "2-sample test for equality of proportions with continuity correction"
                          : "2-sample test for equality of proportions";

  const double alpha       = 1.0 - conf_level;
  const double se_unpooled = std::sqrt(p1 * (1.0 - p1) / n1d + p2 * (1.0 - p2) / n2d);
  const double correction  = correct ? 0.5 * inv_n_sum : 0.0;
  result.has_conf_int      = true;
  if (alternative == Alternative::TwoSided) {
    const double crit     = random::normal_quantile(1.0 - 0.5 * alpha);
    const double half     = crit * se_unpooled + correction;
    result.conf_int_lower = std::clamp(diff - half, -1.0, 1.0);
    result.conf_int_upper = std::clamp(diff + half, -1.0, 1.0);
  } else if (alternative == Alternative::Less) {
    const double crit     = random::normal_quantile(conf_level);
    result.conf_int_lower = -1.0;
    result.conf_int_upper = std::clamp(diff + crit * se_unpooled + correction, -1.0, 1.0);
  } else {
    const double crit     = random::normal_quantile(conf_level);
    result.conf_int_lower = std::clamp(diff - crit * se_unpooled - correction, -1.0, 1.0);
    result.conf_int_upper = 1.0;
  }
  return result;
}

HypothesisTestResult binomial_test(
    std::size_t successes,
    std::size_t n,
    double      p,
    Alternative alternative,
    double      conf_level) {
  detail::check_alternative(alternative);
  detail::check_confidence(conf_level);
  if (!std::isfinite(p) || p < 0 || p > 1)
    throw std::invalid_argument("proportion test: invalid null probability");
  if (n == 0)
    throw std::invalid_argument("binomial_test: n must be > 0");
  if (successes > n)
    throw std::invalid_argument("binomial_test: successes must be <= n");

  HypothesisTestResult result;
  result.statistic   = static_cast<double>(successes);
  result.parameter1  = static_cast<double>(n);
  result.estimate1   = static_cast<double>(successes) / static_cast<double>(n);
  result.alternative = alternative;
  result.method      = "Exact binomial test";

  result.p_value = detail::binomial_test_p(successes, n, p, alternative);

  const double alpha  = 1.0 - conf_level;
  result.has_conf_int = true;
  const double x = static_cast<double>(successes), nd = static_cast<double>(n);
  if (alternative == Alternative::TwoSided) {
    result.conf_int_lower = (successes == 0) ? 0.0 : beta_quantile(0.5 * alpha, x, nd - x + 1.0);
    result.conf_int_upper =
        (successes == n) ? 1.0 : beta_quantile(1.0 - 0.5 * alpha, x + 1.0, nd - x);
  } else if (alternative == Alternative::Less) {
    result.conf_int_lower = 0.0;
    result.conf_int_upper = (successes == n) ? 1.0 : beta_quantile(conf_level, x + 1.0, nd - x);
  } else {
    result.conf_int_lower =
        (successes == 0) ? 0.0 : beta_quantile(1.0 - conf_level, x, nd - x + 1.0);
    result.conf_int_upper = 1.0;
  }
  return result;
}

ProportionDifferenceInterval newcombe_difference_interval(
    std::size_t successes1, std::size_t n1, std::size_t successes2, std::size_t n2,
    double confidence_level) {
  detail::check_confidence(confidence_level);
  if (n1 == 0 || n2 == 0 || successes1 > n1 || successes2 > n2)
    throw std::invalid_argument("newcombe_difference_interval: invalid counts");
  const auto p1 = static_cast<double>(successes1) / n1;
  const auto p2 = static_cast<double>(successes2) / n2;
  const auto z = -random::normal_quantile((1 - confidence_level) / 2);
  double lower1, upper1, lower2, upper2;
  wilson_interval(successes1, n1, z, lower1, upper1);
  wilson_interval(successes2, n2, z, lower2, upper2);
  const auto difference = p1 - p2;
  return {difference, std::max(-1., difference - std::hypot(p1 - lower1, upper2 - p2)),
          std::min(1., difference + std::hypot(upper1 - p1, p2 - lower2))};
}

RiskRatioInterval risk_ratio_score_interval(
    std::size_t successes1, std::size_t n1, std::size_t successes2, std::size_t n2,
    double confidence_level) {
  detail::check_confidence(confidence_level);
  if (n1 == 0 || n2 == 0 || successes1 > n1 || successes2 > n2)
    throw std::invalid_argument("risk_ratio_score_interval: invalid counts");
  constexpr auto infinity = std::numeric_limits<double>::infinity();
  if (successes1 == 0 && successes2 == 0)
    return {std::nullopt, 0, infinity};

  const RiskSample first{static_cast<long double>(successes1), static_cast<long double>(n1 - successes1),
                         static_cast<long double>(n1)};
  const RiskSample second{static_cast<long double>(successes2), static_cast<long double>(n2 - successes2),
                          static_cast<long double>(n2)};
  const auto estimate = (first.successes / first.size) / (second.successes / second.size);
  // For a boundary point estimate use a finite accepted log ratio.
  const auto center = successes1 == 0 ? -512.L : successes2 == 0 ? 512.L : std::log(estimate);
  const auto critical = -random::normal_quantile((1 - confidence_level) / 2);
  if (critical == 0)
    return {static_cast<double>(estimate), static_cast<double>(estimate), static_cast<double>(estimate)};
  const auto lower = successes1 == 0 ? 0 : std::exp(lower_risk_ratio(first, second, center, critical));
  const auto upper = successes2 == 0 ? infinity : std::exp(-lower_risk_ratio(second, first, -center, critical));
  return {static_cast<double>(estimate), static_cast<double>(lower), static_cast<double>(upper)};
}

PoissonRateInterval poisson_rate_interval(std::size_t events, double exposure,
                                         double confidence_level) {
  check_exposure(exposure);
  detail::check_confidence(confidence_level);
  const auto tail = (1 - confidence_level) / 2;
  const auto count = static_cast<double>(events);
  const auto lower = events == 0 ? 0 : random::gamma_quantile(tail, count);
  const auto upper = random::gamma_quantile(tail, count + 1, 1, false);
  return {count / exposure, lower / exposure, upper / exposure};
}

PoissonRateInterval poisson_rate_ratio_interval(std::size_t events1, double exposure1,
                                               std::size_t events2, double exposure2,
                                               double confidence_level) {
  check_exposure(exposure1);
  check_exposure(exposure2);
  detail::check_confidence(confidence_level);
  constexpr auto infinity = std::numeric_limits<double>::infinity();
  if (events1 == 0 && events2 == 0)
    return {std::nullopt, 0, infinity};
  const auto tail = (1 - confidence_level) / 2;
  const auto first = static_cast<double>(events1);
  const auto second = static_cast<double>(events2);
  const auto log_exposure_ratio =
      std::log(static_cast<long double>(exposure2)) - std::log(static_cast<long double>(exposure1));
  const auto estimate = events1 == 0 ? 0.0 : events2 == 0 ? infinity :
      static_cast<double>(std::exp(log_exposure_ratio + std::log(static_cast<long double>(events1) / events2)));
  return {
      estimate,
      events1 == 0 ? 0 : static_cast<double>(std::exp(log_exposure_ratio +
                                                      std::log(beta_lower_odds(tail, first, second + 1)))),
      events2 == 0 ? infinity : static_cast<double>(std::exp(log_exposure_ratio -
                                                             std::log(beta_lower_odds(tail, second, first + 1))))};
}

HypothesisTestResult cochran_armitage_test(const std::vector<std::size_t>& successes,
                                           const std::vector<std::size_t>& trials,
                                           const std::vector<double>& scores, Alternative alternative) {
  detail::check_alternative(alternative);
  if (trials.size() < 2 || successes.size() != trials.size() ||
      (!scores.empty() && scores.size() != trials.size()))
    throw std::invalid_argument("cochran_armitage_test: mismatched inputs or too few groups");
  long double scale = scores.empty() ? trials.size() : 0;
  for (const double score : scores) {
    if (!std::isfinite(score))
      throw std::invalid_argument("cochran_armitage_test: scores must be finite");
    scale = std::max(scale, std::abs(static_cast<long double>(score)));
  }
  if (scale == 0)
    throw std::invalid_argument("cochran_armitage_test: scores must vary");
  /// @brief Returns the scaled score of a category, using unit spacing by default.
  const auto category_score = [&](std::size_t i) {
    return (scores.empty() ? static_cast<long double>(i + 1) : scores[i]) / scale;
  };
  const auto origin = category_score(0);
  long double positive = 0, negative = 0, weighted_score = 0;
  for (std::size_t i = 0; i < trials.size(); ++i) {
    if (trials[i] == 0 || successes[i] > trials[i])
      throw std::invalid_argument("cochran_armitage_test: invalid counts");
    positive += successes[i];
    negative += trials[i] - successes[i];
    weighted_score += trials[i] * (category_score(i) - origin);
  }
  if (!(positive > 0 && negative > 0))
    throw std::invalid_argument("cochran_armitage_test: outcomes must vary");
  const auto total = positive + negative;
  const auto success_probability = positive / total;
  const auto failure_probability = negative / total;
  const auto center = weighted_score / total;
  long double sum_squares = 0, numerator = 0;
  for (std::size_t i = 0; i < trials.size(); ++i) {
    const auto centered_score = (category_score(i) - origin) - center;
    sum_squares += trials[i] * centered_score * centered_score;
    const auto residual = failure_probability * successes[i] -
                          success_probability * (trials[i] - successes[i]);
    numerator += centered_score * residual;
  }
  if (!(sum_squares > 0))
    throw std::invalid_argument("cochran_armitage_test: scores must vary");
  HypothesisTestResult result;
  result.statistic = static_cast<double>(numerator /
      std::sqrt(success_probability * failure_probability * sum_squares));
  result.alternative = alternative;
  result.p_value = detail::normal_test_p(result.statistic, alternative);
  result.method = "Cochran-Armitage trend test";
  return result;
}

} // namespace datamunge::stats
