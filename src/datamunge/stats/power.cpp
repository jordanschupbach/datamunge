/// @file power.cpp
/// @brief Power and integer sample-size inversion for Gaussian and binomial designs.

#include <datamunge/random/noncentral_distributions.hpp>
#include <datamunge/stats/detail/test_common.hpp>
#include <datamunge/stats/power.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Validation, rejection rules, and inversion shared by the design entry points.
namespace {

/// @brief Checks an open-unit-interval significance or power probability.
/// @param p Candidate probability.
/// @throws std::invalid_argument Invalid probability.
void check_power_probability(double p) {
  if (!(p > 0 && p < 1))
    throw std::invalid_argument("power calculation: significance and target power must be in (0,1)");
}

/// @brief Allocates significance to each active tail of the rejection rule.
/// @param alpha Validated significance level.
/// @param two_sided Whether the level is split between two tails.
/// @return Representable tail probability.
double one_tail_level(double alpha, bool two_sided) {
  return two_sided ? std::max(alpha / 2, std::numeric_limits<double>::denorm_min()) : alpha;
}

/// @brief Refines a standard normal quantile using its directly evaluated smaller tail.
/// @param p Lower-tail probability strictly between zero and one.
/// @return Signed standard normal quantile.
long double power_normal_quantile(double p) {
  const long double small = std::min(p, 1 - p);
  auto z = std::abs(static_cast<long double>(random::normal_quantile(p)));
  for (int i = 0; i < 2; ++i) {
    const auto tail = std::erfc(z / std::numbers::sqrt2_v<long double>) / 2;
    const auto density = std::exp(-z * z / 2) / std::sqrt(2 * std::numbers::pi_v<long double>);
    if (density > 0)
      z += (tail - small) / density;
  }
  return std::copysign(z, p - .5);
}

/// @brief Checks a standardized effect and the options controlling the rejection rule.
/// @param effect Signed population effect in SD units.
/// @param alpha Significance level.
/// @param alternative Requested tail or tails.
/// @throws std::invalid_argument Invalid effect, significance level or alternative.
void check_effect_options(double effect, double alpha, Alternative alternative) {
  check_power_probability(alpha);
  detail::check_alternative(alternative);
  if (!std::isfinite(effect))
    throw std::invalid_argument("power calculation: standardized effect must be finite");
}

/// @brief Orients a signed effect toward the rejection region.
/// @param effect First-minus-second contrast from the null.
/// @param alternative One- or two-sided rejection region.
/// @return Absolute effect for two sides and signed effect toward the selected side otherwise.
long double oriented_effect(long double effect, Alternative alternative) {
  return alternative == Alternative::TwoSided ? std::abs(effect) : alternative == Alternative::Less ? -effect : effect;
}

/// @brief Computes strict two-sided or directional noncentral-t power.
/// @param df Residual degrees of freedom.
/// @param information Effective sample size multiplying the squared standardized effect.
/// @param effect Signed standardized population effect.
/// @param alpha Rejection probability at the null.
/// @param alternative Direction of the rejection region.
/// @return Probability of falling in the rejection region.
double t_power(double df, long double information, double effect, double alpha, Alternative alternative) {
  if (effect == 0)
    return alpha;
  const bool two_sided = alternative == Alternative::TwoSided;
  const auto ncp = oriented_effect(effect, alternative) * std::sqrt(information);
  // Keep the smallest representable directional probability at the
  // double-precision boundary of a two-sided level.
  const auto tail = one_tail_level(alpha, two_sided);
  const auto cutoff = random::noncentral_t_quantile(tail, df, 0, false);
  if (std::isinf(cutoff))
    return cutoff > 0 ? 0 : 1;
  if (std::abs(ncp) > std::numeric_limits<double>::max())
    return ncp > 0 ? 1 : 0;
  const auto displacement = static_cast<double>(ncp);
  const auto upper = random::noncentral_t_cdf(cutoff, df, displacement, false);
  if (!two_sided)
    return upper;
  return std::min(1., upper + random::noncentral_t_cdf(-cutoff, df, displacement));
}

/// @brief Computes noncentral t power using pooled residual degrees of freedom.
/// @param first First group size.
/// @param second Second group size.
/// @param effect Signed standardized population effect.
/// @param alpha Significance level.
/// @param alternative Direction of the rejection region.
/// @return Prospective rejection probability.
double pooled_power(std::size_t first, std::size_t second, double effect, double alpha, Alternative alternative) {
  const long double n = first, m = second;
  return t_power(static_cast<double>(n + m - 2), (n / (n + m)) * m, effect, alpha, alternative);
}

/// @brief Identifies an effect that can raise power as information increases.
/// @param effect Signed population standardized effect.
/// @param alternative Direction of the rejection region.
/// @return Whether the effect favors the rejection region.
bool favorable_effect(double effect, Alternative alternative) {
  if (alternative == Alternative::Less)
    return effect < 0;
  return alternative == Alternative::TwoSided ? effect != 0 : effect > 0;
}

/// @brief Reduces positive group allocation units to their coprime ratio.
/// @param allocation Integer first and second allocation units.
/// @return Coprime units in the same group order.
/// @throws std::invalid_argument An allocation unit is zero.
std::pair<std::size_t, std::size_t> fixed_allocation(std::pair<std::size_t, std::size_t> allocation) {
  const auto [first, second] = allocation;
  if (first == 0 || second == 0)
    throw std::invalid_argument("sample size: allocation units must be positive");
  const auto divisor = std::gcd(first, second);
  return {first / divisor, second / divisor};
}

/// @brief Finds the first admissible integer reaching the target.
/// @tparam Power Callable computing power at an admissible count.
/// @param minimum Smallest admissible count.
/// @param maximum Largest representable count.
/// @param target Desired rejection probability.
/// @param favorable Whether power can increase from its minimum-size value.
/// @param power Power evaluated at the count.
/// @return First admissible count meeting the target.
/// @throws std::domain_error The effect cannot attain the requested power.
/// @throws std::overflow_error No representable count reaches the target.
template <class Power>
std::size_t required_size(std::size_t minimum, std::size_t maximum, double target, bool favorable,
                          const Power& power) {
  if (minimum > maximum)
    throw std::overflow_error("sample size: no valid size at the requested allocation");
  if (power(minimum) >= target)
    return minimum;
  if (!favorable)
    throw std::domain_error("sample size: the effect cannot attain target power in the specified direction");

  auto lower = minimum, upper = minimum;
  for (;;) {
    if (upper == maximum)
      throw std::overflow_error("sample size: required size is beyond the representable range");
    upper = upper > maximum / 2 ? maximum : upper * 2;
    if (power(upper) >= target)
      break;
    lower = upper;
  }
  while (upper - lower > 1) {
    const auto mid = std::midpoint(lower, upper);
    (power(mid) >= target ? upper : lower) = mid;
  }
  return upper;
}

/// @brief Integrates a normal alternative over the selected rejection region.
/// @param shift Change in the expected contrast, oriented toward the upper tail.
/// @param boundary Upper rejection boundary on the contrast scale.
/// @param variance Sampling variance under the alternative.
/// @param two_sided Whether to include the symmetric lower rejection tail.
/// @return Probability of rejection, using a strict comparison for a deterministic alternative.
double normal_rejection_power(long double shift, long double boundary, long double variance, bool two_sided) {
  if (variance == 0)
    return shift > boundary ? 1 : 0;
  const auto scale = std::numbers::sqrt2_v<long double> * std::sqrt(variance);
  const auto upper = std::erfc((boundary - shift) / scale) / 2;
  const auto lower = two_sided ? std::erfc((boundary + shift) / scale) / 2 : 0.L;
  return static_cast<double>(std::min(1.L, upper + lower));
}

/// @brief Fixed-risk rejection reference, reusable across sample sizes at one or more allocations.
struct TwoProportionReference {
  std::pair<long double, long double> null_variances; ///< Individual Bernoulli variances under the null.
  std::pair<long double, long double> alt_variances;  ///< Individual Bernoulli variances under the alternative.
  long double shift; ///< Change in first-minus-second risk, oriented toward the selected rejection region.
  long double critical; ///< Standard normal cutoff of the oriented test.
  double alpha;      ///< Nominal rejection probability at the null.
  bool two_sided;    ///< Whether to include the opposite rejection tail.

  /// @brief Evaluates the rejection region under the specified alternative risks.
  /// @param first Positive first group size.
  /// @param second Positive second group size.
  /// @return Directional or two-sided power at this allocation.
  double power(long double first, long double second) const {
    const auto null_variance = null_variances.first / first + null_variances.second / second;
    const auto alternative_variance = alt_variances.first / first + alt_variances.second / second;
    if (shift == 0 && alternative_variance == null_variance)
      return alpha;
    return normal_rejection_power(shift, critical * std::sqrt(null_variance), alternative_variance, two_sided);
  }
};

/// @brief Computes the Bernoulli variance for an admissible probability.
/// @param p Risk in [0,1].
/// @return Bernoulli variance retaining precision near either boundary.
/// @throws std::invalid_argument Risk outside [0,1] or nonfinite.
long double bernoulli_variance(double p) {
  if (!(p >= 0 && p <= 1))
    throw std::invalid_argument("two-proportion power: risks must be in [0,1]");
  return p * (1.L - p);
}

/// @brief Validates risks and sets up the normal rejection cutoffs.
/// @param null_risks Risk pair under the null.
/// @param alt_risks Risk pair under the alternative.
/// @param alpha Nominal significance level.
/// @param alternative Direction of the risk difference.
/// @return Validated reference before inserting group sizes.
/// @throws std::invalid_argument Invalid risks, null variance, level, or direction.
TwoProportionReference two_proportion_reference(std::pair<double, double> null_risks,
                                               std::pair<double, double> alt_risks,
                                               double alpha, Alternative alternative) {
  check_power_probability(alpha);
  detail::check_alternative(alternative);
  const auto first_null = bernoulli_variance(null_risks.first);
  const auto second_null = bernoulli_variance(null_risks.second);
  if (first_null == 0 && second_null == 0)
    throw std::invalid_argument("two-proportion power: null risk difference must have positive variance");
  const auto shift = (static_cast<long double>(alt_risks.first) - null_risks.first) -
                     (static_cast<long double>(alt_risks.second) - null_risks.second);
  const bool two_sided = alternative == Alternative::TwoSided;
  const auto tail = one_tail_level(alpha, two_sided);
  return {{first_null, second_null}, {bernoulli_variance(alt_risks.first), bernoulli_variance(alt_risks.second)},
          oriented_effect(shift, alternative), -power_normal_quantile(tail), alpha, two_sided};
}

} // namespace

double one_sample_t_test_power(std::size_t n, double standardized_effect, double alpha, Alternative alternative) {
  check_effect_options(standardized_effect, alpha, alternative);
  if (n < 2)
    throw std::invalid_argument("one-sample t-test power: need at least two observations");
  return t_power(static_cast<double>(n - 1), static_cast<long double>(n), standardized_effect, alpha, alternative);
}

double paired_t_test_power(std::size_t pairs, double standardized_difference, double alpha, Alternative alternative) {
  return one_sample_t_test_power(pairs, standardized_difference, alpha, alternative);
}

double pooled_t_test_power(std::size_t first, std::size_t second, double standardized_effect,
                           double alpha, Alternative alternative) {
  check_effect_options(standardized_effect, alpha, alternative);
  if (first < 2 || second < 2)
    throw std::invalid_argument("pooled t-test power: need at least two observations per group");
  return pooled_power(first, second, standardized_effect, alpha, alternative);
}

std::size_t one_sample_t_test_sample_size(double standardized_effect, double target_power,
                                          double alpha, Alternative alternative) {
  check_effect_options(standardized_effect, alpha, alternative);
  check_power_probability(target_power);
  return required_size(2, std::numeric_limits<std::size_t>::max(), target_power,
                       favorable_effect(standardized_effect, alternative), [&](std::size_t n) {
    return t_power(static_cast<double>(n - 1), static_cast<long double>(n), standardized_effect, alpha, alternative);
  });
}

std::size_t paired_t_test_sample_size(double standardized_difference, double target_power,
                                      double alpha, Alternative alternative) {
  return one_sample_t_test_sample_size(standardized_difference, target_power, alpha, alternative);
}

std::pair<std::size_t, std::size_t> pooled_t_test_sample_size(
    double standardized_effect, double target_power, std::pair<std::size_t, std::size_t> allocation,
    double alpha, Alternative alternative) {
  check_effect_options(standardized_effect, alpha, alternative);
  check_power_probability(target_power);
  const auto [first, second] = fixed_allocation(allocation);
  const std::size_t minimum = std::min(first, second) == 1 ? 2 : 1;
  const auto maximum = std::numeric_limits<std::size_t>::max() / std::max(first, second);
  const auto multiplier = required_size(minimum, maximum, target_power,
      favorable_effect(standardized_effect, alternative), [&](std::size_t scale) {
        return pooled_power(first * scale, second * scale, standardized_effect, alpha, alternative);
      });
  return {first * multiplier, second * multiplier};
}

double one_way_anova_power(const std::vector<double>& means, const std::vector<std::size_t>& group_sizes,
                           double common_variance, double alpha) {
  check_power_probability(alpha);
  if (means.size() < 2 || means.size() != group_sizes.size() || !(common_variance > 0) ||
      !std::isfinite(common_variance))
    throw std::invalid_argument("ANOVA power: need matched groups and a positive finite variance");
  long double scale = 0, residual_df = 0;
  for (std::size_t i = 0; i < means.size(); ++i) {
    if (!std::isfinite(means[i]) || group_sizes[i] == 0)
      throw std::invalid_argument("ANOVA power: means must be finite and every group size positive");
    scale = std::max(scale, std::abs(static_cast<long double>(means[i])));
    residual_df += group_sizes[i] - 1;
  }
  if (!(residual_df > 0))
    throw std::invalid_argument("ANOVA power: residual degrees of freedom must be positive");
  if (scale == 0)
    return alpha;

  const long double anchor = means.front();
  long double center = 0, weight = 0, between = 0;
  for (std::size_t i = 0; i < means.size(); ++i) {
    const auto difference = static_cast<long double>(means[i]) - anchor;
    const auto value = std::isfinite(difference) ? difference / scale : means[i] / scale - anchor / scale;
    const long double added = group_sizes[i];
    const auto total = weight + added;
    const auto delta = value - center;
    between += delta * delta * (weight / total) * added;
    center += delta * (added / total);
    weight = total;
  }
  if (between == 0)
    return alpha;

  const auto scaled_sd = scale / std::sqrt(static_cast<long double>(common_variance));
  const auto ncp = (between * scaled_sd) * scaled_sd;
  const auto df1 = static_cast<double>(means.size() - 1);
  const auto df2 = static_cast<double>(residual_df);
  const auto cutoff = random::noncentral_f_quantile(alpha, df1, df2, 0, false);
  if (std::isinf(cutoff))
    return 0;
  if (ncp > std::numeric_limits<double>::max())
    return 1;
  return random::noncentral_f_cdf(cutoff, df1, df2, static_cast<double>(ncp), false);
}

double two_proportion_power(std::size_t first, std::size_t second,
                            std::pair<double, double> null_risks, std::pair<double, double> alternative_risks,
                            double alpha, Alternative alternative) {
  if (first == 0 || second == 0)
    throw std::invalid_argument("two-proportion power: group sizes must be positive");
  return two_proportion_reference(null_risks, alternative_risks, alpha, alternative).power(first, second);
}

std::pair<std::size_t, std::size_t> two_proportion_sample_size(
    std::pair<double, double> null_risks, std::pair<double, double> alternative_risks,
    double target_power, std::pair<std::size_t, std::size_t> allocation, double alpha, Alternative alternative) {
  check_power_probability(target_power);
  const auto reference = two_proportion_reference(null_risks, alternative_risks, alpha, alternative);
  const auto [first, second] = fixed_allocation(allocation);
  const auto maximum = std::numeric_limits<std::size_t>::max() / std::max(first, second);
  const auto multiplier = required_size(1, maximum, target_power, reference.shift > 0, [&](std::size_t n) {
    return reference.power(first * n, second * n);
  });
  return {first * multiplier, second * multiplier};
}

std::size_t schoenfeld_event_count(double hazard_ratio, double allocation_fraction, double target_power,
                                   double alpha, Alternative alternative) {
  check_power_probability(alpha);
  check_power_probability(target_power);
  detail::check_alternative(alternative);
  if (!(hazard_ratio > 0) || !std::isfinite(hazard_ratio) ||
      !(allocation_fraction > 0 && allocation_fraction < 1))
    throw std::invalid_argument("Schoenfeld event count: require a positive finite ratio and allocation strictly in (0,1)");
  if (target_power <= alpha)
    return 0;
  const auto log_ratio = std::log(static_cast<long double>(hazard_ratio));
  if (!favorable_effect(static_cast<double>(log_ratio), alternative))
    throw std::domain_error("Schoenfeld event count: the ratio cannot attain target power in the specified direction");
  const auto tail = one_tail_level(alpha, alternative == Alternative::TwoSided);
  const long double allocation = allocation_fraction;
  const auto quantiles = power_normal_quantile(target_power) - power_normal_quantile(tail);
  const auto scaled = quantiles / (std::abs(log_ratio) * std::sqrt(allocation * (1 - allocation)));
  const auto events = std::max(1.L, std::ceil(scaled * scaled));
  if (events >= std::ldexp(1.L, std::numeric_limits<std::size_t>::digits))
    throw std::overflow_error("Schoenfeld event count: required count exceeds std::size_t");
  return static_cast<std::size_t>(events);
}

ClusterDesignEffect cluster_design_effect(const std::vector<std::size_t>& cluster_sizes, double rho) {
  if (cluster_sizes.empty() || !(rho >= 0 && rho <= 1))
    throw std::invalid_argument("cluster design: require nonempty cluster sizes and correlation in [0,1]");
  long double individuals = 0, same_cluster_pairs = 0;
  for (const auto size : cluster_sizes) {
    if (size == 0)
      throw std::invalid_argument("cluster design: cluster sizes must be positive");
    individuals += size;
    same_cluster_pairs += static_cast<long double>(size) * (size - 1);
  }
  const auto effect = 1 + rho * (same_cluster_pairs / individuals);
  return {static_cast<double>(effect), static_cast<double>(individuals / effect)};
}

ClusterRandomizedPower cluster_randomized_mean_power(
    const std::vector<std::size_t>& first, const std::vector<std::size_t>& second,
    double rho, double standardized_effect, double alpha, Alternative alternative) {
  check_effect_options(standardized_effect, alpha, alternative);
  const auto first_arm = cluster_design_effect(first, rho);
  const auto second_arm = cluster_design_effect(second, rho);
  if (standardized_effect == 0)
    return {alpha, first_arm, second_arm};
  const auto variance = 1.L / first_arm.effective_sample_size + 1.L / second_arm.effective_sample_size;
  const bool two_sided = alternative == Alternative::TwoSided;
  const auto boundary = -power_normal_quantile(one_tail_level(alpha, two_sided)) * std::sqrt(variance);
  return {normal_rejection_power(oriented_effect(standardized_effect, alternative), boundary, variance, two_sided),
          first_arm, second_arm};
}

ClusterRandomizedPower cluster_randomized_proportion_power(
    const std::vector<std::size_t>& first, const std::vector<std::size_t>& second, double rho,
    std::pair<double, double> null_risks, std::pair<double, double> alternative_risks,
    double alpha, Alternative alternative) {
  const auto reference = two_proportion_reference(null_risks, alternative_risks, alpha, alternative);
  const auto first_arm = cluster_design_effect(first, rho);
  const auto second_arm = cluster_design_effect(second, rho);
  return {reference.power(first_arm.effective_sample_size, second_arm.effective_sample_size), first_arm, second_arm};
}

} // namespace datamunge::stats
