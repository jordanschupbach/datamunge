/// @file additional_tests.cpp
/// @brief Native classical tests and bounded exhaustive null enumerations.
#include <datamunge/stats/additional_tests.hpp>

#include <datamunge/linalg/least_squares.hpp>
#include <datamunge/stats/detail/ranking.hpp>
#include <datamunge/stats/detail/test_common.hpp>
#include <datamunge/stats/fisher_exact_test.hpp>
#include <datamunge/stats/nonparametric.hpp>

#include <initializer_list>
#include <numbers>
#include <numeric>
#include <random>
#include <string>

namespace datamunge::stats {
namespace {

/// @brief Constructs an owned statistic, tail probability and method label.
/// @param statistic Test-specific numeric statistic.
/// @param p Null-tail probability.
/// @param method Display name.
/// @return An initialized result with unrelated fields at their defaults.
HypothesisTestResult test_result(double statistic, double p, std::string method) {
  HypothesisTestResult result;
  result.statistic = statistic;
  result.p_value = p;
  result.method = std::move(method);
  return result;
}

/// @brief Validates a finite nonempty sample and computes its mean.
/// @param x Sample observations.
/// @return Arithmetic mean in extended precision.
/// @throws std::invalid_argument If the sample is empty or nonfinite.
long double sample_mean(const std::vector<double>& x) {
  detail::check_sample(x);
  long double mean = 0;
  for (double value : x) mean += static_cast<long double>(value) / x.size();
  return mean;
}

/// @brief Validates a known standard deviation and converts it to a standard error.
/// @param sigma Known population standard deviation.
/// @param n Positive sample size.
/// @return Standard deviation of the sample mean.
/// @throws std::invalid_argument If sigma or the computed error is invalid.
long double known_error(double sigma, std::size_t n) {
  if (!(sigma > 0) || !std::isfinite(sigma))
    throw std::invalid_argument("z test: standard deviation must be positive and finite");
  const long double error = static_cast<long double>(sigma) / std::sqrt(static_cast<long double>(n));
  if (!(error > 0)) throw std::invalid_argument("z test: standard error underflow");
  return error;
}

/// @brief Computes the fitted-normal KS statistic, using the sample standard deviation.
/// @param sample At least two finite observations, sorted in place.
/// @return Maximum difference between the empirical and fitted normal CDFs.
/// @throws std::invalid_argument If moments cannot be resolved or the sample is constant.
double fitted_normal_distance(std::vector<double>& sample) {
  const auto moments = detail::test_moments(sample);
  if (!(moments.variance > 0) || !std::isfinite(moments.variance))
    throw std::invalid_argument("Lilliefors test: sample must have finite positive variance");
  const auto scale = std::sqrt(moments.variance);
  std::sort(sample.begin(), sample.end());
  double distance = 0;
  for (std::size_t i = 0; i < sample.size(); ++i) {
    const double z = static_cast<double>((sample[i] - moments.mean) / scale);
    const double cdf = .5 * std::erfc(-z / std::numbers::sqrt2);
    distance = std::max({distance, static_cast<double>(i + 1) / sample.size() - cdf,
                         cdf - static_cast<double>(i) / sample.size()});
  }
  return distance;
}

/// @brief Evaluates a polynomial with ascending coefficients by Horner's rule.
/// @param x Evaluation point.
/// @param coefficients Constant term first.
/// @return Polynomial value.
double polynomial(double x, std::initializer_list<double> coefficients) {
  double value = 0;
  for (auto it = coefficients.end(); it != coefficients.begin();) value = value * x + *--it;
  return value;
}

/// @brief Generates the positive half of Royston's antisymmetric unit-norm weights.
/// @param n Validated sample size in 3–5000.
/// @return Positive contrast weights in outer-pair-first order.
std::vector<double> wilk_weights(std::size_t n) {
  if (n == 3) return {1.0 / std::numbers::sqrt2};
  std::vector<double> weights(n / 2);
  for (std::size_t i = 0; i < weights.size(); ++i)
    weights[i] = -random::normal_quantile((i + .625) / (n + .25));
  const double norm = std::sqrt(2 * std::inner_product(weights.begin(), weights.end(), weights.begin(), 0.0));
  for (double& value : weights) value /= norm;
  const double inverse = 1 / std::sqrt(static_cast<double>(n));
  const double edge = weights[0] + polynomial(inverse, {0, .221157, -.147981, -2.07119, 4.434685, -2.706056});
  const double next = weights[1] + polynomial(inverse, {0, .042981, -.293762, -1.752461, 5.682633, -3.582633});
  const std::size_t adjusted = n > 5 ? 2 : 1;
  const double old_norm = weights[0] * weights[0] + (adjusted == 2 ? weights[1] * weights[1] : 0);
  const double new_norm = edge * edge + (adjusted == 2 ? next * next : 0);
  const double scale = std::sqrt((.5 - new_norm) / (.5 - old_norm));
  for (std::size_t i = adjusted; i < weights.size(); ++i) weights[i] *= scale;
  weights[0] = edge;
  if (adjusted == 2) weights[1] = next;
  return weights;
}

/// @brief Validates both binomial sample sizes without wrapping their sums.
/// @param a First sample successes.
/// @param b Second sample successes.
/// @param c First sample failures.
/// @param d Second sample failures.
/// @return The two positive sample sizes.
/// @throws std::invalid_argument If either sample is empty or the combined size exceeds 256.
std::pair<std::size_t, std::size_t> binomial_sizes(std::size_t a, std::size_t b, std::size_t c, std::size_t d) {
  constexpr std::size_t limit = 256;
  if (a > limit || b > limit || c > limit || d > limit || a + b + c + d > limit ||
      a + c == 0 || b + d == 0)
    throw std::invalid_argument("unconditional exact test: need two nonempty samples with at most 256 total observations");
  return {a + c, b + d};
}

/// @brief Upper envelope of a binomial rejection polynomial over the nuisance probability.
///
/// Bernstein coefficients bound the polynomial on each interval. De Casteljau subdivision
/// tightens the bound, keeping a conservative numerical upper estimate even on the work limit.
/// @param coefficients Nonempty Bernstein coefficients on [0,1].
/// @return Upper estimate of the maximum, clamped to [0,1].
long double maximize_rejection(const std::vector<long double>& coefficients) {
  std::vector<std::vector<long double>> pending{coefficients};
  long double best = std::max(coefficients.front(), coefficients.back());
  long double upper_bound = 0;
  std::size_t subdivisions = 0;
  while (!pending.empty()) {
    auto current = std::move(pending.back());
    pending.pop_back();
    const long double upper = *std::max_element(current.begin(), current.end());
    if (upper <= best * (1 + 1e-10L) || subdivisions >= 4096) {
      upper_bound = std::max(upper_bound, upper);
      continue;
    }
    ++subdivisions;
    std::vector<long double> left(current.size()), right(current.size());
    left.front() = current.front();
    right.back() = current.back();
    for (std::size_t degree = current.size() - 1; degree > 0; --degree) {
      for (std::size_t j = 0; j < degree; ++j) current[j] = (current[j] + current[j + 1]) * .5L;
      left[current.size() - degree] = current.front();
      right[degree - 1] = current[degree - 1];
    }
    best = std::max(best, left.back());
    const auto left_max = *std::max_element(left.begin(), left.end());
    const auto right_max = *std::max_element(right.begin(), right.end());
    if (left_max > right_max) std::swap(left, right);
    pending.push_back(std::move(left));
    pending.push_back(std::move(right));
  }
  return std::clamp(std::max(best, upper_bound), 0.L, 1.L);
}

/// @brief Aggregates conditional rejection fractions for each total success count.
/// @tparam Reject Callable taking the two success counts and returning a rejection flag.
/// @param n First binomial sample size.
/// @param m Second binomial sample size.
/// @param reject Rejection rule, evaluated once per candidate table.
/// @return Maximum unconditional rejection probability over the common success probability.
template<class Reject>
double unconditional_p(std::size_t n, std::size_t m, const Reject& reject) {
  std::vector<long double> log_n(n + 1), log_m(m + 1), log_total(n + m + 1);
  for (std::size_t i = 0; i <= n; ++i) log_n[i] = detail::test_log_choose(n, i);
  for (std::size_t i = 0; i <= m; ++i) log_m[i] = detail::test_log_choose(m, i);
  for (std::size_t i = 0; i <= n + m; ++i) log_total[i] = detail::test_log_choose(n + m, i);
  std::vector<long double> coefficients(n + m + 1);
  for (std::size_t i = 0; i <= n; ++i)
    for (std::size_t j = 0; j <= m; ++j)
      if (reject(i, j)) coefficients[i + j] += std::exp(log_n[i] + log_m[j] - log_total[i + j]);
  return static_cast<double>(maximize_rejection(coefficients));
}

/// @brief Enumerates group-label assignments and compares centered sums of squares.
/// @param groups Two or more nonempty groups.
/// @param max_states Maximum assignments to visit.
/// @param ranks If true, rank before relabeling and report the tie-corrected H statistic.
/// @return Observed statistic, exact p-value and allocation count in parameter1.
/// @throws std::invalid_argument Invalid groups or limit, or a statistic outside double range.
/// @throws std::length_error If the work limit is exceeded.
HypothesisTestResult exact_groups(const std::vector<std::vector<double>>& groups, std::size_t max_states, bool ranks) {
  if (groups.size() < 2 || max_states == 0)
    throw std::invalid_argument("exact relabeling: need two or more groups and a positive work limit");
  std::vector<double> values;
  std::vector<std::size_t> labels;
  for (std::size_t j = 0; j < groups.size(); ++j) {
    detail::check_sample(groups[j]);
    values.insert(values.end(), groups[j].begin(), groups[j].end());
    labels.insert(labels.end(), groups[j].size(), j);
  }
  const double ties = ranks ? detail::tie_correction_sum(values) : 0;
  if (ranks) values = detail::rank_with_ties(values);
  const long double center = sample_mean(values);
  std::vector<long double> sums(groups.size());
  const auto score = [&] {
    std::fill(sums.begin(), sums.end(), 0.L);
    for (std::size_t i = 0; i < values.size(); ++i) sums[labels[i]] += values[i] - center;
    long double result = 0;
    for (std::size_t j = 0; j < groups.size(); ++j) result += sums[j] * sums[j] / groups[j].size();
    return result;
  };
  const long double observed = score();
  if (!std::isfinite(static_cast<double>(observed)))
    throw std::invalid_argument("exact relabeling: statistic exceeds numeric range");
  std::size_t visited = 0, extreme = 0;
  do {
    if (visited == max_states) throw std::length_error("exact relabeling: state limit exceeded");
    ++visited;
    if (score() >= observed * (1 - 1e-14L)) ++extreme;
  } while (std::next_permutation(labels.begin(), labels.end()));
  const long double n = values.size();
  const long double correction = 1 - ties / (n * (n * n - 1));
  const double statistic = ranks ? (correction > 0 ? static_cast<double>(12 * observed / (n * (n + 1) * correction)) : 0)
                                 : static_cast<double>(observed);
  auto result = test_result(statistic, static_cast<double>(extreme) / visited,
                            ranks ? "Exact Kruskal–Wallis permutation test" : "Exact permutation test for means");
  result.parameter1 = static_cast<double>(visited);
  return result;
}

} // namespace

HypothesisTestResult z_test_one_sample(const std::vector<double>& x, double sigma, double mu,
                                       Alternative alternative) {
  if (!std::isfinite(mu)) throw std::invalid_argument("z test: null mean must be finite");
  const auto mean = sample_mean(x);
  const double z = static_cast<double>((mean - mu) / known_error(sigma, x.size()));
  auto result = test_result(z, detail::normal_test_p(z, alternative), "One-sample z test");
  result.estimate1 = static_cast<double>(mean);
  result.alternative = alternative;
  return result;
}

HypothesisTestResult z_test_two_sample(const std::vector<double>& x, const std::vector<double>& y,
                                       double sigma_x, double sigma_y, double difference, Alternative alternative) {
  if (!std::isfinite(difference)) throw std::invalid_argument("z test: null difference must be finite");
  const auto first = sample_mean(x), second = sample_mean(y);
  const auto se = std::hypot(known_error(sigma_x, x.size()), known_error(sigma_y, y.size()));
  const double z = static_cast<double>((first - second - difference) / se);
  auto result = test_result(z, detail::normal_test_p(z, alternative), "Two-sample z test");
  result.estimate1 = static_cast<double>(first);
  result.estimate2 = static_cast<double>(second);
  result.alternative = alternative;
  return result;
}

HypothesisTestResult chow_test(const linalg::DenseMatrix<double>& x, const std::vector<double>& y,
                                const linalg::DenseMatrix<double>& z, const std::vector<double>& w) {
  const auto p = x.cols();
  if (!p || p != z.cols() || x.rows() < p || z.rows() < p || x.rows() != y.size() ||
      z.rows() != w.size() || (x.rows() == p && z.rows() == p))
    throw std::invalid_argument("Chow test: incompatible designs or no residual degrees of freedom");
  detail::check_sample(y);
  detail::check_sample(w);
  linalg::DenseMatrix<double> pooled(x.rows() + z.rows(), p);
  for (std::size_t i = 0; i < pooled.rows(); ++i)
    for (std::size_t j = 0; j < p; ++j) {
      const double value = i < x.rows() ? x(i, j) : z(i - x.rows(), j);
      if (!std::isfinite(value)) throw std::invalid_argument("Chow test: design must be finite");
      pooled(i, j) = value;
    }
  auto responses = y;
  responses.insert(responses.end(), w.begin(), w.end());
  const auto rss = [](const auto& design, const auto& response) {
    try {
      const auto fit = linalg::least_squares(design, response);
      return static_cast<long double>(fit.residual_norm) * fit.residual_norm;
    } catch (const std::runtime_error&) {
      throw std::invalid_argument("Chow test: rank-deficient design");
    }
  };
  const auto separate = rss(x, y) + rss(z, w);
  const auto shared = rss(pooled, responses);
  if (!(separate > 0) || !std::isfinite(shared) || !std::isfinite(separate))
    throw std::invalid_argument("Chow test: unrestricted residual variance must be finite and positive");
  const auto df = (x.rows() - p) + (z.rows() - p);
  const double f = static_cast<double>((std::max(0.L, shared - separate) / p) / (separate / df));
  auto result = test_result(f, detail::f_test_survival(f, p, df), "Chow test");
  result.parameter1 = static_cast<double>(p);
  result.parameter2 = static_cast<double>(df);
  return result;
}

HypothesisTestResult multinomial_test(const std::vector<std::size_t>& observed,
                                       const std::vector<double>& probabilities, std::size_t max_states) {
  if (observed.size() != probabilities.size() || observed.empty() || !max_states)
    throw std::invalid_argument("multinomial test: mismatched counts and probabilities or invalid state limit");
  std::size_t n = 0;
  long double sum = 0;
  bool impossible = false;
  for (std::size_t j = 0; j < observed.size(); ++j) {
    const double p = probabilities[j];
    if (!(p >= 0 && p <= 1) || observed[j] > std::numeric_limits<std::size_t>::max() - n)
      throw std::invalid_argument("multinomial test: invalid probability or count overflow");
    sum += p;
    n += observed[j];
    impossible |= p == 0 && observed[j] > 0;
  }
  if (!n || std::abs(sum - 1) > 1e-12L)
    throw std::invalid_argument("multinomial test: need positive count total and probabilities summing to one");
  constexpr auto method = "Exact multinomial goodness-of-fit test";
  if (impossible) return test_result(0, 0, method);
  std::vector<long double> log_p;
  for (double p : probabilities) if (p > 0) log_p.push_back(std::log(p / sum));
  if (log_p.size() == 1) return test_result(1, 1, method);
  if (n >= max_states) throw std::length_error("multinomial test: state limit exceeded");
  const long double log_factor = std::lgamma(static_cast<long double>(n) + 1);
  long double log_observed = log_factor;
  for (std::size_t j = 0; j < observed.size(); ++j)
    if (observed[j]) log_observed += observed[j] * std::log(probabilities[j] / sum)
                                    - std::lgamma(static_cast<long double>(observed[j]) + 1);
  std::vector<std::size_t> counts(log_p.size());
  counts.back() = n;
  long double tail = 0;
  for (std::size_t state = 0;; ++state) {
    if (state == max_states) throw std::length_error("multinomial test: state limit exceeded");
    long double log_mass = log_factor;
    for (std::size_t j = 0; j < counts.size(); ++j)
      log_mass += counts[j] * log_p[j] - std::lgamma(static_cast<long double>(counts[j]) + 1);
    if (log_mass <= log_observed + 1e-10L) tail += std::exp(log_mass);
    std::size_t j = counts.size() - 1;
    while (j > 0 && !counts[j]) --j;
    if (j == 0) break;
    const auto remaining = counts[j] - 1;
    counts[j] = 0;
    ++counts[j - 1];
    counts.back() = remaining;
  }
  return test_result(static_cast<double>(std::exp(log_observed)), static_cast<double>(std::min(1.L, tail)), method);
}

HypothesisTestResult barnard_exact_test(std::size_t a, std::size_t b, std::size_t c, std::size_t d,
                                        Alternative alternative) {
  detail::check_alternative(alternative);
  const auto [n, m] = binomial_sizes(a, b, c, d);
  const auto score = [=](std::size_t i, std::size_t j) {
    if (i + j == 0 || i + j == n + m) return 0.L;
    const long double p = static_cast<long double>(i + j) / (n + m);
    return (static_cast<long double>(i) / n - static_cast<long double>(j) / m) /
           std::sqrt(p * (1 - p) * (1.L / n + 1.L / m));
  };
  const long double observed = score(a, b);
  const auto order = [=](long double s) {
    return alternative == Alternative::TwoSided ? std::abs(s) : (alternative == Alternative::Less ? -s : s);
  };
  const auto p = unconditional_p(n, m, [&](auto i, auto j) {
    return order(score(i, j)) >= order(observed) - 1e-14L;
  });
  auto result = test_result(static_cast<double>(observed), p, "Barnard unconditional exact test (pooled score)");
  result.alternative = alternative;
  return result;
}

HypothesisTestResult boschloo_exact_test(std::size_t a, std::size_t b, std::size_t c, std::size_t d,
                                         Alternative alternative) {
  detail::check_alternative(alternative);
  const auto [n, m] = binomial_sizes(a, b, c, d);
  const auto one_sided = [&](Alternative side) {
    const double observed = fisher_exact_test_2x2(a, b, c, d, side).p_value;
    const double p = unconditional_p(n, m, [&](auto i, auto j) {
      return fisher_exact_test_2x2(i, j, n - i, m - j, side).p_value <= observed * (1 + 1e-12);
    });
    return std::pair{p, observed};
  };
  auto [p, statistic] = one_sided(alternative == Alternative::Greater ? Alternative::Greater : Alternative::Less);
  if (alternative == Alternative::TwoSided) {
    const auto other = one_sided(Alternative::Greater);
    if (other.first < p) { p = other.first; statistic = other.second; }
    p = std::min(1.0, 2 * p);
  }
  auto result = test_result(statistic, p, "Boschloo unconditional exact test");
  result.alternative = alternative;
  return result;
}

HypothesisTestResult shapiro_wilk_test(const std::vector<double>& sample) {
  detail::check_sample(sample, 3);
  if (sample.size() > 5000) throw std::invalid_argument("Shapiro–Wilk test: at most 5000 observations");
  auto sorted = sample;
  std::sort(sorted.begin(), sorted.end());
  const auto moments = detail::test_moments(sorted);
  if (!(moments.variance > 0) || !std::isfinite(moments.variance)) throw std::invalid_argument("Shapiro–Wilk test: sample must have finite positive variance");
  const auto weights = wilk_weights(sorted.size());
  long double contrast = 0, norm = 0;
  for (std::size_t i = 0; i < weights.size(); ++i) {
    contrast += weights[i] * (static_cast<long double>(sorted[sorted.size() - 1 - i]) - sorted[i]);
    norm += 2.L * weights[i] * weights[i];
  }
  const auto length = std::sqrt(norm * moments.variance * (sorted.size() - 1));
  const double deficit = static_cast<double>(std::clamp((length - contrast) / length * (1 + contrast / length), 0.L, 1.L));
  const double w = 1 - deficit;
  const double n = sorted.size();
  double p = 1;
  if (n == 3) {
    p = std::clamp(6 / std::numbers::pi * (std::asin(std::sqrt(w)) - std::numbers::pi / 3), 0.0, 1.0);
  } else if (n <= 11) {
    const double shifted = -2.273 + .459 * n - std::log(deficit);
    const double mu = polynomial(n, {.544, -.39978, .025054, -6.714e-4});
    const double sigma = std::exp(polynomial(n, {1.3822, -.77857, .062767, -.0020322}));
    p = shifted > 0 ? detail::normal_test_p((-std::log(shifted) - mu) / sigma, Alternative::Greater) : 0;
  } else {
    const double log_n = std::log(n);
    const double mu = polynomial(log_n, {-1.5861, -.31082, -.083751, .0038915});
    const double sigma = std::exp(polynomial(log_n, {-.4803, -.082676, .0030302}));
    p = detail::normal_test_p((std::log(deficit) - mu) / sigma, Alternative::Greater);
  }
  return test_result(w, p, "Shapiro–Wilk normality test");
}

HypothesisTestResult lilliefors_test(const std::vector<double>& sample, std::size_t draws, std::uint64_t seed) {
  detail::check_sample(sample, 4);
  if (!draws || draws == std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("Lilliefors test: invalid simulation count");
  auto copy = sample;
  const double observed = fitted_normal_distance(copy);
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> normal;
  std::size_t extreme = 1;
  for (std::size_t j = 0; j < draws; ++j) {
    for (auto& value : copy) value = normal(rng);
    if (fitted_normal_distance(copy) >= observed) ++extreme;
  }
  auto result = test_result(observed, static_cast<double>(extreme) / (draws + 1), "Lilliefors normality test (Monte Carlo)");
  result.parameter1 = static_cast<double>(draws);
  return result;
}

HypothesisTestResult permutation_test_exact(const std::vector<std::vector<double>>& groups, std::size_t max_states) {
  return exact_groups(groups, max_states, false);
}

HypothesisTestResult kruskal_wallis_test_exact(const std::vector<std::vector<double>>& groups, std::size_t max_states) {
  return exact_groups(groups, max_states, true);
}

HypothesisTestResult fisher_exact_test(const std::vector<std::vector<std::size_t>>& table, std::size_t max_states) {
  if (table.empty() || table.front().empty() || !max_states)
    throw std::invalid_argument("Fisher exact test: invalid table or state limit");
  std::vector<std::size_t> rows, columns(table.front().size());
  std::size_t total = 0;
  long double observed_denominator = 0;
  for (const auto& row : table) {
    if (row.size() != columns.size()) throw std::invalid_argument("Fisher exact test: ragged table");
    std::size_t row_total = 0;
    for (std::size_t j = 0; j < row.size(); ++j) {
      const auto count = row[j];
      if (count > 256 - total) throw std::invalid_argument("Fisher exact test: at most 256 observations");
      total += count;
      row_total += count;
      columns[j] += count;
      observed_denominator += std::lgamma(static_cast<long double>(count) + 1);
    }
    if (row_total) rows.push_back(row_total);
  }
  std::erase(columns, std::size_t{0});
  constexpr auto method = "Fisher–Freeman–Halton exact test";
  if (rows.size() < 2 || columns.size() < 2) return test_result(1, 1, method);
  if (rows.size() > 64 / columns.size())
    throw std::invalid_argument("Fisher exact test: at most 64 nonempty-margin cells");
  std::vector<long double> log_factorial(total + 1);
  for (std::size_t i = 0; i <= total; ++i) log_factorial[i] = std::lgamma(static_cast<long double>(i) + 1);
  long double numerator = -log_factorial[total];
  for (auto count : rows) numerator += log_factorial[count];
  for (auto count : columns) numerator += log_factorial[count];
  long double tail = 0;
  std::size_t states = 0;
  const auto enumerate = [&](const auto& next, std::size_t i, std::size_t j, std::size_t row_left,
                             long double denominator) -> void {
    if (i == rows.size() - 1) {
      if (states == max_states) throw std::length_error("Fisher exact test: state limit exceeded");
      ++states;
      for (auto count : columns) denominator += log_factorial[count];
      if (denominator >= observed_denominator - 1e-10L) tail += std::exp(numerator - denominator);
      return;
    }
    const auto later_capacity = std::accumulate(columns.begin() + j + 1, columns.end(), std::size_t{0});
    const auto lower = row_left > later_capacity ? row_left - later_capacity : 0;
    const auto upper = std::min(row_left, columns[j]);
    for (std::size_t count = lower; count <= upper; ++count) {
      columns[j] -= count;
      if (j + 1 == columns.size()) next(next, i + 1, 0, rows[i + 1], denominator + log_factorial[count]);
      else next(next, i, j + 1, row_left - count, denominator + log_factorial[count]);
      columns[j] += count;
    }
  };
  enumerate(enumerate, 0, 0, rows.front(), 0);
  auto result = test_result(static_cast<double>(std::exp(numerator - observed_denominator)),
                            static_cast<double>(std::min(1.L, tail)), method);
  result.parameter1 = static_cast<double>(states);
  return result;
}

} // namespace datamunge::stats
