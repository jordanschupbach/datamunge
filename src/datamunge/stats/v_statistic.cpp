/// @file v_statistic.cpp
/// @brief One validated exact/Monte Carlo tuple producer, actual traces and variance fixtures.
#include <cmath>
#include <datamunge/stats/v_statistic.hpp>
#include <random>
#include <stdexcept>
namespace datamunge::stats {
/// @brief Validate and store tuple degree and sampling policy.
VStatistic::VStatistic(std::size_t degree, VStatisticOptions options)
    : degree_(degree), options_(options) {
  if (!degree || degree > 64)
    throw std::invalid_argument("VStatistic degree");
  if (options.max_terms > 5000000)
    throw std::length_error("VStatistic draw cap");
}
/// @brief Run without optional record copies through the same producing evaluator.
VStatisticResult VStatistic::run(const std::vector<double>& sample, const SymmetricKernel& kernel)
    const {
  return evaluate(sample, kernel, 0).result;
}
/// @brief Capture actual tuple values and indices through the same evaluator.
VStatisticTrace VStatistic::trace(
    const std::vector<double>& sample,
    const SymmetricKernel&     kernel,
    std::size_t                max_records) const {
  return evaluate(sample, kernel, max_records);
}
/// @brief Enumerate or draw each ordered tuple once, evaluating its callback exactly once.
VStatisticTrace VStatistic::evaluate(
    const std::vector<double>& sample,
    const SymmetricKernel&     kernel,
    std::size_t                max_records) const {
  std::size_t n = sample.size(), m = degree_;
  if (!n || !kernel)
    throw std::invalid_argument("VStatistic empty sample/kernel");
  if (n > 1000000 || max_records > 1000000 / (2 * m))
    throw std::length_error("VStatistic storage cap");
  for (double x : sample)
    if (!std::isfinite(x))
      throw std::invalid_argument("VStatistic sample");
  std::size_t total = 1;
  bool        small = true;
  for (std::size_t k = 0; k < m; ++k) {
    if (total > 5000000 / n) {
      small = false;
      break;
    }
    total *= n;
  }
  bool        exact = options_.max_terms == 0 && small;
  std::size_t terms = exact ? total : (options_.max_terms ? options_.max_terms : 200000);
  if (terms > 20000000 / m)
    throw std::length_error("VStatistic work cap");
  VStatisticTrace out;
  out.result.degree = m;
  out.result.exact  = exact;
  std::vector<std::size_t>                   digit(m);
  std::vector<double>                        tuple(m);
  std::mt19937_64                            rng(options_.seed);
  std::uniform_int_distribution<std::size_t> pick(0, n - 1);
  double                                     sum = 0;
  for (std::size_t t = 0; t < terms; ++t) {
    if (!exact)
      for (auto& d : digit)
        d = pick(rng);
    for (std::size_t k = 0; k < m; ++k)
      tuple[k] = sample[digit[k]];
    double value = kernel(tuple);
    if (!std::isfinite(value))
      throw std::invalid_argument("VStatistic nonfinite kernel");
    sum += value;
    if (!std::isfinite(sum))
      throw std::overflow_error("VStatistic sum");
    if (out.terms.size() < max_records)
      out.terms.push_back({digit, tuple, value});
    ++out.result.num_terms;
    if (exact)
      for (std::size_t k = m; k-- > 0;) {
        if (++digit[k] < n)
          break;
        digit[k] = 0;
      }
  }
  out.result.estimate = sum / terms;
  return out;
}
/// @brief Evaluate the widened finite variance-pair difference without a report-specific replay.
double empirical_variance_kernel(const std::vector<double>& tuple) {
  if (tuple.size() != 2 || !std::isfinite(tuple[0]) || !std::isfinite(tuple[1]))
    throw std::invalid_argument("variance kernel pair");
  long double d     = static_cast<long double>(tuple[0]) - tuple[1];
  double      value = static_cast<double>(d * d / 2);
  if (!std::isfinite(value))
    throw std::overflow_error("variance kernel");
  return value;
}
/// @brief Generate the original200-point normal(0,2) fixture with the supplied local seed.
std::vector<double> v_statistic_example_sample(std::size_t n, std::uint64_t seed) {
  if (n > 1000000)
    throw std::length_error("normal sample cap");
  std::mt19937_64                  rng(seed);
  std::normal_distribution<double> draw(0, 2);
  std::vector<double>              out(n);
  for (auto& x : out)
    x = draw(rng);
  return out;
}
} // namespace datamunge::stats
