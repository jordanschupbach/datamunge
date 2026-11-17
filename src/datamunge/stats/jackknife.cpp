/// @file jackknife.cpp
/// @brief Checked centred summaries and actual complete/sampled deletion executions.
#include <algorithm>
#include <cmath>
#include <datamunge/stats/jackknife.hpp>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
namespace datamunge::stats {
/// @brief Private complete admission, finite summaries and deterministic deletion kernels.
namespace {
/// @brief Reject nonfinite working arithmetic.
/// @param x Working scalar.
/// @return Unchanged finite scalar.
/// @throws std::overflow_error On nonfinite arithmetic.
long double finite(long double x) {
  if (!std::isfinite(x))
    throw std::overflow_error("jackknife summary arithmetic");
  return x;
}
/// @brief Narrow a finite summary only within double range.
/// @param x Wide value, permitting ordinary underflow to zero.
/// @return Finite double.
/// @throws std::overflow_error On nonfinite/out-of-range value.
double narrow(long double x) {
  finite(x);
  if (std::abs(x) > std::numeric_limits<double>::max())
    throw std::overflow_error("jackknife summary range");
  return static_cast<double>(x);
}
/// @brief Validate complete borrowed input and callback before any callback/copy.
/// @param x Borrowed observations.
/// @param statistic Borrowed callable.
/// @param limits Complete input count policy.
/// @throws std::invalid_argument On n<2, empty callable or nonfinite sample.
/// @throws std::length_error On enabled sample count cap.
void validate(
    const std::vector<double>& x,
    const SampleStatistic&     statistic,
    JackknifeLimits            limits) {
  if (limits.max_elements && x.size() > limits.max_elements)
    throw std::length_error("jackknife complete input cap");
  if (x.size() < 2 || !statistic)
    throw std::invalid_argument("jackknife sample/callback");
  for (auto v : x)
    if (!std::isfinite(v))
      throw std::invalid_argument("jackknife finite observations required");
}
/// @brief Execute one actual statistic and require a finite value.
/// @param x Borrowed original/kept sample during callback only.
/// @param statistic Borrowed callable.
/// @return Actual finite callback result.
/// @throws std::domain_error On nonfinite return; callback exceptions propagate.
double evaluate(const std::vector<double>& x, const SampleStatistic& statistic) {
  const double value = statistic(x);
  if (!std::isfinite(value))
    throw std::domain_error("jackknife finite statistic required");
  return value;
}
/// @brief Admit a complete scalar-entry product without wrapping.
/// @param count Number of retained vectors/replicates.
/// @param width Retained entries per record/vector unit.
/// @param limits Complete optional retained-entry cap.
/// @throws std::length_error On count multiplication overflow or enabled cap.
void admit(std::size_t count, std::size_t width, JackknifeLimits limits) {
  if (width && count > std::numeric_limits<std::size_t>::max() / width)
    throw std::length_error("jackknife retained count overflow");
  if (limits.max_retained_values && count * width > limits.max_retained_values)
    throw std::length_error("jackknife complete retained cap");
}
/// @brief Compute exact C(n,d) only while it remains within a size_t subset ceiling.
/// The monotone recurrence through min(d,n-d) permits early rejection. A widened
/// product avoids both intermediate overflow and floating mode misclassification.
/// @param n Complete sample count.
/// @param d Valid deletion count.
/// @param ceiling Positive exact-enumeration budget.
/// @return Exact count when admitted, otherwise nullopt for Monte Carlo mode.
std::optional<std::size_t> combinations(std::size_t n, std::size_t d, std::size_t ceiling) {
  static_assert(std::numeric_limits<std::size_t>::digits <= 64);
  d                   = std::min(d, n - d);
  unsigned __int128 c = 1;
  for (std::size_t i = 0; i < d; ++i) {
    c = c * (n - i) / (i + 1);
    if (c > ceiling)
      return std::nullopt;
  }
  return static_cast<std::size_t>(c);
}
/// @brief Shared delete-one callback execution and centred summary.
/// @param x Borrowed complete input.
/// @param statistic Borrowed statistic.
/// @param limits Complete input/output admission policy.
/// @return Actual leave-one-out/pseudo-values and finite summary.
/// @throws std::invalid_argument On input violations before callbacks.
/// @throws std::domain_error On nonfinite callbacks; callback exceptions propagate.
/// @throws std::length_error On cap/output count overflow.
/// @throws std::overflow_error On unrepresentable arithmetic/output.
/// @throws std::bad_alloc On allocation failure.
JackknifeResult
delete_one(const std::vector<double>& x, const SampleStatistic& statistic, JackknifeLimits limits) {
  validate(x, statistic, limits);
  const auto n = x.size();
  admit(n, 2, limits);
  if (n > std::vector<double>().max_size())
    throw std::length_error("jackknife output size");
  JackknifeResult out;
  out.leave_one_out.resize(n);
  out.pseudo_values.resize(n);
  std::vector<double> kept(n - 1);
  out.estimate           = evaluate(x, statistic);
  long double mean_delta = 0;
  for (std::size_t i = 0; i < n; ++i) {
    std::size_t k = 0;
    for (std::size_t j = 0; j < n; ++j)
      if (j != i)
        kept[k++] = x[j];
    const double value   = evaluate(kept, statistic);
    out.leave_one_out[i] = value;
    const auto delta     = static_cast<long double>(value) - out.estimate;
    mean_delta           = finite(mean_delta + (delta - mean_delta) / (i + 1));
  }
  out.bias           = narrow((n - 1) * mean_delta);
  out.bias_corrected = narrow(static_cast<long double>(out.estimate) - (n - 1) * mean_delta);
  long double spread = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const auto delta     = static_cast<long double>(out.leave_one_out[i]) - out.estimate;
    spread               = finite(std::hypot(spread, delta - mean_delta));
    out.pseudo_values[i] = narrow(static_cast<long double>(out.estimate) - (n - 1) * delta);
  }
  out.standard_error = narrow(spread * std::sqrt(static_cast<long double>(n - 1) / n));
  return out;
}
/// @brief Shared complete/sampled delete-d execution with optional actual records.
/// @param x Borrowed complete sample.
/// @param statistic Borrowed callback.
/// @param options Deletion count, subset ceiling/draw count and seed.
/// @param limits Complete input/retained-entry policies.
/// @param capture Whether actual sorted deletion sets/statistics are retained.
/// @return Actual summary and optional owned same-execution history.
/// @throws std::invalid_argument On input/callback/options before callbacks.
/// @throws std::domain_error On nonfinite callback returns; callback exceptions propagate.
/// @throws std::length_error On caps/count/container overflow.
/// @throws std::overflow_error On unrepresentable arithmetic/output.
/// @throws std::bad_alloc On allocation failure.
DeleteDJackknifeTrace delete_d(
    const std::vector<double>& x,
    const SampleStatistic&     statistic,
    DeleteDJackknifeOptions    options,
    JackknifeLimits            limits,
    bool                       capture) {
  validate(x, statistic, limits);
  const auto n = x.size(), d = options.d;
  if (d == 0 || d >= n || options.max_subsets == 0)
    throw std::invalid_argument("jackknife delete-d controls");
  const auto exact = combinations(n, d, options.max_subsets);
  const auto count = exact.value_or(options.max_subsets);
  if (count == std::numeric_limits<std::size_t>::max())
    throw std::length_error("jackknife callback count");
  DeleteDJackknifeTrace out;
  out.result.d        = d;
  out.result.complete = exact.has_value();
  if (capture) {
    admit(count, d + 1, limits);
    if (count > out.replicates.max_size() || count > out.deletion_sets.max_size())
      throw std::length_error("jackknife trace containers");
    out.replicates.reserve(count);
    out.deletion_sets.reserve(count);
  }
  std::vector<std::size_t> deletion(d), indices;
  if (exact)
    std::iota(deletion.begin(), deletion.end(), 0);
  else {
    indices.resize(n);
    std::iota(indices.begin(), indices.end(), 0);
  }
  std::vector<double> kept;
  kept.reserve(n - d);
  out.result.estimate = evaluate(x, statistic);
  out.evaluations     = 1;
  std::mt19937_64 rng(options.seed);
  long double     mean = 0, m2 = 0;
  for (std::size_t draw = 0; draw < count; ++draw) {
    if (!exact) {
      for (std::size_t i = 0; i < d; ++i) {
        std::uniform_int_distribution<std::size_t> pick(i, n - 1);
        std::swap(indices[i], indices[pick(rng)]);
      }
      std::copy_n(indices.begin(), d, deletion.begin());
      std::sort(deletion.begin(), deletion.end());
    }
    kept.clear();
    std::size_t j = 0;
    for (std::size_t i = 0; i < n; ++i) {
      if (j < d && deletion[j] == i) {
        ++j;
        continue;
      }
      kept.push_back(x[i]);
    }
    const double value = evaluate(kept, statistic);
    ++out.evaluations;
    const auto delta = static_cast<long double>(value) - mean;
    mean             = finite(mean + delta / (draw + 1));
    m2               = finite(m2 + delta * (static_cast<long double>(value) - mean));
    if (capture) {
      out.deletion_sets.push_back(deletion);
      out.replicates.push_back(value);
    }
    if (exact && draw + 1 < count) {
      std::size_t i = d;
      while (i > 0 && deletion[i - 1] == n - d + i - 1)
        --i;
      if (i == 0)
        throw std::logic_error("jackknife exact enumeration invariant");
      ++deletion[i - 1];
      for (std::size_t k = i; k < d; ++k)
        deletion[k] = deletion[k - 1] + 1;
    }
  }
  out.result.num_subsets            = count;
  out.mean_replicate                = narrow(mean);
  out.replicate_population_variance = finite(std::max(0.L, m2 / count));
  out.result.standard_error         = narrow(
      std::sqrt(finite((static_cast<long double>(n - d) / d) * out.replicate_population_variance)));
  return out;
}
} // namespace
JackknifeResult jackknife(const std::vector<double>& sample, const SampleStatistic& statistic) {
  return delete_one(sample, statistic, {0, 0});
}
JackknifeResult jackknife_checked(
    const std::vector<double>& sample,
    const SampleStatistic&     statistic,
    JackknifeLimits            limits) {
  return delete_one(sample, statistic, limits);
}
DeleteDJackknifeResult jackknife_delete_d(
    const std::vector<double>& sample,
    const SampleStatistic&     statistic,
    DeleteDJackknifeOptions    options) {
  return delete_d(sample, statistic, options, {0, 0}, false).result;
}
DeleteDJackknifeTrace jackknife_delete_d_trace(
    const std::vector<double>& sample,
    const SampleStatistic&     statistic,
    DeleteDJackknifeOptions    options,
    JackknifeLimits            limits) {
  return delete_d(sample, statistic, options, limits, true);
}
} // namespace datamunge::stats
