/// @file permutation_test.cpp
/// @brief Canonical producing implementation and actual-site diagnostics; public contracts are in the paired header.
#include <algorithm>
#include <cmath>
#include <datamunge/stats/permutation_test.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace datamunge::stats {

/// @brief File-local validated kernels shared by ordinary and observed entry points.
namespace {

/// @brief Original summation-order mean, with a finite wider fallback on overflow.
double mean(const std::vector<double>& v) {
  if (v.empty())
    return 0;
  const auto ordinary = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
  if (std::isfinite(ordinary))
    return ordinary;
  long double sum = 0;
  for (double x : v)
    sum += static_cast<long double>(x) / v.size();
  auto out = static_cast<double>(sum);
  if (!std::isfinite(out))
    throw std::overflow_error("permutation mean overflow");
  return out;
}

/// @brief Shared producing shuffle/callback kernel with optional assignment observation.
PermutationTrace kernel(
    const std::vector<double>& x,
    const std::vector<double>& y,
    const TwoSampleStatistic&  statistic,
    PermutationTestOptions     options,
    bool                       capture,
    std::size_t                cap) {
  const std::size_t nx = x.size(), ny = y.size();
  if (nx == 0 || ny == 0)
    throw std::invalid_argument("permutation_test_two_sample: both samples must be non-empty");
  if (options.num_permutations == 0)
    throw std::invalid_argument("permutation_test_two_sample: num_permutations must be >= 1");

  if (options.num_permutations == std::numeric_limits<std::size_t>::max() ||
      ny > std::numeric_limits<std::size_t>::max() - nx)
    throw std::length_error("permutation count range");
  if (options.alternative != Alternative::TwoSided && options.alternative != Alternative::Greater &&
      options.alternative != Alternative::Less)
    throw std::invalid_argument("permutation alternative");
  for (const auto* sample : {&x, &y})
    for (double v : *sample)
      if (!std::isfinite(v))
        throw std::invalid_argument("permutation finite samples required");
  if (capture && (nx == std::numeric_limits<std::size_t>::max() ||
                  options.num_permutations > std::numeric_limits<std::size_t>::max() / (nx + 1) ||
                  (cap && options.num_permutations * (nx + 1) > cap)))
    throw std::length_error("permutation history policy");
  const TwoSampleStatistic stat =
      statistic
          ? statistic
          : TwoSampleStatistic([](const std::vector<double>& a, const std::vector<double>& b) {
              return mean(a) - mean(b);
            });

  const double observed = stat(x, y);
  if (!std::isfinite(observed))
    throw std::invalid_argument("permutation nonfinite observed statistic");

  std::vector<double> pool;
  pool.reserve(nx + ny);
  pool.insert(pool.end(), x.begin(), x.end());
  pool.insert(pool.end(), y.begin(), y.end());

  std::mt19937_64          rng(options.seed);
  PermutationTrace         trace;
  auto&                    r = trace.result;
  std::vector<std::size_t> ids(pool.size());
  std::iota(ids.begin(), ids.end(), 0);
  r.observed         = observed;
  r.num_permutations = options.num_permutations;
  r.null_distribution.resize(options.num_permutations);

  std::vector<double> gx(nx), gy(ny);
  std::size_t         extreme = 0;
  for (std::size_t b = 0; b < options.num_permutations; ++b) {
    std::shuffle(ids.begin(), ids.end(), rng); // relabel: first nx -> group x, rest -> group y
    for (std::size_t i = 0; i < nx; ++i)
      gx[i] = pool[ids[i]];
    for (std::size_t i = 0; i < ny; ++i)
      gy[i] = pool[ids[nx + i]];
    const double t = stat(gx, gy);
    if (!std::isfinite(t))
      throw std::invalid_argument("permutation nonfinite sampled statistic");
    r.null_distribution[b] = t;
    const auto before      = extreme;
    switch (options.alternative) {
      case Alternative::Greater:
        extreme += (t >= observed);
        break;
      case Alternative::Less:
        extreme += (t <= observed);
        break;
      case Alternative::TwoSided:
      default:
        extreme += (std::fabs(t) >= std::fabs(observed));
        break;
    }
    if (capture)
      trace.draws.push_back(
          {std::vector<std::size_t>(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(nx)),
           t,
           extreme != before});
  }

  // Monte Carlo p-value with the +1 correction (never returns exactly 0).
  r.p_value = static_cast<double>(1 + extreme) / static_cast<double>(options.num_permutations + 1);
  r.null_mean = mean(r.null_distribution);
  double ss   = 0.0;
  for (const double v : r.null_distribution)
    ss += (v - r.null_mean) * (v - r.null_mean);
  r.null_sd = options.num_permutations > 1
                  ? std::sqrt(ss / static_cast<double>(options.num_permutations - 1))
                  : 0.0;
  if (!std::isfinite(r.null_sd)) {
    long double ss_wide = 0;
    for (double v : r.null_distribution) {
      const long double d = static_cast<long double>(v) - r.null_mean;
      ss_wide += d * d;
    }
    r.null_sd = options.num_permutations > 1
                    ? static_cast<double>(std::sqrt(ss_wide / (options.num_permutations - 1)))
                    : 0;
    if (!std::isfinite(r.null_sd))
      throw std::overflow_error("permutation null SD overflow");
  }
  return trace;
}

} // namespace
PermutationTestResult permutation_test_two_sample(
    const std::vector<double>& x,
    const std::vector<double>& y,
    const TwoSampleStatistic&  s,
    PermutationTestOptions     o) {
  return kernel(x, y, s, o, false, 0).result;
}
PermutationTrace permutation_test_trace(
    const std::vector<double>& x,
    const std::vector<double>& y,
    const TwoSampleStatistic&  s,
    PermutationTestOptions     o,
    std::size_t                cap) {
  return kernel(x, y, s, o, true, cap);
}
} // namespace datamunge::stats
