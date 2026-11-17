/// @file sundaram_sieve.cpp
/// @brief Actual inclusive overflow-safe odd-composite strikes.
#include <datamunge/algorithms/sundaram_sieve.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private actual marker producer.
namespace {
/// @brief Execute validated odd-index geometry and count every actual pair strike.
SundaramRun execute(std::uint64_t limit, const SundaramOptions& o, bool capture) {
  SundaramRun r;
  if (limit < 2)
    return r;
  r.primes.push_back(2);
  if (limit == 2)
    return r;
  auto k = (limit - 1) / 2;
  if (k >= std::numeric_limits<std::size_t>::max() || (o.max_odd_cells && k + 1 > o.max_odd_cells))
    throw std::length_error("Sundaram marker geometry");
  r.marked.assign(static_cast<std::size_t>(k + 1), false);
  for (std::uint64_t i = 1; i <= k; ++i) {
    auto stride = 2 * i + 1;
    if (i > (k - i) / stride)
      break;
    auto index = i + stride * i;
    for (std::uint64_t j = i;; ++j) {
      if (r.strikes == std::numeric_limits<std::size_t>::max() ||
          (o.max_strikes && r.strikes >= o.max_strikes))
        throw std::length_error("Sundaram strikes");
      ++r.strikes;
      bool fresh = !r.marked[index];
      if (fresh)
        ++r.first_marks;
      r.marked[index] = true;
      if (capture) {
        if (o.max_records && r.steps.size() >= o.max_records)
          throw std::length_error("Sundaram history");
        r.steps.push_back({i, j, index, fresh});
      }
      if (stride > k - index)
        break;
      index += stride;
    }
  }
  for (std::uint64_t i = 1; i <= k; ++i) {
    ++r.output_tests;
    if (!r.marked[i])
      r.primes.push_back(2 * i + 1);
  }
  return r;
}
} // namespace
SundaramRun sundaram_run(std::uint64_t limit, const SundaramOptions& options) {
  return execute(limit, options, false);
}
SundaramRun sundaram_trace(std::uint64_t limit, const SundaramOptions& options) {
  return execute(limit, options, true);
}
std::vector<std::uint64_t> sundaram_primes(std::uint64_t limit) {
  return sundaram_run(limit).primes;
}
} // namespace datamunge::algorithms
