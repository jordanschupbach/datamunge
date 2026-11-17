/// @file roulette_wheel.cpp
/// @brief Actual checked raw stream and strict-upper-CDF native selection kernel.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/roulette_wheel.hpp>
#include <limits>
#include <stdexcept>
/// @brief Canonical fitness-proportionate selection implementation.
namespace datamunge::algorithms {
/// @brief Private independent stream producer.
namespace {
/// @brief Validate the model and execute actual uniform/bucket decisions once.
/// @param fitness Complete borrowed weights.
/// @param n Requested draw count.
/// @param seed Raw seed.
/// @param o Input/draw/history policies.
/// @param capture Retain real raw decisions.
/// @return Owned actual stream and model.
RouletteRun solve(
    const std::vector<double>& fitness,
    std::size_t                n,
    std::uint64_t              seed,
    const RouletteOptions&     o,
    bool                       capture) {
  if (fitness.empty())
    throw std::invalid_argument("roulette: empty model");
  if ((o.max_population && fitness.size() > o.max_population) || (o.max_draws && n > o.max_draws) ||
      (capture && o.max_trace_entries && n > o.max_trace_entries))
    throw std::length_error("roulette: complete policy");
  double maximum = 0, total = 0;
  bool   scaled = false;
  for (double f : fitness) {
    if (!std::isfinite(f) || f < 0)
      throw std::invalid_argument("roulette: invalid fitness");
    maximum = std::max(maximum, f);
    total += f;
    if (!std::isfinite(total))
      scaled = true;
  }
  if (maximum == 0)
    throw std::invalid_argument("roulette: zero model");
  RouletteRun out;
  out.scaled = scaled;
  out.cumulative.reserve(fitness.size());
  out.counts.assign(fitness.size(), 0);
  total = 0;
  for (double f : fitness) {
    total += scaled ? f / maximum : f;
    out.cumulative.push_back(total);
  }
  out.total = total;
  out.picks.reserve(n);
  if (capture)
    out.draws.reserve(n);
  std::uint64_t state = seed ? seed : 0x9E3779B97F4A7C15ULL;
  for (std::size_t draw = 0; draw < n; ++draw) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    const auto   raw     = state;
    const double uniform = (raw >> 11) * (1.0 / 9007199254740992.0);
    double       target  = uniform * total;
    if (!(target < total))
      target = std::nextafter(total, 0.);
    std::size_t lo = 0, hi = fitness.size() - 1, comparisons = 0;
    while (lo < hi) {
      if (out.comparisons == std::numeric_limits<std::size_t>::max())
        throw std::length_error("roulette: work capacity");
      ++comparisons;
      ++out.comparisons;
      const auto mid = lo + (hi - lo) / 2;
      if (out.cumulative[mid] <= target)
        lo = mid + 1;
      else
        hi = mid;
    }
    out.picks.push_back(lo);
    ++out.counts[lo];
    if (capture)
      out.draws.push_back(
          {raw,
           uniform,
           target,
           lo,
           lo ? out.cumulative[lo - 1] : 0,
           out.cumulative[lo],
           comparisons});
  }
  return out;
}
} // namespace
RouletteRun roulette_wheel_run(
    const std::vector<double>& f,
    std::size_t                n,
    std::uint64_t              s,
    const RouletteOptions&     o) {
  return solve(f, n, s, o, false);
}
RouletteRun roulette_wheel_trace(
    const std::vector<double>& f,
    std::size_t                n,
    std::uint64_t              s,
    const RouletteOptions&     o) {
  return solve(f, n, s, o, true);
}
std::vector<std::size_t>
roulette_wheel_sample(const std::vector<double>& f, int n, std::uint64_t s) {
  if (n < 0)
    throw std::invalid_argument("roulette: negative count");
  return solve(f, static_cast<std::size_t>(n), s, {}, false).picks;
}
} // namespace datamunge::algorithms
