/// @file lagged_fibonacci_analysis.cpp
/// @brief Complete preallocated actual consuming ring capture and copied-engine parity orbit.
#include <algorithm>
#include <datamunge/random/lagged_fibonacci_analysis.hpp>
#include <limits>
#include <stdexcept>
#include <unordered_map>
namespace datamunge::random {
LaggedFibonacciTrace
lagged_fibonacci_trace(LaggedFibonacci64& generator, std::size_t count, std::size_t max_entries) {
  const auto k = generator.long_lag();
  if (k > std::numeric_limits<std::size_t>::max() / 2 ||
      count > std::numeric_limits<std::size_t>::max() - 2 * k)
    throw std::length_error("LFG capture count overflow");
  const auto capture = 2 * k + count;
  if (max_entries && capture > max_entries)
    throw std::length_error("LFG capture quota");
  if (count > std::vector<LaggedFibonacciStep>().max_size())
    throw std::length_error("LFG record geometry");
  LaggedFibonacciTrace out;
  out.short_lag     = generator.short_lag();
  out.long_lag      = k;
  out.initial_index = generator.ring_index();
  out.initial_state = generator.state_words();
  out.final_state.resize(k);
  out.steps.reserve(count);
  out.captured_entries = capture;
  for (std::size_t i = 0; i < count; ++i)
    out.steps.push_back(generator.next_step());
  std::copy(
      generator.state_words().begin(), generator.state_words().end(), out.final_state.begin());
  out.final_index = generator.ring_index();
  return out;
}
LaggedFibonacciParityOrbit lagged_fibonacci_parity_orbit(
    const LaggedFibonacci64& generator,
    std::size_t              max_states) {
  if (generator.long_lag() > 63)
    throw std::invalid_argument("LFG parity masks require k<=63");
  auto                       g = generator;
  LaggedFibonacciParityOrbit out;
  out.short_lag = g.short_lag();
  out.long_lag  = g.long_lag();
  std::unordered_map<std::uint64_t, std::size_t> seen;
  for (;;) {
    std::uint64_t mask = 0;
    std::size_t   at   = g.ring_index();
    for (std::size_t i = 0; i < g.long_lag(); ++i) {
      mask |= (g.state_words()[at] & 1ULL) << i;
      if (++at == g.long_lag())
        at = 0;
    }
    if (const auto it = seen.find(mask); it != seen.end()) {
      out.complete      = true;
      out.repeat_offset = it->second;
      out.period        = out.masks.size() - it->second;
      return out;
    }
    if (max_states && out.masks.size() >= max_states)
      return out;
    seen.emplace(mask, out.masks.size());
    out.masks.push_back(mask);
    g.next_u64();
  }
}
} // namespace datamunge::random
