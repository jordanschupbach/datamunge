/// @file sfc64_analysis.cpp
/// @brief Single actual raw stream and bit/byte counter producer.
#include <datamunge/random/sfc64_analysis.hpp>
#include <stdexcept>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::random {
/// @brief Private invocation-local checked helpers.
namespace {
/// @brief Generate and summarize actual local engine words once.
/// @param seed Historical seed.
/// @param count Complete sample count.
/// @param o Output/capture policies.
/// @param capture Retain actual raw transitions.
/// @return Owned actual sample and counts.
Sfc64Run sample(std::uint64_t seed, std::size_t count, const Sfc64Options& o, bool capture) {
  if ((o.max_draws && count > o.max_draws) ||
      (capture && o.max_trace_entries && count > o.max_trace_entries) ||
      count > std::vector<std::uint64_t>().max_size() ||
      (capture && count > std::vector<Sfc64Step>().max_size()))
    throw std::length_error("SFC64: complete sample/capture policy");
  Sfc64    generator(seed);
  Sfc64Run out;
  out.initial = generator.state();
  out.outputs.reserve(count);
  if (capture)
    out.steps.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    std::uint64_t value;
    if (capture) {
      auto step = generator.next_trace();
      value     = step.value;
      out.steps.push_back(step);
    } else
      value = generator.next_u64();
    out.outputs.push_back(value);
    ++out.low_bytes[value & 255];
    for (unsigned bit = 0; bit < 64; ++bit)
      out.ones[bit] += (value >> bit) & 1;
  }
  out.final = generator.state();
  return out;
}
} // namespace
Sfc64Run sfc64_run(std::uint64_t seed, std::size_t n, const Sfc64Options& o) {
  return sample(seed, n, o, false);
}
Sfc64Run sfc64_trace(std::uint64_t seed, std::size_t n, const Sfc64Options& o) {
  return sample(seed, n, o, true);
}
} // namespace datamunge::random
