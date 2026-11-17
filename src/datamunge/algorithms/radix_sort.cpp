/// @file radix_sort.cpp
/// @brief Shared stable signed radix producer with actual prefix and identity provenance.
#include <bit>
#include <datamunge/algorithms/radix_sort.hpp>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
/// @brief Canonical signed sorting implementation.
namespace datamunge::algorithms {
/// @brief Private byte-LSD producer.
namespace {
/// @brief Sort owned keys with optional actual pass capture.
/// @param data Owned signed input.
/// @param o Allocation/history policies.
/// @param capture Retain actual passes.
/// @return Owned stable ordering and selected work.
RadixSortResult solve(std::vector<std::int64_t> data, const RadixSortOptions& o, bool capture) {
  const auto n = data.size();
  if ((o.max_keys && n > o.max_keys) || n > std::numeric_limits<std::size_t>::max() / 16)
    throw std::length_error("radix_sort: input/work capacity");
  RadixSortResult r;
  r.origins.resize(n);
  std::iota(r.origins.begin(), r.origins.end(), 0);
  if (n < 2) {
    r.values = std::move(data);
    return r;
  }
  constexpr std::uint64_t    bias = std::uint64_t{1} << 63;
  std::vector<std::uint64_t> keys(n), buffer(n);
  std::vector<std::size_t>   origin_buffer(n);
  for (std::size_t i = 0; i < n; ++i)
    keys[i] = static_cast<std::uint64_t>(data[i]) ^ bias;
  std::size_t history = 0;
  for (unsigned shift = 0; shift < 64; shift += 8) {
    if (capture) {
      if (n > (std::numeric_limits<std::size_t>::max() - 514) / 2)
        throw std::length_error("radix_sort: history capacity");
      const auto cost = 2 * n + 514;
      if (cost > std::numeric_limits<std::size_t>::max() - history ||
          (o.max_trace_entries &&
           (history > o.max_trace_entries || cost > o.max_trace_entries - history)))
        throw std::length_error("radix_sort: history policy");
      history += cost;
    }
    std::array<std::size_t, 257> count{};
    std::array<std::size_t, 256> histogram{};
    for (auto key : keys) {
      ++count[((key >> shift) & 255) + 1];
      ++r.digit_reads;
    }
    if (capture)
      for (std::size_t i = 0; i < 256; ++i)
        histogram[i] = count[i + 1];
    for (std::size_t i = 0; i < 256; ++i) {
      count[i + 1] += count[i];
      ++r.prefix_steps;
    }
    const auto starts = count;
    for (std::size_t i = 0; i < n; ++i) {
      const auto bucket = (keys[i] >> shift) & 255;
      ++r.digit_reads;
      const auto j     = count[bucket]++;
      buffer[j]        = keys[i];
      origin_buffer[j] = r.origins[i];
      ++r.scatters;
    }
    keys.swap(buffer);
    r.origins.swap(origin_buffer);
    if (capture) {
      RadixSortPass pass;
      pass.shift   = shift;
      pass.counts  = histogram;
      pass.offsets = starts;
      pass.origins = r.origins;
      pass.values.reserve(n);
      for (auto key : keys)
        pass.values.push_back(std::bit_cast<std::int64_t>(key ^ bias));
      r.passes.push_back(std::move(pass));
    }
  }
  for (std::size_t i = 0; i < n; ++i)
    data[i] = std::bit_cast<std::int64_t>(keys[i] ^ bias);
  r.values = std::move(data);
  return r;
}
} // namespace
RadixSortResult radix_sort_run(std::vector<std::int64_t> a, const RadixSortOptions& o) {
  return solve(std::move(a), o, false);
}
RadixSortResult radix_sort_trace(std::vector<std::int64_t> a, const RadixSortOptions& o) {
  return solve(std::move(a), o, true);
}
void radix_sort(std::vector<std::int64_t>& a) {
  auto r = radix_sort_run(a, RadixSortOptions{0, 0});
  a      = std::move(r.values);
}
} // namespace datamunge::algorithms
