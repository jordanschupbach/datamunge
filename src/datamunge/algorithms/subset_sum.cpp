/// @file subset_sum.cpp
/// @brief Actual widened two-half enumeration and matching, without signed-overflow UB.
#include <algorithm>
#include <datamunge/algorithms/subset_sum.hpp>
#include <limits>
#include <stdexcept>
#include <utility>
namespace datamunge::algorithms {
/// @brief Private invocation-only exact half sums and checked counters.
namespace {
/// @brief Exact signed accumulator;62 signed64 inputs and one target difference fit.
using Wide = __int128;
/// @brief Render an exact private signed accumulator to owned decimal text.
std::string decimal(Wide value) {
  bool neg = value < 0;
  if (neg)
    value = -value;
  std::string s;
  do {
    s.push_back(char('0' + value % 10));
    value /= 10;
  } while (value);
  if (neg)
    s.push_back('-');
  std::reverse(s.begin(), s.end());
  return s;
}
/// @brief Increment one actual event count under representational capacity.
void increment(std::size_t& n) {
  if (n == std::numeric_limits<std::size_t>::max())
    throw std::length_error("subset event capacity");
  ++n;
}
/// @brief Enumerate and match the actual checked half masks.
SubsetSumRun execute(
    const std::vector<long long>& nums,
    long long                     target,
    const SubsetSumOptions&       o,
    bool                          capture) {
  if (nums.size() > 62)
    throw std::length_error("subset intrinsic62 keys");
  std::size_t   na = nums.size() / 2, nb = nums.size() - na;
  std::uint64_t nr = std::uint64_t{1} << nb, nl = std::uint64_t{1} << na;
  if (nr > std::numeric_limits<std::size_t>::max() ||
      (o.max_half_subsets && nr > o.max_half_subsets))
    throw std::length_error("subset complete half geometry");
  SubsetSumRun r;
  r.left_size         = na;
  std::size_t records = 0;
  /// @brief Bound an actual retained enumeration/probe record.
  auto retain = [&]() {
    if (records == std::numeric_limits<std::size_t>::max() ||
        (o.max_records && records >= o.max_records))
      throw std::length_error("subset history");
    ++records;
  };
  /// @brief Sum selected keys in the actual bit-scan order, counting every tested position.
  auto sum = [&](std::uint64_t mask, std::size_t start, std::size_t n) {
    Wide value = 0;
    for (std::size_t j = 0; j < n; ++j) {
      increment(r.bit_tests);
      if (mask & (std::uint64_t{1} << j)) {
        increment(r.selected_additions);
        value += Wide(nums[start + j]);
      }
    }
    return value;
  };
  std::vector<std::pair<Wide, std::uint64_t>> right;
  right.reserve(static_cast<std::size_t>(nr));
  for (std::uint64_t mask = 0; mask < nr; ++mask) {
    auto value = sum(mask, na, nb);
    right.emplace_back(value, mask);
    ++r.right_subsets;
    if (capture) {
      retain();
      r.right.push_back({mask, decimal(value)});
    }
  }
  /// @brief Count actual sorting comparisons while retaining historical sum-only tie semantics.
  auto less = [&](const auto& a, const auto& b) {
    increment(r.sort_comparisons);
    return a.first < b.first;
  };
  std::sort(right.begin(), right.end(), less);
  for (std::uint64_t mask = 0; mask < nl; ++mask) {
    Wide left = sum(mask, 0, na), needed = Wide(target) - left;
    ++r.left_probes;
    /// @brief Count an actual native lower-bound comparison.
    auto below = [&](const auto& x, Wide value) {
      increment(r.lookup_comparisons);
      return x.first < value;
    };
    auto it    = std::lower_bound(right.begin(), right.end(), needed, below);
    bool found = it != right.end() && it->first == needed;
    if (capture) {
      retain();
      r.probes.push_back({mask, decimal(left), decimal(needed), found, found ? it->second : 0});
    }
    if (found) {
      r.result.found = true;
      for (std::size_t j = 0; j < na; ++j)
        if (mask & (std::uint64_t{1} << j))
          r.result.indices.push_back(j);
      for (std::size_t j = 0; j < nb; ++j)
        if (it->second & (std::uint64_t{1} << j))
          r.result.indices.push_back(na + j);
      break;
    }
  }
  return r;
}
} // namespace
SubsetSumRun subset_sum_run(
    const std::vector<long long>& nums,
    long long                     target,
    const SubsetSumOptions&       options) {
  return execute(nums, target, options, false);
}
SubsetSumRun subset_sum_trace(
    const std::vector<long long>& nums,
    long long                     target,
    const SubsetSumOptions&       options) {
  return execute(nums, target, options, true);
}
SubsetSumResult subset_sum(const std::vector<long long>& nums, long long target) {
  return subset_sum_run(nums, target).result;
}
} // namespace datamunge::algorithms
