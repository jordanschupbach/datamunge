#include <datamunge/algorithms/counting_sort.hpp>
/// @file counting_sort.cpp
/// @brief Shared checked histogram/prefix/reverse-placement kernel for keys and identities.
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <utility>

namespace datamunge::algorithms {
/// @brief Private counting-sort range checks and single executing placement kernel.
namespace {
/// @brief Return ordered input extrema, or zero bounds for empty input.
/// @param data Read-only keys.
/// @return Inclusive minimum/maximum, or {0,0} when empty.
std::pair<std::int64_t,std::int64_t> counting_bounds(const std::vector<std::int64_t>& data) {
  if(data.empty()) return {0,0};
  const auto bounds=std::minmax_element(data.begin(),data.end());
  return {*bounds.first,*bounds.second};
}

/// @brief Run one validated histogram, prefix scan and reverse placement.
/// Validates every key before any size-dependent allocation or budget check.
/// Auxiliary count/cursor arrays cost O(k); diagnostics add owned boundaries
/// and identities, tracing adds n actual events. No state is shared or retained.
/// @param data Read-only input keys.
/// @param lo Inclusive lower bound.
/// @param hi Inclusive upper bound.
/// @param max_buckets Optional complete bucket cap; zero disables it.
/// @param max_placements Optional complete event cap; zero disables it.
/// @param diagnostic Optional owned metadata output; null skips identity/boundary allocation.
/// @param events Optional owned actual-event output; null skips tracing.
/// @param automatic Whether to account for automatic range discovery.
/// @return Owned sorted keys; empty input skips span checks and allocation.
/// @throws std::invalid_argument If bounds invert or a key is out of range.
/// @throws std::length_error If a budget, the hard ceiling or a container limit is exceeded.
/// @throws std::bad_alloc If temporary or result storage cannot be allocated.
std::vector<std::int64_t> counting_kernel(const std::vector<std::int64_t>& data,
    std::int64_t lo,std::int64_t hi,std::size_t max_buckets,std::size_t max_placements,
    CountingSortAnalysis* diagnostic,std::vector<CountingSortPlacement>* events,bool automatic) {
  if(hi<lo) throw std::invalid_argument("counting_sort: inverted range");
  for(const auto value:data)
    if(value<lo || value>hi) throw std::invalid_argument("counting_sort: key outside range");
  if(diagnostic) { diagnostic->min_value=lo;diagnostic->max_value=hi; }
  if(data.empty()) return {};
  const auto span=static_cast<std::uint64_t>(hi)-static_cast<std::uint64_t>(lo);
  if(span>=static_cast<std::uint64_t>(kCountingSortMaxRange))
    throw std::length_error("counting_sort: hard bucket ceiling exceeded");
  if(span>=std::numeric_limits<std::size_t>::max())
    throw std::length_error("counting_sort: bucket count is not representable");
  const auto k=static_cast<std::size_t>(span)+1;
  if(max_buckets && k>max_buckets) throw std::length_error("counting_sort: bucket budget exceeded");
  if(events && max_placements && data.size()>max_placements)
    throw std::length_error("counting_sort: complete placement budget exceeded");
  /// @brief Map a validated key to its exact unsigned offset without signed overflow.
  /// @param value Key already known to lie in [lo,hi].
  /// @return Bucket index below k.
  const auto bucket=[lo](std::int64_t value) {
    return static_cast<std::size_t>(static_cast<std::uint64_t>(value)-static_cast<std::uint64_t>(lo));
  };
  std::vector<std::size_t> counts(k,0),cursor(k,0);
  if(diagnostic) diagnostic->original_indices.resize(data.size());
  if(events) events->reserve(data.size());
  for(const auto value:data) ++counts[bucket(value)];
  std::size_t running=0;
  for(std::size_t b=0;b<k;++b) { running+=counts[b];cursor[b]=running; }
  if(diagnostic) {
    diagnostic->counts=counts;
    diagnostic->ends=cursor;
    diagnostic->starts.resize(k);
    for(std::size_t b=0;b<k;++b) diagnostic->starts[b]=cursor[b]-counts[b];
    diagnostic->work={automatic?data.size():0,data.size(),k,data.size()};
  }
  std::vector<std::int64_t> output(data.size());
  for(std::size_t i=data.size();i-- >0;) {
    const auto b=bucket(data[i]);
    const auto before=cursor[b];
    const auto j=--cursor[b];
    output[j]=data[i];
    if(diagnostic) diagnostic->original_indices[j]=i;
    if(events) events->push_back({i,data[i],b,before,j});
  }
  return output;
}
} // namespace

void counting_sort(std::vector<std::int64_t>& data) {
  const auto [lo,hi]=counting_bounds(data);
  data=counting_kernel(data,lo,hi,0,0,nullptr,nullptr,true);
}
void counting_sort(std::vector<std::int64_t>& data,std::int64_t min_value,std::int64_t max_value) {
  data=counting_kernel(data,min_value,max_value,0,0,nullptr,nullptr,false);
}
CountingSortAnalysis counting_sort_analyze(const std::vector<std::int64_t>& data,std::size_t max_buckets) {
  const auto [lo,hi]=counting_bounds(data);
  CountingSortAnalysis result;
  result.output=counting_kernel(data,lo,hi,max_buckets,0,&result,nullptr,true);
  return result;
}
CountingSortAnalysis counting_sort_analyze(const std::vector<std::int64_t>& data,
    std::int64_t min_value,std::int64_t max_value,std::size_t max_buckets) {
  CountingSortAnalysis result;
  result.output=counting_kernel(data,min_value,max_value,max_buckets,0,&result,nullptr,false);
  return result;
}
CountingSortTrace counting_sort_trace(const std::vector<std::int64_t>& data,
    std::int64_t min_value,std::int64_t max_value,std::size_t max_buckets,std::size_t max_placements) {
  CountingSortTrace result;
  result.analysis.output=counting_kernel(data,min_value,max_value,max_buckets,max_placements,
      &result.analysis,&result.placements,false);
  return result;
}
} // namespace datamunge::algorithms
