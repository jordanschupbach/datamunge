/// @file bucket_sort.cpp
/// @brief One checked scatter/insertion/gather kernel for integer sorting and actual traces.
#include <datamunge/algorithms/bucket_sort.hpp>

#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace datamunge::algorithms {
/// @brief Implementation-only exact scaling and checked trace accumulation.
namespace bucket_sort_detail {
static_assert(std::numeric_limits<int>::digits<=31,"bucket sort requires at most 32-bit int");
static_assert(sizeof(std::size_t)<=sizeof(std::uint64_t),"bucket count must fit uint64_t");

/// @brief Map a valid value using bounded products, without repeating public validation.
/// @param value Value inside the supplied range.
/// @param minimum Lower endpoint.
/// @param span Positive inclusive range length, at most 2^32.
/// @param count Positive bucket count.
/// @return Exact floor(offset*count/span), strictly less than count.
std::size_t index(int value,int minimum,std::uint64_t span,std::size_t count) {
  const auto offset=static_cast<std::uint64_t>(static_cast<std::int64_t>(value)-minimum);
  const auto k=static_cast<std::uint64_t>(count);
  return static_cast<std::size_t>(offset*(k/span)+(offset*(k%span))/span);
}

/// @brief Add a nonnegative trace count or throw before wrapping.
/// @param target Counter to update.
/// @param amount Increment, defaulting to one.
/// @throws std::overflow_error If the result does not fit size_t.
void add(std::size_t& target,std::size_t amount=1) {
  if (amount>std::numeric_limits<std::size_t>::max()-target)
    throw std::overflow_error("bucket sort trace counter overflow");
  target+=amount;
}

/// @brief Run historical early exits and actual stable insertion operations once.
/// @tparam Record Whether to own snapshots and accumulate checked counters.
/// @param values Owned input; becomes the result directly on shortcut.
/// @return Output and, for Record, actual scatter/insert records.
template<bool Record>
BucketSortTrace sort(std::vector<int> values) {
  BucketSortTrace result;
  if constexpr (Record) result.input=values;
  if (!values.empty()) {
    const auto bounds=std::minmax_element(values.begin(),values.end());
    result.minimum=*bounds.first;
    result.maximum=*bounds.second;
    result.span=static_cast<std::uint64_t>(static_cast<std::int64_t>(result.maximum)-result.minimum)+1;
  }
  if (values.size()<=1 || result.minimum==result.maximum) {
    result.output=std::move(values);
    return result;
  }
  result.shortcut=false;
  result.bucket_count=values.size();
  std::vector<std::vector<int>> buckets(result.bucket_count);
  if constexpr (Record) {
    result.bucket_of_input.reserve(values.size());
    result.buckets.resize(result.bucket_count);
  }
  for (const int value:values) {
    const auto destination=index(value,result.minimum,result.span,result.bucket_count);
    buckets[destination].push_back(value);
    if constexpr (Record) result.bucket_of_input.push_back(destination);
  }
  result.output.reserve(values.size());
  for (std::size_t b=0;b<buckets.size();++b) {
    auto& bucket=buckets[b];
    if constexpr (Record) {
      auto& record=result.buckets[b];
      record.input=bucket;
      const auto size=bucket.size();
      if (size) ++result.nonempty_buckets;
      result.max_occupancy=std::max(result.max_occupancy,size);
      if (size && size>std::numeric_limits<std::size_t>::max()/size)
        throw std::overflow_error("bucket sort occupancy square overflow");
      add(result.occupancy_square_sum,size*size);
    }
    for (std::size_t i=1;i<bucket.size();++i) {
      const int key=bucket[i];
      auto j=i;
      while (j>0) {
        if constexpr (Record) add(result.buckets[b].key_comparisons);
        if (!(bucket[j-1]>key)) break;
        bucket[j]=bucket[j-1];
        --j;
        if constexpr (Record) add(result.buckets[b].shifts);
      }
      bucket[j]=key;
      if constexpr (Record) add(result.buckets[b].placements);
    }
    result.output.insert(result.output.end(),bucket.begin(),bucket.end());
    if constexpr (Record) {
      auto& record=result.buckets[b];
      record.output=std::move(bucket);
      add(result.key_comparisons,record.key_comparisons);
      add(result.shifts,record.shifts);
      add(result.placements,record.placements);
    }
  }
  return result;
}
} // namespace bucket_sort_detail

std::size_t bucket_sort_index(int value,int minimum,int maximum,std::size_t bucket_count) {
  if (!bucket_count || minimum>maximum || value<minimum || value>maximum)
    throw std::invalid_argument("bucket sort requires a positive bucket count and an in-range value");
  const auto span=static_cast<std::uint64_t>(static_cast<std::int64_t>(maximum)-minimum)+1;
  return bucket_sort_detail::index(value,minimum,span,bucket_count);
}

std::vector<int> bucket_sort(std::vector<int> a) {
  return bucket_sort_detail::sort<false>(std::move(a)).output;
}

BucketSortTrace bucket_sort_trace(std::vector<int> a) {
  return bucket_sort_detail::sort<true>(std::move(a));
}

BucketSortWorkloads bucket_sort_workloads(std::size_t count,std::uint32_t seed) {
  BucketSortWorkloads result;
  result.uniform.resize(count);
  result.clustered.resize(count);
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> uniform(0,999);
  std::normal_distribution<double> normal(500.0,4.0);
  for (std::size_t i=0;i<count;++i) {
    result.uniform[i]=uniform(rng);
    // Clamp before conversion as well, so even an extreme draw cannot overflow int.
    result.clustered[i]=static_cast<int>(std::clamp(normal(rng),0.0,999.0));
  }
  if (count>=2) {result.clustered[0]=0;result.clustered[1]=999;}
  return result;
}

std::vector<int> bucket_sort_collision_input(std::size_t cluster_size,bool descending) {
  const auto limit=static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (cluster_size>limit-2 || cluster_size+2>limit/(cluster_size+1))
    throw std::length_error("bucket sort collision endpoint does not fit int");
  const auto endpoint=(cluster_size+2)*(cluster_size+1);
  std::vector<int> result;
  result.reserve(cluster_size+2);
  result.push_back(0);
  for (std::size_t i=0;i<cluster_size;++i)
    result.push_back(static_cast<int>(descending?cluster_size-i:i+1));
  result.push_back(static_cast<int>(endpoint));
  return result;
}

} // namespace datamunge::algorithms
