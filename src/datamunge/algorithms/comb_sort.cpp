/// @file comb_sort.cpp
/// @brief One checked gap kernel for legacy output/work/traces and the original bubble baseline.
#include <datamunge/algorithms/comparison_sorts.hpp>
#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace datamunge::algorithms {
/// @brief Implementation-only checked counters and conditional actual comb records.
namespace comb_sort_detail {
/// @brief Add actual work without unsigned wrap.
/// @param count Counter to update.
/// @param amount Actual nonnegative increment.
/// @throws std::overflow_error If the increment cannot fit size_t.
void add(std::size_t& count,std::size_t amount) {
  if(amount>std::numeric_limits<std::size_t>::max()-count)throw std::overflow_error("exchange sort counter overflow");
  count+=amount;
}

/// @brief Execute strict gapped exchanges once with optional work/identity recording.
/// @tparam Count Whether to check and retain actual work counters.
/// @tparam Record Whether to retain actual states and carried original ids; implies Count.
/// @param a Owned input, modified by swaps.
/// @param max_passes Positive online record cap for Record, otherwise ignored.
/// @return Owned sorted values, optional counters and optional actual pass records.
/// @throws std::length_error If a requested record exceeds its cap or storage capacity.
/// @throws std::overflow_error If an actual work counter cannot fit.
/// @throws std::bad_alloc If owned storage cannot be allocated.
template<bool Count,bool Record>
CombSortTrace sort(std::vector<int> a,std::size_t max_passes) {
  static_assert(!Record || Count);
  CombSortTrace result;
  if constexpr(Record) {
    result.input=a;result.original_indices.resize(a.size());
    std::iota(result.original_indices.begin(),result.original_indices.end(),std::size_t{0});
  }
  if(a.size()>1) {
    auto gap=a.size();bool swapped=true;
    while(gap>1 || swapped) {
      if constexpr(Record)
        if(result.passes.size()==max_passes)throw std::length_error("comb sort pass limit");
      gap=comb_sort_next_gap(gap);swapped=false;
      CombSortPass pass;pass.gap=gap;
      for(std::size_t i=0;i<a.size()-gap;++i) {
        if constexpr(Count)add(pass.key_comparisons,1);
        if(a[i]>a[i+gap]) {
          std::swap(a[i],a[i+gap]);swapped=true;
          if constexpr(Count)add(pass.swaps,1);
          if constexpr(Record)std::swap(result.original_indices[i],result.original_indices[i+gap]);
        }
      }
      if constexpr(Count) {
        add(result.work.key_comparisons,pass.key_comparisons);add(result.work.swaps,pass.swaps);
        add(result.work.pass_count,1);if(gap==1)add(result.work.gap_one_passes,1);
      }
      if constexpr(Record) {
        pass.stopped=gap==1 && !swapped;pass.values=a;pass.original_indices=result.original_indices;
        result.passes.push_back(std::move(pass));
      }
    }
  }
  result.work.output=std::move(a);return result;
}
} // namespace comb_sort_detail

std::size_t comb_sort_next_gap(std::size_t gap) noexcept {
  return std::max(std::size_t{1},(gap/13)*10+((gap%13)*10)/13);
}

std::vector<int> comb_sort(std::vector<int> a) {
  return comb_sort_detail::sort<false,false>(std::move(a),0).work.output;
}

ExchangeSortWork comb_sort_work(std::vector<int> a) {
  return comb_sort_detail::sort<true,false>(std::move(a),0).work;
}

CombSortTrace comb_sort_trace(std::vector<int> a,std::size_t max_passes) {
  if(!max_passes)throw std::invalid_argument("comb sort requires a positive pass limit");
  return comb_sort_detail::sort<true,true>(std::move(a),max_passes);
}

ExchangeSortWork bubble_sort_work(std::vector<int> a) {
  ExchangeSortWork result;
  if(a.size()>1) {
    bool swapped;
    do {
      swapped=false;
      for(std::size_t i=0;i<a.size()-1;++i) {
        comb_sort_detail::add(result.key_comparisons,1);
        if(a[i]>a[i+1]) {
          std::swap(a[i],a[i+1]);swapped=true;comb_sort_detail::add(result.swaps,1);
        }
      }
      comb_sort_detail::add(result.pass_count,1);comb_sort_detail::add(result.gap_one_passes,1);
    }while(swapped);
  }
  result.output=std::move(a);return result;
}
} // namespace datamunge::algorithms
