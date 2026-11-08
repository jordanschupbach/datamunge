/// @file cocktail_shaker_sort.cpp
/// @brief One bidirectional adjacent-exchange kernel for legacy output and actual pass records.
#include <datamunge/algorithms/exchange_sorts.hpp>

#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace datamunge::algorithms {
/// @brief Implementation-only checked counts and conditional pass recording.
namespace cocktail_sort_detail {

/// @brief Add an actual nonnegative work amount without wrapping.
/// @param count Counter to update.
/// @param amount Amount to add.
/// @throws std::overflow_error If the sum cannot fit size_t.
void add(std::size_t& count,std::size_t amount) {
  if(amount>std::numeric_limits<std::size_t>::max()-count)
    throw std::overflow_error("cocktail sort counter overflow");
  count+=amount;
}

/// @brief Execute historical strict comparisons once, optionally owning actual pass states.
/// @tparam Record Whether to retain identities/snapshots and check counters.
/// @param a Owned values, modified by adjacent swaps.
/// @param max_passes Positive pass limit for Record; ignored otherwise.
/// @return Sorted values and, for Record, actual work and records.
/// @throws std::length_error If a traced pass exceeds max_passes or storage capacity.
/// @throws std::overflow_error If traced work cannot fit size_t.
/// @throws std::bad_alloc If allocation fails.
template<bool Record>
CocktailSortTrace sort(std::vector<int> a,std::size_t max_passes) {
  CocktailSortTrace result;
  if constexpr(Record) {
    result.input=a;
    result.original_indices.resize(a.size());
    std::iota(result.original_indices.begin(),result.original_indices.end(),std::size_t{0});
  }
  if(a.size()>1) {
    std::size_t lo=0,hi=a.size()-1;
    bool forward=true;
    for(;;) {
      if constexpr(Record)
        if(result.passes.size()==max_passes)throw std::length_error("cocktail sort pass limit");
      CocktailSortPass pass;
      if constexpr(Record) {pass.forward=forward;pass.lo=lo;pass.hi=hi;}
      bool swapped=false;
      // Adjacent exchange keeps the optional identity vector in the same order.
      /// @brief Compare one valid adjacent pair and exchange its values/identities if inverted.
      /// @param left Index of the left member; left+1 is valid.
      const auto compare=[&](std::size_t left) {
        if constexpr(Record)add(pass.key_comparisons,1);
        if(a[left]>a[left+1]) {
          std::swap(a[left],a[left+1]);swapped=true;
          if constexpr(Record) {
            add(pass.swaps,1);
            std::swap(result.original_indices[left],result.original_indices[left+1]);
          }
        }
      };
      if(forward) {
        for(std::size_t i=lo;i<hi;++i)compare(i);
        if(swapped)--hi;
      } else {
        for(std::size_t i=hi;i>lo;--i)compare(i-1);
        ++lo;
      }
      if constexpr(Record) {
        pass.next_lo=lo;pass.next_hi=hi;pass.stopped=!swapped;
        pass.values=a;pass.original_indices=result.original_indices;
        add(result.key_comparisons,pass.key_comparisons);add(result.swaps,pass.swaps);
        result.passes.push_back(std::move(pass));
      }
      if(!swapped)break;
      forward=!forward;
    }
  }
  result.output=std::move(a);
  return result;
}
} // namespace cocktail_sort_detail

std::vector<int> cocktail_shaker_sort(std::vector<int> a) {
  return cocktail_sort_detail::sort<false>(std::move(a),0).output;
}

CocktailSortTrace cocktail_shaker_sort_trace(std::vector<int> a,std::size_t max_passes) {
  if(!max_passes)throw std::invalid_argument("cocktail sort requires a positive pass limit");
  return cocktail_sort_detail::sort<true>(std::move(a),max_passes);
}
} // namespace datamunge::algorithms
