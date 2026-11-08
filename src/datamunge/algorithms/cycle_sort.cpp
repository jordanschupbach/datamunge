/// @file
/// @brief Shared rank-scan cycle-sort kernel, checked actual counters and placement capture.
#include <datamunge/algorithms/cycle_sort.hpp>
#include <algorithm>
#include <cassert>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace datamunge::algorithms {
/// @brief Private rank searches and actual value-cell placement operations.
namespace cycle_sort_detail {
/// @brief Add one actual aggregate contribution without wrapping.
/// @param count Counter to update.
/// @param amount Actual nonnegative contribution; default one.
/// @throws std::length_error Counter would exceed SIZE_MAX.
void add(std::size_t& count,std::size_t amount=1) {
  if(amount>std::numeric_limits<std::size_t>::max()-count) throw std::length_error("cycle_sort: work count overflow");
  count+=amount;
}
/// @brief Local bounded results of one suffix-rank and equal-value search.
struct Target {
  std::size_t rank{0}; ///< First slot for this key in the remaining sorted suffix.
  std::size_t position{0}; ///< Rank advanced past already-correct equal values.
  std::size_t comparisons{0}; ///< Exactly n-start-1 strict less-than tests.
  std::size_t equalities{0}; ///< Actual equality tests, including final false test if searched.
  std::size_t skips{0}; ///< Equal candidates passed; at most n-1.
};
/// @brief Find one actual rank and, when needed, skip correctly occupied equal-key slots.
/// @param a Current working array; its active start can be a duplicate of the held item.
/// @param start First unfixed outer position.
/// @param item Currently held key replacing a[start] in the conceptual suffix multiset.
/// @param initial True for a new outer start, permitting a zero-write correct position.
/// @param work Optional aggregate counters; null avoids aggregate counting.
/// @return Actual bounded search details; initial rank=start skips equality search.
/// @throws std::length_error Aggregate counter overflow.
Target target(const std::vector<int>& a,std::size_t start,int item,bool initial,CycleSortWork* work) {
  Target t;t.rank=start;
  for(std::size_t i=start+1;i<a.size();++i) {++t.comparisons;if(a[i]<item) ++t.rank;}
  t.position=t.rank;
  if(!(initial && t.rank==start)) {
    // The held key has a target block in the conceptual suffix multiset.
    // A nontrivial rotation always has an unfilled cell in that block.
    for(;;) {
      assert(t.position<a.size());++t.equalities;
      if(a[t.position]!=item) break;
      ++t.position;++t.skips;
    }
  }
  if(work) {add(work->rank_scans);add(work->rank_comparisons,t.comparisons);add(work->equality_comparisons,t.equalities);add(work->duplicate_skips,t.skips);}
  return t;
}
/// @brief Execute one sorting run, optionally capturing work and the very same moves.
/// @param a Owned working values, mutated in place.
/// @param work Optional aggregate record; its output vector may alias a.
/// @param trace Optional owned trace with initialized identity vector; requires work.
/// @throws std::length_error Aggregate counter or placement-vector capacity exceeded.
/// @throws std::bad_alloc Placement history allocation failure when traced.
void run(std::vector<int>& a,CycleSortWork* work,CycleSortTrace* trace) {
  const std::size_t n=a.size();if(n<2) return;
  for(std::size_t start=0;start<n-1;++start) {
    if(work) add(work->outer_starts);
    int item=a[start];std::size_t origin=trace?trace->original_indices[start]:0;
    auto t=target(a,start,item,true,work);if(t.rank==start) continue;
    if(work) add(work->cycle_count);
    for(;;) {
      const auto pos=t.position;
      if(trace) {
        trace->placements.push_back({start,t.rank,pos,item,a[pos],origin,trace->original_indices[pos],t.skips,t.comparisons,pos==start});
        std::swap(origin,trace->original_indices[pos]);
      }
      if(work) add(work->array_writes);
      std::swap(item,a[pos]); // one value-array assignment; held-item stores are excluded
      if(pos==start) break;
      t=target(a,start,item,false,work);
    }
  }
}
} // namespace cycle_sort_detail

std::vector<int> cycle_sort(std::vector<int> a) {
  cycle_sort_detail::run(a,nullptr,nullptr);return a;
}
CycleSortWork cycle_sort_work(std::vector<int> a) {
  CycleSortWork w;cycle_sort_detail::run(a,&w,nullptr);w.output=std::move(a);return w;
}
CycleSortTrace cycle_sort_trace(const std::vector<int>& values,std::size_t max_elements) {
  if(max_elements && values.size()>max_elements) throw std::length_error("cycle_sort: complete trace element cap exceeded");
  CycleSortTrace t;t.input=values;t.work.output=values;t.original_indices.resize(values.size());
  std::iota(t.original_indices.begin(),t.original_indices.end(),std::size_t{0});
  cycle_sort_detail::run(t.work.output,&t.work,&t);return t;
}
std::size_t cycle_sort_write_lower_bound(const std::vector<int>& values) {
  auto sorted=values;std::sort(sorted.begin(),sorted.end());std::size_t bound=0;
  for(std::size_t i=0;i<values.size();++i) bound+=values[i]!=sorted[i];
  return bound;
}
} // namespace datamunge::algorithms
