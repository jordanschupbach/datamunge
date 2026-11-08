/// @file binary_search.cpp
/// @brief Shared closed-interval binary-search loop with optional midpoint recording.
#include <datamunge/algorithms/binary_search.hpp>

namespace datamunge::algorithms {
/// @brief Private canonical midpoint-search implementation.
namespace {
/// @brief Search a sorted integer vector; optionally append pre-update snapshots.
/// @param a Borrowed nondecreasing vector.
/// @param key Search key.
/// @param trace Nullable borrowed probe vector, initially empty.
/// @return Owned found/index/probes result.
SearchResult search_loop(const std::vector<int>& a,int key,std::vector<BinarySearchProbe>* trace) {
  SearchResult result;
  std::ptrdiff_t lo=0,hi=static_cast<std::ptrdiff_t>(a.size())-1;
  while (lo<=hi) {
    const auto mid=lo+(hi-lo)/2;
    const int value=a[static_cast<std::size_t>(mid)];
    ++result.probes;
    if (trace) trace->push_back({lo,hi,mid,value});
    if (value==key) {
      result.found=true;
      result.index=mid;
      return result;
    }
    if (value<key) lo=mid+1;
    else hi=mid-1;
  }
  return result;
}
} // namespace

SearchResult binary_search(const std::vector<int>& a,int key) {
  return search_loop(a,key,nullptr);
}

BinarySearchTrace binary_search_trace(const std::vector<int>& a,int key) {
  BinarySearchTrace trace;
  trace.result=search_loop(a,key,&trace.probes);
  return trace;
}
} // namespace datamunge::algorithms
