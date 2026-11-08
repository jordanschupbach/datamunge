#pragma once
/// @file binary_search.hpp
/// @brief Instrumented binary search on ascending integer vectors.
#include <datamunge/algorithms/search_result.hpp>
#include <vector>

namespace datamunge::algorithms {
/// @brief One midpoint inspection of a closed binary-search interval.
/// Values are snapshots before either bound is updated. No input pointer is retained.
/// @par Example
/// @code
/// const auto trace = dm::algorithms::binary_search_trace({1,3,5},5);
/// std::print("{} {}\n", trace.probes.front().index, trace.probes.front().value); // 1 3
/// @endcode
/// @see BinarySearchTrace
struct BinarySearchProbe {
  std::ptrdiff_t lo{0}; ///< Inclusive lower candidate index before this probe.
  std::ptrdiff_t hi{-1}; ///< Inclusive upper candidate index before this probe.
  std::ptrdiff_t index{-1}; ///< Lower midpoint lo+(hi-lo)/2.
  int value{0}; ///< Value at the probed input index.
};

/// @brief Owned search answer and the chronological snapshots from the same binary-search loop.
/// Empty inputs have an empty probe list. result.probes equals probes.size().
/// @par Example
/// @code
/// const auto trace = dm::algorithms::binary_search_trace({1,3,5},5);
/// std::print("{} {}\n", trace.result.index, trace.probes.size()); // 2 2
/// @endcode
/// @see binary_search_trace()
struct BinarySearchTrace {
  SearchResult result; ///< Match flag, input index and probe count.
  std::vector<BinarySearchProbe> probes; ///< One record per element inspection.
};

/// @brief Find an integer key by repeatedly halving a closed interval of a sorted vector.
/// Precondition: a is sorted in nondecreasing order; sortedness is not checked.
/// Returns the first equal midpoint encountered, which need not be the first or
/// last occurrence of a duplicate key. Empty input returns an absent result with
/// zero probes. Uses at most floor(log2(n))+1 element inspections for n>0,
/// O(log(n+1)) word work and O(1) auxiliary storage. No allocation, mutation or
/// shared state; concurrent reads require that the input remain unmodified.
/// @param a Borrowed ascending-sorted vector of int values, possibly empty or duplicated.
/// @param key Integer to locate; the entire int range is supported.
/// @return SearchResult with an input index when found; otherwise index=-1.
/// @par Example
/// @code
/// const auto result = dm::algorithms::binary_search({1,3,5,7,9,11,13,15,17,19},13);
/// std::print("{} {}\n", result.index, result.probes); // 6 4
/// @endcode
/// @par References
/// - The C++ standard algorithms.binary.search section specifies the related
///   partitioned-range membership operation: https://eel.is/c++draft/alg.binary.search
/// @see binary_search_trace(), SearchResult
SearchResult binary_search(const std::vector<int>& a,int key);

/// @brief Record midpoint values and closed candidate intervals in binary_search().
/// Uses the same canonical loop and duplicate-key choice as the unrecorded call.
/// Sortedness remains a precondition. O(log(n+1)) word work and owned trace
/// storage; no input pointers survive the call. Empty input records no probes.
/// @param a Borrowed vector sorted in nondecreasing order, possibly empty.
/// @param key Integer key, including int's minimum and maximum values.
/// @return Owned match result and one snapshot per inspected element.
/// @par Example
/// @code
/// const auto trace = dm::algorithms::binary_search_trace({1,3,5},4);
/// std::print("{} {}\n", trace.result.found, trace.probes.size()); // false 2
/// @endcode
/// @par References
/// - https://eel.is/c++draft/alg.binary.search
/// @see binary_search(), BinarySearchProbe
BinarySearchTrace binary_search_trace(const std::vector<int>& a,int key);
} // namespace datamunge::algorithms
