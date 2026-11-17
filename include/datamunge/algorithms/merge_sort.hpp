#pragma once

/// @file merge_sort.hpp
/// @brief Canonical merge sort API, producing histories and checked native contracts.

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned key and observable payload tag.
/// @details For value traces, tags are original zero-based indices. For keyed-pair traces, tags are original second fields.
/// @par Example
/// @code
/// auto t=dm::algorithms::merge_sort_trace({2,1});
/// std::print("{} {}\n",t.output[0].key,t.output[0].tag); // 1 1
/// @endcode
/// @see merge_sort_trace
/// @par Reference
/// Knuth, The Art of Computer Programming, Volume3: Sorting and Searching, merging and merge sort.
struct MergeSortItem {
  std::int64_t key{0}; ///< Ordering key.
  std::int64_t tag{0}; ///< Payload carried without comparison.
};

/// @brief Actual work of the canonical stable top-down producer.
/// @details Counters exclude allocation, capture overhead and public output conversion.
/// @par Example
/// @code
/// auto a=dm::algorithms::merge_sort_analyze({2,1});
/// std::print("{}\n",a.work.comparisons); // 1
/// @endcode
/// @see merge_sort_trace
/// @par Reference
/// Knuth, The Art of Computer Programming, Volume3: Sorting and Searching, merging and merge sort.
struct MergeSortWork {
  std::size_t comparisons{0};    ///< Actual strict key comparisons.
  std::size_t scratch_writes{0}; ///< Actual merged-element writes into the reusable buffer.
  std::size_t copy_writes{0};    ///< Actual buffer-to-data copy writes.
  std::size_t merges{0};         ///< Actual nontrivial completed intervals.
  std::size_t max_depth{
      0}; ///< Largest active recursive call depth, root one including empty/singleton.
};

/// @brief One actual strict right-front versus left-front comparison.
/// @details The recorded positions refer to the data array before this iteration writes the scratch destination.
/// @par Example
/// @code
/// auto t=dm::algorithms::merge_sort_trace({2,1});
/// std::print("{}\n",t.comparisons[0].take_right); // true
/// @endcode
/// @see merge_sort_trace
/// @par Reference
/// Knuth, The Art of Computer Programming, Volume3: Sorting and Searching, merging and merge sort.
struct MergeSortComparison {
  std::size_t  lo{0};             ///< Inclusive merge start.
  std::size_t  mid{0};            ///< Boundary between sorted child runs.
  std::size_t  hi{0};             ///< Exclusive merge end.
  std::size_t  left{0};           ///< Actual left-front source position.
  std::size_t  right{0};          ///< Actual right-front source position.
  std::int64_t left_key{0};       ///< Actual left key read.
  std::int64_t right_key{0};      ///< Actual right key read.
  bool         take_right{false}; ///< Actual right_key < left_key; ties are false.
};

/// @brief One actual scratch or copyback write.
/// @details Scratch sources refer to the current data array; copyback sources refer to the buffer. The item is copied from that actual source.
/// @par Example
/// @code
/// auto t=dm::algorithms::merge_sort_trace({2,1});
/// std::print("{}\n",t.writes[0].item.key); // 1
/// @endcode
/// @see merge_sort_trace
/// @par Reference
/// Knuth, The Art of Computer Programming, Volume3: Sorting and Searching, merging and merge sort.
struct MergeSortWrite {
  bool        copyback{false}; ///< False for scratch buffer writes, true for data copyback writes.
  std::size_t source{0};       ///< Actual source position in data or buffer as selected by phase.
  std::size_t destination{0};  ///< Actual destination position in buffer or data.
  MergeSortItem item;          ///< Actual copied key/tag item.
};

/// @brief One completed actual merge interval.
/// @details Child intervals have already committed and every current merged element has been copied back.
/// @par Example
/// @code
/// auto t=dm::algorithms::merge_sort_trace({2,1});
/// std::print("{}\n",t.merges[0].hi); // 2
/// @endcode
/// @see merge_sort_trace
/// @par Reference
/// Knuth, The Art of Computer Programming, Volume3: Sorting and Searching, merging and merge sort.
struct MergeSortCommit {
  std::size_t lo{0};          ///< Actual inclusive start.
  std::size_t mid{0};         ///< Actual child boundary.
  std::size_t hi{0};          ///< Actual exclusive end.
  std::size_t first_write{0}; ///< First actual scratch-write record for this merge.
  std::size_t past_write{0};  ///< Past-last copyback-write record for this merge.
};

/// @brief Owned sorted values and actual canonical work.
/// @details Input values are copied before sorting; caller storage is not mutated.
/// @par Example
/// @code
/// auto a=dm::algorithms::merge_sort_analyze({2,1});
/// std::print("{}\n",a.values[0]); // 1
/// @endcode
/// @see merge_sort_trace
/// @par Reference
/// Knuth, The Art of Computer Programming, Volume3: Sorting and Searching, merging and merge sort.
struct MergeSortAnalysis {
  std::vector<std::int64_t> values; ///< Owned sorted values.
  MergeSortWork             work;   ///< Actual same-run operations.
};

/// @brief Owned key/tag input and complete actual stable producing history.
/// @details Logical capture counts two N-item owned endpoints plus each comparison, write and merge record. Fields, capacities, runtime/scratch state and bytes are excluded. Captures complete producing operations and never reconstructs an explanatory sort.
/// @par Example
/// @code
/// auto t=dm::algorithms::merge_sort_trace({2,1});
/// std::print("{}\n",t.captured_entries); // 10
/// @endcode
/// @see merge_sort_trace
/// @par Reference
/// Knuth, The Art of Computer Programming, Volume3: Sorting and Searching, merging and merge sort.
struct MergeSortTrace {
  std::vector<MergeSortItem>       input;       ///< Owned original keys and tags.
  std::vector<MergeSortItem>       output;      ///< Owned actual sorted keys and moved tags.
  MergeSortWork                    work;        ///< Actual same-run counters.
  std::vector<MergeSortComparison> comparisons; ///< Actual strict key comparisons.
  std::vector<MergeSortWrite>  writes; ///< Actual scratch and copyback writes in execution order.
  std::vector<MergeSortCommit> merges; ///< Actual completed merge intervals in postorder.
  std::size_t                  captured_entries{0}; ///< Complete logical capture count.
};

/// @brief Sort integer values ascending in place with the canonical stable merge producer.
/// @details Balanced floor midpoint, one reusable O(n) buffer, strict right-before-left comparison; equal fronts take the left item. Mandatory merge and copy work is Theta(n log n) for n>=2 on every input; key-comparison counts vary. Empty/singleton inputs are unchanged.
/// @param data Mutable integer values.
/// @throws std::length_error If actual operation counts exceed size_t.
/// @par Example
/// @code
/// std::vector<std::int64_t> a{2,1};
/// dm::algorithms::merge_sort(a);
/// std::print("{}\n",a[0]); // 1
/// @endcode
/// @see merge_sort_trace
/// @par Reference
/// Knuth, The Art of Computer Programming, Volume3: Sorting and Searching, merging and merge sort.
void merge_sort(std::vector<std::int64_t>& data);

/// @brief Stable in-place sort of keyed records by first, carrying second unchanged.
/// @details The same templated canonical producer as merge_sort; second is never compared. Equal keys retain original payload order. O(n) scratch and Theta(n log n) mandatory write work for n>=2.
/// @param data Mutable int-key/int-payload pairs.
/// @throws std::length_error If actual operation counts exceed size_t.
/// @par Example
/// @code
/// std::vector<std::pair<int,int>> a{{1,9},{1,3},{0,7}};
/// dm::algorithms::merge_sort_pairs(a);
/// std::print("{} {}\n",a[1].second,a[2].second); // 9 3
/// @endcode
/// @see merge_sort_trace
/// @par Reference
/// Knuth, The Art of Computer Programming, Volume3: Sorting and Searching, merging and merge sort.
void merge_sort_pairs(std::vector<std::pair<int, int>>& data);

/// @brief Copy values and return actual sorted output and canonical operation counts.
/// @details Counts come from the real producer, not an independent sorting implementation.
/// @param data Original arbitrary int64 values.
/// @return Owned actual sorted values and selected operations.
/// @throws std::length_error If actual counts exceed size_t.
/// @par Example
/// @code
/// auto a=dm::algorithms::merge_sort_analyze({2,1});
/// std::print("{}\n",a.work.scratch_writes); // 2
/// @endcode
/// @see merge_sort_trace
/// @par Reference
/// Knuth, The Art of Computer Programming, Volume3: Sorting and Searching, merging and merge sort.
MergeSortAnalysis merge_sort_analyze(const std::vector<std::int64_t>& data);

/// @brief Capture the actual stable sort, using original indices as observable tags.
/// @details Shares the canonical templated sorting body. Admit both owned endpoints before copying, then each actual comparison/write/commit before its operation. Default cap one million; zero disables the optional cap. O(n log n) complete history. No prefix is returned after quota failure.
/// @param data Arbitrary int64 values, not mutated.
/// @param max_entries Complete logical capture cap; zero off.
/// @return Owned original/sorted key-tag endpoints and complete actual producing records.
/// @throws std::length_error On nonrepresentable index/count geometry or complete actual capture admission failure.
/// @par Example
/// @code
/// auto t=dm::algorithms::merge_sort_trace({2,1},10);
/// std::print("{}\n",t.output[0].tag); // 1
/// @endcode
/// @see merge_sort_trace
/// @par Reference
/// Knuth, The Art of Computer Programming, Volume3: Sorting and Searching, merging and merge sort.
MergeSortTrace merge_sort_trace(
    const std::vector<std::int64_t>& data,
    std::size_t                      max_entries = 1000000);

/// @brief Capture actual stable keyed-pair sorting without comparing payloads.
/// @details Owned trace keys/tags widen the original int fields exactly; ordinary pair sorting and this capture use the same templated producing body. Same capture accounting as merge_sort_trace.
/// @param data Original int-key/int-payload pairs, not mutated.
/// @param max_entries Complete logical cap, default one million; zero off.
/// @return Actual keys and payloads plus producing comparisons/writes/commits.
/// @throws std::length_error On count geometry or complete capture admission failure.
/// @par Example
/// @code
/// auto t=dm::algorithms::merge_sort_pairs_trace({{1,9},{1,3},{0,7}});
/// std::print("{} {}\n",t.output[1].tag,t.output[2].tag); // 9 3
/// @endcode
/// @see merge_sort_trace
/// @par Reference
/// Knuth, The Art of Computer Programming, Volume3: Sorting and Searching, merging and merge sort.
MergeSortTrace merge_sort_pairs_trace(
    const std::vector<std::pair<int, int>>& data,
    std::size_t                             max_entries = 1000000);

/// @brief Exact worst-case comparison bound of this balanced top-down merge structure.
/// @details For n>=1, n*ceil(log2(n))-2^ceil(log2(n))+1. Integer bit operations and checked wide arithmetic avoid floating log rounding. Zero for n0 or n1. This is a worst-case bound, not every input's observed comparisons.
/// @param n Number of elements.
/// @return Exact representable worst-case key-comparison bound.
/// @throws std::overflow_error If the bound exceeds size_t.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::merge_sort_comparison_bound(8)); // 17
/// @endcode
/// @see merge_sort_trace
/// @par Reference
/// Knuth, The Art of Computer Programming, Volume3: Sorting and Searching, merging and merge sort.
std::size_t merge_sort_comparison_bound(std::size_t n);

/// @brief Return the original eight-element worked fixture.
/// @details Retains values4,1,3,2,6,5,2,6, including duplicate2 and6.
/// @return Owned original values.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::merge_sort_demo_values().size()); // 8
/// @endcode
/// @see merge_sort_trace
/// @par Reference
/// Knuth, The Art of Computer Programming, Volume3: Sorting and Searching, merging and merge sort.
std::vector<std::int64_t> merge_sort_demo_values();

/// @brief Original seven keyed records with original-position payloads.
/// @return Pairs (3,0),(1,1),(3,2),(2,3),(1,4),(3,5),(2,6).
/// @par Example
/// @code
/// auto t=dm::algorithms::merge_sort_pairs_trace(dm::algorithms::merge_sort_demo_pairs());
/// std::print("{} {}\n",t.output[0].tag,t.output[1].tag); // 1 4
/// @endcode
/// @see merge_sort_pairs_trace
/// @par Reference
/// Original datamunge merge_sort_analysis stability fixture.
std::vector<std::pair<int, int>> merge_sort_demo_pairs();

} // namespace datamunge::algorithms
