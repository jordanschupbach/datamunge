#pragma once
/// @file comb_sort.hpp
/// @brief Checked comb-gap arithmetic, actual sorting work and owned gapped-pass records.
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// @brief Owned sorted values and actual key-operation counts for exchange sorts.
/// Key comparisons count evaluations of left>right, excluding loop control,
/// copying, index initialization and gap arithmetic. Passes include the final
/// zero-swap scan. Empty/singleton inputs perform no passes. Counters are checked
/// before increment. No references or shared state are retained.
/// @par Example
/// @code
/// const auto w=dm::algorithms::comb_sort_work({3,2,1});
/// std::print("{} {} {}\n",w.output,w.key_comparisons,w.swaps); // [1, 2, 3] 3 1
/// @endcode
/// @see comb_sort_work(), bubble_sort_work(), CombSortTrace
struct ExchangeSortWork {
  std::vector<int> output; ///< Owned nondecreasing output, preserving multiplicities.
  std::size_t key_comparisons=0; ///< Actual strict-greater key tests.
  std::size_t swaps=0; ///< Actual exchanges; one nonadjacent swap may remove several inversions.
  std::size_t pass_count=0; ///< Actual complete scans, including the terminating zero-swap scan.
  std::size_t gap_one_passes=0; ///< Scans whose gap is one; every bubble scan has gap one.
};

/// @brief Owned state and work after one actual comb-sort gap pass.
/// Pair indices increase from zero; each pair is (i,i+gap). One pass costs n-gap
/// key comparisons. A zero-swap scan stops only at gap one. Original positions
/// follow actual exchanges: their equal-value order need not remain stable.
/// @par Example
/// @code
/// const auto t=dm::algorithms::comb_sort_trace({2,2,1});
/// std::print("{} {}\n",t.passes[0].gap,t.passes[0].original_indices); // 2 [2, 1, 0]
/// @endcode
/// @see CombSortTrace
struct CombSortPass {
  std::size_t gap=1; ///< Actual positive gap used by this pass.
  std::size_t key_comparisons=0; ///< Actual n-gap key comparisons.
  std::size_t swaps=0; ///< Actual strict-greater exchanges.
  bool stopped=false; ///< True exactly on the final gap-one zero-swap pass.
  std::vector<int> values; ///< Owned complete values after this pass.
  std::vector<std::size_t> original_indices; ///< Owned original positions after these same swaps.
};

/// @brief Owned original values, final work and actual comb-sort pass states.
/// work is identical to comb_sort_work() for the same input. Empty/singleton
/// inputs have no records and identity final indices. Each record owns two
/// length-n snapshots; p passes require O(np) snapshot work/storage. Output-only
/// and work-only calls use constant auxiliary storage beyond their owned input.
/// Equal-value identities may cross indirectly during a gapped exchange.
/// @par Example
/// @code
/// const auto t=dm::algorithms::comb_sort_trace({2,2,1});
/// std::print("{} {}\n",t.work.output,t.original_indices); // [1, 2, 2] [2, 1, 0]
/// @endcode
/// @see comb_sort_trace(), CombSortPass
struct CombSortTrace {
  std::vector<int> input; ///< Owned unmodified input.
  ExchangeSortWork work; ///< Sorted output and checked actual aggregate counters.
  std::vector<std::size_t> original_indices; ///< Final original positions in output order.
  std::vector<CombSortPass> passes; ///< Owned actual records, including the terminal scan.
};

/// @brief Compute max(1,floor(10*gap/13)) exactly without overflowing size_t.
/// Quotient/remainder decomposition avoids forming 10*gap. Zero and one both
/// return one. For gap>1 the result is strictly smaller than gap. No allocation,
/// mutation, exceptions or shared state; O(1) work/storage.
/// @param gap Any size_t value, including zero and SIZE_MAX.
/// @return The exact positive next gap; no 9/10-to-11 adjustment is applied.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::comb_sort_next_gap(7)); // 5
/// @endcode
/// @see comb_sort_work(), comb_sort_trace()
std::size_t comb_sort_next_gap(std::size_t gap) noexcept;

/// @brief Sort and count actual comb-sort operations using the legacy executing kernel.
/// Starts gap=n, shrinks before every pass by exact floor(10*gap/13), bounded
/// below by one, and stops only after a zero-swap gap-one scan. No special gap
/// adjustment is used. Signed extremes are valid; comparisons do not subtract
/// keys. Strict swaps do not make this nonadjacent method stable. Worst-case
/// upper bound O(n^2), sorted-input Theta(n log n); measured random behavior is
/// not a universal linearithmic guarantee. No snapshots or shared state are used.
/// @param a Owned integer input; an lvalue caller keeps its original vector.
/// @return Owned sorted values and checked actual key comparisons, swaps and passes.
/// @throws std::overflow_error If any work counter cannot fit size_t.
/// @throws std::length_error If requested vector storage exceeds capacity.
/// @throws std::bad_alloc If input/output storage cannot be allocated.
/// @par Example
/// @code
/// const auto w=dm::algorithms::comb_sort_work({5,3,8,1,9,2,7});
/// std::print("{} {} {}\n",w.key_comparisons,w.pass_count,w.gap_one_passes); // 23 5 2
/// @endcode
/// @par References
/// - P. E. Black (2022), Comb sort, NIST Dictionary of Algorithms and Data Structures,
///   https://xlinux.nist.gov/dads/HTML/combSort.html (one bubbling scan per gap).
/// @see comb_sort(), comb_sort_trace(), comb_sort_next_gap()
ExchangeSortWork comb_sort_work(std::vector<int> a);

/// @brief Record actual comb-sort passes and carried original identities without replay.
/// Same sorting/counter semantics as comb_sort_work(). The positive online pass
/// cap is checked before each pass, including the final zero-swap pass. Exceeding
/// it throws without returning a partial trace; it does not limit the work-only
/// or legacy call. Records own their values and identities independently.
/// Time includes O(np) snapshot copying; storage is O(np) for p passes.
/// @param a Owned integer input; empty/singleton inputs need no records.
/// @param max_passes Positive record cap, default 1024, checked before every pass.
/// @return Owned input, actual pass states, final identities and checked work.
/// @throws std::invalid_argument If max_passes is zero, even for empty input.
/// @throws std::length_error If the cap or container capacity is exceeded.
/// @throws std::overflow_error If an actual work counter cannot fit size_t.
/// @throws std::bad_alloc If owned storage cannot be allocated.
/// @par Example
/// @code
/// const auto t=dm::algorithms::comb_sort_trace({3,2,1},2);
/// std::print("{} {}\n",t.passes.size(),t.passes.back().stopped); // 2 true
/// @endcode
/// @par References
/// - P. E. Black (2022), Comb sort, https://xlinux.nist.gov/dads/HTML/combSort.html.
/// @see comb_sort_work(), CombSortTrace
CombSortTrace comb_sort_trace(std::vector<int> a,std::size_t max_passes=1024);

/// @brief Sort and count the original report's full-width forward bubble baseline.
/// Repeatedly scans all n-1 adjacent pairs, without shrinking a boundary, until
/// an entire scan has no swaps. This preserves the report's comparator baseline;
/// it is not a boundary-shrinking optimization. Empty/singleton inputs return
/// without scans. Adjacent strict swaps are stable and their count equals strict
/// input inversions. Comparisons=(n-1)*pass_count for n>=2, including the terminal
/// scan. Worst O(n^2), sorted O(n), constant auxiliary state beyond owned input.
/// @param a Owned integer input, allowing the full signed range.
/// @return Sorted values and checked actual comparisons, swaps and complete scans.
/// @throws std::overflow_error If a counter cannot fit size_t.
/// @throws std::length_error If requested vector storage exceeds capacity.
/// @throws std::bad_alloc If input/output storage cannot be allocated.
/// @par Example
/// @code
/// const auto w=dm::algorithms::bubble_sort_work({3,2,1});
/// std::print("{} {} {}\n",w.key_comparisons,w.swaps,w.pass_count); // 6 3 3
/// @endcode
/// @par References
/// - P. E. Black and B. Bockholt, Bubble sort, NIST Dictionary of Algorithms and Data Structures,
///   https://xlinux.nist.gov/dads/HTML/bubblesort.html (the full-width variant is deliberate).
/// @see comb_sort_work()
ExchangeSortWork bubble_sort_work(std::vector<int> a);

} // namespace datamunge::algorithms
