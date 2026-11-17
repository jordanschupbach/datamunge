#pragma once
/// @file odd_even_sort.hpp
/// @brief Stable alternating adjacent comparisons with actual pair/phase histories.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual serial producer work, excluding copying, identity setup, loop tests and allocations.
/// @par Example
/// @code
/// dm::algorithms::OddEvenSortWork w;std::print("{}\n",w.comparisons); // 0
/// @endcode
/// @see odd_even_sort_trace()
struct OddEvenSortWork {
  std::size_t rounds{0};      ///< Complete odd-then-even rounds, including final zero-swap round.
  std::size_t phases{0};      ///< Actual phase visits, exactly twice rounds, even for empty input.
  std::size_t comparisons{0}; ///< Actual strict greater-than key tests.
  std::size_t swaps{0};       ///< Actual strict adjacent inversions exchanged.
};
/// @brief Owned stable sorted values, actual carried original IDs and selected serial work.
/// @par Example
/// @code
/// auto r=dm::algorithms::odd_even_sort_run({2,1,2,1});std::print("{}\n",r.original_indices); // [1, 3, 0, 2]
/// @endcode
/// @see odd_even_sort_trace()
struct OddEvenSortRun {
  std::vector<int> output; ///< Owned nondecreasing values.
  std::vector<std::size_t>
                  original_indices; ///< Original zero-based positions carried by actual swaps.
  OddEvenSortWork work;             ///< Actual selected producer operations.
};
/// @brief Actual values and identity labels before one adjacent comparison.
/// @par Example
/// @code
/// dm::algorithms::OddEvenSortComparison c;std::print("{}\n",c.swapped); // false
/// @endcode
/// @see odd_even_sort_trace()
struct OddEvenSortComparison {
  std::size_t phase{0};       ///< Actual zero-based phase ordinal.
  std::size_t left{0};        ///< Actual compared left slot; right slot is left+1.
  int         left_value{0};  ///< Actual left key before comparison.
  int         right_value{0}; ///< Actual right key before comparison.
  std::size_t left_id{0};     ///< Actual original identity at left before comparison.
  std::size_t right_id{0};    ///< Actual original identity at right before comparison.
  bool        swapped{false}; ///< Actual strict greater-than outcome and exchange flag.
};
/// @brief Actual completed odd/even phase snapshot and selected local work.
/// @par Example
/// @code
/// auto t=dm::algorithms::odd_even_sort_trace({2,1});std::print("{}\n",t.phases[0].odd); // true
/// @endcode
/// @see odd_even_sort_trace()
struct OddEvenSortPhase {
  std::size_t              ordinal{0}; ///< Actual zero-based phase ordinal.
  bool                     odd{false}; ///< True compares left slots1,3,...; false compares0,2,... .
  std::size_t              comparisons{0};   ///< Actual comparisons in this phase.
  std::size_t              swaps{0};         ///< Actual exchanges in this phase.
  std::vector<int>         values;           ///< Owned actual values after the phase.
  std::vector<std::size_t> original_indices; ///< Owned actual carried identities after the phase.
};
/// @brief Complete owned initial values, actual comparisons and completed phase snapshots.
/// Complete logical units include input n, final values/IDs2n, each actual
/// fixed comparison record1 and each phase record1 plus its2n snapshot values.
/// This can require O(n cubed) scalar history for quadratic-time sorting;
/// it is not a byte, allocator-capacity or peak-memory measure.
/// @par Example
/// @code
/// auto t=dm::algorithms::odd_even_sort_trace({2,1});std::print("{}\n",t.run.output); // [1, 2]
/// @endcode
/// @see odd_even_sort_trace()
struct OddEvenSortTrace {
  std::vector<int> input; ///< Owned original values.
  OddEvenSortRun   run;   ///< Actual same-producer final stable values, identities and work.
  std::vector<OddEvenSortComparison> comparisons; ///< Every actual comparison in execution order.
  std::vector<OddEvenSortPhase>
              phases; ///< Every actual completed phase, including final zero-swap round.
  std::size_t captured_entries{
      0}; ///< Actual logical units3n + each comparison1 + each phase(1+2n).
};
/// @brief Execute the stable odd-then-even adjacent-exchange producer.
/// Repeats left slots1,3,... then0,2,... until a complete round has no swap.
/// Strict comparisons preserve equal-key order; all int values are accepted.
/// Even empty/singleton inputs execute one two-phase zero-comparison round.
/// This is the serial early-exit producer, not a fixed-depth or parallel runner.
/// Disjoint pairs permit parallel scheduling conceptually; actual work is serial.
/// Sorting uses O(n squared) worst-case comparisons and O(n) best-case work.
/// Ordinary output uses constant other working state; run adds n carried IDs.
/// No caller mutation, retained references, callbacks or shared mutable state.
/// @param a Owned original integers, empty and signed extremes accepted.
/// @return Owned sorted values.
/// @throws std::length_error If selected history quota or checked count geometry overflows.
/// @throws std::bad_alloc If owned storage cannot be allocated.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::odd_even_sort({3,1,2})); // [1, 2, 3]
/// @endcode
/// @par References
/// - H. W. Lang, Odd-even transposition sort, https://hwlang.de/algorithmen/sortieren/networks/oetsen.htm. The native adaptive loop additionally certifies a zero-swap round.
/// @see odd_even_sort_trace(), OddEvenSortWork
std::vector<int> odd_even_sort(std::vector<int> a);
/// @brief Execute the stable odd-then-even adjacent-exchange producer.
/// Repeats left slots1,3,... then0,2,... until a complete round has no swap.
/// Strict comparisons preserve equal-key order; all int values are accepted.
/// Even empty/singleton inputs execute one two-phase zero-comparison round.
/// This is the serial early-exit producer, not a fixed-depth or parallel runner.
/// Disjoint pairs permit parallel scheduling conceptually; actual work is serial.
/// Sorting uses O(n squared) worst-case comparisons and O(n) best-case work.
/// Ordinary output uses constant other working state; run adds n carried IDs.
/// No caller mutation, retained references, callbacks or shared mutable state.
/// @param a Owned original integers, empty and signed extremes accepted.
/// @return Owned sorted values with actual identities and selected work.
/// @throws std::length_error If selected history quota or checked count geometry overflows.
/// @throws std::bad_alloc If owned storage cannot be allocated.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::odd_even_sort_run({3,1,2}).work.swaps); // 2
/// @endcode
/// @par References
/// - H. W. Lang, Odd-even transposition sort, https://hwlang.de/algorithmen/sortieren/networks/oetsen.htm. The native adaptive loop additionally certifies a zero-swap round.
/// @see odd_even_sort_trace(), OddEvenSortWork
OddEvenSortRun odd_even_sort_run(std::vector<int> a);
/// @brief Capture actual pair decisions and complete phase states from the stable odd-then-even adjacent-exchange producer.
/// Repeats left slots1,3,... then0,2,... until a complete round has no swap.
/// Strict comparisons preserve equal-key order; all int values are accepted.
/// Even empty/singleton inputs execute one two-phase zero-comparison round.
/// This is the serial early-exit producer, not a fixed-depth or parallel runner.
/// Disjoint pairs permit parallel scheduling conceptually; actual work is serial.
/// Sorting uses O(n squared) worst-case comparisons and O(n) best-case work.
/// Ordinary output uses constant other working state; run adds n carried IDs.
/// No caller mutation, retained references, callbacks or shared mutable state.
/// History units3n+comparisons+phases*(1+2n) are admitted online before
/// copying/appending each owned record. Default1000000, zero disables policy.
/// A quota failure returns no partial trace and does not mutate caller inputs.
/// @param a Owned original integers, empty and signed extremes accepted.
/// @param max_entries Complete logical history cap, default1000000, zero disables.
/// @return Owned actual complete history and final stable result.
/// @throws std::length_error If selected history quota or checked count geometry overflows.
/// @throws std::bad_alloc If owned storage cannot be allocated.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::odd_even_sort_trace({3,1,2}).run.output); // [1, 2, 3]
/// @endcode
/// @par References
/// - H. W. Lang, Odd-even transposition sort, https://hwlang.de/algorithmen/sortieren/networks/oetsen.htm. The native adaptive loop additionally certifies a zero-swap round.
/// @see odd_even_sort_trace(), OddEvenSortWork
OddEvenSortTrace odd_even_sort_trace(std::vector<int> a, std::size_t max_entries = 1000000);
/// @brief Return the original seven-value report input, without sorting it.
/// @return Fresh owned vector[5,3,8,1,9,2,7].
/// @throws std::bad_alloc If vector allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::odd_even_sort_demo_input().size()); // 7
/// @endcode
/// @see odd_even_sort_trace()
std::vector<int> odd_even_sort_demo_input();
} // namespace datamunge::algorithms
