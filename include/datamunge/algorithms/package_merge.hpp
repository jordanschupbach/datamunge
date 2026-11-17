#pragma once
/// @file package_merge.hpp
/// @brief Checked positive-weight length-limited coding with actual packages, lists and exact128-bit costs.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned exact unsigned128-bit package/weighted-path cost.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::package_merge_cost_decimal({1,0})); // 18446744073709551616
/// @endcode
/// @see package_merge_trace()
struct PackageMergeCost {
  std::uint64_t high{0}; ///< Most significant64bits of nonnegative exact value.
  std::uint64_t low{0};  ///< Least significant64bits of nonnegative exact value.
};
/// @brief Optional complete symbol, cumulative member-copy and owned-history policies.
/// @par Example
/// @code
/// dm::algorithms::PackageMergeOptions o;std::print("{}\n",o.max_symbols); // 1000000
/// @endcode
/// @see package_merge_trace()
struct PackageMergeOptions {
  std::size_t max_symbols{1000000}; ///< Complete original symbol cap,0 disables.
  std::size_t max_member_copies{
      10000000}; ///< Actual cumulative membership elements copied into working coins,0 disables; not live RAM.
  std::size_t max_trace_entries{
      1000000}; ///< Actual logical owned-history quota,0 disables; ignored without capture.
};
/// @brief Actual selected list/member operations, excluding sort comparisons/allocations/cost formatting.
/// @par Example
/// @code
/// dm::algorithms::PackageMergeWork w;std::print("{}\n",w.packages); // 0
/// @endcode
/// @see package_merge_trace()
struct PackageMergeWork {
  std::size_t levels{0};   ///< Levels used,0empty/1singleton; actual list count otherwise.
  std::size_t packages{0}; ///< Actual paired packages constructed.
  std::size_t member_copies{
      0}; ///< Actual membership elements copied into working singleton/package/list coin buffers.
  std::size_t selected_member_visits{
      0};                      ///< Actual selected coin membership increments of final lengths.
  std::size_t pruned_coins{0}; ///< Actual list tails beyond2N-2 removed after a stable sort.
};
/// @brief Owned actual code lengths, exact weighted cost, feasibility diagnostics and work.
/// @par Example
/// @code
/// auto r=dm::algorithms::package_merge_run({1,1},1);std::print("{}\n",r.lengths); // [1, 1]
/// @endcode
/// @see package_merge_trace()
struct PackageMergeRun {
  std::vector<int> lengths;       ///< Actual original-symbol-order lengths.
  PackageMergeCost weighted_cost; ///< Exact sum(weight*actual_length), checked unsigned128bit.
  int              requested_max_length{0}; ///< Original strictly positive requested length limit.
  std::size_t      effective_levels{0};     ///< Actual min(requested,N-1)forN>=2;0empty/1singleton.
  int              actual_max_length{0};    ///< Actual largest returned length,0empty.
  long double      kraft_sum{
      0}; ///< Presentation sum of2^(-actual length),not an exact arithmetic certificate.
  bool kraft_equality{false}; ///< Exact integer-slot completeness check;falseempty/singleton.
  PackageMergeWork work;      ///< Actual selected producer list/member operations.
};
/// @brief Actual constructed singleton/package with exact cost and original member occurrences.
/// @par Example
/// @code
/// dm::algorithms::PackageMergeCoin c;std::print("{}\n",c.package); // false
/// @endcode
/// @see package_merge_trace()
struct PackageMergeCoin {
  std::size_t id{0}; ///< Actual unique construction ID; singleton IDs0..N-1 reused in later lists.
  PackageMergeCost         cost;    ///< Actual exact unsigned128-bit cost.
  std::vector<std::size_t> members; ///< Actual symbol membership occurrences, duplicates retained.
  bool                     package{false}; ///< True for an actual pair package.
  std::size_t              left{0};  ///< Actual left child coin ID,0 placeholder for singleton.
  std::size_t              right{0}; ///< Actual right child coin ID,0 placeholder for singleton.
  std::size_t              level{0}; ///< Actual zero-based construction level,0 for singleton.
};
/// @brief Actual sorted retained list after optional safe tail pruning.
/// @par Example
/// @code
/// dm::algorithms::PackageMergeLevel l;std::print("{}\n",l.level); // 0
/// @endcode
/// @see package_merge_trace()
struct PackageMergeLevel {
  std::size_t level{0}; ///< Actual zero-based list level, finest initial singleton list0.
  std::vector<std::size_t>
              coins; ///< Actual retained cost-order coin IDs, ties preserve native sort rules.
  std::size_t discarded_unpaired{
      0};                ///< Actual old-list final unpaired coin omitted from packaging,0or1.
  std::size_t pruned{0}; ///< Actual sorted tail coin count beyond2N-2 removed this level.
};
/// @brief Owned original weights and actual package/list/selection histories.
/// @par Example
/// @code
/// auto t=dm::algorithms::package_merge_trace({1,1},1);std::print("{}\n",t.selected.size()); // 2
/// @endcode
/// @see package_merge_trace()
struct PackageMergeTrace {
  std::vector<std::uint64_t> weights; ///< Owned original positive integer weights.
  PackageMergeRun            run;     ///< Actual same-producer lengths/exact cost/diagnostics/work.
  std::vector<PackageMergeCoin>
      coins; ///< Every actual constructed singleton/package and its member occurrences.
  std::vector<PackageMergeLevel> levels;   ///< Every actual sorted/pruned level list.
  std::vector<std::size_t>       selected; ///< Actual2N-2 final cheapest coin IDs, empty forN<2.
  std::size_t captured_entries{0}; ///< Actual root2+2N,coins1+members,levels1+IDs,and selected IDs.
};
/// @brief Format all exact128cost bits as nonnegative decimal without rounding.
/// No input mutation or retained references; O(1) fixed-width arithmetic.
/// @param value Complete exact unsigned high/low cost.
/// @return Owned exact decimal text.
/// @throws std::bad_alloc If decimal text allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::package_merge_cost_decimal({0,42})); // 42
/// @endcode
/// @see PackageMergeCost
std::string package_merge_cost_decimal(PackageMergeCost value);
/// @brief Convert exact128cost to a rounded finite double for presentation only.
/// No input mutation or retained references; O(1) fixed-width arithmetic.
/// @param value Complete exact unsigned high/low cost.
/// @return Rounded display value, not an exact integer certificate.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::package_merge_cost_approximation({0,42})); // 42
/// @endcode
/// @see PackageMergeCost
double package_merge_cost_approximation(PackageMergeCost value);
/// @brief Produce an optimal positive-weight length-limited binary prefix-code length vector.
/// Complete positive uint64weights and max_len>=1 are validated first, including
/// empty/singleton cases. N>=2 requires2^max_len>=N, checked without unsafe shifts.
/// Empty returns no lengths; singleton deliberately uses length1, Kraft sum1/2.
/// ForN>=2, levels=min(max_len,N-1): a positive-weight optimal binary Huffman tree
/// needs no more thanN-1depth. Larger requested caps therefore add no optimum.
/// Pair cost-sorted neighboring coins, merge with original singleton list, stable
/// sort and retain only cheapest2N-2coins per level; later selections cannot need
/// a discarded tail. Initial singleton ties retain the native std::sort behavior,
/// not a promised cross-toolchain ID order. Subsequent stable ties are preserved.
/// Final2N-2cheapest memberships produce actual original-symbol lengths.
/// Package/cost arithmetic is checked unsigned128bit; no uint64 wraparound.
/// Current explicit member lists and stable sorting cost O(NL logN+member copies),
/// not a claimed optimized O(NL) implementation. No inputs mutated or referenced.
/// Default policies cap original symbols1000000 and cumulative working member
/// copies10000000; zero disables either. These are work/output policies, not RAM.
/// @param weights Complete original positive uint64symbol weights.
/// @param max_len Strictly positive maximum binary codeword length.
/// @return Owned actual std::vector<int>.
/// @throws std::invalid_argument If complete weights/cap feasibility checks fail.
/// @throws std::length_error If complete geometry/counters or selected policies fail.
/// @throws std::overflow_error If exact package/weighted cost exceeds128bits.
/// @throws std::bad_alloc If owned storage allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::package_merge({1,1},1)); // [1, 1]
/// @endcode
/// @par References
/// - L. L. Larmore and D. S. Hirschberg(1990), A Fast Algorithm for Optimal Length-Limited Huffman Codes, https://ics.uci.edu/~dhirschb/pubs/LenLimHuff.pdf. Native explicit memberships/stable sorting have separately documented costs.
/// @see PackageMergeRun, PackageMergeTrace
std::vector<int> package_merge(const std::vector<std::uint64_t>& weights, int max_len);
/// @brief Produce an optimal positive-weight length-limited binary prefix-code length vector.
/// Complete positive uint64weights and max_len>=1 are validated first, including
/// empty/singleton cases. N>=2 requires2^max_len>=N, checked without unsafe shifts.
/// Empty returns no lengths; singleton deliberately uses length1, Kraft sum1/2.
/// ForN>=2, levels=min(max_len,N-1): a positive-weight optimal binary Huffman tree
/// needs no more thanN-1depth. Larger requested caps therefore add no optimum.
/// Pair cost-sorted neighboring coins, merge with original singleton list, stable
/// sort and retain only cheapest2N-2coins per level; later selections cannot need
/// a discarded tail. Initial singleton ties retain the native std::sort behavior,
/// not a promised cross-toolchain ID order. Subsequent stable ties are preserved.
/// Final2N-2cheapest memberships produce actual original-symbol lengths.
/// Package/cost arithmetic is checked unsigned128bit; no uint64 wraparound.
/// Current explicit member lists and stable sorting cost O(NL logN+member copies),
/// not a claimed optimized O(NL) implementation. No inputs mutated or referenced.
/// Default policies cap original symbols1000000 and cumulative working member
/// copies10000000; zero disables either. These are work/output policies, not RAM.
/// @param weights Complete original positive uint64symbol weights.
/// @param max_len Strictly positive maximum binary codeword length.
/// @param options Complete symbol/work/history policies, defaults1000000/10000000/1000000;0 disables each.
/// @return Owned actual PackageMergeRun.
/// @throws std::invalid_argument If complete weights/cap feasibility checks fail.
/// @throws std::length_error If complete geometry/counters or selected policies fail.
/// @throws std::overflow_error If exact package/weighted cost exceeds128bits.
/// @throws std::bad_alloc If owned storage allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::package_merge_run({1,1},1).kraft_equality); // true
/// @endcode
/// @par References
/// - L. L. Larmore and D. S. Hirschberg(1990), A Fast Algorithm for Optimal Length-Limited Huffman Codes, https://ics.uci.edu/~dhirschb/pubs/LenLimHuff.pdf. Native explicit memberships/stable sorting have separately documented costs.
/// @see PackageMergeRun, PackageMergeTrace
PackageMergeRun package_merge_run(
    const std::vector<std::uint64_t>& weights,
    int                               max_len,
    PackageMergeOptions               options = {});
/// @brief Capture actual package formation, retained lists and final membership selection.
/// Complete positive uint64weights and max_len>=1 are validated first, including
/// empty/singleton cases. N>=2 requires2^max_len>=N, checked without unsafe shifts.
/// Empty returns no lengths; singleton deliberately uses length1, Kraft sum1/2.
/// ForN>=2, levels=min(max_len,N-1): a positive-weight optimal binary Huffman tree
/// needs no more thanN-1depth. Larger requested caps therefore add no optimum.
/// Pair cost-sorted neighboring coins, merge with original singleton list, stable
/// sort and retain only cheapest2N-2coins per level; later selections cannot need
/// a discarded tail. Initial singleton ties retain the native std::sort behavior,
/// not a promised cross-toolchain ID order. Subsequent stable ties are preserved.
/// Final2N-2cheapest memberships produce actual original-symbol lengths.
/// Package/cost arithmetic is checked unsigned128bit; no uint64 wraparound.
/// Current explicit member lists and stable sorting cost O(NL logN+member copies),
/// not a claimed optimized O(NL) implementation. No inputs mutated or referenced.
/// Default policies cap original symbols1000000 and cumulative working member
/// copies10000000; zero disables either. These are work/output policies, not RAM.
/// History quota counts root requested/effective limits2,weights/lengths2N,
/// each constructed coin1+members, each level1+retained IDs,and selected IDs.
/// Default1000000,0 disables; admission precedes each owned history copy/append.
/// No partial trace escapes failure; detailed memberships can dominate memory.
/// @param weights Complete original positive uint64symbol weights.
/// @param max_len Strictly positive maximum binary codeword length.
/// @param options Complete symbol/work/history policies, defaults1000000/10000000/1000000;0 disables each.
/// @return Owned actual PackageMergeTrace.
/// @throws std::invalid_argument If complete weights/cap feasibility checks fail.
/// @throws std::length_error If complete geometry/counters or selected policies fail.
/// @throws std::overflow_error If exact package/weighted cost exceeds128bits.
/// @throws std::bad_alloc If owned storage allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::package_merge_trace({1,1},1).run.actual_max_length); // 1
/// @endcode
/// @par References
/// - L. L. Larmore and D. S. Hirschberg(1990), A Fast Algorithm for Optimal Length-Limited Huffman Codes, https://ics.uci.edu/~dhirschb/pubs/LenLimHuff.pdf. Native explicit memberships/stable sorting have separately documented costs.
/// @see PackageMergeRun, PackageMergeTrace
PackageMergeTrace package_merge_trace(
    const std::vector<std::uint64_t>& weights,
    int                               max_len,
    PackageMergeOptions               options = {});
/// @brief Return the original nine-symbol positive frequency illustration.
/// @return Fresh owned weights[20,17,6,3,2,2,2,1,1].
/// @throws std::bad_alloc If vector allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::package_merge_demo_weights().size()); // 9
/// @endcode
/// @see package_merge_run()
std::vector<std::uint64_t> package_merge_demo_weights();
} // namespace datamunge::algorithms
