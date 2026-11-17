#pragma once
/// @file postman_sort.hpp
/// @brief Native byte-MSD distribution sorting with actual bucket and insertion records.
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief Selected-operation, retained-entry and insertion-cutoff policies.
/// Zero work/history caps disable those policies; the default cutoff preserves the historical32-key fallback.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::PostmanOptions{}.insertion_cutoff); // 32
/// @endcode
struct PostmanOptions {
  std::size_t max_operations{
      10000000}; ///< Selected scalar loop/copy/compare/move event cap; excludes allocation and counter machinery.
  std::size_t max_trace_entries{
      1000000}; ///< Node records plus256 counts and257 offsets per actual radix node; zero disables cap.
  std::size_t insertion_cutoff{
      32}; ///< Ranges of at most this many keys use insertion sort; zero forces byte passes for nontrivial ranges.
};
/// @brief One actual range visit and its immediately used bucket boundaries.
/// Counts/offsets are retained only for byte-distribution nodes; insertion nodes have zero-filled arrays.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::postman_sort_trace({3,1,2}).nodes[0].insertion); // true
/// @endcode
struct PostmanNode {
  std::size_t lo;        ///< Actual inclusive array index processed by this call.
  std::size_t hi;        ///< Actual exclusive array index processed by this call.
  int         shift;     ///< Actual byte shift24,16,8,0, or-8 at the final fallback.
  bool        insertion; ///< Actual branch selecting the stable insertion fallback.
  std::array<std::size_t, 256>
      counts{}; ///< Actual key counts by current byte, captured before prefix summation.
  std::array<std::size_t, 257>
      offsets{}; ///< Actual prefix offsets relative to lo; last entry equals hi-lo.
};
/// @brief Owned sorted output and selected actual event counts.
/// Counters describe executed native loops, not elapsed-time or asymptotic guarantees.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::postman_sort_run({3,1,2}).sorted); // [1, 2, 3]
/// @endcode
struct PostmanRun {
  std::vector<std::uint32_t> sorted; ///< Owned ascending complete output.
  std::size_t                nodes{
      0}; ///< Actual recursive calls, including insertion fallbacks; empty/singleton inputs have none.
  std::size_t radix_nodes{0};   ///< Actual non-fallback byte passes.
  std::size_t key_scans{0};     ///< Actual histogram reads.
  std::size_t distributions{0}; ///< Actual stable writes into temporary buckets.
  std::size_t copybacks{0};     ///< Actual temporary-to-array writes.
  std::size_t comparisons{0};   ///< Actual insertion key comparisons, excluding loop/bounds tests.
  std::size_t moves{0};         ///< Actual insertion shifts and final key placements.
  std::size_t prefix_steps{0};  ///< Actual count prefix additions (256 per radix node).
  std::size_t operations{0};    ///< Sum of the six selected loop event counters.
};
/// @brief Ordinary output and actual producing node histories.
/// Records come from the same recursion and prefix arrays used for sorting; no second bucketing pass creates history.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::postman_sort_trace({2,1}).nodes.size()); // 1
/// @endcode
struct PostmanTrace {
  PostmanRun               run;   ///< Actual output and selected counts.
  std::vector<PostmanNode> nodes; ///< Actual preorder range visits and used bucket boundaries.
};
/// @brief Sort by the shared native stable byte-MSD producer.
/// No node history retained; default32-key insertion fallback. Fixed32-bit keys use at most four radix levels; work includes256 prefix entries per pass and insertion work on fallback ranges. Peak ordinary auxiliary storage is O(n); owned inputs are unchanged outside this call. State is local.
/// @param a Owned uint32 keys, including duplicates and the full unsigned range.
/// @param options Selected event/history policies and configurable fallback cutoff.
/// @return Owned output and actual selected events.
/// @throws std::length_error If a selected-operation/history policy or representable storage is exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::postman_sort_run({2,1}).sorted); // [1, 2]
/// @endcode
/// @par References
/// - D. Knuth, The Art of Computer Programming, Volume3:Sorting and Searching, section5.2.5 (radix sorting); this producer uses byte-MSD passes and an insertion cutoff.
PostmanRun postman_sort_run(std::vector<std::uint32_t> a, PostmanOptions options = {});
/// @brief Sort by the shared native stable byte-MSD producer.
/// Retains actual preorder nodes and complete used histograms/offsets; caps count logical entries, not bytes. Fixed32-bit keys use at most four radix levels; work includes256 prefix entries per pass and insertion work on fallback ranges. Peak ordinary auxiliary storage is O(n); owned inputs are unchanged outside this call. State is local.
/// @param a Owned uint32 keys, including duplicates and the full unsigned range.
/// @param options Selected event/history policies and configurable fallback cutoff.
/// @return Owned output and actual selected events and histories.
/// @throws std::length_error If a selected-operation/history policy or representable storage is exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::postman_sort_trace({2,1}).run.comparisons); // 1
/// @endcode
/// @par References
/// - D. Knuth, The Art of Computer Programming, Volume3:Sorting and Searching, section5.2.5 (radix sorting); this producer uses byte-MSD passes and an insertion cutoff.
PostmanTrace postman_sort_trace(std::vector<std::uint32_t> a, PostmanOptions options = {});
/// @brief Sort unsigned keys in place by the same checked native MSD producer.
/// Successful return replaces a with the ascending result; errors leave the caller vector unchanged. Uses default policies and cutoff32.
/// @param a Mutable owned key vector; no references retained.
/// @throws std::length_error If a selected-operation/history policy or representable storage is exceeded.
/// @par Example
/// @code
/// std::vector<std::uint32_t>a{2,1};dm::algorithms::postman_sort(a);std::print("{}\n",a); // [1, 2]
/// @endcode
/// @see postman_sort_run(), postman_sort_trace()
void postman_sort(std::vector<std::uint32_t>& a);
/// @brief Generate the original native uniform-word sorting fixture.
/// Each key is the low32 bits of one actual mt19937_64 word; the original count8000/seed22 stream is preserved. This is a controlled synthetic input, not a randomness or benchmark-quality certificate. O(count) work/storage; local RNG only.
/// @param count Requested key count.
/// @param seed Local generator seed, default22.
/// @param max_keys Admitted count, defaultone million; zero disables.
/// @return Owned actual generated uint32 keys.
/// @throws std::length_error If requested count exceeds its policy or vector storage.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::postman_uniform_keys(3).size()); // 3
/// @endcode
std::vector<std::uint32_t>
postman_uniform_keys(std::size_t count, std::uint64_t seed = 22, std::size_t max_keys = 1000000);
} // namespace datamunge::algorithms
