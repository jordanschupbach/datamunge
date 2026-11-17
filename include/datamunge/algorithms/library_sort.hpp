#pragma once
/// @file library_sort.hpp
/// @brief Historical deterministic fixed-capacity gapped insertion, checked geometry and actual shelf/search/shift evidence.
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual producing placement branch in the fixed-capacity shelf algorithm.
/// @par Example
/// @code
/// const auto t=dm::algorithms::library_sort_trace({2,1});
/// std::print("{}\n",t.insertions[0].kind==dm::algorithms::LibrarySortPlacement::Seed); // true
/// @endcode
/// @see LibrarySortInsertion
enum class LibrarySortPlacement {
  Seed,      ///< First input placed at the shelf midpoint.
  Direct,    ///< Desired successor slot was already empty.
  ShiftRight ///< Contiguous occupied run shifted right into a real right gap.
};
/// @brief Actual selected fixed-shelf operations; no timing or amortized-paper complexity claim.
/// @par Example
/// @code
/// const auto a=dm::algorithms::library_sort_analyze({2,1});
/// std::print("{} {}\n",a.work.capacity,a.work.placements); // 6 2
/// @endcode
/// @par References
/// - Bender, Farach-Colton and Mosteiro(2006), Insertion Sort is O(n log n), https://doi.org/10.1007/s00224-005-1237-z; this deterministic fixed-capacity variant does not implement its randomized guarantee.
/// @see library_sort_analyze(), library_sort_trace()
struct LibrarySortWork {
  std::size_t capacity{0}; ///< Actual fixed shelf slots2*n+2 for n>=2;0 on trivial-copy path.
  std::size_t search_iterations{0}; ///< Actual predecessor-search loop iterations.
  std::size_t comparisons{
      0}; ///< Actual occupied value<=new value comparisons in predecessor search.
  std::size_t occupancy_reads{
      0}; ///< Actual selected occupancy tests in predecessor/gap/desired-slot decisions.
  std::size_t shift_writes{
      0}; ///< Actual moved occupied value/flag slot pairs, excluding final placement.
  std::size_t placements{
      0}; ///< Actual input values placed in the shelf, n for n>=2;0 on trivial copy.
  std::size_t output_slots{
      0}; ///< Actual complete shelf slots scanned while collecting final sorted values.
};
/// @brief Owned actual sorted values and same-execution fixed-shelf work.
/// @par Example
/// @code
/// const auto a=dm::algorithms::library_sort_analyze({2,1});
/// std::print("{} {}\n",a.values[0],a.values[1]); // 1 2
/// @endcode
/// @par References
/// - Bender, Farach-Colton and Mosteiro(2006), Insertion Sort is O(n log n), https://doi.org/10.1007/s00224-005-1237-z; this deterministic fixed-capacity variant does not implement its randomized guarantee.
/// @see library_sort_analyze(), library_sort_trace()
struct LibrarySortAnalysis {
  std::vector<int> values; ///< Owned actual ascending values with full original multiplicities.
  LibrarySortWork  work;   ///< Actual selected producing operations.
};
/// @brief Actual input identity, predecessor and chosen gap/placement branch.
/// @par Example
/// @code
/// const auto t=dm::algorithms::library_sort_trace({2,1});
/// std::print("{}\n",t.insertions[0].target); // 3
/// @endcode
/// @par References
/// - Bender, Farach-Colton and Mosteiro(2006), Insertion Sort is O(n log n), https://doi.org/10.1007/s00224-005-1237-z; this deterministic fixed-capacity variant does not implement its randomized guarantee.
/// @see library_sort_analyze(), library_sort_trace()
struct LibrarySortInsertion {
  std::size_t    input_index{0}; ///< Original input index being inserted.
  int            value{0};       ///< Actual original value, including any int extremes.
  std::ptrdiff_t predecessor{
      -1}; ///< Actual last occupied value<=new value slot before moving, or-1 if none/seed.
  std::size_t target{0}; ///< Actual final committed new-value slot after any shift.
  std::size_t gap{0};    ///< Actual originally empty slot used to receive placement or shifted run.
  LibrarySortPlacement kind{
      LibrarySortPlacement::Seed}; ///< Actual producing seed/direct/right branch.
  std::size_t moved{0};            ///< Actual occupied slot-pair moves for this insertion.
};
/// @brief Owned actual fixed-capacity shelf state, with explicit occupancy distinct from placeholder zero values.
/// @par Example
/// @code
/// const auto t=dm::algorithms::library_sort_trace({0,0});
/// std::print("{}\n",t.shelves.back().inserted); // 2
/// @endcode
/// @par References
/// - Bender, Farach-Colton and Mosteiro(2006), Insertion Sort is O(n log n), https://doi.org/10.1007/s00224-005-1237-z; this deterministic fixed-capacity variant does not implement its randomized guarantee.
/// @see library_sort_analyze(), library_sort_trace()
struct LibrarySortShelf {
  std::size_t inserted{0}; ///< Actual input values already placed,0 in initial empty shelf.
  std::vector<int>
      slots; ///< Actual stored values, including unused zero placeholders; consult occupied.
  std::vector<char> occupied; ///< Actual occupancy flags, independent of value content.
};
/// @brief Owned original input, actual insertion branches and every real shelf snapshot.
/// For n>=2 exact complete capture counts n original values, n insertion
/// records and(n+1)*(1+2*(2*n+2)) shelf/slot/flag records. For n<=1 only n
/// original values are captured, matching the actual trivial-copy branch.
/// Admitted before copies/algorithm. Excludes fields/capacities/output/runtime
/// copies and bytes/RAM; default million records,0 optional cap off.
/// @par Example
/// @code
/// const auto t=dm::algorithms::library_sort_trace({2,1});
/// std::print("{}\n",t.shelves.size()); // 3
/// @endcode
/// @par References
/// - Bender, Farach-Colton and Mosteiro(2006), Insertion Sort is O(n log n), https://doi.org/10.1007/s00224-005-1237-z; this deterministic fixed-capacity variant does not implement its randomized guarantee.
/// @see library_sort_analyze(), library_sort_trace()
struct LibrarySortTrace {
  std::vector<int>    input;    ///< Owned complete original int values in arrival order.
  LibrarySortAnalysis analysis; ///< Actual final sorted values and selected work.
  std::vector<LibrarySortInsertion>
      insertions; ///< Actual seed and subsequent producing placement records; empty for n<=1.
  std::vector<LibrarySortShelf>
      shelves; ///< Actual initial empty shelf then every real placement snapshot; empty for n<=1.
  std::size_t captured_entries{
      0}; ///< Complete original/placement/shelf-record/slot/flag logical count.
};
/// @brief Sort ascending using the historical deterministic fixed-capacity gapped insertion variant.
/// First value goes at midpoint of2*n+2 slots; predecessor search scans left
/// over gaps, then a direct or right-shifted placement inserts the next
/// value. No randomization or actual rebalancing occurs. The former rebalance
/// branch was unreachable because2*(m+1)<=2*n<2*n+2. Do not attribute the
/// paper's expected O(n log n) to this code; descending inputs can move a
/// quadratic number of slots. A conservative bound is O(n squared*log(n+1))
/// time/O(n) runtime storage; predecessor gap scans are counted explicitly.
/// By-value copy precedes body guards. No global/shared state.
/// @param input Owned by-value original integers; arbitrary int values/duplicates, empty valid.
/// @return Owned sorted values, preserving multiplicities.
/// @throws std::length_error On complete signed-index/shelf/container/count geometry.
/// @throws std::bad_alloc On runtime/output allocations.
/// @par Example
/// @code
/// const auto v=dm::algorithms::library_sort({3,1,2});
/// std::print("{} {} {}\n",v[0],v[1],v[2]); // 1 2 3
/// @endcode
/// @par References
/// - Bender, Farach-Colton and Mosteiro(2006), Insertion Sort is O(n log n), https://doi.org/10.1007/s00224-005-1237-z; this deterministic fixed-capacity variant does not implement its randomized guarantee.
/// @see LibrarySortTrace
std::vector<int> library_sort(std::vector<int> input);
/// @brief Return actual fixed-capacity gapped sort and selected predecessor/gap/shift work.
/// Same complete signed geometry/tie/placement behavior as library_sort().
/// No optional shelf history or replay; n<=1 returns a copy with zero work.
/// @param input Borrowed complete original integer sequence; empty/one-element copy path valid.
/// @return Owned actual ascending values and real selected operations.
/// @throws std::length_error On complete signed-index/shelf/container/count geometry.
/// @throws std::bad_alloc On runtime/output allocations.
/// @par Example
/// @code
/// const auto a=dm::algorithms::library_sort_analyze({2,1});
/// std::print("{}\n",a.work.output_slots); // 6
/// @endcode
/// @par References
/// - Bender, Farach-Colton and Mosteiro(2006), Insertion Sort is O(n log n), https://doi.org/10.1007/s00224-005-1237-z; this deterministic fixed-capacity variant does not implement its randomized guarantee.
/// @see LibrarySortTrace
LibrarySortAnalysis library_sort_analyze(const std::vector<int>& input);
/// @brief Capture every actual fixed-shelf state and producing gap/placement decision once.
/// Exact complete capture geometry admitted before copies or sorting. Zero
/// values are distinguished from unoccupied placeholder zeros by real flags.
/// Original input order, all gaps and right shifts remain inspectable.
/// @param input Borrowed complete original arbitrary int values.
/// @param max_entries Exact complete logical-record cap, default1000000;0 disables optional cap.
/// @return Owned original/actual shelves/placements/result/work; no invented rebalances.
/// @throws std::length_error On complete signed-index/shelf/container/count geometry.
/// @throws std::bad_alloc On runtime/output allocations.
/// @throws std::length_error On complete capture geometry or quota.
/// @par Example
/// @code
/// const auto t=dm::algorithms::library_sort_trace({2,1},43);
/// std::print("{}\n",t.captured_entries); // 43
/// @endcode
/// @par References
/// - Bender, Farach-Colton and Mosteiro(2006), Insertion Sort is O(n log n), https://doi.org/10.1007/s00224-005-1237-z; this deterministic fixed-capacity variant does not implement its randomized guarantee.
/// @see LibrarySortTrace
LibrarySortTrace library_sort_trace(
    const std::vector<int>& input,
    std::size_t             max_entries = 1000000);
/// @brief Produce the original report's portable32-bit LCG value fixture.
/// State wraps modulo2^32, with multiplier1103515245 and increment12345;
/// each output is(state>>16)%100. Synthetic fixture, not a randomness-quality claim.
/// @param count Nonnegative output count, default40.
/// @param seed Original unsigned32 seed, default20240501.
/// @return Owned deterministic integer values in[0,99], empty when count0.
/// @throws std::length_error On complete output-container geometry.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::library_sort_demo_input().size()); // 40
/// @endcode
/// @see library_sort_trace()
std::vector<int> library_sort_demo_input(std::size_t count = 40, std::uint32_t seed = 20240501);
} // namespace datamunge::algorithms
