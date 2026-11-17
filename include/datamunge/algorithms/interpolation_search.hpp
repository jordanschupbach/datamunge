#pragma once
/// @file interpolation_search.hpp
/// @brief Exact-floor integer interpolation search and actual probe boundaries.
#include <cstddef>
#include <cstdint>
#include <datamunge/algorithms/search_result.hpp>
#include <vector>
namespace datamunge::algorithms {
/// @brief Optional complete input count and actual probe-record ceilings.
/// Zero disables each optional cap. Element admission precedes reads; retained
/// probe admission precedes its candidate inspection/append, but endpoint guards
/// have already executed. Excludes borrowed input, scalar work/result, vector
/// capacity and byte/peak RAM. Untraced calls ignore max_probes; no partial trace
/// is returned on an error. Sortedness remains an unchecked precondition.
/// @par Example
/// @code
/// const dm::algorithms::InterpolationSearchOptions o;std::print("{}\n",o.max_elements); // 1000000
/// @endcode
/// @see interpolation_search_trace()
struct InterpolationSearchOptions {
  std::size_t max_elements{1000000}; ///< Complete borrowed element count; zero disables.
  std::size_t max_probes{1000000};   ///< Actual retained scalar-probe record count; zero disables.
};
/// @brief Selected actual endpoint guards, flat-range tests and candidate key tests.
/// Index checks, arithmetic, allocation, records and physical memory/cache reads
/// are excluded. result.probes counts logical candidate inspections, not all
/// endpoint accesses; a flat-range answer counts one logical probe without a
/// separate candidate comparison, because its successful guards certify equality.
/// @par Example
/// @code
/// const auto r=dm::algorithms::interpolation_search_run({1,3,5},3);
/// std::print("{} {}\n",r.result.probes,r.work.key_comparisons); // 1 1
/// @endcode
/// @see interpolation_search_run()
struct InterpolationSearchWork {
  std::size_t endpoint_guards{
      0};                    ///< Actual low-end/high-end key range tests, short-circuited.
  std::size_t flat_tests{0}; ///< Actually executed low_value==high_value tests after guards pass.
  std::size_t key_comparisons{
      0}; ///< Actually executed candidate==key and, if false, candidate<key tests.
};
/// @brief One actual closed interval and exact-floor candidate before its update.
/// Flat intervals have numerator=denominator=offset=0 and position=lo; no division.
/// For nonflat intervals position=lo+floor(numerator*(hi-lo)/denominator), using
/// an exact widened product. All fields are owned scalar snapshots.
/// @par Example
/// @code
/// const auto t=dm::algorithms::interpolation_search_trace({1,3,5},3);
/// std::print("{} {}\n",t.probes[0].position,t.probes[0].denominator); // 1 4
/// @endcode
/// @see interpolation_search_trace()
struct InterpolationSearchProbe {
  std::ptrdiff_t lo{0};          ///< Actual inclusive low bound before this probe.
  std::ptrdiff_t hi{-1};         ///< Actual inclusive high bound before this probe.
  std::ptrdiff_t position{-1};   ///< Actual inspected input index.
  int            low_value{0};   ///< Actual value at lo.
  int            high_value{0};  ///< Actual value at hi.
  int            value{0};       ///< Actual candidate value.
  std::uint64_t  numerator{0};   ///< Exact nonnegative key-low difference, zero for flat.
  std::uint64_t  denominator{0}; ///< Exact positive high-low difference, zero for flat.
  std::size_t    offset{0};      ///< Exact floored index offset, zero for flat.
  bool           flat{false};    ///< Actual equal-endpoint shortcut, not an interpolated division.
};
/// @brief Owned ordinary search result and selected actual work from one execution.
/// Does not copy/own input values. No mutation, retained input references or shared state.
/// @par Example
/// @code
/// const auto r=dm::algorithms::interpolation_search_run({2,2},2);
/// std::print("{} {}\n",r.result.index,r.result.probes); // 0 1
/// @endcode
/// @see interpolation_search_run()
struct InterpolationSearchRun {
  SearchResult
      result; ///< Actual first encountered matching index, not necessarily first duplicate.
  InterpolationSearchWork work; ///< Actual selected logical tests.
};
/// @brief Actual scalar probe history from the same executing interpolation loop.
/// Empty and out-of-endpoint-range queries retain no probes. No repeated-prefix
/// search or reconstructed intermediate states. Input lifetime ends with the call.
/// @par Example
/// @code
/// const auto t=dm::algorithms::interpolation_search_trace({1,3,5},4);
/// std::print("{} {}\n",t.run.result.found,t.probes.size()); // false 1
/// @endcode
/// @see interpolation_search_trace()
struct InterpolationSearchTrace {
  InterpolationSearchRun run; ///< Final result/work from this same execution.
  std::vector<InterpolationSearchProbe>
      probes; ///< Owned chronological actual candidate boundaries.
};
/// @brief Search a borrowed nondecreasing integer vector by exact interpolation.
/// Precondition: the complete input is sorted nondecreasing; NOT validated, so no
/// O(n) prescan is hidden in the probe bound. Full int keys/differences supported;
/// size must fit ptrdiff_t. Exact widened integer arithmetic implements the floor
/// formula, correcting pre-widening signed overflow and floating boundary errors.
/// Legacy signatures/result layout retained; duplicate choice or probe sequences
/// can differ where the former floating estimate rounded across an integer.
/// Empty returns absent with zero probes; outside-endpoint-range keys use zero
/// logical candidate probes, though endpoint reads/tests occur. A flat interval
/// matching the guards returns lo in one logical probe, without division.
/// At most n candidate probes, O(1) ordinary workspace and O(actual probes)
/// optional history. Expected O(log log n) requires probabilistic distribution
/// assumptions; equal spacing is a different deterministic one-probe example.
/// No caller mutation, retained input references or shared state.
/// @param a Borrowed complete nondecreasing int vector, empty/duplicates allowed.
/// @param key Any int target.
/// Legacy entry point disables optional caps.
/// @return Owned index/found/logical probe count.
/// @throws std::length_error On ptrdiff_t length, enabled cap or counter/capacity exhaustion.
/// @throws std::bad_alloc On trace storage allocation failure; ordinary/work calls allocate no storage.
/// @par Example
/// @code
/// const auto r=dm::algorithms::interpolation_search({1,3,5},3);
/// std::print("{}\n",r.index); // 1
/// @endcode
/// @par References
/// - NIST Dictionary of Algorithms and Data Structures, interpolation search,
///   https://xlinux.nist.gov/dads/HTML/interpolationSearch.html.
/// @see interpolation_search_run(), interpolation_search_trace(), InterpolationSearchOptions
SearchResult interpolation_search(const std::vector<int>& a, int key);

/// @brief Search a borrowed nondecreasing integer vector by exact interpolation.
/// Precondition: the complete input is sorted nondecreasing; NOT validated, so no
/// O(n) prescan is hidden in the probe bound. Full int keys/differences supported;
/// size must fit ptrdiff_t. Exact widened integer arithmetic implements the floor
/// formula, correcting pre-widening signed overflow and floating boundary errors.
/// Legacy signatures/result layout retained; duplicate choice or probe sequences
/// can differ where the former floating estimate rounded across an integer.
/// Empty returns absent with zero probes; outside-endpoint-range keys use zero
/// logical candidate probes, though endpoint reads/tests occur. A flat interval
/// matching the guards returns lo in one logical probe, without division.
/// At most n candidate probes, O(1) ordinary workspace and O(actual probes)
/// optional history. Expected O(log log n) requires probabilistic distribution
/// assumptions; equal spacing is a different deterministic one-probe example.
/// No caller mutation, retained input references or shared state.
/// @param a Borrowed complete nondecreasing int vector, empty/duplicates allowed.
/// @param key Any int target.
/// @param options Complete element/actual-record policies, default million each; zero disables optional caps.
/// @return Owned actual result and selected work.
/// @throws std::length_error On ptrdiff_t length, enabled cap or counter/capacity exhaustion.
/// @throws std::bad_alloc On trace storage allocation failure; ordinary/work calls allocate no storage.
/// @par Example
/// @code
/// const auto r=dm::algorithms::interpolation_search_run({1,3,5},3);
/// std::print("{}\n",r.result.index); // 1
/// @endcode
/// @par References
/// - NIST Dictionary of Algorithms and Data Structures, interpolation search,
///   https://xlinux.nist.gov/dads/HTML/interpolationSearch.html.
/// @see interpolation_search_run(), interpolation_search_trace(), InterpolationSearchOptions
InterpolationSearchRun interpolation_search_run(
    const std::vector<int>&    a,
    int                        key,
    InterpolationSearchOptions options = {});

/// @brief Search a borrowed nondecreasing integer vector by exact interpolation.
/// Precondition: the complete input is sorted nondecreasing; NOT validated, so no
/// O(n) prescan is hidden in the probe bound. Full int keys/differences supported;
/// size must fit ptrdiff_t. Exact widened integer arithmetic implements the floor
/// formula, correcting pre-widening signed overflow and floating boundary errors.
/// Legacy signatures/result layout retained; duplicate choice or probe sequences
/// can differ where the former floating estimate rounded across an integer.
/// Empty returns absent with zero probes; outside-endpoint-range keys use zero
/// logical candidate probes, though endpoint reads/tests occur. A flat interval
/// matching the guards returns lo in one logical probe, without division.
/// At most n candidate probes, O(1) ordinary workspace and O(actual probes)
/// optional history. Expected O(log log n) requires probabilistic distribution
/// assumptions; equal spacing is a different deterministic one-probe example.
/// No caller mutation, retained input references or shared state.
/// @param a Borrowed complete nondecreasing int vector, empty/duplicates allowed.
/// @param key Any int target.
/// @param options Complete element/actual-record policies, default million each; zero disables optional caps.
/// @return Owned actual probes and final result/work.
/// @throws std::length_error On ptrdiff_t length, enabled cap or counter/capacity exhaustion.
/// @throws std::bad_alloc On trace storage allocation failure; ordinary/work calls allocate no storage.
/// @par Example
/// @code
/// const auto r=dm::algorithms::interpolation_search_trace({1,3,5},3);
/// std::print("{}\n",r.run.result.index); // 1
/// @endcode
/// @par References
/// - NIST Dictionary of Algorithms and Data Structures, interpolation search,
///   https://xlinux.nist.gov/dads/HTML/interpolationSearch.html.
/// @see interpolation_search_run(), interpolation_search_trace(), InterpolationSearchOptions
InterpolationSearchTrace interpolation_search_trace(
    const std::vector<int>&    a,
    int                        key,
    InterpolationSearchOptions options = {});

} // namespace datamunge::algorithms
