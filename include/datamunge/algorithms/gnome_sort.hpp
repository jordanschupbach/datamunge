#pragma once
/// @file gnome_sort.hpp
/// @brief Stable integer gnome sort, actual pointer records and adjacent-sort input families.
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {
/// @brief Selected actual operations of an executing gnome sort, initially zero.
/// Counts exclude input copying, identity initialization, allocations, loop tests,
/// FLOPs and wall time. A boundary visit at index zero advances without a key
/// comparison. Every strict adjacent swap moves the pointer backwards once.
/// Diagnostic counters are checked before increment and own no references.
/// @par Example
/// @code
/// const dm::algorithms::GnomeSortWork w{};
/// std::cout << w.visits << " " << w.key_comparisons; // 0 0
/// @endcode
/// @see GnomeSortRun
struct GnomeSortWork {
    std::size_t visits{0}; ///< Executed loop bodies, including boundary visits.
    std::size_t key_comparisons{0}; ///< Actual evaluations of left <= right; excludes boundary visits.
    std::size_t swaps{0}; ///< Strict adjacent inversions exchanged, one per backward step.
    std::size_t forward_steps{0}; ///< Pointer increments, including boundary steps.
    std::size_t backward_steps{0}; ///< Pointer decrements after swaps.
    std::size_t boundary_visits{0}; ///< Visits at index zero, with no key comparison.
};
/// @brief Owned sorted values, identities carried by the actual swaps and selected work.
/// Original indices are zero-based positions in the supplied input. Equal keys
/// retain increasing original indices, including repeated signed extremes. The
/// identity output and its initialization cost O(n); sorting has constant other
/// working state. Copies own both vectors and share no mutable state or aliases.
/// @par Example
/// @code
/// const auto r=dm::algorithms::gnome_sort_run({2,1,2,1});
/// std::print("{} {}\n",r.output,r.original_indices); // [1, 1, 2, 2] [1, 3, 0, 2]
/// @endcode
/// @see gnome_sort_run(), GnomeSortWork
struct GnomeSortRun {
    std::vector<int> output; ///< Owned nondecreasing values, preserving multiplicities.
    std::vector<std::size_t> original_indices; ///< Original input positions in actual final stable order.
    GnomeSortWork work{}; ///< Selected actual operations, excluding copying and identity initialization.
};
/// @brief Metadata or an actual loop visit captured without replaying the sort.
/// Initial and stop records count towards the same complete-record quota as
/// step records, even for empty input. A trace with S loop visits has S+2 records
/// if it completes; that expression is mathematical and is not precomputed.
/// @par Example
/// @code
/// const auto k=dm::algorithms::GnomeSortEventKind::step;
/// std::cout << (k==dm::algorithms::GnomeSortEventKind::step); // 1
/// @endcode
/// @see GnomeSortEvent, gnome_sort_trace()
enum class GnomeSortEventKind {
    initial, ///< Before all visits, with pointer zero and other fields unused.
    step, ///< One actual loop body, with before/after pointer and before-pair data.
    stop ///< After all visits, with both pointer fields equal to the input length.
};
/// @brief One owned fixed-size record from the executing pointer loop.
/// Step ordinal is zero-based among loop bodies. Pair values and original ids
/// describe the pair BEFORE the comparison/exchange, only when compared is true.
/// Boundary visits have compared/swapped false and unused pair fields zero.
/// Initial/stop metadata have unused ordinal/pair/flag fields zero. Metadata
/// pointer fields are both zero for initial and both n for stop. Copies retain
/// no references; unused fields are placeholders, not input observations.
/// @par Example
/// @code
/// const dm::algorithms::GnomeSortEvent e{};
/// std::cout << e.index_before << " " << e.compared; // 0 0
/// @endcode
/// @see GnomeSortEventKind, GnomeSortTrace
struct GnomeSortEvent {
    GnomeSortEventKind kind{GnomeSortEventKind::initial}; ///< Emission point defining field meanings.
    std::size_t ordinal{0}; ///< Zero-based actual loop visit; unused zero for metadata.
    std::size_t index_before{0}; ///< Pointer before this step, or the metadata pointer.
    std::size_t index_after{0}; ///< Pointer after this step, or the metadata pointer.
    bool compared{false}; ///< A key comparison occurred; false for boundary visits and metadata.
    bool swapped{false}; ///< Actual strict adjacent exchange occurred in this step.
    int left{0}; ///< Left value before comparison; unused zero without a comparison.
    int right{0}; ///< Right value before comparison; unused zero without a comparison.
    std::size_t left_original_index{0}; ///< Original identity of left before comparison, or unused zero.
    std::size_t right_original_index{0}; ///< Original identity of right before comparison, or unused zero.
};
/// @brief Owned ordinary result and every online fixed-size state record.
/// No full array snapshots are retained per visit: history storage is O(H)
/// complete records, in addition to O(n) owned values/identities. Inputs, root,
/// ordinary output/identities, constant working state and allocator overhead
/// are outside the record quota. It is not a numeric-entry, byte or peak RAM
/// limit. No replay, truncation or partial trace is returned on quota failure.
/// Copies own all storage; independent calls share no mutable state.
/// @par Example
/// @code
/// const auto t=dm::algorithms::gnome_sort_trace({2,1},6);
/// std::cout << t.run.work.visits << " " << t.events.size(); // 4 6
/// @endcode
/// @see gnome_sort_trace(), GnomeSortRun
struct GnomeSortTrace {
    GnomeSortRun run{}; ///< Ordinary result produced by this same execution.
    std::vector<GnomeSortEvent> events; ///< Initial metadata, every actual step and stopping metadata.
};
/// @brief Return a stable nondecreasing integer copy using the original one-pointer gnome loop.
/// Start at index zero. A boundary visit or left<=right comparison advances;
/// otherwise exchange the adjacent strict inversion and step backwards. Equal
/// values never exchange. Empty input has no visits; a singleton has one boundary
/// visit and no key comparisons. Signed extremes need no arithmetic on keys.
/// For n values with I strict inversions, exactly n+2I loop visits occur: O(n+I)
/// time, O(n) best and O(n^2) worst. The supplied owned buffer is sorted in place,
/// with O(1) auxiliary state, but passing an lvalue makes an O(n) owned copy.
/// Scalar output does not allocate identity/history vectors or update diagnostic counters.
/// No caller lvalue mutation, retained aliases or shared mutable state.
/// @param a Owned integer values; all int values and empty input are accepted.
/// @return Independent nondecreasing values with all original multiplicities.
/// @throws std::bad_alloc The by-value input copy cannot be allocated.
/// @throws std::length_error Requested vector storage exceeds container capacity.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::gnome_sort({3,1,2})); // [1, 2, 3]
/// @endcode
/// @par References
/// - H. Sarbazi-Azad (2000), "Stupid Sort: A new sorting algorithm", Computing
///   Science Newsletter, University of Glasgow599:4,2October2000; bibliography
///   verified via P. E. Black, NIST Dictionary of Algorithms and Data Structures,
///   "gnome sort", https://xlinux.nist.gov/dads/HTML/gnomeSort.html.
///   The retained short-circuit boundary/ordered-pair loop matches that entry.
/// @see gnome_sort_run(), gnome_sort_trace()
std::vector<int> gnome_sort(std::vector<int> a);
/// @brief Execute the shared gnome loop with actual stable identities and checked work.
/// Semantics and pointer steps are those of gnome_sort(). Identities are created
/// once and exchanged alongside actual values. They are output, not per-step
/// snapshots. O(n+I) time and O(n) owned identity output/initialization, with O(1)
/// other working state beyond the owned input/output buffers. No history quota,
/// caller lvalue mutation, retained aliases or shared mutable state.
/// @param a Owned integer input; all int values and empty input accepted.
/// @return Owned sorted values, stable original indices and selected actual counters.
/// @throws std::overflow_error A diagnostic counter cannot fit size_t.
/// @throws std::length_error Requested value/identity storage exceeds capacity.
/// @throws std::bad_alloc Owned input/identity storage allocation fails.
/// @par Example
/// @code
/// const auto r=dm::algorithms::gnome_sort_run({3,1,2});
/// std::cout << r.work.visits << " " << r.work.swaps; // 7 2
/// @endcode
/// @par References
/// - Sarbazi-Azad (2000), "Stupid Sort: A new sorting algorithm"; retained loop
///   as documented by NIST: https://xlinux.nist.gov/dads/HTML/gnomeSort.html.
/// @see gnome_sort(), GnomeSortRun
GnomeSortRun gnome_sort_run(std::vector<int> a);
/// @brief Capture every actual gnome-loop visit and metadata online with a whole-record quota.
/// Uses the same execution as gnome_sort_run(), not a second sorting loop or
/// numerical replay. Before EACH complete append the optional max_records and
/// container capacity are checked, including initial/stop metadata. Default one
/// million complete records; zero turns off only that optional quota. Empty input
/// needs two records (reject1/admit2); [2,1] needs six (reject5/admit6). No ordinary
/// output is returned if a final metadata append exceeds the cap after sorting.
/// No truncation, partial return or caller lvalue mutation. O(n+I+H) time and O(H)
/// additional fixed-size records beyond owned value/identity output. Record count
/// excludes inputs/root/ordinary outputs/working state/allocator; not scalar
/// entries, bytes or peak RAM. Independent calls have no shared mutable state.
/// @param a Owned integer input; all int values and empty input accepted.
/// @param max_records Optional whole-record quota, default1000000; zero disables it.
/// @return Owned run and all actual initial/step/stop records in execution order.
/// @throws std::length_error The next complete record exceeds quota/container capacity, or owned output cannot fit.
/// @throws std::overflow_error A selected diagnostic counter cannot fit size_t.
/// @throws std::bad_alloc Owned input/output/history allocation fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::gnome_sort_trace({},2);
/// std::cout << t.run.output.size() << " " << t.events.size(); // 0 2
/// @endcode
/// @par References
/// - Sarbazi-Azad (2000); NIST "gnome sort", retained single-pointer loop:
///   https://xlinux.nist.gov/dads/HTML/gnomeSort.html.
/// @see GnomeSortTrace, gnome_sort_run()
GnomeSortTrace gnome_sort_trace(std::vector<int> a,std::size_t max_records=1000000);
/// @brief Four owned controlled inputs for integer adjacent-exchange sort experiments.
/// All arrays contain 0 through n-1. ascending has no inversions; descending has
/// n(n-1)/2; adjacent_inversion exchanges the middle adjacent pair when n>=2;
/// last_minimum moves zero from first to last, giving n-1 inversions for n>=1.
/// Empty/singleton arrays coincide. The whole generator retains exactly 4n integer
/// entries, with no input aliases or shared state; copies own their vectors.
/// @par Example
/// @code
/// const auto f=dm::algorithms::adjacent_sort_inputs(4);
/// std::print("{} {}\n",f.ascending,f.last_minimum); // [0, 1, 2, 3] [1, 2, 3, 0]
/// @endcode
/// @see adjacent_sort_inputs()
struct AdjacentSortInputs {
    std::vector<int> ascending; ///< No inversions, values 0 through n-1.
    std::vector<int> descending; ///< Decreasing values, with all distinct pairs inverted.
    std::vector<int> adjacent_inversion; ///< Ascending except one middle adjacent exchange when n>=2.
    std::vector<int> last_minimum; ///< Values 1 through n-1 followed by zero; empty/singleton unchanged.
};
/// @brief Generate four owned adjacent-sort families with a checked whole output count.
/// Validate n-1 fits int, 4n fits size_t and every vector capacity BEFORE allocation.
/// The optional quota counts ALL four arrays together, default one million integer
/// values; zero disables it. Root/vector objects and allocator overhead are not
/// counted: this is not bytes or peak RAM. No partial result or shared mutation.
/// O(n) time and 4n output values. Assignments avoid a final overflowing int increment.
/// @param n Values per array; zero accepted, otherwise n-1 must fit int.
/// @param max_values Optional whole four-array integer-entry quota, default1000000; zero disables it.
/// @return Four independent arrays with known inversion patterns and copied ownership.
/// @throws std::length_error n-1, 4n or any vector capacity is unrepresentable, or 4n exceeds quota.
/// @throws std::bad_alloc Output allocation fails after successful admission.
/// @par Example
/// @code
/// const auto f=dm::algorithms::adjacent_sort_inputs(3,12);
/// std::print("{}\n",f.descending); // [2, 1, 0]
/// @endcode
/// @see AdjacentSortInputs, gnome_sort_run()
AdjacentSortInputs adjacent_sort_inputs(std::size_t n,std::size_t max_values=1000000);
} // namespace datamunge::algorithms
