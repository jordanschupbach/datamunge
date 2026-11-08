#pragma once
/// @file sorted_search.hpp
/// @brief Integer searches on sorted arrays and instrumented complete-tree BFS layouts.
#include <datamunge/algorithms/binary_search.hpp>

#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// @brief *Jump (block) search* on an ascending-sorted vector @p a: step forward in fixed blocks of
///        size @c floor(sqrt(n)) until a block whose last element is >= key is found, then scan that
///        block linearly. It makes @c O(sqrt(n)) comparisons -- worse than binary search but with a
///        purely forward access pattern, historically useful when jumping back is expensive (e.g.
///        tape or singly-linked storage).
SearchResult jump_search(const std::vector<int>& a, int key);

/// @brief *Interpolation (predictive) search* on an ascending-sorted vector @p a: instead of always
///        probing the middle, it estimates the key's position by linear interpolation between the
///        current endpoints' values, @c pos = lo + (key - a[lo]) * (hi - lo) / (a[hi] - a[lo]). On
///        (near-)uniformly distributed data this converges in @c O(log log n) expected comparisons;
///        in the worst case (very skewed data) it degrades to @c O(n). A division-by-zero guard
///        handles equal endpoints. @p a must be sorted in non-decreasing order.
SearchResult interpolation_search(const std::vector<int>& a, int key);

/// @brief Initial exact consecutive Fibonacci sizes for a sorted-array search.
/// For positive length, starts at small=0,middle=1,large=1 and grows until
/// large>=length. Growth_steps counts actual additions, excluded from probes.
/// Empty length has all-zero sizes/counts. Length is checked against ptrdiff_t
/// maximum before arithmetic, making the first covering Fibonacci sum fit size_t.
/// Owns scalar state, no allocation or shared mutation.
/// @par Example
/// @code
/// const auto s=dm::algorithms::fibonacci_search_schedule(10);
/// std::print("{} {} {} {}\n",s.small,s.middle,s.large,s.growth_steps); // 5 8 13 5
/// @endcode
/// @see fibonacci_search_schedule(), FibonacciSearchOutcome
struct FibonacciSearchSchedule {
    std::size_t length=0; ///< Original array length, without allocating an array.
    std::size_t small=0; ///< F(k-2), or zero for empty input.
    std::size_t middle=0; ///< F(k-1), or zero for empty input.
    std::size_t large=0; ///< Smallest covering F(k), or zero for empty input.
    std::size_t growth_steps=0; ///< Actual additions while finding the covering Fibonacci size.
};

/// @brief Search result and actual logical work of the shared Fibonacci kernel.
/// Result.probes counts element inspections including a possible final equality
/// fallback and repeated clamped indices. Comparisons counts executed key-value
/// <, > or == tests: upper main branch one, lower/equal main branch two, fallback
/// one. Transitions counts nonmatching main-loop schedule updates. These exclude
/// index/bound checks, instructions, cache events and timings. No optional history.
/// @par Example
/// @code
/// const auto o=dm::algorithms::fibonacci_search_work({1,3,5,7,9,11,13,15,17,19},13);
/// std::print("{} {} {}\n",o.result.probes,o.comparisons,o.transitions); // 4 6 3
/// @endcode
/// @see FibonacciSearchSchedule, fibonacci_search_work()
struct FibonacciSearchOutcome {
    SearchResult result; ///< Actual found/index/probe result, retaining legacy duplicate choice.
    FibonacciSearchSchedule schedule; ///< Initial covering sizes and actual growth work.
    std::size_t comparisons=0; ///< Actual primitive key-value comparisons only.
    std::size_t transitions=0; ///< Actual nonmatching main-loop schedule updates.
};

/// @brief One actual read and pre/post schedule transition in Fibonacci search.
/// Main decisions are+1 for key above value,-1 for key below value,0 for match.
/// Fallback performs only equality and uses decision2 for a miss; it does not
/// compute an unexecuted ordering comparison. Match and fallback preserve sizes
/// and offset. Fields are owned snapshots, never replayed or borrowed.
/// @par Example
/// @code
/// const auto t=dm::algorithms::fibonacci_search_trace({1,3,5,7,9,11,13,15,17,19},13);
/// std::print("{} {} {}\n",t.probes.front().index,t.probes.front().value,t.probes.front().large_after); // 4 9 8
/// @endcode
/// @see FibonacciSearchTrace
struct FibonacciSearchProbe {
    std::size_t small_before=0; ///< Actual F(k-2) before inspection.
    std::size_t middle_before=0; ///< Actual F(k-1) before inspection.
    std::size_t large_before=0; ///< Actual F(k) before inspection.
    std::ptrdiff_t offset_before=-1; ///< Last eliminated-prefix index; -1 before any upper move.
    std::size_t index=0; ///< Actual clamped zero-based input position inspected.
    int value=0; ///< Actual read value, copied before input may later change.
    int decision=0; ///< +1 upper,-1 lower,0 match,2 final equality-only miss.
    std::size_t comparisons=0; ///< Actual one/two primitive key-value tests at this inspection.
    bool final_check=false; ///< True for the possible one-element equality fallback.
    std::size_t small_after=0; ///< Actual F(k-2) after the chosen main transition.
    std::size_t middle_after=0; ///< Actual F(k-1) after the chosen main transition.
    std::size_t large_after=0; ///< Actual F(k) after the chosen main transition.
    std::ptrdiff_t offset_after=-1; ///< Actual eliminated-prefix index after this inspection.
};

/// @brief Owned actual Fibonacci outcome and chronological element-read records.
/// C=captured_entries=probes.size()=outcome.result.probes counts whole optional
/// records, including repeated positions and final fallback. Metadata, input,
/// capacities, allocators and runtime storage are excluded: C is not peak RAM.
/// No input copy or retained reference. Copies are independent and edits do not
/// recompute outcome. Ordinary/work calls keep no optional history or quota.
/// @par Example
/// @code
/// const auto t=dm::algorithms::fibonacci_search_trace({1,3,5,7,9,11,13,15,17,19},8);
/// std::print("{} {} {}\n",t.outcome.result.found,t.probes.back().final_check,t.captured_entries); // false true 5
/// @endcode
/// @see fibonacci_search_trace(), FibonacciSearchProbe
struct FibonacciSearchTrace {
    int key=0; ///< Owned original query key.
    FibonacciSearchOutcome outcome; ///< Actual answer, initial schedule and work.
    std::vector<FibonacciSearchProbe> probes; ///< Actual read/transition records, no replay.
    std::size_t captured_entries=0; ///< Exact optional whole-record count.
};

/// @brief Compute the covering Fibonacci schedule without allocating an input array.
/// Empty n gives zero sizes/additions. Positive n yields large>=n with consecutive
/// predecessors, using additions only. Rejects lengths above the signed result
/// index domain before growth; never relies on size_t wrap. O(log(n+1)) word work
/// and constant storage, no input mutation or shared state.
/// @param n Hypothetical array length, in [0,PTRDIFF_MAX].
/// @return Owned initial schedule and actual growth-addition count.
/// @throws std::length_error If n exceeds ptrdiff_t maximum or schedule arithmetic cannot fit.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::fibonacci_search_schedule(10).large); // 13
/// @endcode
/// @par References
/// - P. E. Black, Fibonaccian search, NIST DADS (2020), https://xlinux.nist.gov/dads/HTML/fibonaccianSearch.html
///   This variant uses the smaller predecessor offset, clamping and an equality fallback.
/// @see fibonacci_search(), FibonacciSearchSchedule
FibonacciSearchSchedule fibonacci_search_schedule(std::size_t n);

/// @brief Find an exact key in a nondecreasing int vector using Fibonacci offsets.
/// Sortedness is an unchecked precondition; duplicates return the first equal
/// position encountered, not necessarily the first occurrence. The original
/// compiled signature, index choice and probe count are retained. Probe indices
/// clamp at n-1 and may repeat; an equality-only fallback may add one read.
/// Empty input returns absent/-1/zero probes. Safe offset arithmetic avoids the
/// old -1-to-unsigned wrap idiom; signed index capacity is checked before growth.
/// O(log(n+1)) word work, constant storage, additions/subtractions for scheduling
/// with no midpoint division. No hardware speed or cache claim. Calls share no
/// state; concurrent input reads require that the input remain unmodified.
/// @param a Borrowed nondecreasing int vector, possibly empty or duplicated.
/// @param key Any int value, including INT_MIN and INT_MAX.
/// @return Original SearchResult semantics; probes are reads, not unique positions or comparisons.
/// @throws std::length_error If input length cannot fit the signed result index domain.
/// @par Example
/// @code
/// const auto r=dm::algorithms::fibonacci_search({1,3,5,7,9,11,13,15,17,19},13);
/// std::print("{} {}\n",r.index,r.probes); // 6 4
/// @endcode
/// @par References
/// - P. E. Black, Fibonaccian search, NIST DADS (2020), https://xlinux.nist.gov/dads/HTML/fibonaccianSearch.html
///   Smaller-predecessor offset/clamping/fallback variant; no general optimality claim.
/// @see fibonacci_search_work(), fibonacci_search_trace(), binary_search()
SearchResult fibonacci_search(const std::vector<int>& a,int key);

/// @brief Run the ordinary Fibonacci kernel and expose actual scheduling/comparison work.
/// Shares exact result, duplicate choice, sorted-input precondition and checked
/// arithmetic with fibonacci_search(). No optional history allocation or quota.
/// Initial additions, element reads, primitive key comparisons and actual main
/// transitions remain separate logical counts; no timing or cache inference.
/// @param a Borrowed nondecreasing int vector; sortedness is unchecked.
/// @param key Any int value.
/// @return Owned actual result and work, with zero counts for empty input.
/// @throws std::length_error If input length exceeds the signed index domain.
/// @par Example
/// @code
/// const auto o=dm::algorithms::fibonacci_search_work({1,3,5,7,9,11,13,15,17,19},20);
/// std::print("{} {} {}\n",o.result.probes,o.comparisons,o.transitions); // 5 5 5
/// @endcode
/// @par References
/// - https://xlinux.nist.gov/dads/HTML/fibonaccianSearch.html ; variant as fibonacci_search().
/// @see FibonacciSearchOutcome, fibonacci_search()
FibonacciSearchOutcome fibonacci_search_work(const std::vector<int>& a,int key);

/// @brief Capture actual Fibonacci reads and chosen transitions during the same search pass.
/// Shares ordinary/work semantics and unchecked sorted-input precondition. Admits
/// one whole probe record before reading the next input element; default quota1m,
/// zero disables only optional quota. Captures the equality fallback too. No
/// replay, input copy, shared state or partial result on errors. O(P) owned history
/// for P actual reads, while ordinary/work calls retain constant storage.
/// @param a Borrowed nondecreasing int vector, possibly empty/duplicated.
/// @param key Any int value.
/// @param max_entries Maximum whole optional probe records, default1000000; zero disables quota.
/// @return Owned actual result and pre/post snapshots; C equals actual probe count.
/// @throws std::length_error If signed-index capacity, vector limits or quota are exceeded.
/// @throws std::bad_alloc If optional history storage fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::fibonacci_search_trace({1,3,5,7,9,11,13,15,17,19},20,5);
/// std::print("{} {}\n",t.probes.back().index,t.captured_entries); // 9 5
/// @endcode
/// @par References
/// - https://xlinux.nist.gov/dads/HTML/fibonaccianSearch.html ; variant as fibonacci_search().
/// @see FibonacciSearchTrace, fibonacci_search_work()
FibonacciSearchTrace fibonacci_search_trace(const std::vector<int>& a,int key,std::size_t max_entries=1000000);

/// @brief One actual in-order assignment from sorted input to a complete-tree node.
/// Owns its integer value and both zero-based positions; retains no input pointer.
/// @par Example
/// @code
/// const auto t=dm::algorithms::eytzinger_layout_trace({1,3,5});
/// std::print("{} {} {}\n",t.placements.front().sorted_index,t.placements.front().layout_index,t.placements.front().value); // 0 1 1
/// @endcode
/// @see EytzingerLayoutTrace
struct EytzingerPlacement {
    std::size_t sorted_index=0; ///< Zero-based source rank assigned at this actual visit.
    std::size_t layout_index=0; ///< Zero-based destination in the breadth-first layout.
    int value=0; ///< Owned value assigned to that destination.
};

/// @brief Owned complete-tree layout and chronological placements from its actual construction.
/// Empty input gives empty vectors and zero capture. Logical optional capture C=N
/// counts whole placement records only; ordinary output layout, metadata, vector
/// capacities and allocator/runtime bytes are excluded. This is not peak RAM.
/// O(N) output/history storage plus O(log(N+1)) recursion; no replay or input alias.
/// @par Example
/// @code
/// const auto t=dm::algorithms::eytzinger_layout_trace({1,3,5});
/// std::print("{} {}\n",t.layout.front(),t.captured_entries); // 3 3
/// @endcode
/// @see eytzinger_layout_trace(), EytzingerPlacement
struct EytzingerLayoutTrace {
    std::vector<int> layout; ///< Owned breadth-first complete-tree values.
    std::vector<EytzingerPlacement> placements; ///< One actual assignment per input, in source-rank order.
    std::size_t captured_entries=0; ///< Whole optional placement records; equals placements.size().
};

/// @brief One actual element inspection and chosen transition in an exact-match tree search.
/// Owns its snapshot, with no input alias. Match terminates immediately. A missing
/// chosen child terminates an unsuccessful search without accessing outside layout.
/// @par Example
/// @code
/// const auto t=dm::algorithms::eytzinger_search_trace({3,1,5},5);
/// std::print("{} {} {}\n",t.probes.front().index,t.probes.front().direction,t.probes.front().next_index); // 0 1 2
/// @endcode
/// @see EytzingerSearchTrace
struct EytzingerProbe {
    std::size_t index=0; ///< Zero-based layout position actually inspected.
    int value=0; ///< Value read at index before choosing the transition.
    int direction=0; ///< -1 for key smaller/left, 0 for equality/match, +1 for larger/right.
    std::ptrdiff_t next_index=-1; ///< Chosen existing child, or -1 for match/missing child termination.
};

/// @brief Owned exact-match answer and actual chronological probe snapshots from one execution.
/// Empty layout has no probes. Logical optional capture C=P is whole probe records,
/// with P=result.probes=probes.size(); input, metadata/capacity/allocator/runtime
/// bytes are excluded. No peak-memory claim. Stored values survive input mutation.
/// @par Example
/// @code
/// const auto t=dm::algorithms::eytzinger_search_trace({3,1,5},4);
/// std::print("{} {} {}\n",t.result.found,t.result.index,t.captured_entries); // false -1 2
/// @endcode
/// @see eytzinger_search_trace(), EytzingerProbe
struct EytzingerSearchTrace {
    int key=0; ///< Owned query key.
    std::size_t layout_size=0; ///< Original input length; the input itself is not copied.
    SearchResult result; ///< Matching layout position or -1, and actual element-inspection count.
    std::vector<EytzingerProbe> probes; ///< One snapshot per actual inspection, without replay.
    std::size_t captured_entries=0; ///< Whole optional probe records; equals probes.size().
};

/// @brief Build a complete binary search tree in breadth-first (Eytzinger) array order.
/// Precondition: sorted is nondecreasing; not validated. A complete tree fills its
/// last level left to right; in-order traversal assigns successive source values.
/// The root need not be a median for incomplete last levels. Duplicates retain
/// in-order source rank. Empty input yields empty output. O(N) assignments/output
/// storage and O(log(N+1)) recursion, no optional history/quota. Child existence is
/// checked before index multiplication. Input is unchanged, output owns values;
/// independent calls share no mutable state. No cache/timing benefit is promised.
/// @param sorted Borrowed nondecreasing int vector, possibly empty or duplicated.
/// @return Owned permutation in zero-based breadth-first node order.
/// @throws std::length_error If positions or output length cannot be represented.
/// @throws std::bad_alloc If output storage cannot be allocated.
/// @par Example
/// @code
/// const auto l=dm::algorithms::eytzinger_layout({1,3,5,7,9,11,13,15,17,19});
/// std::print("{} {} {}\n",l.front(),l[1],l.back()); // 13 7 9
/// @endcode
/// @par References
/// - Khuong and Morin (2017), Array Layouts for Comparison-Based Searching, section3.2.
///   https://arxiv.org/abs/1509.05053 . This is the ordinary complete-tree BFS layout.
/// @see eytzinger_search(), eytzinger_layout_trace()
std::vector<int> eytzinger_layout(const std::vector<int>& sorted);

/// @brief Find the first equal node encountered in a breadth-first binary search tree.
/// Precondition: layout has the tree order produced by eytzinger_layout(); unchecked.
/// Tests equality first, then moves left for a smaller key or right for a larger
/// one; a missing child returns absent. Duplicate matches need not be the first
/// sorted occurrence. Found index is a layout position, not source rank. Empty
/// input returns false/-1/zero probes. At most floor(log2(N))+1 inspected elements
/// for N>0, O(log(N+1)) fixed-word work and constant auxiliary storage, no history.
/// A probe is an element read, not a primitive comparison, cache miss or time unit.
/// Input is unmodified; concurrent reads require it remain unmodified. No shared
/// mutable state, prefetching, branch-free code or timing guarantee is supplied.
/// @param layout Borrowed complete-tree BFS int vector, possibly empty/duplicated.
/// @param key Any int value, including its minimum and maximum.
/// @return Owned match flag, layout index (or -1) and actual probe count.
/// @throws std::length_error If input positions cannot fit the result index type.
/// @par Example
/// @code
/// const auto r=dm::algorithms::eytzinger_search({13,7,17,3,11,15,19,1,5,9},9);
/// std::print("{} {} {}\n",r.found,r.index,r.probes); // true 9 4
/// @endcode
/// @par References
/// - Khuong and Morin (2017), Array Layouts for Comparison-Based Searching, section3.2.
///   https://arxiv.org/abs/1509.05053 . This exact-match branchy variant differs from
///   their lower-bound interface and tuned prefetching/branch-free implementations.
/// @see eytzinger_layout(), eytzinger_search_trace(), SearchResult
SearchResult eytzinger_search(const std::vector<int>& layout, int key);

/// @brief Capture actual sorted-rank placements while building the ordinary Eytzinger layout.
/// Shares eytzinger_layout()'s kernel/preconditions/output, without replay. Known
/// C=N whole placement records (EytzingerLayoutTrace) is admitted before either
/// output/history allocation. Default quota one million; zero disables only quota.
/// O(N) additional owned history, no partial result on exceptions; independent
/// calls share no mutable state and retain no input reference.
/// @param sorted Borrowed nondecreasing int vector; sortedness is unchecked.
/// @param max_entries Whole placement-record limit, default1000000; zero disables quota.
/// @return Owned layout, actual assignments and exact optional record count.
/// @throws std::length_error If quota, vector capacity or representable positions are exceeded.
/// @throws std::bad_alloc If output or optional history cannot be allocated.
/// @par Example
/// @code
/// const auto t=dm::algorithms::eytzinger_layout_trace({1,3,5},3);
/// std::print("{} {}\n",t.layout[0],t.placements.back().layout_index); // 3 2
/// @endcode
/// @par References
/// - https://arxiv.org/abs/1509.05053 ; see eytzinger_layout() for the layout definition.
/// @see EytzingerLayoutTrace, eytzinger_layout()
EytzingerLayoutTrace eytzinger_layout_trace(const std::vector<int>& sorted,std::size_t max_entries=1000000);

/// @brief Retain actual node reads and chosen children during the ordinary exact-match search.
/// Shares eytzinger_search()'s equality-first kernel, input precondition, duplicate
/// choice and probe count. Optional C=P whole probe records (EytzingerSearchTrace)
/// are admitted one at a time before reading the next element. Default quota one
/// million; zero disables only quota. O(P) owned history, no input copy, replay or
/// partial result. Empty input consumes zero quota. No cache/timing claim.
/// @param layout Borrowed complete-tree BFS int vector; tree order is unchecked.
/// @param key Any int value to locate.
/// @param max_entries Whole probe-record limit, default1000000; zero disables quota.
/// @return Owned result, query metadata, actual probes and exact capture count.
/// @throws std::length_error If quota, vector capacity or representable positions are exceeded.
/// @throws std::bad_alloc If optional history cannot be allocated.
/// @par Example
/// @code
/// const auto t=dm::algorithms::eytzinger_search_trace({3,1,5},5,2);
/// std::print("{} {} {}\n",t.result.index,t.captured_entries,t.probes.back().direction); // 2 2 0
/// @endcode
/// @par References
/// - https://arxiv.org/abs/1509.05053 ; exact-match branchy variant as eytzinger_search().
/// @see EytzingerSearchTrace, eytzinger_search()
EytzingerSearchTrace eytzinger_search_trace(const std::vector<int>& layout,int key,std::size_t max_entries=1000000);

} // namespace datamunge::algorithms
