#pragma once
/// @file jump_search.hpp
/// @brief Checked integer block-search schedules and actual read/branch histories.
#include <cstddef>
#include <datamunge/algorithms/search_result.hpp>
#include <vector>
namespace datamunge::algorithms {
/// @brief Exact integer schedule for a possible sorted-array length, without allocation or reads.
/// Positive length uses floor(sqrt(n)), at least1. Empty input has all-zero fields.
/// max_probes is ceil(n/block_size)+block_size for positive n, a conservative
/// read bound including possible repeated block-end/scan inspection.
/// @par Example
/// @code
/// const auto s=dm::algorithms::jump_search_schedule(10);
/// std::print("{} {} {}\n",s.block_size,s.block_count,s.max_probes); // 3 4 7
/// @endcode
/// @see jump_search_schedule()
struct JumpSearchSchedule {
  std::size_t length{0};      ///< Admitted length fitting ptrdiff_t.
  std::size_t block_size{0};  ///< Exact floor square root, or zero for empty input.
  std::size_t block_count{0}; ///< Ceil(length/block_size), zero for empty input.
  std::size_t max_probes{0};  ///< Conservative combined block and scan read bound.
};
/// @brief Selected actual logical comparisons/reads, not machine memory traffic or time.
/// A block read performs one >=key test. A scan read performs ==key, then >key
/// only on inequality. One value is captured per logical probe; repeated visits
/// to the same array element count again. Preparation/sortedness checks excluded.
/// @par Example
/// @code
/// const auto a=dm::algorithms::jump_search_analyze({1,3,5,7,9,11,13,15,17,19},13);
/// std::print("{} {}\n",a.work.block_reads,a.work.scan_reads); // 3 1
/// @endcode
/// @see JumpSearchAnalysis
struct JumpSearchWork {
  std::size_t block_reads{0};   ///< Actual block-end logical probes and >= tests.
  std::size_t scan_reads{0};    ///< Actual linear-scan logical probes and == tests.
  std::size_t greater_tests{0}; ///< Actual >key tests after failed scan equality.
};
/// @brief Owned scalar search answer, exact schedule and producing selected work.
/// No references to caller input. result.probes equals block_reads+scan_reads.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::jump_search_analyze({1,3,5},3).result.index); // 1
/// @endcode
/// @see jump_search_analyze()
struct JumpSearchAnalysis {
  SearchResult       result;   ///< Actual first-match answer, absent index-1, logical probe count.
  JumpSearchSchedule schedule; ///< Exact schedule prepared before reading.
  JumpSearchWork     work;     ///< Actual selected logical work.
};
/// @brief Actual branch after one block-end or linear-scan read.
/// Advance/Select are block >= comparisons; Continue/Found/Passed are scan
/// equality and optional greater comparisons. No additional relation test is done.
/// @par Example
/// @code
/// const auto t=dm::algorithms::jump_search_trace({1,3,5},3);
/// std::print("{}\n",t.probes.back().action==dm::algorithms::JumpSearchAction::Found); // true
/// @endcode
/// @see JumpSearchProbe
enum class JumpSearchAction {
  Advance,  ///< Block end below key; advance to next block.
  Select,   ///< Block end >=key; start scanning from this block's beginning.
  Continue, ///< Scan value below key; continue linearly.
  Found,    ///< Scan equality succeeds; return this first occurrence.
  Passed    ///< Scan value above key; terminate absent.
};
/// @brief Owned actual inspected value, active block and resulting branch.
/// Half-open block bounds are before this inspection. Select followed by a scan
/// can move the read index backward or inspect the same endpoint twice.
/// @par Example
/// @code
/// const auto t=dm::algorithms::jump_search_trace({1,3,5,7,9,11,13,15,17,19},13);
/// std::print("{} {}\n",t.probes[2].index,t.probes[3].index); // 8 6
/// @endcode
/// @see JumpSearchTrace
struct JumpSearchProbe {
  std::size_t      block_begin{0}; ///< Active block's inclusive lower bound.
  std::size_t      block_end{0};   ///< Active block's exclusive upper bound, clipped to length.
  std::size_t      index{0};       ///< Actual inspected original array index.
  int              value{0};       ///< Actual read snapshot, not a caller reference.
  JumpSearchAction action{JumpSearchAction::Advance}; ///< Actual selected branch.
};
/// @brief Owned same-execution analysis and chronological reads, without replay/input copy.
/// History costs O(probes) storage. The optional ceiling counts complete records,
/// excluding capacities, byte counts and peak RAM. No partial result on failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::jump_search_trace({1},1);
/// std::print("{} {}\n",t.analysis.result.probes,t.probes.size()); // 2 2
/// @endcode
/// @see jump_search_trace()
struct JumpSearchTrace {
  JumpSearchAnalysis           analysis; ///< Actual result/schedule/work from this execution.
  std::vector<JumpSearchProbe> probes;   ///< Owned actual read records in order.
};
/// @brief Prepare floor-square-root blocks using division-safe integer arithmetic.
/// O(log(n+1)) word operations, constant storage, no array access or floating sqrt.
/// Length must fit ptrdiff_t before any signed result index could be produced.
/// @param length Possible array length, zero allowed.
/// @return Exact empty or positive schedule, including conservative read bound.
/// @throws std::length_error On signed-length/bound representability limits.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::jump_search_schedule(16).block_size); // 4
/// @endcode
/// @see jump_search()
JumpSearchSchedule jump_search_schedule(std::size_t length);
/// @brief Find the first equal value by block-end probes followed by an ordered block scan.
/// Nondecreasing sortedness is an unchecked precondition. Supports all int keys,
/// duplicates and empty input. Reads original block endpoints, then revisits the
/// selected block from its beginning; it is NOT a strictly forward-only access
/// pattern. Equal endpoints may be read twice. Safe clipped offsets preserve
/// the original scan/tie/probe order on ordinary lengths. O(sqrt(n)+log(n+1))
/// word work including schedule, O(1) auxiliary storage, no allocation/mutation/
/// shared state. Concurrent reads require immutable input.
/// @param a Borrowed nondecreasing int vector, possibly empty.
/// @param key Any int key.
/// @return Owned first occurrence/index/probe count, or absent index-1.
/// @throws std::length_error On signed-length/schedule representability limits.
/// @par Example
/// @code
/// const auto r=dm::algorithms::jump_search({1,3,5,7,9,11,13,15,17,19},13);
/// std::print("{} {}\n",r.index,r.probes); // 6 4
/// @endcode
/// @see jump_search_trace(), jump_search_schedule()
SearchResult jump_search(const std::vector<int>& a, int key);
/// @brief Return result, exact schedule and selected work without event allocation.
/// Same sortedness/tie/read/arithmetic contracts as jump_search(). No mutation or
/// retained references; constant query storage and schedule-plus-probe word work.
/// @param a Borrowed nondecreasing int vector.
/// @param key Any int key.
/// @return Owned scalar analysis from one execution.
/// @throws std::length_error On signed-length/schedule representability failures.
/// @par Example
/// @code
/// const auto a=dm::algorithms::jump_search_analyze({2,2,2},2);
/// std::print("{}\n",a.result.index); // 0
/// @endcode
/// @see jump_search_trace()
JumpSearchAnalysis jump_search_analyze(const std::vector<int>& a, int key);
/// @brief Capture actual block and scan inspections from the shared producing kernel.
/// Sortedness remains unchecked. Admit each record before its read, throwing on
/// enabled ceiling or vector-capacity limits. Zero disables only the optional
/// ceiling. No repeated-prefix execution, input copying, views or partial output.
/// @param a Borrowed nondecreasing int vector.
/// @param key Any int key.
/// @param max_entries Actual complete-record ceiling, default million; zero disables.
/// @return Owned actual analysis and chronological inspected values/branches.
/// @throws std::length_error On enabled ceiling or signed/size/capacity limits.
/// @throws std::bad_alloc On history allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::jump_search_trace({},0,1).probes.size()); // 0
/// @endcode
/// @see JumpSearchProbe, JumpSearchAnalysis
JumpSearchTrace
jump_search_trace(const std::vector<int>& a, int key, std::size_t max_entries = 1000000);
} // namespace datamunge::algorithms
