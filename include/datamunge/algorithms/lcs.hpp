#pragma once
/// @file lcs.hpp
/// @brief Full-table integer-sequence LCS with owned producing table and traceback evidence.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual selected full-table and traceback operations; excludes validation/allocation/copies.
/// @par Example
/// @code
/// const auto a=dm::algorithms::lcs_analyze({1,2},{2});
/// std::print("{}\n",a.work.cells); // 2
/// @endcode
/// @see LCSAnalysis
struct LCSWork {
  std::size_t cells{0};           ///< Actual nonboundary DP cells evaluated, n*m.
  std::size_t traceback_steps{0}; ///< Actual moves until either prefix index reaches zero.
  std::size_t matches{0};         ///< Actual matching diagonal moves/output elements.
};
/// @brief Actual traceback direction; ties take Up, preserving historical output selection.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lcs_trace({1},{1});
/// std::print("{}\n",t.steps[0].move==dm::algorithms::LCSMove::Match); // true
/// @endcode
/// @see LCSStep
enum class LCSMove {
  Match, ///< Matching diagonal consumes both equal elements.
  Up,    ///< Drop the last a prefix element; preferred on unequal-value table ties.
  Left   ///< Drop the last b prefix element when left length is strictly larger.
};
/// @brief Actual prefix coordinate and direction chosen during one producing traceback move.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lcs_trace({1,2},{2});
/// std::print("{} {}\n",t.steps[0].i,t.steps[0].j); // 2 1
/// @endcode
/// @see LCSTrace
struct LCSStep {
  std::size_t i{0};                 ///< Actual a prefix length before move.
  std::size_t j{0};                 ///< Actual b prefix length before move.
  LCSMove     move{LCSMove::Match}; ///< Actual producing direction.
};
/// @brief Owned actual selected optimal subsequence and same-execution work.
/// @par Example
/// @code
/// const auto a=dm::algorithms::lcs_analyze({1,2},{2});
/// std::print("{}\n",a.sequence[0]); // 2
/// @endcode
/// @see lcs_analyze()
struct LCSAnalysis {
  std::vector<int> sequence; ///< Owned actual optimal subsequence in forward input order.
  LCSWork          work;     ///< Actual selected DP/traceback work.
};
/// @brief Owned original inputs, actual producing full DP table and complete actual traceback moves.
/// Table contains (n+1)*(m+1) cells including zero boundaries. Steps stop when
/// either prefix is empty, so the implicit remaining boundary moves are absent.
/// Complete conservative capture admission counts n+m+table_cells+n+m before
/// allocation/evaluation: input records, table cells and at most n+m moves.
/// captured_entries reports the actual logical count using steps.size instead
/// of that admitted upper bound. Excludes fields/capacities/output/runtime
/// copies and bytes/RAM. Default million-entry cap; zero disables it only.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lcs_trace({1},{1});
/// std::print("{} {}\n",t.table[1][1],t.steps.size()); // 1 1
/// @endcode
/// @see lcs_trace()
struct LCSTrace {
  std::vector<int>              a;     ///< Owned original first integer sequence.
  std::vector<int>              b;     ///< Owned original second integer sequence.
  std::vector<std::vector<int>> table; ///< Actual producing prefix-length DP table.
  std::vector<LCSStep> steps;    ///< Actual chronological traceback moves, corner toward boundary.
  LCSAnalysis          analysis; ///< Owned actual forward result and selected work.
  std::size_t          captured_entries{0}; ///< Actual input/table/move logical records.
};
/// @brief Return one longest common integer subsequence via a full prefix-length dynamic program.
/// Equal elements take a matching diagonal; unequal ties choose the upper cell.
/// Empty inputs are valid. Elements are arbitrary ints, not decoded Unicode.
/// O((n+1)*(m+1)) time/space plus output; geometry/int length checked first.
/// No input mutation or shared state. All results own their elements.
/// @param a Borrowed original first integer sequence.
/// @param b Borrowed original second integer sequence.
/// @return Owned actual deterministically selected longest common subsequence.
/// @throws std::length_error On table geometry/container limits or possible int-length overflow.
/// @throws std::bad_alloc On table/output allocation failure.
/// @par Example
/// @code
/// const auto a=dm::algorithms::longest_common_subsequence({1,2},{2});
/// std::print("{}\n",a[0]); // 2
/// @endcode
/// @par References
/// - Hirschberg(1975), A Linear Space Algorithm for Computing Maximal Common Subsequences, https://doi.org/10.1145/360825.360861. This API retains the full-table baseline, not that paper's linear-space construction.
/// @see lcs_trace(), lcs_analyze()
std::vector<int> longest_common_subsequence(const std::vector<int>& a, const std::vector<int>& b);
/// @brief Return the actual full-table LCS result and selected work without retaining its table or moves.
/// Same tie, integer, empty and geometry contracts as longest_common_subsequence().
/// The complete DP table is still used at runtime; count-only is not linear space.
/// @param a Borrowed original first sequence.
/// @param b Borrowed original second sequence.
/// @return Owned actual result and selected real cell/move counts.
/// @throws std::length_error On complete table/int geometry.
/// @throws std::bad_alloc On runtime/output allocation failure.
/// @par Example
/// @code
/// const auto a=dm::algorithms::lcs_analyze({1,2},{2});
/// std::print("{} {}\n",a.sequence.size(),a.work.cells); // 1 2
/// @endcode
/// @par References
/// - Hirschberg(1975), https://doi.org/10.1145/360825.360861; full-table baseline here.
/// @see longest_common_subsequence(), lcs_trace()
LCSAnalysis lcs_analyze(const std::vector<int>& a, const std::vector<int>& b);
/// @brief Capture the actual producing LCS table and moves with preadmitted complete capture geometry.
/// Same algorithm/ties as ordinary/count-only calls; no independently rebuilt
/// display table or traceback replay. Conservative admission uses n+m+table_cells+n+m
/// before copies/computation, even when the actual traceback will be shorter.
/// @param a Borrowed original first sequence.
/// @param b Borrowed original second sequence.
/// @param max_entries Conservative logical admission cap, default1000000;0 disables only cap.
/// @return Owned inputs, actual table/moves/result/work and actual logical record count.
/// @throws std::length_error On full table/int/capture geometry or conservative quota.
/// @throws std::bad_alloc On runtime/capture allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lcs_trace({1},{1},8);
/// std::print("{}\n",t.captured_entries); // 7
/// @endcode
/// @par References
/// - Hirschberg(1975), https://doi.org/10.1145/360825.360861; full-table baseline here.
/// @see LCSTrace, lcs_analyze()
LCSTrace
lcs_trace(const std::vector<int>& a, const std::vector<int>& b, std::size_t max_entries = 1000000);
} // namespace datamunge::algorithms
