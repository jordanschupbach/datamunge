#pragma once
/// @file longest_common_substring.hpp
/// @brief Rolling-row common-suffix DP with actual selected occurrence identities and owned trace.
#include <cstddef>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual common-suffix DP work, excluding guards, allocations and copies.
/// @par Example
/// @code
/// auto a=dm::algorithms::lcsubstring_analyze({1,2},{2});
/// std::print("{}\n",a.work.cells); // 2
/// @endcode
/// @par References
/// - NIST DADS, Longest Common Substring, https://xlinux.nist.gov/dads/HTML/longestCommonSubstring.html; this implementation uses rolling common-suffix rows.
/// @see lcsubstring_analyze(), lcsubstring_trace()
struct LCSubstringWork {
  std::size_t cells{0};   ///< Actual nonboundary comparisons, n*m.
  std::size_t matches{0}; ///< Actual equal-element diagonal extensions.
  std::size_t resets{0};  ///< Actual unequal-element zero assignments.
  std::size_t records{0}; ///< Actual strict best-length improvements.
};
/// @brief Owned longest contiguous shared run and the actual first row-major maximizing occurrence.
/// @par Example
/// @code
/// auto a=dm::algorithms::lcsubstring_analyze({1,2,1},{1});
/// std::print("{} {}\n",a.a_start,a.b_start); // 0 0
/// @endcode
/// @par References
/// - NIST DADS, Longest Common Substring, https://xlinux.nist.gov/dads/HTML/longestCommonSubstring.html; this implementation uses rolling common-suffix rows.
/// @see lcsubstring_analyze(), lcsubstring_trace()
struct LCSubstringAnalysis {
  std::vector<int> sequence;   ///< Actual selected contiguous values, empty if no match.
  std::size_t      a_start{0}; ///< Actual zero-based start in a; zero for empty output.
  std::size_t      b_start{0}; ///< Actual zero-based start in b; zero for empty output.
  std::size_t      length{0};  ///< Actual selected common contiguous length.
  LCSubstringWork  work;       ///< Actual producing DP operations.
};
/// @brief One actual cell write and its strict running-maximum decision.
/// @par Example
/// @code
/// auto t=dm::algorithms::lcsubstring_trace({1},{1});
/// std::print("{}\n",t.cells.front().value); // 1
/// @endcode
/// @par References
/// - NIST DADS, Longest Common Substring, https://xlinux.nist.gov/dads/HTML/longestCommonSubstring.html; this implementation uses rolling common-suffix rows.
/// @see lcsubstring_analyze(), lcsubstring_trace()
struct LCSubstringCell {
  std::size_t i;     ///< Actual one-based row prefix length.
  std::size_t j;     ///< Actual one-based column prefix length.
  bool        equal; ///< Actual input equality comparison.
  std::size_t
      diagonal;      ///< Actual previous-row previous-column value read on equality, zero on reset.
  std::size_t value; ///< Actual written common-suffix length.
  bool        new_best; ///< Actual strict best-length improvement; ties do not replace it.
};
/// @brief Owned original inputs, actual completed rows and all producing cell writes.
/// @par Example
/// @code
/// auto t=dm::algorithms::lcsubstring_trace({1,2},{2});
/// std::print("{} {}\n",t.rows.size(),t.cells.size()); // 3 2
/// @endcode
/// @par References
/// - NIST DADS, Longest Common Substring, https://xlinux.nist.gov/dads/HTML/longestCommonSubstring.html; this implementation uses rolling common-suffix rows.
/// @see lcsubstring_analyze(), lcsubstring_trace()
struct LCSubstringTrace {
  std::vector<int> a; ///< Original first sequence, unchanged and independently owned.
  std::vector<int> b; ///< Original second sequence, unchanged and independently owned.
  std::vector<std::vector<std::size_t>>
                               rows;  ///< Actual initial zero row then each committed DP row.
  std::vector<LCSubstringCell> cells; ///< Actual chronological cell writes, without table replay.
  LCSubstringAnalysis analysis; ///< Actual selected occurrence and work from the same producer.
  std::size_t         captured_entries{
      0}; ///< Logical original elements, row cells and cell records admitted before capture.
};
/// @brief Find a longest contiguous common integer run, preserving first row-major strict-maximum ties.
/// @param a First arbitrary-int sequence, borrowed unchanged.
/// @param b Second arbitrary-int sequence, borrowed unchanged; fixes rolling-row orientation.
/// @return Selected contiguous run, empty when there are no equal elements.
/// @note O(n*m) time, O(m) auxiliary row storage plus output; no subsequence skipping.
/// Empty inputs are valid. Complete size/count geometry is checked before allocation.
/// @throws std::length_error For unrepresentable row, work or iterator geometry.
/// @code
/// auto s=dm::algorithms::longest_common_substring({1,2,3},{0,2,3});
/// std::print("{}\n",s.size()); // 2
/// @endcode
std::vector<int> longest_common_substring(const std::vector<int>& a, const std::vector<int>& b);
/// @brief Run the same rolling-row producer and retain its actual maximizing start indices and work.
/// @param a First arbitrary-int sequence.
/// @param b Second arbitrary-int sequence.
/// @return Owned selected values, actual occurrence positions and selected counts.
/// @throws std::length_error For unrepresentable complete geometry.
/// @code
/// auto a=dm::algorithms::lcsubstring_analyze({2,3,1},{2,3});
/// std::print("{} {} {}\n",a.a_start,a.b_start,a.length); // 0 0 2
/// @endcode
LCSubstringAnalysis lcsubstring_analyze(const std::vector<int>& a, const std::vector<int>& b);
/// @brief Capture actual cell writes and committed rolling rows from one complete producer execution.
/// @param a First arbitrary-int sequence.
/// @param b Second arbitrary-int sequence.
/// @param max_entries Optional complete logical capture quota, default one million; zero disables it.
/// @return Owned inputs, actual rows/cells and same-run result, never prefix reconstruction.
/// @throws std::length_error Before allocation/capture if n+m+(n+1)*(m+1)+n*m exceeds the cap or geometry.
/// @note Counts logical elements/records, not fields, row headers, capacities, output/runtime copies or bytes.
/// Unlike ordinary rolling storage, complete capture uses O(n*m+n+m) storage.
/// @code
/// auto t=dm::algorithms::lcsubstring_trace({1,2},{2});
/// std::print("{}\n",t.captured_entries); // 11
/// @endcode
LCSubstringTrace lcsubstring_trace(
    const std::vector<int>& a,
    const std::vector<int>& b,
    std::size_t             max_entries = 1000000);
/// @brief Convert each string byte to an integer0..255, preserving NULs and avoiding signed-char ambiguity.
/// @param text Original arbitrary bytes, not decoded Unicode characters.
/// @return Owned unsigned byte values in original order.
/// @code
/// auto bytes=dm::algorithms::lcsubstring_bytes("ABC");
/// std::print("{}\n",bytes.front()); // 65
/// @endcode
std::vector<int> lcsubstring_bytes(const std::string& text);
} // namespace datamunge::algorithms
