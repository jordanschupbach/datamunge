#pragma once
/// @file shortest_common_supersequence.hpp
/// @brief Checked actual LCS cells and tie-deterministic shortest-supersequence backtracking.
#include <cstddef>
#include <vector>
/// @brief Canonical integer sequence combination.
namespace datamunge::algorithms {
/// @brief Input, intrinsic table and optional retained-history bounds; zero disables a cap.
/// @par Example
/// @code
/// const dm::algorithms::SCSOptions value{};
/// std::print("{}\n",value.max_cells); // 1000000
/// @endcode
struct SCSOptions {
  std::size_t max_length{10000};  ///< Complete length of either borrowed sequence.
  std::size_t max_cells{1000000}; ///< Complete intrinsic (n+1)*(m+1) LCS cells.
  std::size_t max_trace_entries{
      1000000}; ///< Captured input scalars, table cells and actual backtrack records.
};
/// @brief Actual branch used to emit one reverse-order supersequence value.
/// @par Example
/// @code
/// const auto value=dm::algorithms::SCSChoice::Shared;
/// std::print("{}\n",static_cast<int>(value)); // 0
/// @endcode
enum class SCSChoice {
  Shared, ///< Equal values consume both sequences.
  First,  ///< Emit from the first sequence; wins a nonmatching DP tie.
  Second  ///< Emit from the second sequence.
};
/// @brief One actual backtrack emission before reversing the output.
/// @par Example
/// @code
/// const dm::algorithms::SCSStep value{};
/// std::print("{}\n",value.value); // 0
/// @endcode
struct SCSStep {
  std::size_t first_size{};  ///< Actual remaining first prefix length before this step.
  std::size_t second_size{}; ///< Actual remaining second prefix length before this step.
  SCSChoice   choice{};      ///< Actual shared/first/second producing branch.
  int         value{};       ///< Actual emitted integer value.
};
/// @brief Owned actual shortest supersequence and selected DP/backtrack work.
/// @par Example
/// @code
/// const dm::algorithms::SCSRun value{};
/// std::print("{}\n",value.lcs_length); // 0
/// @endcode
struct SCSRun {
  std::vector<int> sequence;     ///< Actual final forward-order shortest supersequence.
  std::size_t      lcs_length{}; ///< Actual final longest-common-subsequence length.
  std::size_t
      equality_tests{}; ///< Actual input-value equality checks during table filling and backtracking.
  std::size_t
      score_comparisons{}; ///< Actual upper/left cell comparisons for unequal DP/backtrack choices.
  std::vector<int> first;  ///< Owned original first input, only captured by trace.
  std::vector<int> second; ///< Owned original second input, only captured by trace.
  std::vector<std::vector<int>>
                       table; ///< Actual optional complete LCS table, including zero border.
  std::vector<SCSStep> steps; ///< Actual optional reverse-emission history, not a report replay.
};
/// @brief Construct one actual shortest common supersequence.
/// Fills the canonical LCS-length table and backtracks it once. Equal integer values are emitted once while consuming both inputs. On an unequal upper/left score tie, emit the first input value, preserving historical output ordering. Empty inputs are supported; output contains each input as a subsequence and length n+m-LCS. Time O(n*m+n+m), working O((n+1)*(m+1)); result O(n+m). Trace owns original inputs, the actual complete table and reverse-emission records. Captured units are n+m+table_cells+actual_backtrack_records, admitted before their copies/emissions; zero disables only a policy. Complete intrinsic table geometry is checked before allocation, even for ordinary runs. Inputs borrowed unchanged; independent calls share no state.
/// @param first Borrowed complete first integer sequence.
/// @param second Borrowed complete second sequence.
/// @param options Input, intrinsic table and optional capture policies.
/// @return Owned actual shortest sequence, LCS length, selected work and optional histories.
/// @throws std::length_error Input/int-index, table, output or history capacity/policy exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::shortest_common_supersequence_run({1,2},{2,3});
/// std::print("{}\n",r.sequence); // [1, 2, 3]
/// @endcode
/// @par References
/// - Hirschberg (1975), A linear space algorithm for computing maximal common subsequences; full-table LCS variant here. https://doi.org/10.1145/360825.360861
/// @see shortest_common_supersequence()
SCSRun shortest_common_supersequence_run(
    const std::vector<int>& first,
    const std::vector<int>& second,
    const SCSOptions&       options = {});
/// @brief Construct one actual shortest common supersequence with its native table and backtrack history.
/// Fills the canonical LCS-length table and backtracks it once. Equal integer values are emitted once while consuming both inputs. On an unequal upper/left score tie, emit the first input value, preserving historical output ordering. Empty inputs are supported; output contains each input as a subsequence and length n+m-LCS. Time O(n*m+n+m), working O((n+1)*(m+1)); result O(n+m). Trace owns original inputs, the actual complete table and reverse-emission records. Captured units are n+m+table_cells+actual_backtrack_records, admitted before their copies/emissions; zero disables only a policy. Complete intrinsic table geometry is checked before allocation, even for ordinary runs. Inputs borrowed unchanged; independent calls share no state.
/// @param first Borrowed complete first integer sequence.
/// @param second Borrowed complete second sequence.
/// @param options Input, intrinsic table and optional capture policies.
/// @return Owned actual shortest sequence, LCS length, selected work and optional histories.
/// @throws std::length_error Input/int-index, table, output or history capacity/policy exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::shortest_common_supersequence_trace({1,2},{2,3});
/// std::print("{}\n",r.sequence); // [1, 2, 3]
/// @endcode
/// @par References
/// - Hirschberg (1975), A linear space algorithm for computing maximal common subsequences; full-table LCS variant here. https://doi.org/10.1145/360825.360861
/// @see shortest_common_supersequence()
SCSRun shortest_common_supersequence_trace(
    const std::vector<int>& first,
    const std::vector<int>& second,
    const SCSOptions&       options = {});
/// @brief Return the checked canonical shortest supersequence through the preserved signature.
/// Uses the same table/backtrack producer and default bounds, preserving ordinary tie outputs. No optional captured input/table/history copies are retained.
/// @param a Complete first input.
/// @param b Complete second input.
/// @return Owned forward-order shortest common supersequence.
/// @throws std::length_error Default intrinsic or output policy/capacity exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::shortest_common_supersequence({1,2},{2,3})); // [1, 2, 3]
/// @endcode
/// @par References
/// - Hirschberg (1975), A linear space algorithm for computing maximal common subsequences; full-table LCS variant here. https://doi.org/10.1145/360825.360861
/// @see shortest_common_supersequence_run()
std::vector<int> shortest_common_supersequence(
    const std::vector<int>& a,
    const std::vector<int>& b);
} // namespace datamunge::algorithms
