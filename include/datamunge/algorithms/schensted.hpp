#pragma once
/// @file schensted.hpp
/// @brief Actual row insertion, bumped values and paired insertion/recording tableaux.
#include <cstddef>
#include <datamunge/plot/diagram.hpp>
#include <vector>
/// @brief Canonical integer row insertion and owned histories.
namespace datamunge::algorithms {
/// @brief Owned insertion and recording tableaux, preserving the legacy field order.
/// @par Example
/// @code
/// const dm::algorithms::YoungTableaux value{};
/// std::print("{}\n",value.p.size()); // 0
/// @endcode
struct YoungTableaux {
  std::vector<std::vector<int>>
      p; ///< Actual insertion values; weakly increasing rows and strictly increasing columns.
  std::vector<std::vector<int>> q; ///< Actual one-based insertion-step labels, same row shape as p.
};
/// @brief Input, actual row-scan and optional row-copy bounds; zero disables a cap.
/// @par Example
/// @code
/// const dm::algorithms::SchenstedOptions value{};
/// std::print("{}\n",value.max_values); // 100000
/// @endcode
struct SchenstedOptions {
  std::size_t max_values{
      100000}; ///< Complete integer sequence length, also bounded by int label capacity.
  std::size_t max_comparisons{10000000}; ///< Actual value<=carried comparisons during row scans.
  std::size_t max_trace_entries{
      1000000}; ///< Actual before/after row scalars plus one header per row decision.
};
/// @brief One actual row replacement or terminal append during an insertion.
/// @par Example
/// @code
/// const dm::algorithms::SchenstedStep value{};
/// std::print("{}\n",value.bumped); // false
/// @endcode
struct SchenstedStep {
  std::size_t input_index{}; ///< Original zero-based insertion event.
  std::size_t row{};         ///< Actual zero-based row receiving the carried value.
  std::size_t column{};      ///< Actual first-greater slot or append position.
  int         carried{};     ///< Actual value entering this row.
  bool        bumped{};      ///< True when an existing greater value was replaced.
  int outgoing{}; ///< Actual bumped value; zero sentinel on terminal append, with bumped=false.
  std::size_t      comparisons{}; ///< Actual row element comparisons for this decision.
  std::vector<int> before;        ///< Actual row before mutation; empty for a new row.
  std::vector<int> after;         ///< Actual row after its replacement/append.
};
/// @brief Owned actual tableau pair and selected insertion work.
/// @par Example
/// @code
/// const dm::algorithms::SchenstedRun value{};
/// std::print("{}\n",value.bumps); // 0
/// @endcode
struct SchenstedRun {
  YoungTableaux tableau;     ///< Actual completed P and Q, with matching partition shape.
  std::size_t comparisons{}; ///< Actual row value comparisons, including the first failing <= test.
  std::size_t bumps{};       ///< Actual replaced values carried into a later row.
  std::size_t appends{};     ///< Actual terminal appends, exactly one per source value.
  std::vector<SchenstedStep>
      steps; ///< Actual optional chronological row decisions, never replayed.
};
/// @brief Build the native row-insertion tableau pair.
/// Inserts source values in order. Within each row, replace the first value strictly greater than the carried value and bump that old value to the next row; append when no value is greater, recording step+1 at the new Q box. Preserves legacy outputs for all ordinary integer sequences. Distinct inputs give standard tableaux and first-row length equal to strict LIS length; with duplicates P rows are weakly increasing and first-row length equals longest nondecreasing subsequence length instead. Empty inputs give empty P/Q. Direct row scanning costs O(N^2) worst-case, O(N) intrinsic data; actual before/after row histories can cost more and are independently bounded. Selected comparisons count actual row element tests, not allocation or layout. Captured logical units are before_row_size+after_row_size+1, admitted before mutation/copies. Inputs borrowed unchanged, outputs own all values and labels.
/// @param values Complete integer sequence, duplicates allowed.
/// @param options Input, actual comparison and row-history policies.
/// @return Owned actual P/Q and observed scans/bumps/appends, plus optional real row histories.
/// @throws std::length_error Input label, comparison, history or container policy/capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::schensted_run({3,1,2});
/// std::print("{} {}\n",r.tableau.p[0],r.tableau.p[1]); // [1, 2] [3]
/// @endcode
/// @par References
/// - Schensted (1961), Longest Increasing and Decreasing Subsequences. https://www.cambridge.org/core/journals/canadian-journal-of-mathematics/article/longest-increasing-and-decreasing-subsequences/B5098D9BC8B226C575402B971852C05E
/// @see rsk_insert(), schensted_diagram()
SchenstedRun schensted_run(const std::vector<int>& values, const SchenstedOptions& options = {});
/// @brief Build the native row-insertion tableau pair with actual bumped-row histories.
/// Inserts source values in order. Within each row, replace the first value strictly greater than the carried value and bump that old value to the next row; append when no value is greater, recording step+1 at the new Q box. Preserves legacy outputs for all ordinary integer sequences. Distinct inputs give standard tableaux and first-row length equal to strict LIS length; with duplicates P rows are weakly increasing and first-row length equals longest nondecreasing subsequence length instead. Empty inputs give empty P/Q. Direct row scanning costs O(N^2) worst-case, O(N) intrinsic data; actual before/after row histories can cost more and are independently bounded. Selected comparisons count actual row element tests, not allocation or layout. Captured logical units are before_row_size+after_row_size+1, admitted before mutation/copies. Inputs borrowed unchanged, outputs own all values and labels.
/// @param values Complete integer sequence, duplicates allowed.
/// @param options Input, actual comparison and row-history policies.
/// @return Owned actual P/Q and observed scans/bumps/appends, plus optional real row histories.
/// @throws std::length_error Input label, comparison, history or container policy/capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::schensted_trace({3,1,2});
/// std::print("{} {}\n",r.tableau.p[0],r.tableau.p[1]); // [1, 2] [3]
/// @endcode
/// @par References
/// - Schensted (1961), Longest Increasing and Decreasing Subsequences. https://www.cambridge.org/core/journals/canadian-journal-of-mathematics/article/longest-increasing-and-decreasing-subsequences/B5098D9BC8B226C575402B971852C05E
/// @see rsk_insert(), schensted_diagram()
SchenstedRun schensted_trace(const std::vector<int>& values, const SchenstedOptions& options = {});
/// @brief Return the canonical tableau pair through the preserved row-insertion signature.
/// Uses the same bounded producer, preserving ordinary input order/tie behavior. A sequence with repeated values is supported as semistandard insertion; it is not claimed to be a permutation bijection.
/// @param permutation Complete input integers, distinct for the ordinary permutation correspondence.
/// @return Owned actual insertion/recording tableau pair.
/// @throws std::length_error Default source/work policies or int recording-label capacity exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::rsk_insert({3,1,2}).p[0]); // [1, 2]
/// @endcode
/// @par References
/// - Schensted (1961), Longest Increasing and Decreasing Subsequences. https://www.cambridge.org/core/journals/canadian-journal-of-mathematics/article/longest-increasing-and-decreasing-subsequences/B5098D9BC8B226C575402B971852C05E
/// @see schensted_run()
YoungTableaux rsk_insert(const std::vector<int>& permutation);
/// @brief Draw both actual tableau grids with one native editable concept diagram.
/// Validates identical partition-shaped P/Q row lengths, nonempty rows and the complete cell policy before drawing. Labels are supplied actual integer entries. This validates display shape, not a caller-edited insertion proof; empty tableaux receive an explicit empty label. Time/storage O(cells+label text), inputs unchanged, geometry owned.
/// @param tableau Actual pair or a structurally valid edited pair.
/// @param max_cells Combined P+Q cell bound, default10000; zero disables.
/// @return Owned editable native diagram showing actual P and Q side by side.
/// @throws std::invalid_argument Different or nonpartition row shapes.
/// @throws std::length_error Combined cell count or plot policy exceeded.
/// @par Example
/// @code
/// auto d=dm::algorithms::schensted_diagram(dm::algorithms::rsk_insert({3,1,2}));
/// d.save_svg("schensted_example.svg");
/// @endcode
/// @par References
/// - Schensted (1961), Longest Increasing and Decreasing Subsequences. https://www.cambridge.org/core/journals/canadian-journal-of-mathematics/article/longest-increasing-and-decreasing-subsequences/B5098D9BC8B226C575402B971852C05E
/// @see schensted_trace(), graphics::Diagram
graphics::Diagram schensted_diagram(const YoungTableaux& tableau, std::size_t max_cells = 10000);
} // namespace datamunge::algorithms
