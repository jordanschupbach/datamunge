#pragma once

/// @file warnsdorff.hpp
/// @brief Observable deterministic knight-tour attempts and rectangular-grid readout diagnostics.
#include <datamunge/algorithms/tour_search.hpp>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/plot/plot.hpp>
#include <string>

namespace datamunge::algorithms {

/// @brief Owned Warnsdorff attempt, retaining a legal partial path even when the heuristic gets stuck.
///
/// completed means every cell was visited, not that a closing move was made.
/// closed additionally checks the final-to-first knight edge. trace records
/// actual decisions: step, row, column, legal_moves, minimum_degree, ties,
/// selected_row, selected_column and action (move/stuck). A completed path
/// has one fewer decision than vertices; a failed path includes its final
/// stuck decision. path_table holds step, row, column and row_major_index.
/// No input data are retained; fields own their contents.
/// @par Example
/// @code
/// const auto attempt=dm::algorithms::warnsdorff_attempt(8,8,3,3);
/// const auto visited=attempt.path.size(); // 62, retained despite failure
/// const bool complete=attempt.completed; // false
/// @endcode
/// @see warnsdorff_attempt(), plot_knight_tour()
struct WarnsdorffAttempt {
  int rows{0}; ///< Board height; zero only in a default-constructed result.
  int columns{0}; ///< Board width.
  bool completed{false}; ///< Whether the actual path covers all cells.
  bool closed{false}; ///< Whether a complete path also has a legal final-to-first knight move.
  std::vector<std::pair<int,int>> path; ///< Unique visited (row,column) cells, beginning at the requested start.
  dstruct::DataFrame trace; ///< Owned actual minimum-degree decisions, including a terminal dead end when present.
  dstruct::DataFrame path_table; ///< Owned path coordinates and row-major indices, in visit order.
  std::size_t candidate_probes{0}; ///< All eight candidate-direction checks per decision, including invalid destinations.
  std::size_t degree_probes{0}; ///< Eight onward-direction checks per legal unvisited candidate.
};

/// @brief Run the canonical minimum-onward-degree heuristic with a full trace and a retained partial path.
///
/// The current cell is already visited. Among its legal unvisited neighbours,
/// choose the lexicographically smallest (onward_degree,row,column), where
/// degree counts unvisited legal neighbours. There is no random tie breaking,
/// lookahead beyond degree, restart or backtracking. This is a heuristic for a
/// Hamiltonian path, not an impossibility test or a closed-tour solver.
/// O(N) time/storage for N=rows*columns; no shared mutable state.
/// @param rows Positive board height.
/// @param columns Positive board width.
/// @param start_row Zero-based start row, default zero.
/// @param start_column Zero-based start column, default zero.
/// @param max_cells Positive board allocation limit, default 1000000; checked before allocation.
/// @return Owned attempt; a nonempty valid prefix remains available on failure.
/// @throws std::invalid_argument If dimensions, start or max_cells are invalid.
/// @throws std::length_error If the cell product exceeds the limit or supported storage range.
/// @par Example
/// @code
/// const auto attempt=dm::algorithms::warnsdorff_attempt(8,8);
/// const auto visited=attempt.path.size(); // 64
/// const bool complete=attempt.completed; // true
/// @endcode
/// @par References
/// - S. L. Marateck (2008), https://arxiv.org/abs/0803.4321;
///   coordinate ties here differ from that paper's move-order permutations.
/// @see warnsdorff_knight_tour(), warnsdorff_start_profile()
WarnsdorffAttempt warnsdorff_attempt(int rows,int columns,int start_row=0,int start_column=0,
                                    std::size_t max_cells=1000000);

/// @brief Attempt the same deterministic rule from every cell of a rectangular board.
///
/// Rows are start-row-major; each attempt starts with a fresh unvisited board.
/// Results measure this exact tie policy, not tour existence or success rates
/// of randomised policies. O(N^2) time and O(N) temporary/result storage.
/// @param rows Positive board height.
/// @param columns Positive board width.
/// @param max_starts Positive maximum number of starts/cells, default 256.
/// @return Owned start_row, start_column, colour, visited_cells, completed,
/// closed, candidate_probes and degree_probes columns; colours are (row+column) modulo two.
/// @throws std::invalid_argument If dimensions or the limit are invalid.
/// @throws std::length_error If the start count exceeds max_starts or supported storage.
/// @par Example
/// @code
/// const auto starts=dm::algorithms::warnsdorff_start_profile(8,8);
/// const auto count=starts.nrows(); // 64
/// @endcode
/// @see warnsdorff_attempt()
dstruct::DataFrame warnsdorff_start_profile(int rows,int columns,std::size_t max_starts=256);

/// @brief Draw a legal complete or partial knight path over a chequered board.
///
/// Coordinates are column on x and row on y (rows increase upwards in this
/// Cartesian display). Start/end markers are green/red; the path is blue.
/// Unvisited cells are pale yellow.
/// Optional visit numbers are zero-based. Inputs are borrowed and unchanged;
/// the returned plot owns its geometry. O(N+L log L) time and O(N+L) plot
/// storage for N board cells and L path vertices, including validation.
/// @param attempt Valid dimensions and a nonempty unique legal knight path; flags are not trusted for validation.
/// @param annotate Draw visit numbers, default true.
/// @param max_cells Positive maximum displayed board size, default 4096.
/// @return Owned board plot; caller can set title, canvas size and axis labels.
/// @throws std::invalid_argument If dimensions, path or the limit are invalid.
/// @throws std::length_error If the displayed board exceeds max_cells.
/// @par Example
/// @code
/// auto plot=dm::algorithms::plot_knight_tour(dm::algorithms::warnsdorff_attempt(8,8));
/// plot.title("A deterministic knight path");
/// @endcode
/// @see warnsdorff_attempt()
graphics::RPlot plot_knight_tour(const WarnsdorffAttempt& attempt,bool annotate=true,
                                  std::size_t max_cells=4096);

/// @brief Read one numeric grid row in row-major and complete knight-tour order.
///
/// predictors defines the original row-major grid order, with exactly rows*columns
/// unique numeric names. Validates the actual path as a full permutation rather
/// than trusting completed metadata. No interpolation, averaging or value change
/// occurs. O(N log N) validation and O(N) output storage; inputs are borrowed.
/// @param data Frame whose records are numeric grids.
/// @param image_row Zero-based record index.
/// @param predictors Ordered row-major grid column names, all required finite/non-null.
/// @param attempt Complete legal knight path covering the declared rectangle.
/// @return Owned step, tour_row, tour_column, row_major_value and tour_value columns.
/// @throws std::invalid_argument If path, shape, names, types or cells are invalid.
/// @throws std::out_of_range If image_row or a required column is absent.
/// @throws std::length_error If the declared board exceeds supported storage.
/// @par Example
/// @code
/// auto digits=dm::datasets::digits(); auto columns=digits.columns();
/// columns.erase(std::remove(columns.begin(),columns.end(),"digit"),columns.end());
/// const auto values=dm::algorithms::knight_tour_grid_values(
///     digits,0,columns,dm::algorithms::warnsdorff_attempt(8,8));
/// const auto pixels=values.nrows(); // 64
/// @endcode
/// @see knight_tour_grid_summary()
dstruct::DataFrame knight_tour_grid_values(const dstruct::DataFrame& data,std::size_t image_row,
                                            const std::vector<std::string>& predictors,
                                            const WarnsdorffAttempt& attempt);

/// @brief Compare sums and adjacent-value total variation under row-major and complete knight-tour readout.
///
/// Returns one row per input grid. Sums are invariant in exact arithmetic;
/// sum_difference is tour_sum minus value_sum and may reflect floating roundoff.
/// Total variation sums absolute differences between successive values, without
/// a closing edge. The geometric heuristic does not optimise these data values.
/// Rejects null/non-finite cells and arithmetic overflow. Empty input returns
/// an empty table with the same schema. O(N log N+n*N) time and O(N+n) storage
/// for n records, including path/name validation. No shared mutable state.
/// @param data Numeric-grid records, borrowed and unchanged.
/// @param predictors Unique row-major grid columns, exactly rows*columns names.
/// @param attempt Complete legal knight path; actual coverage is checked.
/// @return Owned image_row, value_sum, tour_sum, sum_difference,
/// row_major_variation and tour_variation columns.
/// @throws std::invalid_argument If path, schema, types or cells are invalid.
/// @throws std::out_of_range If a predictor column is absent.
/// @throws std::length_error If the declared board exceeds supported storage.
/// @throws std::overflow_error If sums or total variations become non-finite.
/// @par Example
/// @code
/// auto digits=dm::datasets::digits(); auto columns=digits.columns();
/// columns.erase(std::remove(columns.begin(),columns.end(),"digit"),columns.end());
/// const auto summary=dm::algorithms::knight_tour_grid_summary(
///     digits,columns,dm::algorithms::warnsdorff_attempt(8,8));
/// const auto images=summary.nrows(); // 1797
/// @endcode
/// @see knight_tour_grid_values(), warnsdorff_attempt()
dstruct::DataFrame knight_tour_grid_summary(const dstruct::DataFrame& data,
                                             const std::vector<std::string>& predictors,
                                             const WarnsdorffAttempt& attempt);

} // namespace datamunge::algorithms
