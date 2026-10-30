#pragma once

/// @file sudoku.hpp
/// @brief Sudoku as a binary constraint-satisfaction problem: domains, peer arcs, propagation
///        with AC-3, an exact solver, and a board picture with pencil marks.
///
/// A puzzle is an 81-character string read row by row: `1`..`9` for a given digit and `.` or
/// `0` for a blank (the format of Norvig's collections and of datasets::sudoku()). Cells are
/// numbered 0..80 the same way, so cell `c` is in row `c / 9` and column `c % 9`.
///
/// @code
/// namespace sudoku = dm::algorithms::sudoku;
/// const std::string puzzle = dm::datasets::sudoku().string_at("puzzle", 0);
/// const auto propagated = sudoku::propagate(puzzle);         // AC-3 on the 1620 peer arcs
/// std::print("{}\n", sudoku::to_string(propagated.domains));  // solved cells, '.' elsewhere
/// sudoku::board(propagated.domains, puzzle).save_svg("board.svg");
/// @endcode

#include <datamunge/algorithms/arc_consistency.hpp>
#include <datamunge/plot/plot.hpp>

#include <optional>
#include <string>
#include <utility>
#include <vector>

/// @brief Sudoku encoded as a constraint-satisfaction problem over 81 cells.
namespace datamunge::algorithms::sudoku {

/// @brief The starting domain of each cell: `{d}` for a given digit d, `{1, ..., 9}` for a blank.
/// @param puzzle 81 characters, `1`-`9` given, `.` or `0` blank.
/// @return 81 sorted domains.
/// @throws std::invalid_argument if @p puzzle is not 81 characters of `1`-`9`, `.` or `0`.
///
/// @par Example
/// @code
/// const auto d = dm::algorithms::sudoku::domains(std::string(81, '.'));   // 81 x {1..9}
/// @endcode
std::vector<std::vector<int>> domains(const std::string& puzzle);

/// @brief The 20 peers of @p cell: the other cells in its row, column and 3x3 box, sorted.
/// @throws std::out_of_range if @p cell is not in [0, 81).
///
/// @par Example
/// @code
/// const auto p = dm::algorithms::sudoku::peers(0);   // 1..8, 9, 10, 11, 18, 19, 20, 27, ...
/// @endcode
std::vector<int> peers(int cell);

/// @brief Every directed peer arc (i, j), i.e. 81 x 20 = 1620 arcs, ordered by i then j.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::algorithms::sudoku::arcs().size());   // 1620
/// @endcode
std::vector<std::pair<int, int>> arcs();

/// @brief The Sudoku constraint for arc (i, j): peers must hold different digits (`a != b`).
///        The cell indices are ignored because every arc carries the same constraint.
[[nodiscard]] bool compatible(int i, int a, int j, int b);

/// @brief Runs ac3() on the puzzle's domains, peer arcs and compatible(): the arc-consistent
///        reduction of the puzzle, with AC-3's work counters.
/// @throws std::invalid_argument for a malformed puzzle (see domains()).
///
/// @par Example
/// @code
/// const auto r = dm::algorithms::sudoku::propagate(puzzle);
/// std::print("{} candidates left\n", dm::algorithms::sudoku::candidates(r.domains));
/// @endcode
Ac3Result propagate(const std::string& puzzle);

/// @brief The board as 81 characters: the digit of each cell whose domain is a single value,
///        `.` for any other cell (several candidates, or none).
/// @throws std::invalid_argument if @p domains does not have 81 entries.
std::string to_string(const std::vector<std::vector<int>>& domains);

/// @brief Total number of candidate digits over all 81 cells (81 when solved, 729 when empty).
[[nodiscard]] long long candidates(const std::vector<std::vector<int>>& domains);

/// @brief Number of cells whose domain is a single digit.
[[nodiscard]] int solved_cells(const std::vector<std::vector<int>>& domains);

/// @brief true if @p grid is 81 digits forming a valid solution of @p puzzle: it agrees with
///        every given and each row, column and box holds 1..9 exactly once.
[[nodiscard]] bool is_solution(const std::string& puzzle, const std::string& grid);

/// @brief Solves the puzzle exactly, by Knuth's Algorithm X on the standard exact-cover
///        encoding (729 rows, one per cell and digit; 324 columns, one per cell, row-digit,
///        column-digit and box-digit pair). Returns the first solution found.
/// @return The 81-digit solution, or std::nullopt if the puzzle has none.
/// @throws std::invalid_argument for a malformed puzzle.
///
/// @par Example
/// @code
/// const auto solution = dm::algorithms::sudoku::solve(puzzle);
/// @endcode
std::optional<std::string> solve(const std::string& puzzle);

/// @brief Draws a board with pencil marks: givens in black, other single-candidate cells in
///        blue, and the remaining candidates of each unsolved cell as small grey digits in a 3x3
///        layout (an empty domain shades the cell red). @p highlight cells are tinted, and the
///        first of them more strongly, e.g. to show a cell and its peers.
/// @param domains 81 domains, as from domains() or propagate().
/// @param puzzle The original puzzle, to tell givens apart; empty treats no cell as given.
/// @param highlight Cells to tint (first one darker); may be empty.
/// @param pixels The board's width and height.
/// @return A square RPlot with no axes, ready to title or save.
/// @throws std::invalid_argument if @p domains does not have 81 entries or @p puzzle is malformed.
///
/// @par Example
/// @code
/// const auto r = dm::algorithms::sudoku::propagate(puzzle);
/// dm::algorithms::sudoku::board(r.domains, puzzle).title("After AC-3").save_svg("ac3.svg");
/// @endcode
graphics::RPlot board(const std::vector<std::vector<int>>& domains, const std::string& puzzle = "",
                      const std::vector<int>& highlight = {}, int pixels = 540);

} // namespace datamunge::algorithms::sudoku
