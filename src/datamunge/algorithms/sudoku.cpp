/// @file sudoku.cpp
/// @brief Sudoku CSP encoding, AC-3 propagation, exact-cover solving and the pencil-mark board.

#include <datamunge/algorithms/exact_cover.hpp>
#include <datamunge/algorithms/sudoku.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace datamunge::algorithms::sudoku {

/// @brief Parsing, box arithmetic and drawing helpers.
namespace {

/// @brief The given digit at each cell (0 for a blank), after validating the puzzle string.
std::vector<int> givens(const std::string& puzzle) {
  if (puzzle.size() != 81)
    throw std::invalid_argument("sudoku puzzle must have 81 characters, got " + std::to_string(puzzle.size()));
  std::vector<int> digits(81, 0);
  for (std::size_t c = 0; c < 81; ++c) {
    const char ch = puzzle[c];
    if (ch >= '1' && ch <= '9')
      digits[c] = ch - '0';
    else if (ch != '.' && ch != '0')
      throw std::invalid_argument(std::string("sudoku puzzle has invalid character '") + ch + "'");
  }
  return digits;
}

/// @brief The 3x3 box (0..8, row-major) holding @p cell.
int box_of(int cell) { return (cell / 9) / 3 * 3 + (cell % 9) / 3; }

/// @brief Throws unless @p domains has one entry per cell.
void require_board(const std::vector<std::vector<int>>& domains) {
  if (domains.size() != 81)
    throw std::invalid_argument("sudoku domains must have 81 entries, got " + std::to_string(domains.size()));
}

} // namespace

std::vector<std::vector<int>> domains(const std::string& puzzle) {
  const auto digits = givens(puzzle);
  std::vector<std::vector<int>> out(81);
  for (std::size_t c = 0; c < 81; ++c)
    out[c] = digits[c] ? std::vector<int>{digits[c]} : std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9};
  return out;
}

std::vector<int> peers(int cell) {
  if (cell < 0 || cell >= 81) throw std::out_of_range("sudoku cell must be in [0, 81)");
  std::vector<int> out;
  for (int other = 0; other < 81; ++other) {
    if (other == cell) continue;
    if (other / 9 == cell / 9 || other % 9 == cell % 9 || box_of(other) == box_of(cell)) out.push_back(other);
  }
  return out;
}

std::vector<std::pair<int, int>> arcs() {
  std::vector<std::pair<int, int>> out;
  out.reserve(81 * 20);
  for (int i = 0; i < 81; ++i)
    for (int j : peers(i)) out.emplace_back(i, j);
  return out;
}

bool compatible(int /*i*/, int a, int /*j*/, int b) { return a != b; }

Ac3Result propagate(const std::string& puzzle) { return ac3(domains(puzzle), arcs(), compatible); }

std::string to_string(const std::vector<std::vector<int>>& domains) {
  require_board(domains);
  std::string out(81, '.');
  for (std::size_t c = 0; c < 81; ++c)
    if (domains[c].size() == 1) out[c] = static_cast<char>('0' + domains[c][0]);
  return out;
}

long long candidates(const std::vector<std::vector<int>>& domains) {
  long long n = 0;
  for (const auto& d : domains) n += static_cast<long long>(d.size());
  return n;
}

int solved_cells(const std::vector<std::vector<int>>& domains) {
  return static_cast<int>(std::count_if(domains.begin(), domains.end(), [](const auto& d) { return d.size() == 1; }));
}

bool is_solution(const std::string& puzzle, const std::string& grid) {
  const auto given = givens(puzzle);
  if (grid.size() != 81) return false;
  std::vector<int> g(81);
  for (std::size_t c = 0; c < 81; ++c) {
    if (grid[c] < '1' || grid[c] > '9') return false;
    g[c] = grid[c] - '0';
    if (given[c] && given[c] != g[c]) return false;
  }
  for (int unit = 0; unit < 9; ++unit) {
    unsigned row = 0, col = 0, box = 0;
    for (int k = 0; k < 9; ++k) {
      row |= 1u << g[static_cast<std::size_t>(unit * 9 + k)];
      col |= 1u << g[static_cast<std::size_t>(k * 9 + unit)];
      box |= 1u << g[static_cast<std::size_t>((unit / 3 * 3 + k / 3) * 9 + unit % 3 * 3 + k % 3)];
    }
    if (row != 0x3FEu || col != 0x3FEu || box != 0x3FEu) return false;
  }
  return true;
}

std::optional<std::string> solve(const std::string& puzzle) {
  const auto given = givens(puzzle);
  // One exact-cover row per allowed (cell, digit); four constraint columns per row.
  std::vector<std::vector<int>> rows;
  std::vector<std::pair<int, int>> choice;
  for (int c = 0; c < 81; ++c)
    for (int d = 1; d <= 9; ++d) {
      if (given[static_cast<std::size_t>(c)] && given[static_cast<std::size_t>(c)] != d) continue;
      const int r = c / 9, col = c % 9, b = box_of(c);
      rows.push_back({c, 81 + 9 * r + d - 1, 162 + 9 * col + d - 1, 243 + 9 * b + d - 1});
      choice.emplace_back(c, d);
    }
  const auto cover = exact_cover(324, rows);
  if (!cover.solved) return std::nullopt;
  std::string out(81, '.');
  for (int row : cover.rows) {
    const auto [c, d] = choice[static_cast<std::size_t>(row)];
    out[static_cast<std::size_t>(c)] = static_cast<char>('0' + d);
  }
  return out;
}

graphics::RPlot board(const std::vector<std::vector<int>>& domains, const std::string& puzzle,
                      const std::vector<int>& highlight, int pixels) {
  require_board(domains);
  const std::vector<int> given = puzzle.empty() ? std::vector<int>(81, 0) : givens(puzzle);
  const double margin = 16, top = 52;
  const double cell_px = (pixels - 2 * margin) / 9.0;
  auto plot = graphics::RPlot::create();
  plot.size(static_cast<std::size_t>(pixels), static_cast<std::size_t>(pixels - 2 * margin + margin + top))
      .margins(margin, margin, top, margin)
      .x_limits(0, 9)
      .y_limits(0, 9)
      .hide_axes()
      .show_grid(false)
      .title("")
      .x_label("")
      .y_label("")
      .background({255, 255, 255});

  // Cell shading: highlights and empty (contradictory) domains.
  const auto shade = [&](int cell, graphics::RGB color) {
    const double x = cell % 9, y = static_cast<double>(8 - cell / 9);
    plot.polygon({x, x + 1, x + 1, x}, {y, y, y + 1, y + 1}, color, true);
  };
  for (std::size_t k = 0; k < highlight.size(); ++k)
    shade(highlight[k], k == 0 ? graphics::RGB{37, 99, 235, 0.45} : graphics::RGB{37, 99, 235, 0.16});
  for (int c = 0; c < 81; ++c)
    if (domains[static_cast<std::size_t>(c)].empty()) shade(c, {220, 38, 38, 0.35});

  // Grid: thin cell lines, thick box lines.
  std::vector<double> x0, y0, x1, y1, bx0, by0, bx1, by1;
  for (int k = 0; k <= 9; ++k) {
    auto& ax0 = k % 3 ? x0 : bx0;
    auto& ay0 = k % 3 ? y0 : by0;
    auto& ax1 = k % 3 ? x1 : bx1;
    auto& ay1 = k % 3 ? y1 : by1;
    ax0.insert(ax0.end(), {double(k), 0.0});
    ay0.insert(ay0.end(), {0.0, double(k)});
    ax1.insert(ax1.end(), {double(k), 9.0});
    ay1.insert(ay1.end(), {9.0, double(k)});
  }
  plot.segments(x0, y0, x1, y1, {148, 163, 184}, 1.0);
  plot.segments(bx0, by0, bx1, by1, {17, 24, 39}, 3.0);

  // Digits: big for single values, a 3x3 block of small pencil marks otherwise.
  for (int c = 0; c < 81; ++c) {
    const auto& d = domains[static_cast<std::size_t>(c)];
    const double x = c % 9, y = static_cast<double>(8 - c / 9);
    if (d.size() == 1) {
      const bool is_given = given[static_cast<std::size_t>(c)] != 0;
      plot.text(x + 0.5, y + 0.5, std::to_string(d[0]), is_given ? graphics::RGB{17, 24, 39} : graphics::RGB{37, 99, 235},
                0.58 * cell_px);
    } else {
      for (int v : d)
        plot.text(x + ((v - 1) % 3 + 0.5) / 3.0, y + 1 - (static_cast<double>((v - 1) / 3) + 0.5) / 3.0, std::to_string(v),
                  {100, 116, 139}, 0.27 * cell_px);
    }
  }
  return plot;
}

} // namespace datamunge::algorithms::sudoku
