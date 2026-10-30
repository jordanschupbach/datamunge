/// @file warnsdorff.cpp
/// @brief Shared minimum-degree knight-tour attempts, start profiles and grid readout diagnostics.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <datamunge/algorithms/warnsdorff.hpp>
#include <limits>
#include <set>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Local board/path/schema validation and fixed knight geometry.
namespace {
/// @brief The eight knight displacements; coordinate ties do not depend on this enumeration order.
constexpr std::array<std::pair<int, int>, 8> kKnightMoves{
    {{2, 1}, {2, -1}, {1, 2}, {1, -2}, {-1, 2}, {-1, -2}, {-2, 1}, {-2, -1}}};
/// @brief Check positive dimensions and multiply them without signed or allocation-size overflow.
/// @param rows Board height.
/// @param columns Board width.
/// @param limit Positive maximum cells.
/// @return Safe board cell count.
/// @throws std::invalid_argument If dimensions or limit are zero/negative.
/// @throws std::length_error If the product is not representable or exceeds the limit/storage
/// range.
std::size_t knight_board_cells(int rows, int columns, std::size_t limit) {
  if (rows <= 0 || columns <= 0 || limit == 0)
    throw std::invalid_argument("Warnsdorff: invalid board or cell limit");
  const auto R = static_cast<std::size_t>(rows), C = static_cast<std::size_t>(columns);
  if (R > std::numeric_limits<std::size_t>::max() / C)
    throw std::length_error("Warnsdorff: cell product overflow");
  const auto cells = R * C;
  if (cells > limit || cells > std::vector<unsigned char>().max_size())
    throw std::length_error("Warnsdorff: board cell limit exceeded");
  return cells;
}
/// @brief Test the knight edge relation for valid nonnegative int coordinates.
/// @param a First valid board coordinate.
/// @param b Second valid board coordinate.
/// @return True for absolute displacements (1,2) or (2,1).
bool knight_edge(const std::pair<int, int>& a, const std::pair<int, int>& b) {
  const int dr = std::abs(a.first - b.first), dc = std::abs(a.second - b.second);
  return (dr == 1 && dc == 2) || (dr == 2 && dc == 1);
}
/// @brief Validate actual path geometry/uniqueness rather than trusting status flags.
/// @param attempt Borrowed dimensions and path.
/// @param complete Require every board cell to be present.
/// @param limit Positive permitted board size.
/// @return Board cell count.
/// @throws std::invalid_argument If path or dimensions are invalid/incomplete.
/// @throws std::length_error If the board exceeds the limit.
std::size_t
validate_knight_path(const WarnsdorffAttempt& attempt, bool complete, std::size_t limit) {
  const auto cells = knight_board_cells(attempt.rows, attempt.columns, limit);
  if (attempt.path.empty() || attempt.path.size() > cells ||
      (complete && attempt.path.size() != cells))
    throw std::invalid_argument("knight path: invalid or incomplete coverage");
  std::set<std::pair<int, int>> seen;
  for (std::size_t i = 0; i < attempt.path.size(); ++i) {
    const auto p = attempt.path[i];
    if (p.first < 0 || p.first >= attempt.rows || p.second < 0 || p.second >= attempt.columns ||
        !seen.insert(p).second)
      throw std::invalid_argument("knight path: out-of-board or repeated cell");
    if (i && !knight_edge(attempt.path[i - 1], p))
      throw std::invalid_argument("knight path: illegal edge");
  }
  return cells;
}
/// @brief Validate grid schema and form the complete path's row-major permutation.
/// @param data Borrowed numeric-grid frame.
/// @param predictors Unique row-major column names.
/// @param attempt Complete legal path and dimensions.
/// @return Owned row-major indices in tour order.
/// @throws std::invalid_argument If the path, grid shape, duplicate names or types are invalid.
/// @throws std::out_of_range If a required column is absent.
/// @throws std::length_error If the board dimensions cannot fit supported storage.
std::vector<std::size_t> knight_grid_permutation(
    const dstruct::DataFrame&       data,
    const std::vector<std::string>& predictors,
    const WarnsdorffAttempt&        attempt) {
  const auto cells = validate_knight_path(attempt, true, std::numeric_limits<std::size_t>::max());
  if (predictors.size() != cells ||
      std::set<std::string>(predictors.begin(), predictors.end()).size() != cells)
    throw std::invalid_argument("knight grid: wrong number of unique row-major columns");
  for (const auto& name : predictors)
    (void)data.get_column(name);
  std::vector<std::size_t> order;
  order.reserve(cells);
  for (const auto p : attempt.path)
    order.push_back(static_cast<std::size_t>(p.first) * attempt.columns + p.second);
  return order;
}
/// @brief Copy one finite non-null numeric grid record in its original row-major order.
/// @param data Borrowed numeric frame.
/// @param row Required record index.
/// @param predictors Required numeric columns in grid order.
/// @return Owned numeric values.
/// @throws std::invalid_argument If a selected value is null or non-finite, or a column has wrong
/// type.
/// @throws std::out_of_range If row or a required column is absent.
std::vector<double> knight_grid_row(
    const dstruct::DataFrame&       data,
    std::size_t                     row,
    const std::vector<std::string>& predictors) {
  if (row >= data.nrows())
    throw std::out_of_range("knight grid: image row is absent");
  std::vector<double> values;
  values.reserve(predictors.size());
  for (const auto& name : predictors) {
    const auto value = data.get_column(name).at(row);
    if (!std::isfinite(value))
      throw std::invalid_argument("knight grid: non-finite value");
    values.push_back(value);
  }
  return values;
}
} // namespace

WarnsdorffAttempt
warnsdorff_attempt(int rows, int columns, int start_row, int start_column, std::size_t max_cells) {
  const auto cells = knight_board_cells(rows, columns, max_cells);
  if (start_row < 0 || start_row >= rows || start_column < 0 || start_column >= columns)
    throw std::invalid_argument("Warnsdorff: invalid starting coordinate");
  WarnsdorffAttempt result;
  result.rows    = rows;
  result.columns = columns;
  std::vector<unsigned char> seen(cells, 0);
  result.path.push_back({start_row, start_column});
  seen[static_cast<std::size_t>(start_row) * columns + start_column] = 1;
  std::vector<double> step, current_row, current_column, legal_count, minimum, ties, selected_row,
      selected_column;
  std::vector<std::string> action;
  /// @brief Whether a wide intermediate coordinate is in bounds and not visited.
  /// @param r Candidate row, allowing negative/out-of-int-range intermediates.
  /// @param c Candidate column.
  /// @return True for a legal unvisited board cell.
  const auto unvisited = [&](std::int64_t r, std::int64_t c) {
    return r >= 0 && r < rows && c >= 0 && c < columns &&
           !seen[static_cast<std::size_t>(r) * columns + static_cast<std::size_t>(c)];
  };
  while (result.path.size() < cells) {
    const auto [r, c] = result.path.back();
    std::tuple<int, int, int> best{99, -1, -1};
    int                       legal = 0, tie_count = 0;
    for (const auto [dr, dc] : kKnightMoves) {
      ++result.candidate_probes;
      const std::int64_t nr = std::int64_t(r) + dr, nc = std::int64_t(c) + dc;
      if (!unvisited(nr, nc))
        continue;
      ++legal;
      int degree = 0;
      for (const auto [a, b] : kKnightMoves) {
        ++result.degree_probes;
        degree += unvisited(nr + a, nc + b);
      }
      if (degree < std::get<0>(best))
        tie_count = 1;
      else if (degree == std::get<0>(best))
        ++tie_count;
      best = std::min(
          best, std::tuple<int, int, int>{degree, static_cast<int>(nr), static_cast<int>(nc)});
    }
    const auto [degree, nr, nc] = best;
    step.push_back(result.path.size() - 1);
    current_row.push_back(r);
    current_column.push_back(c);
    legal_count.push_back(legal);
    minimum.push_back(nr < 0 ? -1 : degree);
    ties.push_back(tie_count);
    selected_row.push_back(nr);
    selected_column.push_back(nc);
    action.push_back(nr < 0 ? "stuck" : "move");
    if (nr < 0)
      break;
    result.path.push_back({nr, nc});
    seen[static_cast<std::size_t>(nr) * columns + nc] = 1;
  }
  result.completed = result.path.size() == cells;
  result.closed    = result.completed && result.path.size() > 1 &&
                  knight_edge(result.path.front(), result.path.back());
  result.trace.add_column("step", step);
  result.trace.add_column("row", current_row);
  result.trace.add_column("column", current_column);
  result.trace.add_column("legal_moves", legal_count);
  result.trace.add_column("minimum_degree", minimum);
  result.trace.add_column("ties", ties);
  result.trace.add_column("selected_row", selected_row);
  result.trace.add_column("selected_column", selected_column);
  result.trace.add_column("action", action);
  std::vector<double> index, path_row, path_column, row_major;
  for (std::size_t i = 0; i < result.path.size(); ++i) {
    const auto p = result.path[i];
    index.push_back(i);
    path_row.push_back(p.first);
    path_column.push_back(p.second);
    row_major.push_back(static_cast<std::size_t>(p.first) * columns + p.second);
  }
  result.path_table.add_column("step", index);
  result.path_table.add_column("row", path_row);
  result.path_table.add_column("column", path_column);
  result.path_table.add_column("row_major_index", row_major);
  return result;
}

std::vector<std::pair<int, int>>
warnsdorff_knight_tour(int rows, int columns, int start_row, int start_column) {
  auto result = warnsdorff_attempt(
      rows, columns, start_row, start_column, std::numeric_limits<std::size_t>::max());
  if (!result.completed)
    return {};
  return std::move(result.path);
}

dstruct::DataFrame warnsdorff_start_profile(int rows, int columns, std::size_t max_starts) {
  (void)knight_board_cells(rows, columns, max_starts);
  std::vector<double> sr, sc, colour, visited, complete, closed, candidates, degrees;
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < columns; ++c) {
      const auto attempt = warnsdorff_attempt(rows, columns, r, c, max_starts);
      sr.push_back(r);
      sc.push_back(c);
      colour.push_back((r & 1) ^ (c & 1));
      visited.push_back(attempt.path.size());
      complete.push_back(attempt.completed);
      closed.push_back(attempt.closed);
      candidates.push_back(attempt.candidate_probes);
      degrees.push_back(attempt.degree_probes);
    }
  dstruct::DataFrame out;
  out.add_column("start_row", sr);
  out.add_column("start_column", sc);
  out.add_column("colour", colour);
  out.add_column("visited_cells", visited);
  out.add_column("completed", complete);
  out.add_column("closed", closed);
  out.add_column("candidate_probes", candidates);
  out.add_column("degree_probes", degrees);
  return out;
}

graphics::RPlot
plot_knight_tour(const WarnsdorffAttempt& attempt, bool annotate, std::size_t max_cells) {
  const auto                 cells = validate_knight_path(attempt, false, max_cells);
  std::vector<unsigned char> visited(cells, 0);
  for (const auto p : attempt.path)
    visited[static_cast<std::size_t>(p.first) * attempt.columns + p.second] = 1;
  auto plot = graphics::RPlot::create();
  for (int r = 0; r < attempt.rows; ++r)
    for (int c = 0; c < attempt.columns; ++c) {
      const unsigned char shade  = ((r & 1) ^ (c & 1)) ? 230 : 250;
      const auto          colour = visited[static_cast<std::size_t>(r) * attempt.columns + c]
                                       ? graphics::RGB{shade, shade, shade}
                                       : graphics::RGB{254, 240, 138};
      plot.polygon(
          {c - 0.5, c + 0.5, c + 0.5, c - 0.5}, {r - 0.5, r - 0.5, r + 0.5, r + 0.5}, colour, true);
    }
  std::vector<double> x, y;
  for (const auto p : attempt.path) {
    x.push_back(p.second);
    y.push_back(p.first);
  }
  plot.line(x, y, "", {37, 99, 235}, 1)
      .points({x.front()}, {y.front()}, "", {74, 222, 128}, 8)
      .points({x.back()}, {y.back()}, "", {248, 113, 113}, 8);
  if (annotate)
    for (std::size_t i = 0; i < x.size(); ++i)
      plot.text(x[i], y[i], std::to_string(i), {15, 15, 15}, 11);
  plot.x_limits(-0.5, attempt.columns - 0.5)
      .y_limits(-0.5, attempt.rows - 0.5)
      .show_grid(false)
      .x_label("column")
      .y_label("row");
  return plot;
}

dstruct::DataFrame knight_tour_grid_values(
    const dstruct::DataFrame&       data,
    std::size_t                     image_row,
    const std::vector<std::string>& predictors,
    const WarnsdorffAttempt&        attempt) {
  const auto          order  = knight_grid_permutation(data, predictors, attempt);
  const auto          values = knight_grid_row(data, image_row, predictors);
  std::vector<double> step, row, column, toured;
  for (std::size_t i = 0; i < order.size(); ++i) {
    step.push_back(i);
    row.push_back(attempt.path[i].first);
    column.push_back(attempt.path[i].second);
    toured.push_back(values[order[i]]);
  }
  dstruct::DataFrame out;
  out.add_column("step", step);
  out.add_column("tour_row", row);
  out.add_column("tour_column", column);
  out.add_column("row_major_value", values);
  out.add_column("tour_value", toured);
  return out;
}

dstruct::DataFrame knight_tour_grid_summary(
    const dstruct::DataFrame&       data,
    const std::vector<std::string>& predictors,
    const WarnsdorffAttempt&        attempt) {
  const auto          order = knight_grid_permutation(data, predictors, attempt);
  std::vector<double> image, sum, tour_sum, difference, variation, tour_variation;
  for (std::size_t r = 0; r < data.nrows(); ++r) {
    const auto values = knight_grid_row(data, r, predictors);
    double     total = 0, toured_total = 0, tv = 0, toured_tv = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
      total += values[i];
      toured_total += values[order[i]];
      if (i) {
        tv += std::abs(values[i] - values[i - 1]);
        toured_tv += std::abs(values[order[i]] - values[order[i - 1]]);
      }
    }
    const double delta = toured_total - total;
    if (!std::isfinite(total) || !std::isfinite(toured_total) || !std::isfinite(delta) ||
        !std::isfinite(tv) || !std::isfinite(toured_tv))
      throw std::overflow_error("knight grid: non-finite sum or variation");
    image.push_back(r);
    sum.push_back(total);
    tour_sum.push_back(toured_total);
    difference.push_back(delta);
    variation.push_back(tv);
    tour_variation.push_back(toured_tv);
  }
  dstruct::DataFrame out;
  out.add_column("image_row", image);
  out.add_column("value_sum", sum);
  out.add_column("tour_sum", tour_sum);
  out.add_column("sum_difference", difference);
  out.add_column("row_major_variation", variation);
  out.add_column("tour_variation", tour_variation);
  return out;
}
} // namespace datamunge::algorithms
