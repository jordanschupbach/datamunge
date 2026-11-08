/// @file flood_fill.cpp
/// @brief Shared checked exact-value iterative flood-fill kernel and owned evidence.
#include <algorithm>
#include <datamunge/algorithms/flood_fill.hpp>
#include <limits>
#include <stdexcept>
#include <utility>
namespace datamunge::algorithms {
/// @brief Private checked preparation and actual traversal implementation.
namespace {
/// @brief Add two entry/counter quantities without wrapping.
/// @param a First quantity.
/// @param b Second quantity.
/// @return Checked sum.
/// @throws std::length_error Sum is unrepresentable.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("flood fill: entry sum overflow");
  return a + b;
}
/// @brief Multiply entry quantities without wrapping.
/// @param a First factor.
/// @param b Second factor.
/// @return Checked product.
/// @throws std::length_error Product is unrepresentable.
std::size_t product(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::length_error("flood fill: entry product overflow");
  return a * b;
}
/// @brief Increment one actual counter without wrapping.
/// @param n Mutable counter.
/// @throws std::overflow_error Counter already equals SIZE_MAX.
void increment(std::size_t& n) {
  if (n == std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("flood fill: counter overflow");
  ++n;
}
/// @brief Validate all rows and determine the seed/no-op state once.
/// @param grid Read-only input.
/// @param sr Seed row.
/// @param sc Seed column.
/// @param value Replacement.
/// @return Prepared scalar metadata without mutation.
/// @throws std::invalid_argument Ragged rows.
/// @throws std::length_error Cell product overflow.
FloodFillWork
prepare(const std::vector<std::vector<int>>& grid, std::size_t sr, std::size_t sc, int value) {
  FloodFillWork w;
  w.rows = grid.size();
  w.cols = grid.empty() ? 0 : grid.front().size();
  for (const auto& row : grid)
    if (row.size() != w.cols)
      throw std::invalid_argument("flood fill: ragged grid");
  w.cells     = product(w.rows, w.cols);
  w.seed_row  = sr;
  w.seed_col  = sc;
  w.new_value = value;
  if (!w.cells)
    return w;
  if (sr >= w.rows || sc >= w.cols) {
    w.stop = FloodFillStop::SeedOutside;
    return w;
  }
  w.has_target = true;
  w.target     = grid[sr][sc];
  w.stop       = w.target == value ? FloodFillStop::SameValue : FloodFillStop::Filled;
  return w;
}
/// @brief Traverse once, optionally appending actual evidence to an owned result.
/// @param grid Mutable validated grid.
/// @param w Prepared metadata, updated with actual counters.
/// @param eight Eight-direction mode.
/// @param trace Optional separate result history, null for ordinary work.
/// @throws std::length_error Stack or history capacity exhausted.
/// @throws std::overflow_error Actual counter overflow.
/// @throws std::bad_alloc Stack or history allocation fails; earlier paint may remain.
void traverse(
    std::vector<std::vector<int>>& grid,
    FloodFillWork&                 w,
    bool                           eight,
    FloodFillTrace*                trace) {
  if (w.stop != FloodFillStop::Filled)
    return;
  std::vector<std::pair<std::size_t, std::size_t>> stack{{w.seed_row, w.seed_col}};
  grid[w.seed_row][w.seed_col] = w.new_value;
  w.discovered                 = 1;
  w.peak_stack                 = 1;
  while (!stack.empty()) {
    const auto [r, c] = stack.back();
    stack.pop_back();
    increment(w.filled);
    FloodFillVisit visit{r, c, stack.size(), 0, trace ? trace->probes.size() : 0, 0};
    const auto     neighbors = flood_fill_neighborhood(w.rows, w.cols, r, c, eight);
    for (std::size_t k = 0; k < neighbors.count; ++k) {
      const auto p = neighbors.candidates[k];
      increment(w.neighbor_slots);
      FloodFillProbe probe{p, 0, false};
      if (p.in_bounds) {
        increment(w.in_bounds_neighbors);
        probe.value_before = grid[p.row][p.col];
        probe.accepted     = probe.value_before == w.target;
        if (probe.accepted) {
          increment(w.accepted_neighbors);
          increment(w.discovered);
          grid[p.row][p.col] = w.new_value;
          stack.emplace_back(p.row, p.col);
          w.peak_stack = std::max(w.peak_stack, stack.size());
        }
      }
      if (trace)
        trace->probes.push_back(probe);
    }
    if (trace) {
      visit.stack_after_scan = stack.size();
      visit.probe_end        = trace->probes.size();
      trace->visits.push_back(visit);
    }
  }
}
} // namespace
FloodFillNeighborhood flood_fill_neighborhood(
    std::size_t rows,
    std::size_t cols,
    std::size_t row,
    std::size_t col,
    bool        eight) noexcept {
  FloodFillNeighborhood n;
  if (row >= rows || col >= cols)
    return n;
  /// @brief Ordered row offsets for four and eight connectivity.
  constexpr std::array<int, 8> dr4{-1, 1, 0, 0}, dr8{-1, -1, -1, 0, 0, 1, 1, 1};
  /// @brief Ordered column offsets for four and eight connectivity.
  constexpr std::array<int, 8> dc4{0, 0, -1, 1}, dc8{-1, 0, 1, -1, 1, -1, 0, 1};
  const auto&                  dr = eight ? dr8 : dr4;
  const auto&                  dc = eight ? dc8 : dc4;
  n.count                         = eight ? 8 : 4;
  for (std::size_t k = 0; k < n.count; ++k) {
    auto& p     = n.candidates[k];
    p.direction = k;
    if ((dr[k] < 0 && row == 0) || (dr[k] > 0 && row == rows - 1) || (dc[k] < 0 && col == 0) ||
        (dc[k] > 0 && col == cols - 1))
      continue;
    p.in_bounds = true;
    p.row       = dr[k] < 0 ? row - 1 : dr[k] > 0 ? row + 1 : row;
    p.col       = dc[k] < 0 ? col - 1 : dc[k] > 0 ? col + 1 : col;
  }
  return n;
}
FloodFillWork flood_fill_work(
    std::vector<std::vector<int>>& grid,
    std::size_t                    sr,
    std::size_t                    sc,
    int                            value,
    bool                           eight) {
  auto w = prepare(grid, sr, sc, value);
  traverse(grid, w, eight, nullptr);
  return w;
}
std::size_t flood_fill(
    std::vector<std::vector<int>>& grid,
    std::size_t                    sr,
    std::size_t                    sc,
    int                            value,
    bool                           eight) {
  return flood_fill_work(grid, sr, sc, value, eight).filled;
}
FloodFillTrace flood_fill_trace(
    const std::vector<std::vector<int>>& grid,
    std::size_t                          sr,
    std::size_t                          sc,
    int                                  value,
    bool                                 eight,
    std::size_t                          cap) {
  auto       w      = prepare(grid, sr, sc, value);
  const bool active = w.stop == FloodFillStop::Filled;
  const auto slots  = active ? product(eight ? 8 : 4, w.cells) : 0;
  const auto bound  = add(product(2, w.rows), product(active ? (eight ? 11 : 7) : 2, w.cells));
  if ((cap && bound > cap) || (active && (w.cells > std::vector<FloodFillVisit>().max_size() ||
                                          slots > std::vector<FloodFillProbe>().max_size())))
    throw std::length_error("flood fill: complete trace entry cap or container limit");
  FloodFillTrace t;
  t.work   = w;
  t.input  = grid;
  t.output = grid;
  traverse(t.output, t.work, eight, &t);
  return t;
}
std::vector<std::vector<int>> flood_fill_example() {
  return {
      {1, 1, 1, 0, 0, 0, 0, 0},
      {1, 1, 1, 0, 0, 0, 0, 0},
      {1, 1, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 1, 1, 0},
      {0, 0, 0, 0, 0, 1, 1, 0},
      {0, 0, 0, 0, 0, 0, 1, 0}};
}
} // namespace datamunge::algorithms
