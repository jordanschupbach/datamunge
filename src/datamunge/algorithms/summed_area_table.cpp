/// @file summed_area_table.cpp
/// @brief Actual checked integral-image recurrence and inclusive rectangle reads.
#include <cmath>
#include <datamunge/algorithms/summed_area_table.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private finite arithmetic policy.
namespace {
/// @brief Reject a nonfinite actual intermediate.
double finite(double x) {
  if (!std::isfinite(x))
    throw std::overflow_error("summed-area arithmetic");
  return x;
}
} // namespace
SummedAreaTable::SummedAreaTable(
    const std::vector<std::vector<double>>& grid,
    const SummedAreaOptions&                options)
    : rows_(grid.size()), cols_(grid.empty() ? 0 : grid.front().size()) {
  for (const auto& row : grid) {
    if (row.size() != cols_)
      throw std::invalid_argument("summed-area ragged grid");
    for (double v : row)
      if (!std::isfinite(v))
        throw std::invalid_argument("summed-area finite grid");
  }
  auto cap = std::numeric_limits<std::size_t>::max();
  if (rows_ == cap || cols_ == cap || rows_ + 1 > cap / (cols_ + 1))
    throw std::length_error("summed-area geometry");
  auto cells = (rows_ + 1) * (cols_ + 1);
  if (options.max_cells && cells > options.max_cells)
    throw std::length_error("summed-area cells");
  if (options.capture && options.max_records && rows_ * cols_ > options.max_records)
    throw std::length_error("summed-area history");
  s_.assign(rows_ + 1, std::vector<double>(cols_ + 1));
  for (std::size_t r = 0; r < rows_; ++r)
    for (std::size_t c = 0; c < cols_; ++c) {
      auto   above = s_[r][c + 1], left = s_[r + 1][c], diagonal = s_[r][c];
      double value     = finite(finite(finite(grid[r][c] + above) + left) - diagonal);
      s_[r + 1][c + 1] = value;
      if (options.capture)
        steps_.push_back({r, c, grid[r][c], above, left, diagonal, value});
    }
}
SummedAreaQuery
SummedAreaTable::query(std::size_t r0, std::size_t c0, std::size_t r1, std::size_t c1) const {
  if (r0 > r1 || c0 > c1 || r1 >= rows_ || c1 >= cols_)
    throw std::out_of_range("summed-area inclusive rectangle");
  SummedAreaQuery q{s_[r1 + 1][c1 + 1], s_[r0][c1 + 1], s_[r1 + 1][c0], s_[r0][c0]};
  q.value = finite(finite(finite(q.bottom_right - q.above) - q.left) + q.diagonal);
  return q;
}
double SummedAreaTable::rect_sum(std::size_t r0, std::size_t c0, std::size_t r1, std::size_t c1)
    const {
  return query(r0, c0, r1, c1).value;
}
double SummedAreaTable::rect_mean(std::size_t r0, std::size_t c0, std::size_t r1, std::size_t c1)
    const {
  return rect_sum(r0, c0, r1, c1) / static_cast<double>((r1 - r0 + 1) * (c1 - c0 + 1));
}
std::size_t SummedAreaTable::rows() const noexcept {
  return rows_;
}
std::size_t SummedAreaTable::cols() const noexcept {
  return cols_;
}
const std::vector<std::vector<double>>& SummedAreaTable::prefix() const noexcept {
  return s_;
}
const std::vector<SummedAreaStep>& SummedAreaTable::steps() const noexcept {
  return steps_;
}
std::vector<std::vector<double>> summed_area_example_grid() {
  std::vector<std::vector<double>> g(6, std::vector<double>(6));
  for (std::size_t r = 0; r < 6; ++r)
    for (std::size_t c = 0; c < 6; ++c)
      g[r][c] = r * 6 + c + 1;
  return g;
}
} // namespace datamunge::algorithms
