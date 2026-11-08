#pragma once
/// @file translated_grid_2d.hpp
/// @brief Deterministic translated rectangular grids for native spatial examples.
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>
namespace datamunge::datasets {
/// @brief Replicate a two-dimensional rectangular grid about each supplied centre.
/// Rows are ordered by centre, grid row, then grid column. Coordinate formulas are
/// centre[0]+x_step*column+x_offset and centre[1]+y_step*row+y_offset, evaluated in
/// double. Negative/zero steps are allowed. This is a deterministic lattice, not a
/// Gaussian sample. Empty centres or a zero axis count return empty after validation.
/// Inputs are borrowed/unchanged; owned output uses O(k*rows*columns) work/storage,
/// with no RNG or shared state. Generated nonfinite coordinates throw.
/// @param centres Finite two-coordinate centres, possibly empty.
/// @param columns Number of columns per translated grid.
/// @param rows Number of rows per translated grid.
/// @param x_step Finite signed horizontal spacing.
/// @param y_step Finite signed vertical spacing.
/// @param x_offset Finite horizontal offset from each centre; default zero.
/// @param y_offset Finite vertical offset from each centre; default zero.
/// @return Owned two-coordinate rows in centre/row/column order.
/// @throws std::invalid_argument Centre width differs from two or any scalar is nonfinite.
/// @throws std::length_error Nonempty total point count overflows size_t.
/// @throws std::overflow_error A generated coordinate is nonfinite.
/// @par Example
/// @code
/// const auto p=dm::datasets::translated_grid_2d({{0,0}},2,2,1,2);
/// std::print("{}\n",p); // [[0, 0], [1, 0], [0, 2], [1, 2]]
/// @endcode
/// @see gaussian_blobs()
inline std::vector<std::vector<double>> translated_grid_2d(const std::vector<std::vector<double>>& centres,
    std::size_t columns,std::size_t rows,double x_step,double y_step,double x_offset=0,double y_offset=0) {
  if(!std::isfinite(x_step) || !std::isfinite(y_step) || !std::isfinite(x_offset) || !std::isfinite(y_offset))
    throw std::invalid_argument("translated_grid_2d: nonfinite spacing/offset");
  for(const auto& centre:centres)
    if(centre.size()!=2 || !std::isfinite(centre[0]) || !std::isfinite(centre[1]))
      throw std::invalid_argument("translated_grid_2d: invalid centre");
  std::vector<std::vector<double>> out;
  if(centres.empty() || !columns || !rows) return out;
  const auto maximum=std::numeric_limits<std::size_t>::max();
  if(rows>maximum/columns || centres.size()>maximum/(rows*columns))
    throw std::length_error("translated_grid_2d: point count overflow");
  out.reserve(centres.size()*rows*columns);
  for(const auto& centre:centres) for(std::size_t row=0;row<rows;++row) for(std::size_t column=0;column<columns;++column) {
    const double x=centre[0]+x_step*column+x_offset,y=centre[1]+y_step*row+y_offset;
    if(!std::isfinite(x) || !std::isfinite(y)) throw std::overflow_error("translated_grid_2d: nonfinite coordinate");
    out.push_back({x,y});
  }
  return out;
}
} // namespace datamunge::datasets
