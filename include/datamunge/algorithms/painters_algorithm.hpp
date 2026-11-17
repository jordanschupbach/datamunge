#pragma once
/// @file painters_algorithm.hpp
/// @brief Stable object-depth ordering and clipped rectangle overdraw, with actual write history.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Inclusive integer-pixel rectangle with one constant camera depth.
/// Larger depth is farther; equal-depth objects retain input order, so the later
/// input wins an overlap. Reversed bounds paint nothing. No 3D geometry or depth
/// interpolation is performed. Instances own no external resources.
/// @par Example
/// @code
/// const dm::algorithms::DepthRect r{0,0,1,1,3,2};std::print("{}\n",r.color); // 2
/// @endcode
struct DepthRect {
  int    x0;    ///< Inclusive left pixel, clipped to the raster.
  int    y0;    ///< Inclusive top pixel, clipped to the raster.
  int    x1;    ///< Inclusive right pixel; less than x0 denotes an empty rectangle.
  int    y1;    ///< Inclusive bottom pixel; less than y0 denotes an empty rectangle.
  double depth; ///< Finite sorting key; greater is painted earlier.
  int    color; ///< Arbitrary integer color id, including background zero.
};
/// @brief One actual pixel assignment in stable back-to-front traversal.
/// @par Example
/// @code
/// const auto t=dm::algorithms::painter_trace({{0,0,0,0,1,7}},1,1);std::print("{}\n",t.writes[0].after); // 7
/// @endcode
struct PainterWrite {
  std::size_t object; ///< Original rectangle index.
  int         x;      ///< Clipped pixel column.
  int         y;      ///< Clipped pixel row.
  int         before; ///< Value immediately before assignment.
  int         after;  ///< Assigned rectangle color, possibly equal to before.
};
/// @brief Owned final raster, actual order and optional pixel-write history.
/// Working raster is height by width; history is empty for painter_render().
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::painter_trace({},0,0).writes.size()); // 0
/// @endcode
struct PainterTrace {
  std::vector<int>              order;  ///< Original indices in stable descending depth order.
  std::vector<std::vector<int>> raster; ///< Row-major final color ids; zero background.
  std::vector<PainterWrite>     writes; ///< Actual assignments, in execution order.
};
/// @brief Return stable descending indices for complete finite depth keys.
/// O(n log n) comparisons, O(n) storage; borrows input without mutation.
/// @param depths Finite depths, with at most INT_MAX entries; empty is valid.
/// @return Owned original indices; equal-depth indices increase.
/// @throws std::invalid_argument If any depth is nonfinite.
/// @throws std::length_error If indices cannot be represented by int.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::painter_order({1,3,3})); // [1, 2, 0]
/// @endcode
/// @par References
/// - MIT 6.838, Painter's Algorithm, https://groups.csail.mit.edu/graphics/classes/6.838/S98/meetings/m23/painters.html
std::vector<int> painter_order(const std::vector<double>& depths);
/// @brief Render constant-depth inclusive rectangles with clipped integer overdraw.
/// Equal depths use input order; color zero can erase. No per-pixel depth test.
/// Empty/reversed/offscreen rectangles do not write. O(n log n + writes + wh)
/// time and O(n+wh) storage. Borrowed input is unchanged; no shared mutable state.
/// @param rects Finite-depth rectangle objects, at most INT_MAX.
/// @param width Nonnegative raster width, in pixels.
/// @param height Nonnegative raster height, in pixels.
/// @return Owned height rows, each width entries, initially background zero.
/// @throws std::invalid_argument If dimensions are negative or depths nonfinite.
/// @throws std::length_error If dimensions/storage/index ranges cannot be represented.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::painter_render({{0,0,0,0,1,7}},1,1)[0]); // [7]
/// @endcode
/// @see painter_trace()
std::vector<std::vector<int>>
painter_render(const std::vector<DepthRect>& rects, int width, int height);
/// @brief Retain actual assignments from the same producer as painter_render().
/// No raster replay is used. Logical history cap counts pixel-write records only,
/// not bytes or raster storage. O(writes) additional owned history; input unchanged.
/// @param rects Finite constant-depth rectangles.
/// @param width Nonnegative raster width.
/// @param height Nonnegative raster height.
/// @param max_writes Maximum retained writes, default one million; zero disables policy.
/// @return Owned order, final raster and every producing write.
/// @throws std::invalid_argument If dimensions/depths are invalid.
/// @throws std::length_error If history policy or representable storage is exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::painter_trace({{0,0,1,0,1,7}},2,1).writes.size()); // 2
/// @endcode
PainterTrace painter_trace(
    const std::vector<DepthRect>& rects,
    int                           width,
    int                           height,
    std::size_t                   max_writes = 1000000);
} // namespace datamunge::algorithms
