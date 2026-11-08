#pragma once
/// @file bresenham.hpp
/// @brief Integer all-octant line rasterisation with checked output sizes and actual decisions.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace datamunge::algorithms {
/// @brief Owned integer pixel-centre coordinates (x,y), shared by Bresenham results.
/// No geometric cell-area coverage or intensity is implied by a selected centre.
/// @par Example
/// @code
/// const dm::algorithms::GridPoint point{2,3};
/// std::print("{} {}\n",point.first,point.second); // 2 3
/// @endcode
/// @see bresenham_line()
using GridPoint=std::pair<int,int>;

/// @brief One actual current point/error and the coordinate moves selected after it.
/// Both tests use the saved doubled_error, even when the x move changes error.
/// The final endpoint has terminal=true and no outgoing move flags. All fields
/// are owned values; the point is the centre selected by the canonical loop.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::bresenham_line_trace(0,0,2,1);
/// std::print("{} {}\n",trace.steps[0].step_x,trace.steps[0].step_y); // true true
/// @endcode
/// @see BresenhamTrace, bresenham_line_trace()
struct BresenhamStep {
    GridPoint point; ///< Current selected pixel centre, before the recorded moves.
    std::int64_t error{0}; ///< Current integer error, initially dx-dy.
    std::int64_t doubled_error{0}; ///< Saved 2*error used for both comparisons.
    bool step_x{false}; ///< Move by sx after this point when doubled_error>=-dy.
    bool step_y{false}; ///< Move by sy after this point when doubled_error<=dx.
    bool terminal{false}; ///< Final endpoint; no further coordinate move is performed.
};

/// @brief Owned raster points and actual integer decisions from the ordinary loop.
/// dx/dy are nonnegative coordinate extents. Move counters exclude the terminal
/// record; a diagonal move increments both axis counters and diagonal_moves.
/// Recording uses O(max(dx,dy)+1) storage beyond the returned points.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::bresenham_line_trace(0,0,2,1);
/// std::print("{} {} {}\n",trace.points.size(),trace.x_moves,trace.diagonal_moves); // 3 2 1
/// @endcode
/// @see bresenham_line_trace(), BresenhamStep
struct BresenhamTrace {
    std::vector<GridPoint> points; ///< Complete inclusive ordered raster, identical to ordinary output.
    std::vector<BresenhamStep> steps; ///< One actual pre-movement record per selected point.
    std::int64_t dx{0}; ///< Absolute horizontal endpoint difference, widened before subtraction.
    std::int64_t dy{0}; ///< Absolute vertical endpoint difference, widened before subtraction.
    std::size_t x_moves{0}; ///< Number of horizontal coordinate updates.
    std::size_t y_moves{0}; ///< Number of vertical coordinate updates.
    std::size_t diagonal_moves{0}; ///< Number of iterations updating both coordinates.
};

/// @brief Count inclusive Bresenham points without constructing or walking the raster.
/// Returns max(abs(x1-x0),abs(y1-y0))+1 using widened arithmetic. This is O(1)
/// work/storage with no mutation, allocation or shared state. The uint64 result
/// can represent the entire inclusive span of 32-bit int endpoints.
/// @param x0 Initial x coordinate.
/// @param y0 Initial y coordinate.
/// @param x1 Final x coordinate.
/// @param y1 Final y coordinate.
/// @return Number of selected centres, one for coincident endpoints.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::bresenham_point_count(0,0,14,6)); // 15
/// @endcode
/// @see bresenham_line(), bresenham_line_trace()
inline std::uint64_t bresenham_point_count(int x0,int y0,int x1,int y1) noexcept {
    return static_cast<std::uint64_t>(std::max(
        std::abs(static_cast<std::int64_t>(x1)-x0),
        std::abs(static_cast<std::int64_t>(y1)-y0)))+1;
}

/// @brief Private common integer raster loop; record allocation is compiled out for ordinary calls.
namespace bresenham_detail {
/// @brief Execute the existing inclusive all-octant recurrence with widened arithmetic.
/// @tparam Record Whether to retain decisions and move counters.
/// @param x0 Initial x coordinate.
/// @param y0 Initial y coordinate.
/// @param x1 Final x coordinate.
/// @param y1 Final y coordinate.
/// @param max_points Optional maximum complete-output count, zero meaning unlimited.
/// @return Owned points, and records/counters only when Record is true.
/// @throws std::length_error Before allocation if capacity or a positive limit is exceeded.
/// Allocation errors propagate; no partial raster is returned.
template<bool Record>
inline BresenhamTrace raster(int x0,int y0,int x1,int y1,std::size_t max_points) {
    static_assert(std::numeric_limits<int>::digits<=31,"Bresenham requires int coordinates of at most 32 bits");
    BresenhamTrace trace;
    trace.dx=std::abs(static_cast<std::int64_t>(x1)-x0);
    trace.dy=std::abs(static_cast<std::int64_t>(y1)-y0);
    const auto count=bresenham_point_count(x0,y0,x1,y1);
    if ((max_points && count>max_points) || count>trace.points.max_size())
        throw std::length_error("Bresenham output exceeds point limit or vector capacity");
    if constexpr (Record) {
        if (count>trace.steps.max_size()) throw std::length_error("Bresenham trace exceeds vector capacity");
        trace.steps.reserve(static_cast<std::size_t>(count));
    }
    trace.points.reserve(static_cast<std::size_t>(count));
    const int sx=x0<x1?1:-1,sy=y0<y1?1:-1;
    auto error=trace.dx-trace.dy;
    for (;;) {
        trace.points.emplace_back(x0,y0);
        const bool terminal=x0==x1 && y0==y1;
        const auto doubled=2*error;
        const bool step_x=!terminal && doubled>=-trace.dy;
        const bool step_y=!terminal && doubled<=trace.dx;
        if constexpr (Record) {
            trace.steps.push_back({{x0,y0},error,doubled,step_x,step_y,terminal});
            trace.x_moves+=step_x;
            trace.y_moves+=step_y;
            trace.diagonal_moves+=step_x && step_y;
        }
        if (terminal) break;
        if (step_x) {error-=trace.dy;x0+=sx;}
        if (step_y) {error+=trace.dx;y0+=sy;}
    }
    return trace;
}
} // namespace bresenham_detail

/// @brief Rasterise an integer segment to one centre per major-axis position, including both endpoints.
/// Preserves the original four-argument signature and directed all-octant tie
/// rule: step x when saved 2*error>=-dy and y when saved 2*error<=dx. Exactly
/// halfway cases move the minor coordinate towards the destination. Reversing
/// a segment can therefore choose a different tied centre. Coincident endpoints
/// return one point. This is not a supercover of all geometrically touched cells
/// and does not provide antialiasing. Arithmetic is widened before differences
/// and doubling, covering all 32-bit int endpoints. O(max(dx,dy)+1) work and
/// owned output storage, O(1) auxiliary state; no trace allocation or shared/input
/// mutation. No clipping is performed. Large spans may fail to allocate.
/// @param x0 Initial x coordinate.
/// @param y0 Initial y coordinate.
/// @param x1 Final x coordinate.
/// @param y1 Final y coordinate.
/// @return Owned ordered pixel centres, inclusive and 8-connected.
/// @throws std::length_error If the complete point count exceeds vector capacity.
/// Allocation errors propagate; no partial vector is returned.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::bresenham_line(0,0,2,1)); // [(0, 0), (1, 1), (2, 1)]
/// @endcode
/// @par References
/// - J. E. Bresenham (1965), Algorithm for computer control of a digital plotter,
///   IBM Systems Journal 4(1), 25-30, doi:10.1147/sj.41.0025. This is the
///   symmetric all-octant error recurrence, retaining its directed inclusive ties.
/// @see bresenham_line_trace(), bresenham_point_count()
inline std::vector<GridPoint> bresenham_line(int x0,int y0,int x1,int y1) {
    return bresenham_detail::raster<false>(x0,y0,x1,y1,0).points;
}

/// @brief Record actual saved-error decisions from the same kernel as bresenham_line().
/// Returns the same centres and adds one owned pre-movement record per point;
/// terminal flags have no outgoing moves. O(max(dx,dy)+1) time and storage,
/// without shared/input mutation or extra geometric replay. A positive point
/// limit rejects an oversized complete output before allocation; zero is unlimited.
/// A limit failure never returns a misleading truncated segment.
/// @param x0 Initial x coordinate.
/// @param y0 Initial y coordinate.
/// @param x1 Final x coordinate.
/// @param y1 Final y coordinate.
/// @param max_points Maximum complete-output count, or zero (default) for no caller limit.
/// @return Owned inclusive points, decisions, extents and move counts.
/// @throws std::length_error If a positive limit or vector capacity would be exceeded.
/// Allocation errors propagate, with no partial result.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::bresenham_line_trace(0,0,2,1,3);
/// std::print("{} {}\n",trace.steps.size(),trace.y_moves); // 3 1
/// @endcode
/// @par References
/// - J. E. Bresenham (1965), doi:10.1147/sj.41.0025; records this library's
///   all-octant integer recurrence rather than a separate floating-point reconstruction.
/// @see bresenham_line(), BresenhamTrace, bresenham_point_count()
inline BresenhamTrace bresenham_line_trace(int x0,int y0,int x1,int y1,std::size_t max_points=0) {
    return bresenham_detail::raster<true>(x0,y0,x1,y1,max_points);
}
} // namespace datamunge::algorithms
