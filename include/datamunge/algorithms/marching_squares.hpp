#pragma once
/// @file marching_squares.hpp
/// @brief Checked row-major contour cells with actual crossed-edge interpolation and explicit saddle connectivity.
#include <array>
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief One actual contour segment{x0,y0,x1,y1} in grid-index coordinates, column=x and row=y.
/// @par Example
/// @code
/// auto s=dm::algorithms::marching_squares({{1,0},{0,0}},.5);
/// std::print("{}\n",s.front()[0]); // 0
/// @endcode
/// @par References
/// - Nielson and Hamann(1991), The Asymptotic Decider, https://doi.org/10.1109/VISUAL.1991.175782; this API exposes fixed saddle pairings, not that asymptotic decider.
/// @see marching_squares_analyze(), marching_squares_trace()
using ContourSegment = std::array<double, 4>;
/// @brief Explicit topology choice for alternating case5/10 corner classifications; not an asymptotic decider.
/// @par Example
/// @code
/// auto rule=dm::algorithms::MarchingSaddleConnectivity::AboveDisconnected;
/// std::print("{}\n",int(rule)); // 0
/// @endcode
/// @par References
/// - Nielson and Hamann(1991), The Asymptotic Decider, https://doi.org/10.1109/VISUAL.1991.175782; this API exposes fixed saddle pairings, not that asymptotic decider.
/// @see marching_squares_analyze(), marching_squares_trace()
enum class MarchingSaddleConnectivity {
  AboveDisconnected, ///< Historical fixed pairs isolate each above-threshold corner; default preserves ordinary output.
  AboveConnected ///< Alternative fixed pairs connect above-threshold corners through the cell interior.
};
/// @brief Actual selected cell/classification/interpolation/emission work, excluding validation/copies.
/// @par Example
/// @code
/// auto t=dm::algorithms::marching_squares_trace({{1,0},{0,0}},.5);
/// std::print("{}\n",t.analysis.work.interpolations); // 2
/// @endcode
/// @par References
/// - Nielson and Hamann(1991), The Asymptotic Decider, https://doi.org/10.1109/VISUAL.1991.175782; this API exposes fixed saddle pairings, not that asymptotic decider.
/// @see marching_squares_analyze(), marching_squares_trace()
struct MarchingSquaresWork {
  std::size_t cells{0}; ///< Actual visited cells,(rows-1)*(cols-1) when both dimensions exceed1.
  std::size_t classifications{0}; ///< Actual corner comparisons, four per visited cell.
  std::size_t interpolations{
      0}; ///< Actual crossed-edge interpolation evaluations, never unused constant edges.
  std::size_t saddle_cells{0}; ///< Actual alternating corner cases5 or10.
  std::size_t segments{
      0}; ///< Actual emitted contour segments, including any zero-length tie segments.
};
/// @brief Owned actual row-major contour segments and selected producing work.
/// @par Example
/// @code
/// auto t=dm::algorithms::marching_squares_trace({{1,0},{0,0}},.5);
/// std::print("{}\n",t.analysis.segments.size()); // 1
/// @endcode
/// @par References
/// - Nielson and Hamann(1991), The Asymptotic Decider, https://doi.org/10.1109/VISUAL.1991.175782; this API exposes fixed saddle pairings, not that asymptotic decider.
/// @see marching_squares_analyze(), marching_squares_trace()
struct MarchingSquaresAnalysis {
  std::vector<ContourSegment>
      segments; ///< Actual independent emitted cell segments, not stitched/oriented/deduplicated curves.
  MarchingSquaresWork work; ///< Actual same-run classification/interpolation/emission work.
};
/// @brief One actual interpolation on a classified crossing edge.
/// @par Example
/// @code
/// auto t=dm::algorithms::marching_squares_trace({{1,0},{0,0}},.5);
/// std::print("{}\n",t.crossings.front().fraction); // 0.5
/// @endcode
/// @par References
/// - Nielson and Hamann(1991), The Asymptotic Decider, https://doi.org/10.1109/VISUAL.1991.175782; this API exposes fixed saddle pairings, not that asymptotic decider.
/// @see marching_squares_analyze(), marching_squares_trace()
struct MarchingCrossing {
  std::size_t row;    ///< Actual cell row.
  std::size_t column; ///< Actual cell column.
  unsigned
      edge; ///< Actual edge0:row0 horizontal,1:column1 vertical,2:row1 horizontal,3:column0 vertical.
  double first_value;  ///< Actual first edge sample in the native oriented edge order.
  double second_value; ///< Actual second edge sample.
  double
      fraction; ///< Actual finite interpolation fraction, with positive common scaling only if raw differences fail.
  std::array<double, 2> point; ///< Actual interpolated grid-coordinate point.
  bool scaled; ///< Whether nonfinite raw differences required positive common scaling.
};
/// @brief One actual completed cell classification and segment-output span.
/// @par Example
/// @code
/// auto t=dm::algorithms::marching_squares_trace({{1,0},{0,0}},.5);
/// std::print("{}\n",t.cells.front().case_index); // 1
/// @endcode
/// @par References
/// - Nielson and Hamann(1991), The Asymptotic Decider, https://doi.org/10.1109/VISUAL.1991.175782; this API exposes fixed saddle pairings, not that asymptotic decider.
/// @see marching_squares_analyze(), marching_squares_trace()
struct MarchingCell {
  std::size_t row;    ///< Actual cell row.
  std::size_t column; ///< Actual cell column.
  std::array<double, 4>
           values;     ///< Actual corner values in row0-left,row0-right,row1-right,row1-left order.
  unsigned case_index; ///< Actual four-bit mask from corner>=iso, in bits1,2,4,8.
  std::array<bool, 4> crossed; ///< Actual crossed-edge flags; only these edge points are valid.
  std::array<std::array<double, 2>, 4>
      points; ///< Actual interpolated points on flagged edges; unflagged entries are zero placeholders.
  std::size_t segment_begin; ///< Actual segment output size before this cell.
  std::size_t segment_end;   ///< Actual segment output size after this cell.
};
/// @brief Owned original grid and actual cell/crossing records from one contour producer.
/// @par Example
/// @code
/// auto t=dm::algorithms::marching_squares_trace({{1,0},{0,0}},.5);
/// std::print("{} {}\n",t.analysis.segments.size(),t.crossings.size()); // 1 2
/// @endcode
/// @par References
/// - Nielson and Hamann(1991), The Asymptotic Decider, https://doi.org/10.1109/VISUAL.1991.175782; this API exposes fixed saddle pairings, not that asymptotic decider.
/// @see marching_squares_analyze(), marching_squares_trace()
struct MarchingSquaresTrace {
  std::vector<std::vector<double>>
         grid;   ///< Complete original finite rectangular grid, independently owned.
  double iso{0}; ///< Actual finite requested level.
  MarchingSaddleConnectivity connectivity{
      MarchingSaddleConnectivity::AboveDisconnected}; ///< Actual explicit saddle rule.
  std::vector<MarchingCell>
      cells; ///< Actual row-major classifications and output spans, including empty cases.
  std::vector<MarchingCrossing>
      crossings; ///< Actual crossed-edge evaluations in native edge order, no interpolation replay.
  MarchingSquaresAnalysis analysis; ///< Actual same-run segments and work.
  std::size_t             captured_entries{
      0}; ///< Input scalar cells plus actual cell/crossing/emitted-segment records.
};
/// @brief Extract historical fixed-topology cell segments from a completely validated finite grid.
/// @param grid Rectangular finite row-major samples; empty, zero-width or one-dimensional grids yield no cells.
/// @param iso Finite threshold, with equality classified above.
/// @return Actual row-major segments; zero-length/duplicate tie geometry is retained, without curve stitching.
/// @throws std::invalid_argument For any ragged/nonfinite input or nonfinite iso, even if no cells exist.
/// @throws std::length_error For unrepresentable cell/output geometry or grid coordinates beyond exact integer doubles.
/// @note O(rows*columns) validation/extraction work plus output; no saddle asymptotic-decider or exact-level-set topology guarantee.
/// @code
/// auto s=dm::algorithms::marching_squares({{1,0},{0,0}},.5);
/// std::print("{}\n",s.size()); // 1
/// @endcode
std::vector<ContourSegment> marching_squares(
    const std::vector<std::vector<double>>& grid,
    double                                  iso);
/// @brief Run the same cell producer with an explicit saddle connectivity and retain its actual work.
/// @param grid Complete finite rectangular row-major samples.
/// @param iso Finite requested level.
/// @param connectivity AboveDisconnected preserves old pairs; AboveConnected selects the other saddle pairs.
/// @return Actual cell segments and selected work, without history.
/// @throws std::invalid_argument For complete invalid grid/level/rule inputs.
/// @throws std::length_error For unrepresentable complete geometry.
MarchingSquaresAnalysis marching_squares_analyze(
    const std::vector<std::vector<double>>& grid,
    double                                  iso,
    MarchingSaddleConnectivity connectivity = MarchingSaddleConnectivity::AboveDisconnected);
/// @brief Capture actual classifications, crossed-edge interpolation and segment emission without reconstruction.
/// @param grid Complete finite rectangular original samples.
/// @param iso Finite requested level.
/// @param connectivity Explicit saddle rule, default preserving old ordinary pairs.
/// @param max_entries Optional actual logical capture cap, default one million; zero disables it.
/// @return Owned original grid and complete actual producer records.
/// @throws std::invalid_argument Before capture on invalid complete inputs.
/// @throws std::length_error Before the next input/record copy exceeds its cap or geometry.
/// @note Counts original scalar samples and each real cell/crossing/emitted-segment record, excluding fields,
/// row headers, capacities, returned/runtime copies and physical bytes. Future cases are not fabricated; complete capture or exception.
MarchingSquaresTrace marching_squares_trace(
    const std::vector<std::vector<double>>& grid,
    double                                  iso,
    MarchingSaddleConnectivity connectivity = MarchingSaddleConnectivity::AboveDisconnected,
    std::size_t                max_entries  = 1000000);
/// @brief Owned prescribed radial field, actual contour and measured true-circle midpoint deviations.
/// @par Example
/// @code
/// auto s=dm::algorithms::marching_radial_study();
/// std::print("{}\n",s.midpoint_errors.size()); // 44
/// @endcode
/// @par References
/// - Nielson and Hamann(1991), The Asymptotic Decider, https://doi.org/10.1109/VISUAL.1991.175782; this API exposes fixed saddle pairings, not that asymptotic decider.
/// @see marching_squares_analyze(), marching_squares_trace()
struct MarchingRadialStudy {
  std::vector<std::vector<double>>
      grid; ///< Actual Euclidean-distance samples centered at(size-1)/2 in grid coordinates.
  MarchingSquaresAnalysis contour; ///< Actual native contour at the prescribed nonnegative radius.
  std::vector<double>
      midpoint_errors; ///< Actual absolute radial deviations at returned segment midpoints, not a uniform error certificate.
  double mean_midpoint_error{
      0}; ///< Mean observed deviation; zero placeholder when no segments exist.
  double maximum_midpoint_error{
      0}; ///< Largest observed deviation; zero placeholder when no segments exist.
};
/// @brief Sample a native radial field and measure actual segment-midpoint deviations from its true circle.
/// @param size Square-grid sample count per axis, default21 retaining the original center(10,10).
/// @param radius Finite nonnegative contour radius in grid-index units, default6 retaining the original figure.
/// @return Owned sampled field, actual historical-topology contour and its measured midpoint errors.
/// @throws std::invalid_argument For negative/nonfinite radius.
/// @throws std::length_error For unrepresentable allocation/coordinate geometry.
/// @note Empty/no-cell/no-crossing studies have no midpoint observations; mean/max zero are explicit placeholders then.
/// @code
/// auto s=dm::algorithms::marching_radial_study();
/// std::print("{}\n",s.contour.segments.size()); // 44
/// @endcode
MarchingRadialStudy marching_radial_study(std::size_t size = 21, double radius = 6);
} // namespace datamunge::algorithms
