#pragma once

/// @file midpoint_circle.hpp
/// @brief Canonical midpoint circle API, producing histories and checked native contracts.

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// @brief One integer pixel centre as (x,y).
/// @par Example
/// @code
/// dm::algorithms::CirclePoint p{3,4};
/// std::print("{}\n",p.first); // 3
/// @endcode
/// @see midpoint_circle
/// @par Reference
/// Integer midpoint rasterization uses pixel centres on the square lattice.
using CirclePoint = std::pair<int, int>;

/// @brief Actual constant-state rasterization counts.
/// @details Reflection emissions retain duplicates at axes and diagonals.
/// @par Example
/// @code
/// auto a=dm::algorithms::midpoint_circle_analyze(0,0,0);
/// std::print("{} {}\n",a.iterations,a.points.size()); // 1 8
/// @endcode
/// @see midpoint_circle_analyze
/// @par Reference
/// Foley et al., Computer Graphics: Principles and Practice, midpoint circle algorithm.
struct MidpointCircleAnalysis {
  std::vector<CirclePoint>
      points; ///< Actual eight reflection emissions per visited octant pixel, in legacy order.
  std::size_t iterations{0};      ///< Actual visited x>=y octant states.
  std::size_t east_steps{0};      ///< Actual d<0 decisions; x remains fixed.
  std::size_t southeast_steps{0}; ///< Actual d>=0 decisions; x is decremented.
};

/// @brief One actual pre-emission state and its following integer update.
/// @details y increases on every iteration, including the final update after the last emitted octant pixel.
/// @par Example
/// @code
/// auto t=dm::algorithms::midpoint_circle_trace(0,0,0);
/// std::print("{} {}\n",t.steps[0].d,t.steps[0].d_after); // 1 6
/// @endcode
/// @see MidpointCircleTrace
/// @par Reference
/// Standard midpoint circle recurrence d=1-r, then east or southeast update.
struct MidpointCircleStep {
  int          x{0};        ///< Actual octant x before emission.
  int          y{0};        ///< Actual octant y before emission.
  std::int64_t d{0};        ///< Actual integer decision before branch selection.
  bool         east{false}; ///< Actual old decision d<0.
  int x_after{0}; ///< Actual x after the branch, possibly -1 in the radius-zero final state.
  int y_after{0}; ///< Actual y after the mandatory increment.
  std::int64_t d_after{0};     ///< Actual updated decision.
  std::size_t  first_point{0}; ///< First of eight same-run reflection emissions.
};

/// @brief Complete owned integer-circle execution, with actual reflection output and transitions.
/// @details The quota counts three input scalars, actual point records and actual step records;
/// record fields, capacities, runtime state and bytes are excluded.
/// @par Example
/// @code
/// auto t=dm::algorithms::midpoint_circle_trace(0,0,0);
/// std::print("{}\n",t.captured_entries); // 12
/// @endcode
/// @see midpoint_circle_trace
/// @par Reference
/// Captured standard midpoint circle producing recurrence.
struct MidpointCircleTrace {
  int                    cx{0};          ///< Owned input centre x.
  int                    cy{0};          ///< Owned input centre y.
  int                    radius{0};      ///< Owned original radius, negative means no emissions.
  MidpointCircleAnalysis analysis;       ///< Same-run raw points and actual work.
  std::vector<MidpointCircleStep> steps; ///< Actual pre-state, branch and post-state records.
  std::size_t captured_entries{0};       ///< Complete logical count, three plus nine per iteration.
};

/// @brief Emit integer circle pixels using the legacy midpoint recurrence and eight-way reflection.
/// @details Starts x=r,y=0,d=1-r. Emit (cx+x,cy+y), (cx-x,cy+y),
/// (cx+x,cy-y), (cx-x,cy-y), (cx+y,cy+x), (cx-y,cy+x),
/// (cx+y,cy-x), (cx-y,cy-x) while x>=y. Increment y; if old d<0
/// add 2*y+1, otherwise decrement x and add 2*(y-x)+1.
/// Signed 64-bit decisions avoid legacy int overflow. Duplicate axes/diagonal
/// emissions are intentional; this is a sequence, not a mathematical set.
/// Negative radius returns empty. Time O(r+1), O(1) state plus O(r+1) output.
/// @param cx Centre x in integer pixel coordinates.
/// @param cy Centre y in integer pixel coordinates.
/// @param r Integer radius; negative gives empty output.
/// @return Actual reflection emissions in original order, including eight copies for radius zero.
/// @throws std::overflow_error Before work if any bounding coordinate cx±r or cy±r is outside int.
/// @throws std::length_error If output geometry exceeds representable container sizes.
/// @par Example
/// @code
/// auto p=dm::algorithms::midpoint_circle(0,0,0);
/// std::print("{}\n",p.size()); // 8
/// @endcode
/// @see midpoint_circle_analyze, midpoint_circle_trace
/// @par Reference
/// Foley et al., Computer Graphics: Principles and Practice, midpoint rasterization.
std::vector<CirclePoint> midpoint_circle(int cx, int cy, int r);

/// @brief Return raw emissions and actual branch counts from the same canonical producer.
/// @param cx Integer centre x.
/// @param cy Integer centre y.
/// @param r Integer radius; negative is empty.
/// @return Actual points, iteration count, east decisions and southeast decisions.
/// @throws std::overflow_error If bounding coordinates cannot be represented as int.
/// @throws std::length_error If output geometry cannot be represented.
/// @par Example
/// @code
/// auto a=dm::algorithms::midpoint_circle_analyze(0,0,0);
/// std::print("{}\n",a.southeast_steps); // 1
/// @endcode
/// @see midpoint_circle, midpoint_circle_trace
/// @par Reference
/// Standard midpoint circle integer recurrence.
MidpointCircleAnalysis midpoint_circle_analyze(int cx, int cy, int r);

/// @brief Capture actual integer states, branches and reflected emissions in one run.
/// @details Admission of three input scalars occurs first; each complete next iteration
/// admits eight output points and one step before emission. Default cap is one million;
/// zero disables that optional cap. Cap failure throws without returning a prefix.
/// @param cx Integer centre x.
/// @param cy Integer centre y.
/// @param r Integer radius; negative yields no iterations and three captured entries.
/// @param max_entries Optional complete logical entry cap, zero disables it.
/// @return Owned inputs and actual same-run pre/post states and points.
/// @throws std::overflow_error Before capture if bounding coordinates exceed int.
/// @throws std::length_error If geometry or actual complete capture admission exceeds its cap.
/// @par Example
/// @code
/// auto t=dm::algorithms::midpoint_circle_trace(0,0,0,12);
/// std::print("{}\n",t.steps.size()); // 1
/// @endcode
/// @see MidpointCircleTrace, midpoint_circle_analyze
/// @par Reference
/// Captured standard midpoint circle producer.
MidpointCircleTrace midpoint_circle_trace(int cx, int cy, int r, std::size_t max_entries = 1000000);

/// @brief Native summary of actual emitted pixels and measured radial residuals.
/// @details Sorting/deduplication adds O(r log(r+1)) work to the O(r+1) producer.
/// Distances are observations from double hypot, not an all-radius exact error certificate.
/// @par Example
/// @code
/// auto s=dm::algorithms::midpoint_circle_study(0,0,0);
/// std::print("{} {}\n",s.unique_points.size(),s.duplicate_emissions); // 1 7
/// @endcode
/// @see midpoint_circle_study
/// @par Reference
/// Standard midpoint rasterization and Euclidean radial distance.
struct MidpointCircleStudy {
  MidpointCircleAnalysis analysis; ///< Canonical same-run raw emissions and selected work.
  std::vector<CirclePoint>
      unique_points; ///< Lexicographically sorted distinct actual emitted pixels.
  std::vector<double>
      radial_errors; ///< Absolute hypot(dx,dy)-radius for each unique pixel, in matching order.
  std::size_t duplicate_emissions{0}; ///< Raw emission count minus distinct actual pixel count.
  double      maximum_radial_error{
      0}; ///< Largest observed residual; zero placeholder when there are no points.
};

/// @brief Summarize distinct actual emitted pixels and their measured radial residuals.
/// @param cx Integer centre x.
/// @param cy Integer centre y.
/// @param r Integer radius, negative yields empty pixels and a zero maximum placeholder.
/// @return Canonical output, sorted distinct pixels, duplicate count and measured errors.
/// @throws std::overflow_error If bounding coordinates exceed int.
/// @throws std::length_error If output geometry cannot be represented.
/// @par Example
/// @code
/// auto s=dm::algorithms::midpoint_circle_study(0,0,12);
/// std::print("{}\n",s.analysis.points.size()); // 72
/// @endcode
/// @see MidpointCircleStudy, midpoint_circle_analyze
/// @par Reference
/// Euclidean radial error of actual standard midpoint rasterization pixels.
MidpointCircleStudy midpoint_circle_study(int cx, int cy, int r);

/// @brief Original report radius, twelve pixels.
/// @return The integer 12.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::midpoint_circle_demo_radius()); // 12
/// @endcode
/// @see midpoint_circle
/// @par Reference
/// Original datamunge midpoint_circle_analysis worked example.
int midpoint_circle_demo_radius();

} // namespace datamunge::algorithms
