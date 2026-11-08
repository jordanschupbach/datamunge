#pragma once
/// @file
/// @brief Checked accumulated DDA rasterisation, actual samples and exact rational comparisons.
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace datamunge::algorithms {
/// @brief Owned integer pixel-centre pair (x,y), retaining the original DDA alias.
/// Centres do not imply cell-area coverage, clipping, antialiasing or intensity.
/// @par Example
/// @code
/// const dm::algorithms::DDAPoint p{2,-1};std::print("{}\n",p); // (2, -1)
/// @endcode
/// @see dda_line()
using DDAPoint=std::pair<int,int>;

/// @brief Actual pre-rounding accumulator and its exact-rational sample comparison.
/// index is the zero-based major-axis sample. point rounds the stored double
/// coordinates halfway away from zero; exact_point rounds the exact rational
/// endpoint interpolation using the same tie rule. Drift and pixel errors are
/// double diagnostics computed against integer quotient/remainder pieces of the
/// exact sample, avoiding subtraction of two large nearly equal coordinates.
/// They are approximations to real errors, not directed interval bounds.
/// Owned values have no caller lifetime dependency or shared mutable state.
/// @par Example
/// @code
/// const auto t=dm::algorithms::dda_line_trace(0,0,4,2);
/// std::print("{} {} {}\n",t.samples[1].y,t.samples[1].point,t.samples[1].exact_point); // 0.5 (1, 1) (1, 1)
/// @endcode
/// @see DdaTrace, dda_exact_sample()
struct DdaSample {
  std::uint64_t index{0}; ///< Zero-based interpolation index, inclusive through steps.
  double x{0}; ///< Actual current x accumulator before rounding or the next addition.
  double y{0}; ///< Actual current y accumulator before rounding or the next addition.
  DDAPoint point; ///< Actual rounded accumulator centre, also in the trace points vector.
  DDAPoint exact_point; ///< Exact rational nearest centre, with signed ties away from zero.
  double x_drift{0}; ///< Actual x accumulator minus exact rational x, approximately.
  double y_drift{0}; ///< Actual y accumulator minus exact rational y, approximately.
  double x_error{0}; ///< Actual rounded x minus exact rational x, approximately.
  double y_error{0}; ///< Actual rounded y minus exact rational y, approximately.
};

/// @brief Owned ordinary DDA points, actual accumulators and checked work diagnostics.
/// The floating kernel is exactly the repeated-addition implementation, without
/// endpoint snapping, clamping or replacement by the rational reference. One
/// record per returned centre; no replay. Counts exclude final unused additions:
/// coordinate_additions=2*steps and roundings=2*(steps+1), including the coincident
/// case with zero additions and two rounds. Counters exclude reference/diagnostic
/// arithmetic, allocation and copying. Storage and work are O(steps+1), with
/// O(1) kernel state. Calls are independent and returned containers own all data.
/// @par Example
/// @code
/// const auto t=dm::algorithms::dda_line_trace(0,0,4,2);
/// std::print("{} {} {}\n",t.points.size(),t.coordinate_additions,t.roundings); // 5 8 10
/// @endcode
/// @see dda_line_trace(), DdaSample
struct DdaTrace {
  std::vector<DDAPoint> points; ///< Complete actual rounded accumulator sequence.
  std::vector<DdaSample> samples; ///< Actual pre-addition states and rational comparisons.
  std::int64_t dx{0}; ///< Signed x1-x0, widened before subtraction.
  std::int64_t dy{0}; ///< Signed y1-y0, widened before subtraction.
  std::uint64_t steps{0}; ///< max(abs(dx),abs(dy)); zero for coincident endpoints.
  double x_increment{0}; ///< Rounded double dx/steps, zero when steps is zero.
  double y_increment{0}; ///< Rounded double dy/steps, zero when steps is zero.
  std::uint64_t coordinate_additions{0}; ///< Actual nonterminal x and y additions, 2*steps.
  std::uint64_t roundings{0}; ///< Two actual coordinate-round calls per sample.
  std::uint64_t differing_samples{0}; ///< Samples whose rounded centre differs from exact_point.
  double max_coordinate_error{0}; ///< Maximum absolute approximate x/y pixel error.
  double max_accumulator_drift{0}; ///< Maximum absolute approximate x/y accumulation error.
};

/// @brief Count inclusive DDA samples without allocating or walking the line.
/// Widened differences cover all int coordinates up to 32 bits. No mutation,
/// shared state or allocation; O(1) time/storage. Includes one coincident point.
/// @param x0 Initial horizontal coordinate.
/// @param y0 Initial vertical coordinate.
/// @param x1 Final horizontal coordinate.
/// @param y1 Final vertical coordinate.
/// @return max(abs(x1-x0),abs(y1-y0))+1, in uint64.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::dda_point_count(0,0,14,6)); // 15
/// @endcode
/// @see dda_line(), dda_line_trace()
std::uint64_t dda_point_count(int x0,int y0,int x1,int y1) noexcept;

/// @brief Round one exact rational endpoint interpolation to its nearest centre.
/// At N=max(abs(dx),abs(dy)), sample i is (x0+i*dx/N,y0+i*dy/N).
/// Uses unsigned 64-bit products, integer quotient/remainder and signed
/// halfway-away-from-zero rounding, with no floating selection or accumulation.
/// The largest product is (2^32-1)^2, which fits uint64 for 32-bit int.
/// Coincident endpoints accept only index zero. O(1) work/storage, no allocation,
/// input mutation or shared state. The result lies within the endpoint rectangle.
/// @param x0 Initial horizontal coordinate.
/// @param y0 Initial vertical coordinate.
/// @param x1 Final horizontal coordinate.
/// @param y1 Final vertical coordinate.
/// @param index Sample in inclusive range 0..N.
/// @return Exact nearest integer pair at that rational parameter value.
/// @throws std::out_of_range index exceeds N.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::dda_exact_sample(0,0,12,1,6)); // (6, 1)
/// @endcode
/// @see dda_line_exact(), dda_line_trace()
DDAPoint dda_exact_sample(int x0,int y0,int x1,int y1,std::uint64_t index);

/// @brief Rasterise integer endpoints using the original repeated-double-addition DDA.
/// Retains the four-argument signature, DDAPoint and accumulator output on safe
/// original inputs. Uses N=max(abs(dx),abs(dy)), initial exact integer doubles,
/// increments double(dx)/N and double(dy)/N, then rounds each current coordinate
/// halfway away from zero. Coincident endpoints return their single centre.
/// Both coordinate differences and loop/count arithmetic are widened. No clipping,
/// endpoint snapping or tie repair is performed: small floating drift may select
/// the other side of a rational tie, and sufficiently large drift may miss the
/// final minor endpoint. Returned centres need not equal dda_line_exact().
/// O(N+1) time and owned output storage, O(1) auxiliary state; no trace/reference
/// arithmetic or input/shared mutation. No optional allocation cap; large requests
/// can exhaust memory. Floating results assume ordinary non-fast-math evaluation;
/// the current floating environment is neither changed nor restored.
/// @param x0 Initial horizontal coordinate, any int up to 32 bits.
/// @param y0 Initial vertical coordinate.
/// @param x1 Final horizontal coordinate.
/// @param y1 Final vertical coordinate.
/// @return Complete N+1 actual accumulator centres, or an exception; never truncates.
/// @throws std::length_error Complete output exceeds container capacity.
/// @throws std::overflow_error Any rounded accumulator is nonfinite or outside int.
/// @throws std::bad_alloc Allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::dda_line(0,0,4,2)); // [(0, 0), (1, 1), (2, 1), (3, 2), (4, 2)]
/// @endcode
/// @par References
/// - C. Flanagan, Line Drawing on Raster Displays, Digital Differential Analyser section, https://www.cs.helsinki.fi/group/goa/mallinnus/lines/gsoft2.html . This variant uses double, both coordinates, inclusive major-axis sampling and signed ties away from zero, rather than floor(v+0.5).
/// @see dda_line_trace(), dda_line_exact(), dda_point_count()
std::vector<DDAPoint> dda_line(int x0,int y0,int x1,int y1);

/// @brief Record actual DDA accumulators, rounding and rational-reference differences.
/// Uses the same executing kernel as dda_line(), without replay. Retains every
/// original accumulator choice and adds exact integer nearest-centre comparisons
/// plus approximate drift/coordinate-error diagnostics. A positive complete-point
/// limit is checked before all allocation; zero disables only that optional cap.
/// No truncated raster or partial trace is returned on failure. O(N+1) time and
/// storage; reference work is additional to the recorded floating operation counts.
/// @param x0 Initial horizontal coordinate.
/// @param y0 Initial vertical coordinate.
/// @param x1 Final horizontal coordinate.
/// @param y1 Final vertical coordinate.
/// @param max_points Maximum complete output points; default 1000000, zero disables.
/// @return Owned actual points, samples, increments, work counts and diagnostics.
/// @throws std::length_error Optional cap or either vector capacity is exceeded before allocation.
/// @throws std::overflow_error Any rounded accumulator is nonfinite or outside int.
/// @throws std::bad_alloc Allocation fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::dda_line_trace(0,0,4,2,5);
/// std::print("{} {}\n",t.differing_samples,t.max_coordinate_error); // 0 0.5
/// @endcode
/// @par References
/// - C. Flanagan, Line Drawing on Raster Displays, https://www.cs.helsinki.fi/group/goa/mallinnus/lines/gsoft2.html ; records this library's inclusive double variant.
/// @see dda_line(), DdaTrace, dda_exact_sample()
DdaTrace dda_line_trace(int x0,int y0,int x1,int y1,std::size_t max_points=1000000);

/// @brief Rasterise exact rational major-axis samples with signed nearest-centre ties.
/// Calls the same integer quantizer as dda_exact_sample() at every index. Both
/// endpoints are included; for distinct endpoints the centres are 8-connected,
/// with one unique centre per major-axis coordinate. Coordinate error relative
/// to each exact sample is at most one half; minor rounding also bounds distance
/// to the geometric segment by one half. Reversal returns the reversed sequence.
/// This is a mathematical comparison API, not a replacement for the repeated
/// double accumulator. It is not a cell supercover or antialiased raster.
/// O(N+1) time/output storage and O(1) auxiliary state; integer products fit uint64
/// for all supported endpoints. Independent calls own results and share no state.
/// @param x0 Initial horizontal coordinate.
/// @param y0 Initial vertical coordinate.
/// @param x1 Final horizontal coordinate.
/// @param y1 Final vertical coordinate.
/// @param max_points Complete output cap, default 1000000; zero disables only optional cap.
/// @return Complete owned exact-sample nearest centres.
/// @throws std::length_error Optional point cap or vector capacity exceeded before allocation.
/// @throws std::bad_alloc Allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::dda_line_exact(0,0,2,-1)); // [(0, 0), (1, -1), (2, -1)]
/// @endcode
/// @see dda_line(), dda_exact_sample()
std::vector<DDAPoint> dda_line_exact(int x0,int y0,int x1,int y1,std::size_t max_points=1000000);
} // namespace datamunge::algorithms
