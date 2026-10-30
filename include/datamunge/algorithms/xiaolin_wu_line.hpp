#pragma once

/// @file xiaolin_wu_line.hpp
/// @brief Checked two-pixel antialiasing, endpoint policies and DataFrame polyline rasterisation.
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/plot/plot.hpp>
#include <string>
#include <vector>

namespace datamunge::algorithms {

/// @brief One additive antialiasing contribution at an integer pixel centre.
///
/// Brightness is a linear ink weight, not exact geometric area coverage or
/// gamma-encoded display luminance. Contributions at the same coordinate may
/// be added; their sum can exceed one. The zero-initialised value has no ink.
/// @par Example
/// @code
/// const dm::algorithms::WuPixel pixel{2,3,0.25};
/// const double ink = pixel.brightness; // 0.25
/// @endcode
/// @see xiaolin_wu_line(), wu_pixel_table()
struct WuPixel {
  int    x{0};          ///< Horizontal integer pixel centre; positive is right.
  int    y{0};          ///< Vertical integer pixel centre; positive is up.
  double brightness{0}; ///< This contribution's weight in [0,1].
};

/// @brief Major-axis endpoint convention for two-pixel antialiasing.
/// @par Example
/// @code
/// const auto policy = dm::algorithms::WuEndpointPolicy::ClippedColumns;
/// const auto pixels = dm::algorithms::xiaolin_wu_line(0,0,2,1,policy);
/// @endcode
/// @see xiaolin_wu_line()
enum class WuEndpointPolicy {
  FullColumns,   ///< Unit ink at every rounded major-axis coordinate, including endpoints (legacy).
  ClippedColumns ///< Weight each column by its width intersecting the segment's major-axis
                 ///< interval.
};

/// @brief Rasterise with the legacy full-column endpoint convention.
///
/// Preserves the four-argument API and its integer-endpoint two-pixel pairs.
/// Equivalent to the explicit-policy overload with FullColumns. Endpoints
/// are not half weighted; this is a simplified Wu variant, not an exact
/// coverage integral. O(P) time and output storage, O(1) auxiliary storage,
/// where P is the number of rounded major-axis steps. No shared mutable state.
/// @param x0 Finite first horizontal coordinate in [INT_MIN+2,INT_MAX-2].
/// @param y0 Finite first vertical coordinate in the same range.
/// @param x1 Finite second horizontal coordinate in the same range.
/// @param y1 Finite second vertical coordinate in the same range.
/// @return Owned adjacent-pixel pairs in increasing major-axis order, including zero weights.
/// @throws std::invalid_argument If any coordinate is non-finite or outside the safe int range.
/// @par Example
/// @code
/// const auto pixels = dm::algorithms::xiaolin_wu_line(0,0,12,5);
/// const auto count = pixels.size(); // 26
/// @endcode
/// @par References
/// - X. Wu (1991), An Efficient Antialiasing Technique, Computer Graphics 25(4), 143-152,
///   https://doi.org/10.1145/127719.122734 (two-pixel split; simplified endpoints here).
/// @see WuEndpointPolicy, wu_pixel_plot()
std::vector<WuPixel> xiaolin_wu_line(double x0, double y0, double x1, double y1);

/// @brief Rasterise using an explicit full-column or clipped-column endpoint convention.
///
/// Swaps axes for steep lines, orders the endpoints by major coordinate, and
/// evaluates the minor coordinate at each rounded major centre. For fractional
/// part f and column weight w the two contributions are w*(1-f), w*f.
/// ClippedColumns uses length([i-1/2,i+1/2] intersect [major0,major1]); for
/// separated endpoints this is the usual Wu endpoint gap weighting. It also
/// handles both endpoints in one column without double counting. Empty-width
/// columns are omitted. FullColumns uses w=1. Rounding is std::lround (ties
/// away from zero). Coincident endpoints explicitly draw a unit-weight point
/// split across the two adjacent minor-axis centres, under either policy.
/// O(P) time/output, O(1) auxiliary storage. Floating arithmetic may introduce
/// rounding error; this implements a distance-weight filter, not area coverage.
/// @param x0 Finite first x in [INT_MIN+2,INT_MAX-2].
/// @param y0 Finite first y in the same range.
/// @param x1 Finite second x in the same range.
/// @param y1 Finite second y in the same range.
/// @param endpoints Endpoint policy; unknown enum values are rejected.
/// @return Owned pixel pairs; duplicate coordinates across separate calls are not merged.
/// @throws std::invalid_argument If a coordinate or policy is invalid.
/// @par Example
/// @code
/// const auto pixels = dm::algorithms::xiaolin_wu_line(0,0,2,1,
///     dm::algorithms::WuEndpointPolicy::ClippedColumns);
/// const double first_pair_weight = pixels[0].brightness + pixels[1].brightness; // 0.5
/// @endcode
/// @par References
/// - X. Wu (1991), https://doi.org/10.1145/127719.122734
/// @see wu_polyline(), wu_pixel_table()
std::vector<WuPixel>
xiaolin_wu_line(double x0, double y0, double x1, double y1, WuEndpointPolicy endpoints);

/// @brief Map two numeric DataFrame columns into a pixel viewport and connect successive rows.
///
/// Each axis is scaled independently into [1,width-2] or [1,height-2], leaving
/// room for adjacent pixels. A constant axis maps to the viewport's centre.
/// Row order is preserved; there is no sorting, missing-value removal or data
/// smoothing. Empty input returns no contributions; a single row draws a point.
/// Zero-length consecutive segments draw point markers. Contributions from
/// adjacent segments are retained separately, so joins can accumulate ink.
/// O(n+P) time/output, O(n) temporary coordinates, for n rows and P raster steps.
/// The input is borrowed and unchanged; no shared mutable state.
/// @param data Frame with finite numeric coordinates (no missing cells).
/// @param x_column Horizontal column name.
/// @param y_column Vertical column name.
/// @param width Viewport width in pixels, at least 4.
/// @param height Viewport height in pixels, at least 4.
/// @param endpoints Endpoint policy, default ClippedColumns.
/// @return Owned pixel contributions in segment order; coordinates use y-up convention.
/// @throws std::invalid_argument If dimensions, columns, cells or endpoint policy are invalid.
/// @par Example
/// @code
/// const auto pixels = dm::algorithms::wu_polyline(dm::datasets::mtcars(),
///     "wt","mpg",96,32);
/// const bool nonempty = !pixels.empty(); // true
/// @endcode
/// @see xiaolin_wu_line(), wu_pixel_plot()
std::vector<WuPixel> wu_polyline(
    const dstruct::DataFrame& data,
    const std::string&        x_column,
    const std::string&        y_column,
    int                       width,
    int                       height,
    WuEndpointPolicy          endpoints = WuEndpointPolicy::ClippedColumns);

/// @brief Sum antialiasing contributions by integer pixel coordinate.
///
/// Retains zero-weight coordinates. Rows are ordered by x, then y. Summed
/// weight can exceed one; no saturation or gamma conversion is applied.
/// O(P log P) time and O(P) storage; input is borrowed and unchanged.
/// @param pixels Contributions with finite brightness in [0,1].
/// @return Owned x, y and weight columns; empty input gives a zero-row table.
/// @throws std::invalid_argument If a contribution has an invalid brightness.
/// @par Example
/// @code
/// const auto table = dm::algorithms::wu_pixel_table({{0,0,0.75},{0,0,0.5}});
/// const double weight = table.double_at("weight",0); // 1.25
/// @endcode
/// @see wu_pixel_plot()
dstruct::DataFrame wu_pixel_table(const std::vector<WuPixel>& pixels);

/// @brief Plot accumulated pixel ink as filled grey cells on white, with positive y upwards.
///
/// Merges duplicates, saturates weight at one, and maps grey=round(255*(1-weight)).
/// This linear grey mapping performs no gamma correction; cells describe the
/// numerical weights, not a calibrated display response. Axis scaling is caller-controlled;
/// use a matching canvas aspect ratio for square display cells. O(P log P) time and
/// O(P) result storage. The returned plot owns its data; calls share no state.
/// @param pixels Nonempty valid contributions, including any zero weights for bounds.
/// @return Owned plot with grid hidden and cell-centred bounds; callers can set titles/size.
/// @throws std::invalid_argument If pixels is empty or a brightness is invalid.
/// @par Example
/// @code
/// auto plot = dm::algorithms::wu_pixel_plot(dm::algorithms::xiaolin_wu_line(0,0,4,2));
/// plot.title("Antialiased line");
/// @endcode
/// @see wu_pixel_table(), wu_polyline()
graphics::RPlot wu_pixel_plot(const std::vector<WuPixel>& pixels);

} // namespace datamunge::algorithms
