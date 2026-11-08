#pragma once
#include <datamunge/algorithms/bilinear_interpolation.hpp>

#include <vector>

namespace datamunge::algorithms {

/// @brief *Piecewise linear interpolation*: connect consecutive sample points @c (xs[i], ys[i]) with
///        straight segments and read off the value at @p x. It is the simplest continuous
///        interpolant -- exact for data that lies on a line, @c C^0 (continuous but kinked at the
///        nodes), and the basis of everything from lookup tables to graphics. Outside the sample
///        range it *extrapolates* along the nearest end segment.
///
/// @param xs the sample abscissae, strictly increasing (size >= 2).
/// @param ys the sample values (same size as @p xs).
/// @param x the query point.
/// @return the interpolated value at @p x.
double linear_interpolate(const std::vector<double>& xs, const std::vector<double>& ys, double x);

/// @brief *Cubic Hermite interpolation*: on each interval fit the unique cubic that matches both the
///        *values* @c ys and the *slopes* @p slopes at the two endpoints, using the Hermite basis
///        @c h00,h10,h01,h11. The result is @c C^1 (continuous value and first derivative) and lets
///        you sculpt the curve by choosing tangents -- the mechanism behind Catmull-Rom splines and
///        animation easing curves. Supplying different slopes at the same points yields visibly
///        different curves through them.
///
/// @param xs the node abscissae, strictly increasing (size >= 2).
/// @param ys the node values (same size as @p xs).
/// @param slopes the desired derivative @c dy/dx at each node (same size as @p xs).
/// @param x the query point (clamped to the node range).
/// @return the interpolated value at @p x.
double hermite_interpolate(const std::vector<double>& xs, const std::vector<double>& ys,
                           const std::vector<double>& slopes, double x);

/// @brief *Monotone cubic interpolation* (Fritsch-Carlson, 1980): a cubic Hermite interpolant whose
///        node slopes are *chosen* so the curve never overshoots -- if the data is monotone on an
///        interval, so is the interpolant. It computes secant slopes, averages them for initial
///        tangents, then *limits* each tangent into the Fritsch-Carlson monotonicity region
///        (@c alpha^2+beta^2 <= 9). This removes the spurious wiggles a natural cubic spline
///        introduces near sharp changes, at the cost of @c C^1 (not @c C^2) smoothness -- the right
///        choice for monotone data like cumulative distributions.
///
/// @param xs the node abscissae, strictly increasing (size >= 2).
/// @param ys the node values (same size as @p xs).
/// @param x the query point (clamped to the node range).
/// @return the interpolated value at @p x.
double monotone_cubic_interpolate(const std::vector<double>& xs, const std::vector<double>& ys, double x);

} // namespace datamunge::algorithms
