#pragma once
/// @file bilinear_interpolation.hpp
/// @brief Checked tensor-product linear interpolation on rectangular grids.
#include <vector>

namespace datamunge::algorithms {
/// @brief Interpolate a rectangular grid by two successive linear blends.
/// Rectilinear axes may be unevenly spaced. Values use x-major layout:
/// z[i*ys.size()+j] belongs to (xs[i],ys[j]). Reproduces a+b*x+c*y+d*x*y
/// up to floating error. Queries outside either axis extrapolate using the
/// nearest end cell; they are not clamped. Inside a cell, the four nonnegative
/// corner weights sum to one, so the result remains in the corner-value range.
/// Validation costs O(nx+ny+nx*ny) per call, followed by binary cell lookup and
/// constant arithmetic; O(1) auxiliary storage, no mutation or shared state.
/// Comparable to SciPy RegularGridInterpolator(method="linear", bounds_error=false,
/// fill_value=None), restricted to two increasing axes and finite real data.
/// @param xs Finite, strictly increasing x coordinates, size nx >= 2.
/// @param ys Finite, strictly increasing y coordinates, size ny >= 2.
/// @param z Exactly nx*ny finite samples in x-major order.
/// @param x Finite query abscissa, including outside the grid.
/// @param y Finite query ordinate, including outside the grid.
/// @return Interpolated or extrapolated value, in the units of z.
/// @throws std::invalid_argument For short, nonfinite, nonincreasing axes,
/// mismatched/nonfinite samples or nonfinite queries.
/// @throws std::overflow_error If interpolation coordinates or the result are not representable.
/// @par Example
/// @code
/// const double value = dm::algorithms::bilinear_interpolate(
///     {0,2}, {0,4}, {0,4,2,10}, 0.5, 3);
/// std::print("{}\n", value); // 4.25
/// @endcode
/// @par References
/// - A. Weiser and S. E. Zarantonello (1988), A note on piecewise linear and
///   multilinear table interpolation in many dimensions, Mathematics of Computation
///   50(181), 189–196, doi:10.1090/S0025-5718-1988-0917826-0.
/// - https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html
/// @see linear_interpolate(), bicubic_interpolate(), resample_grid()
double bilinear_interpolate(const std::vector<double>& xs, const std::vector<double>& ys,
                            const std::vector<double>& z, double x, double y);
} // namespace datamunge::algorithms
