#pragma once

/// @file bicubic_interpolation.hpp
/// @brief Bicubic interpolation on a regular 2-D grid by Keys' cubic convolution, and whole-grid
///        resampling by nearest-neighbour, bilinear or bicubic interpolation.
///
/// Bicubic convolution blends the 4x4 neighbourhood of a query with the separable kernel
/// \f$w_a\f$ of cubic_convolution_kernel(), a piecewise cubic on \f$|t|\le1\f$ and \f$1<|t|<2\f$.
/// Every \f$a\f$ gives a \f$C^1\f$ interpolant that passes through the samples and reproduces
/// constants. Only \f$a=-1/2\f$ (Keys 1981) also reproduces linear and quadratic functions, making
/// the error \f$O(h^3)\f$; any other \f$a\f$ leaves a bias \f$-(2a+1)t(1-t)(1-2t)hf'\f$ and is
/// only first order. Indices outside the grid are clamped to
/// the edge, so near the border the accuracy drops to first order.

#include <cmath>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// @brief Implementation helpers shared with tricubic_interpolation.hpp.
namespace detail {

/// @brief Keys cubic-convolution kernel with \f$a=-1/2\f$.
/// @param t Offset in grid units.
/// @return \f$w_{-1/2}(t)\f$.
inline double bicubic_kernel(double t) {
    constexpr double a = -0.5;
    t                  = std::fabs(t);
    if (t <= 1.0) return ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0;
    if (t < 2.0) return (((t - 5.0) * t + 8.0) * t - 4.0) * a;
    return 0.0;
}

/// @brief Clamps a grid index into [0, n-1] so queries near the border stay in range.
/// @param i Index, possibly negative or past the end.
/// @param n Number of grid lines.
/// @return The clamped index.
inline std::size_t bicubic_clamp(long i, std::size_t n) {
    if (i < 0) return 0;
    if (i >= static_cast<long>(n)) return n - 1;
    return static_cast<std::size_t>(i);
}

} // namespace detail

/// @brief Keys' cubic-convolution kernel \f$w_a(t)\f$ with free parameter @p a.
///
/// \f$w_a(0)=1\f$ and \f$w_a(k)=0\f$ at other integers, so the interpolant passes through the
/// samples. Common choices: \f$a=-1/2\f$ (Keys, third order), \f$a=-3/4\f$ (sharper, used by
/// some image libraries, but first order), \f$a=-1\f$ (sharper still).
///
/// @param t Offset in grid units.
/// @param a Kernel parameter (the slope \f$w_a'(1)\f$).
/// @return The kernel value.
double cubic_convolution_kernel(double t, double a = -0.5);

/// @brief Bicubic (cubic-convolution) interpolation at one point.
///
/// @param xs Grid abscissae, uniformly spaced and increasing (at least 2).
/// @param ys Grid ordinates, uniformly spaced and increasing (at least 2).
/// @param z Values, row-major: @c z[i*ny + j] is the value at @c (xs[i], ys[j]).
/// @param x,y The query point.
/// @param a Kernel parameter; see cubic_convolution_kernel().
/// @return The interpolated value.
///
/// @par Example
/// @code
/// const double v = dm::algorithms::bicubic_interpolate(xs, ys, z, 1.25, 0.4);
/// @endcode
double bicubic_interpolate(const std::vector<double>& xs, const std::vector<double>& ys,
                           const std::vector<double>& z, double x, double y, double a = -0.5);

/// @brief Interpolation rule used by resample_grid().
enum class GridInterpolation {
    Nearest,  ///< Value of the nearest grid node (piecewise constant, first order).
    Bilinear, ///< Bilinear blend of the 2x2 surrounding nodes (continuous, second order).
    Bicubic,  ///< Cubic convolution over the 4x4 surrounding nodes (\f$C^1\f$; third order for a = -1/2, first order otherwise).
};

/// @brief Evaluates a gridded function on a new grid.
///
/// Typical uses are image magnification and reconstructing a field from a coarser sampling.
///
/// @param xs,ys Source grid lines, uniformly spaced and increasing (at least 2 each).
/// @param z Source values, row-major: @c z[i*ys.size() + j].
/// @param new_xs,new_ys Target grid lines (any order or spacing).
/// @param method Interpolation rule.
/// @param a Kernel parameter for GridInterpolation::Bicubic.
/// @return Target values, row-major: element @c i*new_ys.size() + j is the value at
///         @c (new_xs[i], new_ys[j]).
/// @throws std::invalid_argument if a source axis has fewer than 2 lines or z has the wrong size.
std::vector<double> resample_grid(const std::vector<double>& xs, const std::vector<double>& ys,
                                  const std::vector<double>& z, const std::vector<double>& new_xs,
                                  const std::vector<double>& new_ys, GridInterpolation method,
                                  double a = -0.5);

} // namespace datamunge::algorithms
