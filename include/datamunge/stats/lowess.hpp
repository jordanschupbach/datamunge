#pragma once

/// @file lowess.hpp
/// @brief Robust locally weighted scatterplot smoothing, R's `lowess()` (Cleveland 1979).

#include <vector>

namespace datamunge::stats {

/// @brief Smoothed curve returned by `lowess`: the x values sorted increasingly, and the
///        smoothed y at each.
struct LowessFit {
    std::vector<double> x; ///< Sorted x values.
    std::vector<double> y; ///< Smoothed values at each x.
};

/// @brief Robust locally weighted regression smoother, matching R's
///        `lowess(x, y, f, iter, delta)`.
///
/// At each x a straight line is fitted by weighted least squares to the nearest
/// \f$\lceil fn\rceil\f$ points, with tricube weights \f$(1-(d/h)^3)^3\f$ in the distance \f$d\f$
/// scaled by the largest distance \f$h\f$ in the window. Then @p iter robustness passes refit
/// with each point's weight multiplied by the bisquare \f$(1-(r/6M)^2)^2\f$ of its residual
/// \f$r\f$, where \f$M\f$ is the median absolute residual, so outliers lose their pull. Points
/// within @p delta of the last fitted x are linearly interpolated instead of fitted.
/// @param x Predictor values (any order).
/// @param y Response values, same length.
/// @param f Smoother span: the fraction of points in each local fit (R default 2/3).
/// @param iter Number of robustness iterations (R default 3).
/// @param delta Interpolation distance; negative (the default) uses R's 1% of the x range.
/// @return The sorted x values and the smoothed y values.
/// @throws std::invalid_argument If the lengths differ, or f is not positive.
///
/// @par Example
/// @code
/// const auto fit = dm::stats::lowess(dm::c(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), dm::c(1.0, 3.0, 2.0, 5.0, 4.0, 6.0));
/// std::print("{::.3f}\n", fit.y);  // [1.339, 2.141, 3.157, 3.843, 4.859, 5.661]
/// @endcode
///
/// @par References
/// - W. S. Cleveland (1979). "Robust locally weighted regression and smoothing scatterplots".
///   Journal of the American Statistical Association 74(368), 829-836.
///   doi:10.1080/01621459.1979.10481038
/// - W. S. Cleveland (1981). "LOWESS: A program for smoothing scatterplots by robust locally
///   weighted regression". The American Statistician 35(1), 54. doi:10.2307/2683591
/// - R `lowess()`: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/lowess.html
///
/// @see LM::plot_residuals_vs_fitted()
LowessFit lowess(const std::vector<double>& x, const std::vector<double>& y, double f = 2.0 / 3.0, int iter = 3,
                 double delta = -1.0);

} // namespace datamunge::stats
