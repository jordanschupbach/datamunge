#pragma once

#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

/// @brief *Lagrange interpolation*: evaluates, at the point @p x, the unique polynomial of degree
///        @c <n that passes through the @c n data points @c (xs[i], ys[i]). It sums the data values
///        weighted by the Lagrange basis polynomials
///        @c L_j(x) = prod_{k != j} (x - xs[k]) / (xs[j] - xs[k]), each of which is 1 at @c xs[j] and
///        0 at every other node. Runs in @c O(n^2) per evaluation. The @p xs must be distinct.
///
/// @param xs the (distinct) node abscissae.
/// @param ys the node values.
/// @param x  the point at which to evaluate the interpolating polynomial.
/// @return the interpolated value @c P(x).
double lagrange_interpolate(const std::vector<double>& xs, const std::vector<double>& ys, double x);

/// @brief *Neville's algorithm*: evaluates the same interpolating polynomial as
///        @ref lagrange_interpolate at @p x, but via a numerically stable recurrence that builds a
///        triangular table of successively higher-degree interpolants -- combining two
///        neighbouring lower-degree estimates at each step. It is the method of choice when only the
///        /value/ (not the polynomial's coefficients) is needed, and it naturally yields an error
///        estimate from the spread of the table. Runs in @c O(n^2) per evaluation.
///
/// @param xs the (distinct) node abscissae.
/// @param ys the node values.
/// @param x  the evaluation point.
/// @return the interpolated value @c P(x).
double neville_interpolate(const std::vector<double>& xs, const std::vector<double>& ys, double x);

/// @brief A piecewise-cubic spline: the knot abscissae @ref xs and, for each of the @c n-1
///        intervals, the coefficients of @c S_i(t) = a_i + b_i t + c_i t^2 + d_i t^3 with
///        @c t = x - xs[i]. Produced by @ref natural_cubic_spline and evaluated by
///        @ref cubic_spline_eval.
struct CubicSpline {
    std::vector<double> xs; ///< knot abscissae (length n).
    std::vector<double> a;  ///< constant coefficients (length n-1); a_i = ys[i].
    std::vector<double> b;  ///< linear coefficients (length n-1).
    std::vector<double> c;  ///< quadratic coefficients (length n-1).
    std::vector<double> d;  ///< cubic coefficients (length n-1).
};

/// @brief Builds the *natural cubic spline* interpolating the points @c (xs[i], ys[i]): the unique
///        piecewise-cubic that passes through every point, has continuous first and second
///        derivatives across the interior knots (@c C^2), and satisfies the *natural* boundary
///        condition @c S''=0 at both ends. Interior second-derivative "moments" are found by solving
///        a symmetric tridiagonal system (Thomas algorithm) in @c O(n) time. Unlike a single
///        high-degree polynomial, the spline resists the wild oscillation (Runge's phenomenon) of
///        equispaced interpolation.
///
/// @param xs the knot abscissae, strictly increasing (length n >= 2).
/// @param ys the values at the knots (length n).
/// @return the fitted @ref CubicSpline.
CubicSpline natural_cubic_spline(const std::vector<double>& xs, const std::vector<double>& ys);

/// @brief Evaluates a @ref CubicSpline at @p x, locating the containing interval by binary search
///        (clamping to the end intervals for @p x outside the knot range) and evaluating that
///        interval's cubic.
///
/// @param spline a spline from @ref natural_cubic_spline.
/// @param x      the evaluation point.
/// @return the spline value @c S(x).
double cubic_spline_eval(const CubicSpline& spline, double x);

/// @brief A point in the plane, used for Bezier control points and curve samples.
struct PlanarPoint {
    double x{0.0};
    double y{0.0};
};

/// @brief *De Casteljau's algorithm*: evaluates the Bezier curve defined by the @p control points at
///        parameter @p t in @c [0,1], by repeated linear interpolation. It builds a pyramid of
///        points, each level replacing consecutive points by their @c (1-t):(t) blend, until a
///        single point -- the curve at @p t -- remains. It is the numerically stable standard for
///        Bezier evaluation and equals the Bernstein-polynomial form
///        @c sum_i C(n,i) t^i (1-t)^{n-i} P_i. Runs in @c O(n^2) per evaluation. The curve passes
///        through the first and last control points and lies within their convex hull.
///
/// @param control the Bezier control points (at least one).
/// @param t       the curve parameter in @c [0,1].
/// @return the point on the Bezier curve at @p t.
PlanarPoint de_casteljau(const std::vector<PlanarPoint>& control, double t);

} // namespace datamunge::algorithms
