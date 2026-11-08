#pragma once

#include <datamunge/algorithms/cubic_spline.hpp>
#include <datamunge/algorithms/de_casteljau.hpp>
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


} // namespace datamunge::algorithms
