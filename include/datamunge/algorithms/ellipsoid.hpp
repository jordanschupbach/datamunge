#pragma once

#include <functional>
#include <vector>

namespace datamunge::algorithms {

/// @brief The result of an ellipsoid-method minimization: the best point @ref x found, its objective
///        @ref value, and the number of @ref iterations run.
struct EllipsoidResult {
    std::vector<double> x;             ///< the best (lowest-objective) point found.
    double              value{0.0};    ///< the objective value at @ref x.
    int                 iterations{0}; ///< iterations performed.
};

/// @brief The *ellipsoid method* for convex minimization (Shor/Yudin-Nemirovski; Khachiyan 1979 used
///        it to prove linear programming is in P). It maintains an *ellipsoid guaranteed to contain a
///        minimizer* and, each step, evaluates a *subgradient* at the ellipsoid's centre: the
///        subgradient's halfspace tells which half of the ellipsoid to discard, and the smallest
///        ellipsoid enclosing that half becomes the next iterate. The volume shrinks by a fixed
///        factor @c e^{-1/(2n)} every step, giving *geometric* convergence regardless of
///        conditioning -- polynomial-time in theory, though slow in practice compared with
///        interior-point methods. It needs only a subgradient oracle, so it handles nonsmooth convex
///        objectives.
///
/// @param f the convex objective to minimize.
/// @param subgradient a subgradient oracle: returns a subgradient of @p f at a point.
/// @param center the centre of the initial ellipsoid (a ball of the given @p radius).
/// @param radius the radius of the initial enclosing ball (must contain a minimizer).
/// @param max_iter the iteration cap.
/// @param tol stop when the scaled subgradient step falls below this.
/// @return the best @ref EllipsoidResult found.
EllipsoidResult ellipsoid_minimize(const std::function<double(const std::vector<double>&)>&              f,
                                   const std::function<std::vector<double>(const std::vector<double>&)>& subgradient,
                                   std::vector<double> center, double radius, int max_iter = 1000,
                                   double tol = 1e-10);

} // namespace datamunge::algorithms
