#pragma once

#include <functional>

namespace datamunge::algorithms {

/// @brief The outcome of a ternary search: the located extremum's @ref x, the function @ref value
///        there, and the number of @ref iterations performed.
struct TernaryResult {
    double x{0.0};       ///< the argument at which the extremum was found.
    double value{0.0};   ///< the function value @c f(x) at that point.
    int    iterations{0}; ///< number of interval-narrowing steps taken.
};

/// @brief *Ternary search* for the *maximum* of a *unimodal* function on @c [lo,hi] (one that strictly
///        increases and then strictly decreases). Each step splits the interval into thirds at
///        @c m1 and @c m2; comparing @c f(m1) and @c f(m2) reveals which outer third *cannot* contain
///        the peak, so it is discarded and the interval shrinks by a factor of @c 2/3. After each
///        step the bracket provably still contains the maximum, giving *logarithmic* convergence.
///        (Unlike gradient methods it needs no derivative, only unimodality.)
///
/// @param f the unimodal objective.
/// @param lo,hi the bracket known to contain the peak.
/// @param tol stop once the bracket is narrower than this.
/// @param max_iter iteration cap.
/// @return the @ref TernaryResult at the located maximum.
TernaryResult ternary_search_max(const std::function<double(double)>& f, double lo, double hi,
                                 double tol = 1e-9, int max_iter = 200);

/// @brief *Ternary search* for the *minimum* of a unimodal function (one that strictly decreases and
///        then strictly increases) on @c [lo,hi]. Identical to @ref ternary_search_max but discarding
///        the outer third that cannot contain the *valley*.
///
/// @param f the unimodal objective.
/// @param lo,hi the bracket known to contain the valley.
/// @param tol stop once the bracket is narrower than this.
/// @param max_iter iteration cap.
/// @return the @ref TernaryResult at the located minimum.
TernaryResult ternary_search_min(const std::function<double(double)>& f, double lo, double hi,
                                 double tol = 1e-9, int max_iter = 200);

} // namespace datamunge::algorithms
