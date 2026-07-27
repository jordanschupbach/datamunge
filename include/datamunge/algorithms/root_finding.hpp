#pragma once

#include <functional>

namespace datamunge::algorithms {

/// @brief The outcome of a one-dimensional root search: whether it @ref converged, the final
///        @ref root estimate, the number of @ref iterations taken, and the @ref residual
///        @c |f(root)| there.
struct RootResult {
    bool   converged{false}; ///< whether the tolerance was met.
    double root{0.0};        ///< the final estimate of the root.
    int    iterations{0};    ///< number of iterations performed.
    double residual{0.0};    ///< |f(root)| at the returned estimate.
};

/// @brief The *bisection method*: given a continuous @p f that changes sign across @c [a,b] (so a
///        root is bracketed), repeatedly halve the interval, keeping the half that still brackets
///        the root. It converges *linearly* -- one bit of accuracy per step -- but is utterly
///        reliable: it cannot diverge and always brackets a true root. Stops when @c |f(midpoint)|
///        or the half-width falls below @p tol.
///
/// @param f the continuous function.
/// @param a,b endpoints bracketing the root (@c f(a) and @c f(b) must have opposite signs).
/// @param tol convergence tolerance on the residual / half-interval width.
/// @param max_iter iteration cap.
/// @return the @ref RootResult; @c converged is false if @c [a,b] does not bracket a sign change.
RootResult bisection(const std::function<double(double)>& f, double a, double b, double tol = 1e-10,
                     int max_iter = 200);

/// @brief *Newton's method* (Newton-Raphson): from a guess @p x0, repeatedly step to the zero of the
///        tangent line, @c x <- x - f(x)/f'(x). It converges *quadratically* near a simple root (the
///        number of correct digits roughly doubles each step) but needs the derivative @p df and can
///        diverge from a poor start or where @c f'(x)=0.
///
/// @param f the function.
/// @param df its derivative.
/// @param x0 the initial guess.
/// @param tol convergence tolerance on the residual.
/// @param max_iter iteration cap.
/// @return the @ref RootResult; @c converged is false if a zero derivative is hit or the cap is
///         reached without meeting @p tol.
RootResult newton_raphson(const std::function<double(double)>& f, const std::function<double(double)>& df,
                          double x0, double tol = 1e-10, int max_iter = 100);

/// @brief The *secant method*: like Newton's, but approximates the derivative with a finite
///        difference over the two most recent points, needing no explicit @c f'. It converges
///        *superlinearly* (order ≈ 1.618, the golden ratio) using one function evaluation per step.
///
/// @param f the function.
/// @param x0,x1 two initial points (need not bracket the root).
/// @param tol convergence tolerance on the residual.
/// @param max_iter iteration cap.
/// @return the @ref RootResult.
RootResult secant(const std::function<double(double)>& f, double x0, double x1, double tol = 1e-10,
                  int max_iter = 100);

/// @brief *Ridders' method*: a bracketing root-finder that, per step, evaluates @c f at the interval
///        midpoint and then applies an exponential correction that fits @c f·e^{Q} to a straight
///        line, landing far closer to the root than bisection. It keeps the guaranteed convergence
///        of a bracketing method while achieving *quadratic* convergence, using two function
///        evaluations per iteration. Requires a sign-changing bracket @c [a,b].
///
/// @param f the continuous function.
/// @param a,b endpoints bracketing the root.
/// @param tol convergence tolerance on the residual / bracket width.
/// @param max_iter iteration cap.
/// @return the @ref RootResult; @c converged is false if @c [a,b] does not bracket a sign change.
RootResult ridders(const std::function<double(double)>& f, double a, double b, double tol = 1e-10,
                   int max_iter = 100);

} // namespace datamunge::algorithms
