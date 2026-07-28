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

/// @brief The *false position method* (regula falsi) with the *Illinois* acceleration: a bracketing
///        root-finder that, instead of bisecting, connects the endpoints with a secant line and takes
///        its x-intercept as the next estimate. Plain regula falsi can stall (one endpoint stays
///        fixed forever, retaining the bracket's guarantee but converging only linearly); the
///        *Illinois* modification halves the stale endpoint's function value whenever it is kept
///        twice, restoring superlinear convergence while never losing the bracket.
///
/// @param f the continuous function.
/// @param a,b endpoints bracketing the root (@c f(a) and @c f(b) must have opposite signs).
/// @param tol convergence tolerance on the residual.
/// @param max_iter iteration cap.
/// @return the @ref RootResult; @c converged is false if @c [a,b] does not bracket a sign change.
RootResult false_position(const std::function<double(double)>& f, double a, double b, double tol = 1e-10,
                          int max_iter = 200);

/// @brief *Halley's method*: a third-order (cubically convergent) root-finder that uses the first
///        *and second* derivatives. It fits a hyperbola (rather than Newton's tangent line) to the
///        function at each iterate, giving @c x_{n+1}=x_n-2 f f'/(2 f'^2 - f f''). Near a simple root
///        it roughly *triples* the number of correct digits each step -- faster than Newton -- at the
///        cost of needing @c f''.
///
/// @param f the function.
/// @param df its first derivative.
/// @param d2f its second derivative.
/// @param x0 the initial guess.
/// @param tol convergence tolerance on the residual.
/// @param max_iter iteration cap.
/// @return the @ref RootResult.
RootResult halley(const std::function<double(double)>& f, const std::function<double(double)>& df,
                  const std::function<double(double)>& d2f, double x0, double tol = 1e-10,
                  int max_iter = 100);

/// @brief *Muller's method*: generalizes the secant method by fitting a *parabola* through the three
///        most recent points and taking the root of that parabola nearest the last iterate (choosing
///        the quadratic-formula sign that maximizes the denominator for stability). It converges at
///        order ~1.84, needs no derivative, and -- because a parabola can cross zero where a line
///        cannot -- can find roots that trap the secant method (this real-valued implementation
///        clamps a negative discriminant to zero, so it targets real roots).
///
/// @param f the function.
/// @param x0,x1,x2 three initial points.
/// @param tol convergence tolerance on the residual.
/// @param max_iter iteration cap.
/// @return the @ref RootResult.
RootResult muller(const std::function<double(double)>& f, double x0, double x1, double x2,
                  double tol = 1e-10, int max_iter = 100);

/// @brief The *ITP method* (Interpolate-Truncate-Project; Oliveira & Takahashi, 2020): a modern
///        bracketing root-finder that is *minmax optimal* in worst-case iterations yet *superlinear*
///        on average -- getting the best of bisection and secant methods simultaneously. Each step
///        computes the regula-falsi point, *truncates* it toward the interval midpoint by a
///        shrinking margin, then *projects* it into a minmax interval around the midpoint, guaranteeing
///        it never does worse than bisection while usually doing much better.
///
/// @param f the continuous function.
/// @param a,b endpoints bracketing the root.
/// @param tol convergence tolerance on the half-interval width.
/// @param max_iter iteration cap.
/// @return the @ref RootResult; @c converged is false if @c [a,b] does not bracket a sign change.
RootResult itp(const std::function<double(double)>& f, double a, double b, double tol = 1e-10,
               int max_iter = 100);

} // namespace datamunge::algorithms
