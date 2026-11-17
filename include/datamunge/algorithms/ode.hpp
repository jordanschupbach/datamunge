#include <datamunge/algorithms/velocity_verlet.hpp>
#pragma once

/// @file ode.hpp
/// @brief One-step ODE integrators (forward and backward Euler, RK4, velocity Verlet), an adaptive
///        backward Euler for stiff problems, and the stability functions of the one-step methods.

#include <complex>
#include <datamunge/algorithms/euler_method.hpp>
#include <datamunge/algorithms/runge_kutta4.hpp>
#include <functional>
#include <limits>
#include <vector>

namespace datamunge::algorithms {

/// @brief The *backward (implicit) Euler method*: steps @c y_{n+1} = y_n + h·f(t_{n+1}, y_{n+1}),
///        evaluating the slope at the *arrival* point. Because @c y_{n+1} appears on both sides, each
///        step solves an implicit equation -- here by *Newton's method* with a central-difference
///        Jacobian. The reward is *A-stability*: on a stiff decaying problem it stays bounded for
///        *any* step size, where forward Euler explodes. It is first-order accurate like forward
///        Euler, but stable.
///
/// @param f the ODE system @c y'=f(t,y).
/// @param y0 the initial state at @p t0.
/// @param t0,t1 the integration interval.
/// @param steps the number of equal steps; must be >= 1.
/// @param newton_tol convergence tolerance for the per-step Newton solve.
/// @param newton_max_iter iteration cap for the per-step Newton solve.
/// @return the sampled @ref OdeSolution with @c steps+1 points.
OdeSolution backward_euler(
    const OdeSystem&    f,
    std::vector<double> y0,
    double              t0,
    double              t1,
    int                 steps,
    double              newton_tol      = 1e-12,
    int                 newton_max_iter = 50);

/// @brief Work counters and the trajectory of an adaptive integration.
struct AdaptiveOdeResult {
  OdeSolution         solution;   ///< accepted time points and states.
  std::vector<double> step_sizes; ///< size of each accepted step (one fewer than time points).
  int                 accepted          = 0; ///< steps accepted.
  int                 rejected          = 0; ///< steps rejected by the error test.
  long                f_evaluations     = 0; ///< calls of f, including those for Jacobians.
  long                newton_iterations = 0; ///< Newton iterations over all implicit solves.
};

/// @brief Backward Euler with step-size control by step doubling.
///
/// Each attempt takes one step of size @c h and two of size @c h/2 from the same point. Their
/// difference estimates the local error of the half steps (first-order local error constants
/// cancel to leave @c y_{h/2,h/2} - y_h ≈ h^2 y''/4). A step is accepted when
/// @c max_i |err_i| / (atol + rtol·|y_i|) <= 1, keeping the two-half-step result, and the next step
/// is scaled by @c 0.9/sqrt(err) (clamped to [0.2, 2]) as the local error is @c O(h^2).
/// Each implicit step is solved by Newton's method to an absolute tolerance of
/// @c min(1e-12, atol/100).
///
/// @param f      the ODE system.
/// @param y0     the initial state.
/// @param t0,t1  the interval (t1 > t0).
/// @param rtol   relative tolerance.
/// @param atol   absolute tolerance.
/// @param h0     the first trial step.
/// @param h_max  the largest step allowed.
/// @return the accepted trajectory and work counters.
/// @throws std::invalid_argument if t1 <= t0, a tolerance is not positive, or h0 <= 0.
AdaptiveOdeResult backward_euler_adaptive(
    const OdeSystem&    f,
    std::vector<double> y0,
    double              t0,
    double              t1,
    double              rtol,
    double              atol,
    double              h0,
    double              h_max = std::numeric_limits<double>::infinity());

/// One-step methods whose stability function stability_function() evaluates.
enum class OdeMethod {
  ForwardEuler,  ///< \f$R(z)=1+z\f$.
  BackwardEuler, ///< \f$R(z)=1/(1-z)\f$.
  Trapezoidal,   ///< \f$R(z)=(1+z/2)/(1-z/2)\f$ (Crank-Nicolson).
  RungeKutta4    ///< \f$R(z)=1+z+z^2/2+z^3/6+z^4/24\f$.
};

/// @brief The stability function \f$R(z)\f$ of a one-step method: applied to \f$y'=\lambda y\f$ with
///        step \f$h\f$, the method gives \f$y_{n+1}=R(h\lambda)\,y_n\f$. The method is stable for
///        that step when \f$|R(h\lambda)|\le1\f$.
std::complex<double> stability_function(OdeMethod method, std::complex<double> z);

} // namespace datamunge::algorithms
