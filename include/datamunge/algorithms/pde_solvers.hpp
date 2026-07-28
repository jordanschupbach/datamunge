#pragma once

#include <functional>
#include <vector>

namespace datamunge::algorithms {

/// @brief The *finite difference method* for the 1-D Poisson boundary-value problem
///        @c -u''(x) = f(x) on @c [0,1] with Dirichlet data @c u(0)=alpha, @c u(1)=beta. It replaces
///        the second derivative by the central difference
///        @c u''(x_i) ≈ (u_{i-1} - 2u_i + u_{i+1})/h^2, turning the ODE into a *tridiagonal* linear
///        system for the interior values, solved directly by the Thomas algorithm in @c O(n). The
///        error is @c O(h^2). This is the archetype of discretising a differential operator on a grid.
///
/// @param f the source term.
/// @param alpha,beta the Dirichlet boundary values at @c x=0 and @c x=1.
/// @param n the number of interior grid points (spacing @c h=1/(n+1)).
/// @return the solution at all @c n+2 grid points (boundaries included).
std::vector<double> finite_difference_poisson(const std::function<double(double)>& f, double alpha,
                                              double beta, int n);

/// @brief The *Crank-Nicolson method* for the diffusion (heat) equation @c u_t = D·u_xx on @c [0,L]
///        with zero Dirichlet boundaries. It averages the explicit and implicit (forward/backward
///        Euler) spatial stencils at consecutive time levels, giving a scheme that is
///        *second-order in both time and space* and *unconditionally stable* -- no step-size
///        restriction, unlike the explicit method. Each step solves a tridiagonal system (Thomas).
///
/// @param u0 the initial profile @c u(x,0).
/// @param L the domain length.
/// @param D the diffusion coefficient.
/// @param nx the number of interior spatial points.
/// @param dt the time step.
/// @param nsteps the number of time steps to advance.
/// @return the solution profile at the final time, at all @c nx+2 grid points.
std::vector<double> crank_nicolson_heat(const std::function<double(double)>& u0, double L, double D,
                                        int nx, double dt, int nsteps);

/// @brief The *Lax-Wendroff scheme* for the linear advection (one-way wave) equation
///        @c u_t + a·u_x = 0 on a *periodic* domain. It is derived from a second-order Taylor
///        expansion in time (using the PDE to convert @c u_tt into @c a^2·u_xx), yielding an explicit
///        update that is *second-order accurate* in space and time -- far less diffusive than
///        first-order upwind. It is stable when the Courant number @c |a|·dt/dx <= 1 (the CFL
///        condition).
///
/// @param u0 the initial values on the periodic grid.
/// @param a the advection speed.
/// @param dx the grid spacing.
/// @param dt the time step (must satisfy @c |a|·dt/dx <= 1).
/// @param nsteps the number of time steps.
/// @return the solution on the grid after @p nsteps steps.
std::vector<double> lax_wendroff_advection(const std::vector<double>& u0, double a, double dx, double dt,
                                           int nsteps);

/// @brief The *trapezoidal rule for ODEs*: the implicit integrator
///        @c y_{n+1} = y_n + (h/2)·(f(t_n,y_n) + f(t_{n+1},y_{n+1})) for @c y'=f(t,y). Averaging the
///        slopes at both ends of the step makes it *second-order accurate* and *A-stable* -- a strict
///        improvement over backward Euler's first order, and the ODE counterpart of Crank-Nicolson.
///        Each step solves the implicit equation by fixed-point iteration (Newton for a scalar).
///
/// @param f the ODE right-hand side.
/// @param y0 the initial value at @p t0.
/// @param t0,t1 the integration interval.
/// @param steps the number of equal steps.
/// @return the times and values @c (t_i, y_i) of the trajectory (@c steps+1 points).
std::vector<std::pair<double, double>> trapezoidal_ode(const std::function<double(double, double)>& f,
                                                       double y0, double t0, double t1, int steps);

} // namespace datamunge::algorithms
