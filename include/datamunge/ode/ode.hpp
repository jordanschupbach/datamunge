#pragma once

// Convenience umbrella header -- basic numerical ODE support: a director-enabled RHS
// abstraction plus a solver offering the Runge-Kutta family (Euler, midpoint, classical RK4,
// adaptive Dormand-Prince RK45) and linear multistep methods (Adams-Bashforth,
// Adams-Bashforth-Moulton predictor-corrector), and a fixed set of named example systems for
// bindings without director support.

#include <datamunge/ode/ode_solver.hpp>
#include <datamunge/ode/rhs.hpp>
