#pragma once

#include <vector>

namespace datamunge::ode {

/// @brief Right-hand side of a first-order ODE system dy/dt = f(t, y): R x R^n -> R^n.
///        Director-enabled only in the Python binding, matching datamunge::filter::VectorFunction
///        (the closest existing precedent -- a vector<double>-returning pure-virtual director):
///        every other SWIG backend either has no director support at all, or -- confirmed by a
///        real compile failure while wiring this class up in R -- cannot generate a working
///        director wrapper for a virtual method that returns std::vector<double>. Every other
///        binding can only use ODESolver::solve_builtin()'s fixed set of named systems, since a
///        live user-supplied RHS requires exactly that subclassing.
class RHS {
 public:
  virtual ~RHS() = default;
  virtual std::vector<double> evaluate(double t, const std::vector<double>& y) = 0;
};

} // namespace datamunge::ode
