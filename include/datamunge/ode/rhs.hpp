#pragma once
/// @file rhs.hpp
/// @brief Canonical finite-state derivative interface for native forward ODE integration.
#include <vector>
namespace datamunge::ode {
/// @brief Caller-defined derivative callback dy/dt=f(t,y), without retained solver-owned references.
/// The solver requires finite matching-dimensional returns and guards stages/call
/// budgets before invoking it. User callback exceptions/effects propagate and are
/// not rolled back. Thread safety belongs to the callback implementation.
/// Configured SWIG director support is backend-specific; named builtin systems
/// remain available independently of subclassing support. No all-binding claim.
/// @par Example
/// @code
/// struct F:dm::ode::RHS{std::vector<double>evaluate(double,const std::vector<double>&y)override{return {-y[0]};}};F f;std::print("{}\n",f.evaluate(0,{1})[0]); // -1
/// @endcode
/// @see ODESolver
class RHS {
 public:
  /// @brief Destroy a derivative implementation through the canonical virtual interface.
  virtual ~RHS() = default;
  /// @brief Evaluate a complete derivative vector at one actual time/state.
  /// @param t Actual finite stage time supplied by solver.
  /// @param y Actual finite fixed-dimensional stage,borrowed only for this call.
  /// @return Owned finite derivative vector with the same dimension as y.
  /// @par Example
  /// @code
  /// struct F:dm::ode::RHS{std::vector<double>evaluate(double,const std::vector<double>&y)override{return {-y[0]};}};F f;std::print("{}\n",f.evaluate(0,{2})[0]); // -2
  /// @endcode
  /// @see ODESolver::solve()
  virtual std::vector<double> evaluate(double t, const std::vector<double>& y) = 0;
};
} // namespace datamunge::ode
