#pragma once
/// @file ode_types.hpp
/// @brief Shared first-order ODE callback and owned trajectory records, preserving their original layouts.
#include <functional>
#include <vector>
namespace datamunge::algorithms {
/// @brief First-order ODE callback f(t,y), returning the state derivative vector.
/// Receives time and a borrowed const state vector valid only during the call.
/// Return entries have state/time units and the same dimension as y. Euler
/// validates shape/finiteness; validation by other solvers follows their contracts.
/// Higher-order equations can be rewritten as first-order vector systems. The
/// callback may own captured state; its exceptions propagate and concurrent use
/// follows that captured state's thread-safety. No solver retains the callback.
/// @par Example
/// @code
/// const dm::algorithms::OdeSystem f=[](double,const std::vector<double>& y){return std::vector<double>{-y[0]};};
/// std::print("{}\n",f(0,{2})[0]); // -2
/// @endcode
/// @see OdeSolution, euler_method()
using OdeSystem=std::function<std::vector<double>(double,const std::vector<double>&)>;

/// @brief Owned sampled ODE trajectory with original time/state field order.
/// t[i] is the time of y[i]; each state row follows the solver's initial dimension.
/// This mutable record does not itself validate matching lengths, finiteness or
/// monotonicity. Those guarantees follow the producing solver. Copies own every
/// time and state value; no input/callback storage is borrowed. Times and state
/// coordinates retain their input units. Default construction is an empty record.
/// @par Example
/// @code
/// const dm::algorithms::OdeSolution s{{0,1},{{2},{3}}};
/// std::print("{} {}\n",s.t.back(),s.y.back()[0]); // 1 3
/// @endcode
/// @see OdeSystem, euler_method(), backward_euler(), runge_kutta4()
struct OdeSolution {
    std::vector<double> t; ///< Owned sample times.
    std::vector<std::vector<double>> y; ///< Owned state rows, aligned with t.
};
} // namespace datamunge::algorithms
