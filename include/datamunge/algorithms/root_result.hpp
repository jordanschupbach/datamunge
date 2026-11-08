#pragma once
/// @file root_result.hpp
/// @brief Shared result record for one-dimensional root-search algorithms.
namespace datamunge::algorithms {
/// @brief Owned root estimate, stopping flag, iteration count and absolute callback residual.
/// The meaning of converged is method-specific: a residual or interval test can
/// pass without establishing a mathematical root if continuity or function
/// evaluation assumptions fail. No input or callback pointer is retained.
/// @par Example
/// @code
/// const auto r=dm::algorithms::bisection([](double x){return x-1;},0,2);
/// std::print("{} {}\n",r.converged,r.root); // true 1
/// @endcode
/// @see bisection(), bisection_trace(), newton_raphson()
struct RootResult {
  bool converged{false}; ///< Whether the selected method's stopping criterion passed.
  double root{0}; ///< Last estimate; some failed methods retain their default zero.
  int iterations{0}; ///< Number of method iterations, excluding initial endpoint checks.
  double residual{0}; ///< Absolute callback value at the estimate when an estimate was evaluated.
};
} // namespace datamunge::algorithms
