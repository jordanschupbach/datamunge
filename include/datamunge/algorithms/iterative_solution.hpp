#pragma once
/// @file iterative_solution.hpp
/// @brief Preserved owned iterative-solver result layout.
#include <vector>
/// @brief Canonical checked native producers and owned values.
namespace datamunge::algorithms {
/// @brief The result of an iterative linear solve: whether it @ref converged to the tolerance, the
///        approximate solution @ref x, the number of @ref iterations, and the final @ref residual
///        @c ||A x - b||.
/// @brief IterativeSolution owns actual producer values or explicit invocation policies; fields state units and defaults.
/// @par Example
/// @code
/// const dm::algorithms::IterativeSolution value{};
/// std::print("{}\n",value.iterations); // 0
/// @endcode
struct IterativeSolution {
  bool                converged{false}; ///< Whether the tolerance was reached.
  std::vector<double> x;                ///< The approximate solution.
  int                 iterations{0};    ///< Iterations performed.
  double              residual{0.0};    ///< Final residual norm ||A x - b||.
};

} // namespace datamunge::algorithms
