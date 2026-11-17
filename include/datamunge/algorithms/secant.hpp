#pragma once
/// @file secant.hpp
/// @brief Checked actual unbracketed secant proposals, callbacks and explicit failure outcomes.
#include <cstddef>
#include <datamunge/algorithms/root_result.hpp>
#include <functional>
#include <vector>
/// @brief Canonical scalar root iteration and owned producing histories.
namespace datamunge::algorithms {
/// @brief Actual secant stopping branch, distinct from a universal convergence guarantee.
/// @par Example
/// @code
/// const auto value=dm::algorithms::SecantStop::Residual;
/// std::print("{}\n",static_cast<int>(value)); // 0
/// @endcode
enum class SecantStop {
  Residual,        ///< Actual callback zero or positive residual threshold passed.
  Horizontal,      ///< Retained callback values were equal, so the secant denominator vanished.
  Stagnation,      ///< A finite proposal rounded back to the latest point without residual success.
  ArithmeticRange, ///< Actual widened proposal was nonfinite or outside finite double range.
  Budget           ///< Actual correction-attempt budget exhausted.
};
/// @brief Residual convergence and actual callback/capture policies.
/// @par Example
/// @code
/// const dm::algorithms::SecantOptions value{};
/// std::print("{}\n",value.max_iterations); // 100
/// @endcode
struct SecantOptions {
  std::size_t max_iterations{
      100}; ///< Actual correction-attempt budget, including zero; must fit int.
  double tolerance{
      1e-10}; ///< Finite nonnegative absolute callback tolerance; zero accepts exact represented zero only.
  std::size_t max_evaluations{
      1000000}; ///< Actual callback invocations, including both initial queries; zero disables.
  std::size_t max_trace_entries{
      1000000}; ///< One retained actual evaluation/proposal record; zero disables.
};
/// @brief One actual synchronous scalar callback return.
/// @par Example
/// @code
/// const dm::algorithms::SecantEvaluation value{};
/// std::print("{}\n",value.x); // 0
/// @endcode
struct SecantEvaluation {
  double x{};     ///< Actual queried finite position.
  double value{}; ///< Actual finite callback result.
};
/// @brief One actual retained two-point proposal and optional distinct candidate query.
/// @par Example
/// @code
/// const dm::algorithms::SecantStep value{};
/// std::print("{}\n",value.evaluated); // false
/// @endcode
struct SecantStep {
  double previous{};   ///< Actual earlier retained point.
  double current{};    ///< Actual latest retained point.
  double f_previous{}; ///< Actual retained earlier callback result.
  double f_current{};  ///< Actual latest callback result.
  long double
      normalized_denominator{}; ///< Actual f_current/scale-f_previous/scale; zero for a horizontal secant.
  long double
      proposal{}; ///< Actual widened proposed position; can be nonfinite on an explicit arithmetic-range stop.
  bool   evaluated{}; ///< Whether this proposal became a real finite callback query.
  double value{};     ///< Actual candidate callback result, zero sentinel when evaluated=false.
};
/// @brief Owned actual latest point/residual and selected secant work.
/// @par Example
/// @code
/// const dm::algorithms::SecantRun value{};
/// std::print("{}\n",value.proposals); // 0
/// @endcode
struct SecantRun {
  RootResult
      result; ///< Matching actual queried point/residual; iterations counts completed correction queries.
  SecantStop stop{
      SecantStop::Budget}; ///< Actual stopping outcome, residual alone sets converged=true.
  std::size_t
      proposals{}; ///< Actual denominator/proposal attempts, including a terminal failed attempt.
  std::size_t
      function_evaluations{}; ///< Actual two initial plus completed correction callback calls.
  std::vector<SecantEvaluation> evaluations; ///< Actual optional chronological queried values.
  std::vector<SecantStep>
      steps; ///< Actual optional proposal records; no repeated-prefix simulations.
};
/// @brief Execute checked unbracketed secant iteration.
/// Validates finite initial points, nonempty callback, nonnegative finite tolerance and an int-compatible attempt budget before any callback. Queries both initial points; an initial actual residual success returns its matching point. Each attempt normalizes finite callback values in widened arithmetic before forming the ratio, then uses a convex combination for interpolation or a widened difference for extrapolation. Horizontal secants, a proposal rounded to the current point, unrepresentable arithmetic and exhausted budgets return explicit unconverged outcomes at the last actually queried point. Finite callback returns are required; exceptions propagate synchronously. No sign-changing bracket or global convergence guarantee is assumed; local superlinear order requires a simple regular root and suitable starting points. Time O(actual callback work+attempts), O(1) work plus optional actual query/proposal records, admitted before their work. Inputs unchanged, outputs owned; no retained callback/shared state.
/// @param f Borrowed synchronous finite scalar function.
/// @param x0 Finite earlier initial point.
/// @param x1 Finite latest initial point.
/// @param options Residual, actual callback and capture controls.
/// @return Owned actual point/residual, explicit stop, observed work and optional producing histories.
/// @throws std::invalid_argument Malformed input controls or nonfinite callback result.
/// @throws std::length_error Actual callback/history/container capacity or policy exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::secant_run([](double x){return x*x-2;},1,2);
/// std::print("{}\n",r.result.converged); // true
/// @endcode
/// @par References
/// - Traub (1964), Iterative Methods for the Solution of Equations, secant analysis.
/// @see secant(), ridders_run()
SecantRun secant_run(
    const std::function<double(double)>& f,
    double                               x0,
    double                               x1,
    const SecantOptions&                 options = {});
/// @brief Execute checked unbracketed secant iteration with actual proposals and callback histories.
/// Validates finite initial points, nonempty callback, nonnegative finite tolerance and an int-compatible attempt budget before any callback. Queries both initial points; an initial actual residual success returns its matching point. Each attempt normalizes finite callback values in widened arithmetic before forming the ratio, then uses a convex combination for interpolation or a widened difference for extrapolation. Horizontal secants, a proposal rounded to the current point, unrepresentable arithmetic and exhausted budgets return explicit unconverged outcomes at the last actually queried point. Finite callback returns are required; exceptions propagate synchronously. No sign-changing bracket or global convergence guarantee is assumed; local superlinear order requires a simple regular root and suitable starting points. Time O(actual callback work+attempts), O(1) work plus optional actual query/proposal records, admitted before their work. Inputs unchanged, outputs owned; no retained callback/shared state.
/// @param f Borrowed synchronous finite scalar function.
/// @param x0 Finite earlier initial point.
/// @param x1 Finite latest initial point.
/// @param options Residual, actual callback and capture controls.
/// @return Owned actual point/residual, explicit stop, observed work and optional producing histories.
/// @throws std::invalid_argument Malformed input controls or nonfinite callback result.
/// @throws std::length_error Actual callback/history/container capacity or policy exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::secant_trace([](double x){return x*x-2;},1,2);
/// std::print("{}\n",r.result.converged); // true
/// @endcode
/// @par References
/// - Traub (1964), Iterative Methods for the Solution of Equations, secant analysis.
/// @see secant(), ridders_run()
SecantRun secant_trace(
    const std::function<double(double)>& f,
    double                               x0,
    double                               x1,
    const SecantOptions&                 options = {});
/// @brief Return checked native secant diagnostics through the preserved signature.
/// Uses the same producer/default callback bounds. Negative budgets and nonfinite controls now reject; zero attempts still diagnose the initial points. iterations records actual completed correction queries, replacing the former loop-index offset.
/// @param f Borrowed finite scalar callback.
/// @param x0 Finite earlier initial point.
/// @param x1 Finite latest initial point.
/// @param tol Finite nonnegative residual tolerance, default1e-10.
/// @param max_iter Nonnegative correction-attempt budget, default100.
/// @return Matching actual returned point/residual and convergence flag.
/// @throws std::invalid_argument Invalid callback/points/tolerance/budget or nonfinite return.
/// @throws std::length_error Default callback policy exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::secant([](double x){return x-2;},0,4).root); // 2
/// @endcode
/// @par References
/// - Traub (1964), Iterative Methods for the Solution of Equations, secant analysis.
/// @see secant_run()
RootResult secant(
    const std::function<double(double)>& f,
    double                               x0,
    double                               x1,
    double                               tol      = 1e-10,
    int                                  max_iter = 100);
} // namespace datamunge::algorithms
