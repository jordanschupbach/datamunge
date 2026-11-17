#pragma once
/// @file ternary_search.hpp
/// @brief Checked actual unimodal brackets, sampled values and explicit floating termination.
#include <cstddef>
#include <functional>
#include <vector>
namespace datamunge::algorithms {
/// @brief Own the TernaryResult contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TernaryResult r{};
/// (void)r;
/// @endcode
/// @see ternary_search_max_run()
struct TernaryResult {
  double x{};          ///< Actual midpoint of returned bracket.
  double value{};      ///< Actual final objective evaluation at x.
  int    iterations{}; ///< Actual completed bracket reductions.
};
/// @brief Own the TernaryOptions contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TernaryOptions r{};
/// (void)r;
/// @endcode
/// @see ternary_search_max_run()
struct TernaryOptions {
  double tolerance{1e-9}; ///< Finite nonnegative absolute interval-width tolerance.
  int    max_iterations{
      200}; ///< Nonnegative completed-reduction cap; zero evaluates only final midpoint.
  std::size_t max_records{1000000}; ///< Actual captured reductions; zero disables policy.
};
/// @brief Own the TernaryStep contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TernaryStep r{};
/// (void)r;
/// @endcode
/// @see ternary_search_max_run()
struct TernaryStep {
  double lo_before{};    ///< Actual left endpoint before this comparison.
  double hi_before{};    ///< Actual right endpoint before this comparison.
  double first{};        ///< Actual first interior evaluation argument.
  double second{};       ///< Actual second interior evaluation argument.
  double first_value{};  ///< Actual first callback value.
  double second_value{}; ///< Actual second callback value.
  bool   kept_left{};    ///< Actual comparison selected [lo,second] instead of[first,hi].
  double lo_after{};     ///< Actual retained left endpoint.
  double hi_after{};     ///< Actual retained right endpoint.
};
/// @brief Own the TernaryRun contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TernaryRun r{};
/// (void)r;
/// @endcode
/// @see ternary_search_max_run()
struct TernaryRun {
  TernaryResult result;      ///< Actual final midpoint evaluation and completed reductions.
  double        lo{};        ///< Actual final left endpoint.
  double        hi{};        ///< Actual final right endpoint.
  bool          converged{}; ///< Actual final widened width<=tolerance.
  bool
      precision_limited{}; ///< No representable strict interval reduction remained before tolerance.
  std::size_t
      evaluations{}; ///< Actual callback invocations: two per executed comparison plus final midpoint.
  std::vector<TernaryStep> steps; ///< Optional actual comparison/bracket histories.
};
/// @brief Find a unimodal maximum with actual bracket diagnostics.
/// Requires a nonempty callback, finite ordered lo<=hi, finite nonnegative tolerance and nonnegative iteration cap. Under strict unimodality, each comparison discards an outer third without discarding the extremum; arbitrary functions and broad flat shoulders have no such guarantee. Ties keep the left interval. Evaluate the first interior point before the second, then evaluate the final midpoint once; callbacks may observe this explicit order. Objective values must be finite; callback exceptions propagate and callback side effects are not rolled back. Stable widened span/interpolation supports finite extreme endpoints. Precision exhaustion and iteration exhaustion remain distinct from tolerance convergence. Owned histories and no shared state; O(iterations) evaluations/time, optional linear history.
/// @param f Borrowed objective callback, finite at every sampled point.
/// @param lo Finite initial lower endpoint.
/// @param hi Finite initial upper endpoint at least lo.
/// @param options Absolute width tolerance, completed-reduction cap and capture policy.
/// @return Owned actual final bracket, midpoint result, convergence/precision flags and optional history.
/// @throws std::invalid_argument Invalid callback, bracket or controls.
/// @throws std::domain_error Actual callback result is nonfinite.
/// @throws std::length_error Actual capture/counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::ternary_search_max_run([](double x){return -x*x;},-1,1);
/// std::print("{}\n",r.converged); // true
/// @endcode
/// @see ternary_search_max()
TernaryRun ternary_search_max_run(
    const std::function<double(double)>& f,
    double                               lo,
    double                               hi,
    const TernaryOptions&                options = {});
/// @brief Find a unimodal maximum with actual bracket diagnostics.
/// Requires a nonempty callback, finite ordered lo<=hi, finite nonnegative tolerance and nonnegative iteration cap. Under strict unimodality, each comparison discards an outer third without discarding the extremum; arbitrary functions and broad flat shoulders have no such guarantee. Ties keep the left interval. Evaluate the first interior point before the second, then evaluate the final midpoint once; callbacks may observe this explicit order. Objective values must be finite; callback exceptions propagate and callback side effects are not rolled back. Stable widened span/interpolation supports finite extreme endpoints. Precision exhaustion and iteration exhaustion remain distinct from tolerance convergence. Owned histories and no shared state; O(iterations) evaluations/time, optional linear history.
/// @param f Borrowed objective callback, finite at every sampled point.
/// @param lo Finite initial lower endpoint.
/// @param hi Finite initial upper endpoint at least lo.
/// @param options Absolute width tolerance, completed-reduction cap and capture policy.
/// @return Owned actual final bracket, midpoint result, convergence/precision flags and optional history.
/// @throws std::invalid_argument Invalid callback, bracket or controls.
/// @throws std::domain_error Actual callback result is nonfinite.
/// @throws std::length_error Actual capture/counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::ternary_search_max_trace([](double x){return -x*x;},-1,1);
/// std::print("{}\n",r.converged); // true
/// @endcode
/// @see ternary_search_max()
TernaryRun ternary_search_max_trace(
    const std::function<double(double)>& f,
    double                               lo,
    double                               hi,
    const TernaryOptions&                options = {});
/// @brief Preserve the value-only unimodal max search signature.
/// Requires a nonempty callback, finite ordered lo<=hi, finite nonnegative tolerance and nonnegative iteration cap. Under strict unimodality, each comparison discards an outer third without discarding the extremum; arbitrary functions and broad flat shoulders have no such guarantee. Ties keep the left interval. Evaluate the first interior point before the second, then evaluate the final midpoint once; callbacks may observe this explicit order. Objective values must be finite; callback exceptions propagate and callback side effects are not rolled back. Stable widened span/interpolation supports finite extreme endpoints. Precision exhaustion and iteration exhaustion remain distinct from tolerance convergence. Owned histories and no shared state; O(iterations) evaluations/time, optional linear history.
/// @param f Borrowed objective callback.
/// @param lo Finite lower endpoint.
/// @param hi Finite upper endpoint.
/// @param tol Finite nonnegative width tolerance, default1e-9.
/// @param max_iter Nonnegative actual reduction cap, default200.
/// @return Actual final midpoint, callback value and completed reductions; use run for convergence flags.
/// @throws std::invalid_argument Malformed callback/bracket/controls.
/// @throws std::domain_error Nonfinite callback result.
/// @throws std::length_error Counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::ternary_search_max([](double x){return x;},0,0);
/// std::print("{}\n",r.value); // 0
/// @endcode
/// @see ternary_search_max_run()
TernaryResult ternary_search_max(
    const std::function<double(double)>& f,
    double                               lo,
    double                               hi,
    double                               tol      = 1e-9,
    int                                  max_iter = 200);
/// @brief Find a unimodal minimum with actual bracket diagnostics.
/// Requires a nonempty callback, finite ordered lo<=hi, finite nonnegative tolerance and nonnegative iteration cap. Under strict unimodality, each comparison discards an outer third without discarding the extremum; arbitrary functions and broad flat shoulders have no such guarantee. Ties keep the left interval. Evaluate the first interior point before the second, then evaluate the final midpoint once; callbacks may observe this explicit order. Objective values must be finite; callback exceptions propagate and callback side effects are not rolled back. Stable widened span/interpolation supports finite extreme endpoints. Precision exhaustion and iteration exhaustion remain distinct from tolerance convergence. Owned histories and no shared state; O(iterations) evaluations/time, optional linear history.
/// @param f Borrowed objective callback, finite at every sampled point.
/// @param lo Finite initial lower endpoint.
/// @param hi Finite initial upper endpoint at least lo.
/// @param options Absolute width tolerance, completed-reduction cap and capture policy.
/// @return Owned actual final bracket, midpoint result, convergence/precision flags and optional history.
/// @throws std::invalid_argument Invalid callback, bracket or controls.
/// @throws std::domain_error Actual callback result is nonfinite.
/// @throws std::length_error Actual capture/counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::ternary_search_min_run([](double x){return x*x;},-1,1);
/// std::print("{}\n",r.converged); // true
/// @endcode
/// @see ternary_search_min()
TernaryRun ternary_search_min_run(
    const std::function<double(double)>& f,
    double                               lo,
    double                               hi,
    const TernaryOptions&                options = {});
/// @brief Find a unimodal minimum with actual bracket diagnostics.
/// Requires a nonempty callback, finite ordered lo<=hi, finite nonnegative tolerance and nonnegative iteration cap. Under strict unimodality, each comparison discards an outer third without discarding the extremum; arbitrary functions and broad flat shoulders have no such guarantee. Ties keep the left interval. Evaluate the first interior point before the second, then evaluate the final midpoint once; callbacks may observe this explicit order. Objective values must be finite; callback exceptions propagate and callback side effects are not rolled back. Stable widened span/interpolation supports finite extreme endpoints. Precision exhaustion and iteration exhaustion remain distinct from tolerance convergence. Owned histories and no shared state; O(iterations) evaluations/time, optional linear history.
/// @param f Borrowed objective callback, finite at every sampled point.
/// @param lo Finite initial lower endpoint.
/// @param hi Finite initial upper endpoint at least lo.
/// @param options Absolute width tolerance, completed-reduction cap and capture policy.
/// @return Owned actual final bracket, midpoint result, convergence/precision flags and optional history.
/// @throws std::invalid_argument Invalid callback, bracket or controls.
/// @throws std::domain_error Actual callback result is nonfinite.
/// @throws std::length_error Actual capture/counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::ternary_search_min_trace([](double x){return x*x;},-1,1);
/// std::print("{}\n",r.converged); // true
/// @endcode
/// @see ternary_search_min()
TernaryRun ternary_search_min_trace(
    const std::function<double(double)>& f,
    double                               lo,
    double                               hi,
    const TernaryOptions&                options = {});
/// @brief Preserve the value-only unimodal min search signature.
/// Requires a nonempty callback, finite ordered lo<=hi, finite nonnegative tolerance and nonnegative iteration cap. Under strict unimodality, each comparison discards an outer third without discarding the extremum; arbitrary functions and broad flat shoulders have no such guarantee. Ties keep the left interval. Evaluate the first interior point before the second, then evaluate the final midpoint once; callbacks may observe this explicit order. Objective values must be finite; callback exceptions propagate and callback side effects are not rolled back. Stable widened span/interpolation supports finite extreme endpoints. Precision exhaustion and iteration exhaustion remain distinct from tolerance convergence. Owned histories and no shared state; O(iterations) evaluations/time, optional linear history.
/// @param f Borrowed objective callback.
/// @param lo Finite lower endpoint.
/// @param hi Finite upper endpoint.
/// @param tol Finite nonnegative width tolerance, default1e-9.
/// @param max_iter Nonnegative actual reduction cap, default200.
/// @return Actual final midpoint, callback value and completed reductions; use run for convergence flags.
/// @throws std::invalid_argument Malformed callback/bracket/controls.
/// @throws std::domain_error Nonfinite callback result.
/// @throws std::length_error Counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::ternary_search_min([](double x){return x;},0,0);
/// std::print("{}\n",r.value); // 0
/// @endcode
/// @see ternary_search_min_run()
TernaryResult ternary_search_min(
    const std::function<double(double)>& f,
    double                               lo,
    double                               hi,
    double                               tol      = 1e-9,
    int                                  max_iter = 200);
} // namespace datamunge::algorithms
