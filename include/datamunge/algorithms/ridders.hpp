#pragma once
/// @file ridders.hpp
/// @brief Checked normalized Ridders interpolation with actual callbacks, brackets and stop reasons.
#include <cstddef>
#include <datamunge/algorithms/root_result.hpp>
#include <functional>
#include <vector>
/// @brief Canonical scalar bracket solvers and actual histories.
namespace datamunge::algorithms {
/// @brief Actual stopping reason; interval resolution does not certify a small residual.
/// @par Example
/// @code
/// const auto kind=dm::algorithms::RiddersStop::NoBracket;
/// std::print("{}\n",static_cast<int>(kind)); // 0
/// @endcode
enum class RiddersStop {
  NoBracket,  ///< Nonzero endpoint values have equal signs.
  Residual,   ///< Actual zero or positive residual threshold passed.
  Interval,   ///< Positive interval tolerance passed on the actual current bracket.
  Stagnation, ///< Midpoint rounded to an endpoint before a threshold passed.
  Budget      ///< Requested actual iteration budget exhausted.
};
/// @brief Convergence controls and actual callback/capture policies.
/// @par Example
/// @code
/// const dm::algorithms::RiddersOptions value{};
/// std::print("{}\n",value.max_iterations); // 100
/// @endcode
struct RiddersOptions {
  std::size_t max_iterations{
      100}; ///< Actual completed Ridders iterations, including zero; must fit int.
  double residual_tolerance{
      1e-10}; ///< Finite nonnegative residual threshold; zero means exact represented callback zero only.
  double      x_tolerance{1e-10}; ///< Finite nonnegative bracket width threshold; zero disables.
  std::size_t max_evaluations{
      1000000}; ///< Actual endpoint/midpoint/correction callback calls; zero disables.
  std::size_t max_trace_entries{
      1000000}; ///< One record per actual evaluation and completed iteration; zero disables.
};
/// @brief One actual synchronous callback result.
/// @par Example
/// @code
/// const dm::algorithms::RiddersEvaluation value{};
/// std::print("{}\n",value.x); // 0
/// @endcode
struct RiddersEvaluation {
  double x{};     ///< Actual queried scalar position.
  double value{}; ///< Actual finite callback result.
};
/// @brief One actual midpoint/correction decision and its resulting bracket.
/// @par Example
/// @code
/// const dm::algorithms::RiddersStep value{};
/// std::print("{}\n",value.correction_evaluated); // false
/// @endcode
struct RiddersStep {
  double a{};              ///< Actual ordered lower endpoint before the step.
  double b{};              ///< Actual ordered upper endpoint before the step.
  double fa{};             ///< Actual callback value retained at a.
  double fb{};             ///< Actual callback value retained at b.
  double midpoint{};       ///< Actual queried finite midpoint.
  double midpoint_value{}; ///< Actual returned midpoint value.
  long double
      normalized_radicand{}; ///< Actual scaled fc^2-fa*fb used by interpolation; zero if midpoint already stops.
  double candidate{};       ///< Actual accepted correction or midpoint fallback/reuse.
  double candidate_value{}; ///< Actual returned candidate value or reused midpoint result.
  bool
      correction_evaluated{}; ///< Whether a distinct corrected point caused a second real callback.
  bool   safeguarded{}; ///< Whether an exterior/nonfinite correction fell back to the midpoint.
  double a_after{}; ///< Actual ordered lower bracket after the step; unchanged on residual success.
  double b_after{}; ///< Actual ordered upper bracket after the step.
};
/// @brief Owned actual root result, final bracket and selected callback work.
/// @par Example
/// @code
/// const dm::algorithms::RiddersRun value{};
/// std::print("{}\n",value.function_evaluations); // 0
/// @endcode
struct RiddersRun {
  RootResult
      result; ///< Actual selected point/residual; converged means residual or interval criterion passed.
  RiddersStop stop{RiddersStop::NoBracket}; ///< Actual stopping branch.
  bool
      bracketed{}; ///< Endpoint zero or actual opposite endpoint signs, distinct from continuity proof.
  double      lower{};                        ///< Actual final ordered lower endpoint.
  double      upper{};                        ///< Actual final ordered upper endpoint.
  std::size_t function_evaluations{};         ///< Actual synchronous callback invocations.
  std::vector<RiddersEvaluation> evaluations; ///< Actual optional chronological queries/results.
  std::vector<RiddersStep>       steps;       ///< Actual optional completed iteration records.
};
/// @brief Solve a finite scalar bracket with normalized Ridders interpolation.
/// Orders finite endpoints, validates callback/controls before calls, and queries both actual endpoint values. Equal nonzero signs return NoBracket without a false root certificate. Uses midpoint plus the Ridders exponential correction; scales finite endpoint/midpoint values in widened arithmetic before sqrt(fc^2-fa*fb) to avoid overflowing raw products. A correction that rounds outside the open bracket falls back to the actual midpoint; midpoint/candidate equality reuses the one real evaluation. Exact midpoint zeros stop before another query. Retains true finite brackets and distinguishes residual success, width resolution, midpoint stagnation and exhausted budget. Width convergence alone does not certify a root of a discontinuous callback. A zero iteration budget still diagnoses both endpoints and returns the best actual endpoint. Time O(actual callback work+iterations), O(1) working storage plus optional actual evaluation/step records. History admission reserves each step/query before that work; callback is synchronous/not retained, and its exceptions propagate. Input endpoints unchanged; outputs owned.
/// @param f Borrowed synchronous finite-valued scalar callback; continuity/sign assumptions belong to the caller.
/// @param a Finite first endpoint.
/// @param b Finite second endpoint, either order.
/// @param options Finite nonnegative convergence and actual work/history controls.
/// @return Owned actual point/residual, bracket, stop reason, callback count and optional histories.
/// @throws std::invalid_argument Empty callback, nonfinite input/return, invalid tolerance or unrepresentable iteration count.
/// @throws std::length_error Actual callback/history/container policy exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::ridders_run([](double x){return x*x-2;},1,2);
/// std::print("{}\n",r.result.converged); // true
/// @endcode
/// @par References
/// - C. J. F. Ridders (1979), A new algorithm for computing a single root of a real continuous function, doi:10.1109/TCS.1979.1084580. This variant normalizes products and safeguards rounded corrections.
/// @see ridders(), bisection_trace()
RiddersRun ridders_run(
    const std::function<double(double)>& f,
    double                               a,
    double                               b,
    const RiddersOptions&                options = {});
/// @brief Solve a finite scalar bracket with normalized Ridders interpolation and actual histories.
/// Orders finite endpoints, validates callback/controls before calls, and queries both actual endpoint values. Equal nonzero signs return NoBracket without a false root certificate. Uses midpoint plus the Ridders exponential correction; scales finite endpoint/midpoint values in widened arithmetic before sqrt(fc^2-fa*fb) to avoid overflowing raw products. A correction that rounds outside the open bracket falls back to the actual midpoint; midpoint/candidate equality reuses the one real evaluation. Exact midpoint zeros stop before another query. Retains true finite brackets and distinguishes residual success, width resolution, midpoint stagnation and exhausted budget. Width convergence alone does not certify a root of a discontinuous callback. A zero iteration budget still diagnoses both endpoints and returns the best actual endpoint. Time O(actual callback work+iterations), O(1) working storage plus optional actual evaluation/step records. History admission reserves each step/query before that work; callback is synchronous/not retained, and its exceptions propagate. Input endpoints unchanged; outputs owned.
/// @param f Borrowed synchronous finite-valued scalar callback; continuity/sign assumptions belong to the caller.
/// @param a Finite first endpoint.
/// @param b Finite second endpoint, either order.
/// @param options Finite nonnegative convergence and actual work/history controls.
/// @return Owned actual point/residual, bracket, stop reason, callback count and optional histories.
/// @throws std::invalid_argument Empty callback, nonfinite input/return, invalid tolerance or unrepresentable iteration count.
/// @throws std::length_error Actual callback/history/container policy exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::ridders_trace([](double x){return x*x-2;},1,2);
/// std::print("{}\n",r.result.converged); // true
/// @endcode
/// @par References
/// - C. J. F. Ridders (1979), A new algorithm for computing a single root of a real continuous function, doi:10.1109/TCS.1979.1084580. This variant normalizes products and safeguards rounded corrections.
/// @see ridders(), bisection_trace()
RiddersRun ridders_trace(
    const std::function<double(double)>& f,
    double                               a,
    double                               b,
    const RiddersOptions&                options = {});
/// @brief Return the checked Ridders result through the preserved signature.
/// Delegates once to the shared producer. The legacy tol sets both finite nonnegative residual and interval tolerances; max_iter must be nonnegative. Same-sign endpoints return unconverged with the best actually evaluated endpoint rather than an unqueried zero sentinel.
/// @param f Finite synchronous scalar callback.
/// @param a Finite first endpoint.
/// @param b Finite second endpoint.
/// @param tol Finite nonnegative residual/width tolerance, default1e-10.
/// @param max_iter Nonnegative completed iteration budget, default100.
/// @return Actual point, residual, iteration count and convergence flag.
/// @throws std::invalid_argument Malformed input, callback result or negative budget.
/// @throws std::length_error Default callback policy exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::ridders([](double x){return x-2;},0,4).root); // 2
/// @endcode
/// @see ridders_run()
RootResult ridders(
    const std::function<double(double)>& f,
    double                               a,
    double                               b,
    double                               tol      = 1e-10,
    int                                  max_iter = 100);
} // namespace datamunge::algorithms
