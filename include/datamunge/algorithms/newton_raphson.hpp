#pragma once
/// @file newton_raphson.hpp
/// @brief Checked scalar Newton steps, actual callbacks and tangent-geometry study.
#include <cstddef>
#include <datamunge/algorithms/root_result.hpp>
#include <functional>
#include <string_view>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual numerical stop, not an existence/global-convergence certificate.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::newton_raphson_stop_name(dm::algorithms::NewtonRaphsonStop::ZeroDerivative)); // zero-derivative
/// @endcode
/// @see newton_raphson_stop_name()
enum class NewtonRaphsonStop {
  Residual,       ///< Actual strict absolute residual is below tolerance.
  ZeroDerivative, ///< Actual requested derivative is exactly zero before division.
  IterationLimit  ///< Requested update budget exhausted and final residual failed.
};
/// @brief Return the stable display name of a numerical stop.
/// @param stop Actual enum; unknown cast values allowed.
/// @return residual, zero-derivative, iteration-limit or unknown.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::newton_raphson_stop_name(dm::algorithms::NewtonRaphsonStop::Residual)); // residual
/// @endcode
/// @see NewtonRaphsonStop
std::string_view newton_raphson_stop_name(NewtonRaphsonStop stop) noexcept;
/// @brief Actual legacy-layout result, stop and selected callback/update work.
/// Legacy result.iterations counts one-based loop inspections, not updates; an
/// initial accepted root reports1 iteration and0 updates when budget is positive.
/// Zero budget still calls f once at x0 and reports0 legacy iterations.
/// @par Example
/// @code
/// auto r=dm::algorithms::newton_raphson_run([](double x){return x-1;},[](double){return 1.;},1);std::print("{} {}\n",r.function_calls,r.updates); // 1 0
/// @endcode
/// @see newton_raphson_trace()
struct NewtonRaphsonRun {
  RootResult result; ///< Actual root/residual/legacy inspection iterations and convergence flag.
  NewtonRaphsonStop stop{
      NewtonRaphsonStop::IterationLimit}; ///< Actual first numerical stopping route.
  std::size_t function_calls{0};          ///< Actual f callbacks, including final cap check.
  std::size_t derivative_calls{
      0}; ///< Actual df callbacks, excluding residual-accepted/cap-final inspections.
  std::size_t inspections{0}; ///< Actual f-value residual checks, including final cap check.
  std::size_t updates{0};     ///< Actual x<-x-f/df assignments.
};
/// @brief Actual function or derivative callback in sequential execution order.
/// @par Example
/// @code
/// auto t=dm::algorithms::newton_raphson_trace([](double x){return x-1;},[](double){return 1.;},0);std::print("{}\n",t.evaluations[1].derivative); // true
/// @endcode
/// @see newton_raphson_trace()
struct NewtonRaphsonEvaluation {
  double x{0};              ///< Actual callback argument.
  double value{0};          ///< Actual finite callback return value.
  bool   derivative{false}; ///< True df, false f.
};
/// @brief Actual residual inspection and any requested derivative/tangent step.
/// @par Example
/// @code
/// auto t=dm::algorithms::newton_raphson_trace([](double x){return x*x-2;},[](double x){return 2*x;},2);std::print("{}\n",t.steps[0].next); // 1.5
/// @endcode
/// @see newton_raphson_trace()
struct NewtonRaphsonStep {
  double x{0};          ///< Actual inspected point.
  double fx{0};         ///< Actual f callback value at this point.
  double derivative{0}; ///< Actual df if derivative_evaluated; zero otherwise.
  double correction{0}; ///< Actual f/df quotient if updated; zero placeholder otherwise.
  double next{0};       ///< Actual assigned next point if updated, inspected point otherwise.
  bool   derivative_evaluated{false}; ///< Whether df was called during this inspection.
  bool   updated{false};              ///< Whether a tangent correction was assigned.
  bool   at_limit{false}; ///< Actual final cap inspection, with no requested derivative.
};
/// @brief Owned original controls, actual callback sequence and actual inspections.
/// Capture counts the three input controls and one unit per callback/inspection
/// record. Admission occurs before that record/callback; no partial trace is
/// returned, though earlier callback side effects remain on a later failure.
/// @par Example
/// @code
/// auto t=dm::algorithms::newton_raphson_trace([](double x){return x-1;},[](double){return 1.;},1);std::print("{}\n",t.captured_entries); // 5
/// @endcode
/// @see newton_raphson_trace()
struct NewtonRaphsonTrace {
  double           initial{0};        ///< Original finite starting point.
  double           tolerance{0};      ///< Original finite nonnegative absolute residual tolerance.
  int              max_iterations{0}; ///< Original nonnegative update cap.
  NewtonRaphsonRun run;               ///< Actual same-producer result and work.
  std::vector<NewtonRaphsonEvaluation> evaluations; ///< Actual sequential f/df calls, no replay.
  std::vector<NewtonRaphsonStep>       steps; ///< Actual residual inspections in execution order.
  std::size_t captured_entries{0}; ///< Actual logical history units3+callbacks+inspections.
};
/// @brief Compute scalar Newton steps with actual strict-residual and zero-derivative stops.
/// Uses x_next=x-f(x)/df(x), preserving legacy callback order and RootResult
/// layout. Quadratic convergence is local near a simple root under smoothness
/// and a suitable initial point; multiple roots, cycles/divergence and an
/// incorrect supplied derivative have no global safeguard or guarantee.
/// Validate both callables, finite x0/tol, tol>=0 and0<=max_iter<INT_MAX before
/// callbacks. tol0 disables residual acceptance. Exact zero df returns a
/// numerical failure. Nonfinite callbacks/intermediates fail explicitly.
/// Every loop inspection calls f first; residual success skips df. At the cap,
/// call f once at the last assigned point, including cap0, then test residual.
/// O(I) producer work plus actual callback costs; no caller object mutation
/// beyond callback effects. Calls are sequential; caller controls callback safety.
/// @param f Nonempty scalar function callback, finite returns required.
/// @param df Nonempty supplied derivative callback, finite returns required.
/// @param x0 Finite initial point.
/// @param tol Finite nonnegative absolute residual threshold, default1e-10.
/// @param max_iter Nonnegative update cap below INT_MAX, default100.
/// @return Owned actual RootResult result; residual convergence is not a root certificate.
/// @throws std::invalid_argument If complete controls/callables are invalid before callbacks.
/// @throws std::domain_error If f or df returns NaN/infinity.
/// @throws std::overflow_error If a correction or assigned point is nonfinite.
/// @throws std::length_error If count geometry is exceeded.
/// @throws std::bad_alloc If allocation fails; caller callback exceptions propagate.
/// @par Example
/// @code
/// auto r=dm::algorithms::newton_raphson([](double x){return x-1;},[](double){return 1.;},0);
/// std::print("{}\n",r.root); // 1
/// @endcode
/// @see NewtonRaphsonRun, NewtonRaphsonTrace
RootResult newton_raphson(
    const std::function<double(double)>& f,
    const std::function<double(double)>& df,
    double                               x0,
    double                               tol      = 1e-10,
    int                                  max_iter = 100);
/// @brief Compute scalar Newton steps with actual strict-residual and zero-derivative stops.
/// Uses x_next=x-f(x)/df(x), preserving legacy callback order and RootResult
/// layout. Quadratic convergence is local near a simple root under smoothness
/// and a suitable initial point; multiple roots, cycles/divergence and an
/// incorrect supplied derivative have no global safeguard or guarantee.
/// Validate both callables, finite x0/tol, tol>=0 and0<=max_iter<INT_MAX before
/// callbacks. tol0 disables residual acceptance. Exact zero df returns a
/// numerical failure. Nonfinite callbacks/intermediates fail explicitly.
/// Every loop inspection calls f first; residual success skips df. At the cap,
/// call f once at the last assigned point, including cap0, then test residual.
/// O(I) producer work plus actual callback costs; no caller object mutation
/// beyond callback effects. Calls are sequential; caller controls callback safety.
/// @param f Nonempty scalar function callback, finite returns required.
/// @param df Nonempty supplied derivative callback, finite returns required.
/// @param x0 Finite initial point.
/// @param tol Finite nonnegative absolute residual threshold, default1e-10.
/// @param max_iter Nonnegative update cap below INT_MAX, default100.
/// @return Owned actual NewtonRaphsonRun result; residual convergence is not a root certificate.
/// @throws std::invalid_argument If complete controls/callables are invalid before callbacks.
/// @throws std::domain_error If f or df returns NaN/infinity.
/// @throws std::overflow_error If a correction or assigned point is nonfinite.
/// @throws std::length_error If count geometry is exceeded.
/// @throws std::bad_alloc If allocation fails; caller callback exceptions propagate.
/// @par Example
/// @code
/// auto r=dm::algorithms::newton_raphson_run([](double x){return x-1;},[](double){return 1.;},0);
/// std::print("{}\n",r.result.root); // 1
/// @endcode
/// @see NewtonRaphsonRun, NewtonRaphsonTrace
NewtonRaphsonRun newton_raphson_run(
    const std::function<double(double)>& f,
    const std::function<double(double)>& df,
    double                               x0,
    double                               tol      = 1e-10,
    int                                  max_iter = 100);
/// @brief Capture actual checked scalar Newton callback and tangent decisions.
/// Uses x_next=x-f(x)/df(x), preserving legacy callback order and RootResult
/// layout. Quadratic convergence is local near a simple root under smoothness
/// and a suitable initial point; multiple roots, cycles/divergence and an
/// incorrect supplied derivative have no global safeguard or guarantee.
/// Validate both callables, finite x0/tol, tol>=0 and0<=max_iter<INT_MAX before
/// callbacks. tol0 disables residual acceptance. Exact zero df returns a
/// numerical failure. Nonfinite callbacks/intermediates fail explicitly.
/// Every loop inspection calls f first; residual success skips df. At the cap,
/// call f once at the last assigned point, including cap0, then test residual.
/// O(I) producer work plus actual callback costs; no caller object mutation
/// beyond callback effects. Calls are sequential; caller controls callback safety.
/// @param f Nonempty scalar function callback, finite returns required.
/// @param df Nonempty supplied derivative callback, finite returns required.
/// @param x0 Finite initial point.
/// @param tol Finite nonnegative absolute residual threshold, default1e-10.
/// @param max_iter Nonnegative update cap below INT_MAX, default100.
/// @param max_entries Online actual logical history cap, default1000000, zero disables.
/// @return Owned actual history with same-producer result and work.
/// @throws std::invalid_argument If complete controls/callables are invalid before callbacks.
/// @throws std::domain_error If f or df returns NaN/infinity.
/// @throws std::overflow_error If a correction or assigned point is nonfinite.
/// @throws std::length_error If count geometry or online capture quota is exceeded.
/// @throws std::bad_alloc If allocation fails; caller callback exceptions propagate.
/// @par Example
/// @code
/// auto r=dm::algorithms::newton_raphson_trace([](double x){return x-1;},[](double){return 1.;},0);
/// std::print("{}\n",r.run.result.root); // 1
/// @endcode
/// @see NewtonRaphsonRun, NewtonRaphsonTrace
NewtonRaphsonTrace newton_raphson_trace(
    const std::function<double(double)>& f,
    const std::function<double(double)>& df,
    double                               x0,
    double                               tol         = 1e-10,
    int                                  max_iter    = 100,
    std::size_t                          max_entries = 1000000);
/// @brief Original square-root-of-two experiment with actual state/tangent plotting data.
/// @par Example
/// @code
/// auto s=dm::algorithms::newton_raphson_square_study();std::print("{}\n",s.trace.run.updates); // 4
/// @endcode
/// @see newton_raphson_trace()
struct NewtonRaphsonSquareStudy {
  NewtonRaphsonTrace
         trace;        ///< Actual canonical x squared minus2 run and all requested derivatives.
  double reference{0}; ///< Native sqrt(2) comparison value.
  std::vector<double> curve_x;          ///< Fixed51-point1.15..2.15 plot grid.
  std::vector<double> curve_y;          ///< Actual x squared minus2 values on the plot grid.
  std::vector<double> states;           ///< Actual inspected x points from the producer.
  std::vector<double> values;           ///< Their actual f callback values.
  std::vector<double> tangent_x;        ///< Actual updated inspection starting x points.
  std::vector<double> tangent_y;        ///< Actual f callback values at tangent starts.
  std::vector<double> tangent_next;     ///< Actual assigned tangent x-axis crossings.
  std::vector<double> indices;          ///< Actual zero-based inspected-state positions.
  std::vector<double> reference_errors; ///< Actual absolute state-minus-sqrt2 errors.
};
/// @brief Produce the original Newton x squared minus2 example without replaying its update loop.
/// Native callback f=x*x-2 and df=2*x; curve points are display samples only.
/// Original default x0=2,tol1e-10 gives4 updates and5 inspections; it does not
/// claim four updates reach the nearest representable sqrt2. No forced dataset.
/// @param x0 Finite seed, default2.
/// @param tol Finite nonnegative absolute residual threshold, default1e-10.
/// @param max_iter Update budget, default100, valid as newton_raphson_trace().
/// @return Owned actual run, states, tangent endpoints and analytic comparison.
/// @throws std::invalid_argument If complete controls are invalid.
/// @throws std::domain_error If an actual callback is nonfinite.
/// @throws std::overflow_error If an actual update is nonfinite.
/// @throws std::length_error If the default trace policy/count geometry is exceeded.
/// @par Example
/// @code
/// auto s=dm::algorithms::newton_raphson_square_study();std::print("{}\n",s.states.front()); // 2
/// @endcode
/// @see NewtonRaphsonSquareStudy
NewtonRaphsonSquareStudy
newton_raphson_square_study(double x0 = 2, double tol = 1e-10, int max_iter = 100);
} // namespace datamunge::algorithms
