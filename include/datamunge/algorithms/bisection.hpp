#pragma once
/// @file bisection.hpp
/// @brief Finite-value bisection with separate position/residual tolerances and bracket traces.
#include <datamunge/algorithms/root_result.hpp>
#include <cstddef>
#include <functional>
#include <limits>
#include <string_view>
#include <vector>

namespace datamunge::algorithms {
/// @brief Reason the canonical bisection loop stopped, distinct from a root certificate.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::bisection_stop_name(dm::algorithms::BisectionStop::Interval)); // interval
/// @endcode
/// @see BisectionTrace, bisection_stop_name()
enum class BisectionStop {
  NoBracket, ///< Nonzero endpoint callback values have the same sign.
  Endpoint, ///< A callback returned zero at an endpoint.
  ExactValue, ///< A callback returned zero at a midpoint; this need not be exact mathematical zero.
  Residual, ///< A positive residual tolerance passed.
  Interval, ///< The rounded maximum midpoint-to-endpoint distance is below the position tolerance.
  Stagnation, ///< Midpoint rounded to an endpoint before a stopping tolerance passed.
  IterationLimit ///< The requested midpoint-evaluation budget was exhausted.
};

/// @brief Independent stopping controls for a recorded bisection search.
/// Position tolerance has x's units; residual tolerance has f(x)'s units. A zero
/// residual tolerance disables that test, while callback zeros still stop.
/// @par Example
/// @code
/// const dm::algorithms::BisectionOptions options{1e-8,0,100};
/// std::print("{}\n",options.max_iterations); // 100
/// @endcode
/// @see bisection_trace(), bisection()
struct BisectionOptions {
  double x_tolerance{1e-10}; ///< Positive finite maximum-distance tolerance in coordinate units.
  double residual_tolerance{0}; ///< Nonnegative finite tolerance in function units; zero disables it.
  int max_iterations{200}; ///< Positive midpoint-evaluation budget.
};

/// @brief Pre-update bracket, callback values and midpoint from one bisection iteration.
/// Records numerical values without retaining the callback. Endpoint signs are
/// opposite when this record is created. Radius is rounded max(midpoint-lower,
/// upper-midpoint), accounting for midpoint rounding to a noncentral double.
/// @par Example
/// @code
/// const auto t=dm::algorithms::bisection_trace([](double x){return x*x-2;},1,2);
/// std::print("{} {}\n",t.steps.front().midpoint,t.steps.front().radius); // 1.5 0.5
/// @endcode
/// @see BisectionTrace
struct BisectionStep {
  double lower{0}; ///< Lower endpoint before the update.
  double upper{0}; ///< Upper endpoint before the update.
  double f_lower{0}; ///< Finite callback value at lower.
  double f_upper{0}; ///< Finite callback value at upper.
  double midpoint{0}; ///< Overflow-safe midpoint, possibly rounded to an endpoint.
  double f_midpoint{0}; ///< Finite callback value at midpoint.
  double radius{0}; ///< Rounded maximum distance from midpoint to either endpoint.
};

/// @brief Owned bisection result, stopping reason and chronological bracket snapshots.
/// Terminal endpoints describe the bracket before the returned estimate's test;
/// a zero endpoint collapses it. Radius is meaningful for evaluated estimates,
/// and remains infinity for a same-sign rejected bracket. It is a rounded interval
/// distance, not an independently certified floating-point interval bound.
/// @par Example
/// @code
/// const auto t=dm::algorithms::bisection_trace([](double x){return x-1;},0,2);
/// std::print("{} {}\n",t.result.root,t.evaluations); // 1 3
/// @endcode
/// @see bisection_trace(), BisectionStop
struct BisectionTrace {
  RootResult result; ///< Estimate, convergence flag, iteration count and callback residual.
  BisectionStop stop{BisectionStop::NoBracket}; ///< Actual stopping reason.
  std::vector<BisectionStep> steps; ///< One snapshot per midpoint evaluation.
  std::size_t evaluations{0}; ///< Callback calls, including initial endpoints that were evaluated.
  double lower{0}; ///< Terminal recorded lower endpoint.
  double upper{0}; ///< Terminal recorded upper endpoint.
  double radius{std::numeric_limits<double>::infinity()}; ///< Rounded distance bound for the estimate.
};

/// @brief Return a stable name for a bisection stopping reason.
/// The string view refers to a program-lifetime literal; no allocation or shared mutation.
/// @param stop Reason enumerator; other representable enum values map to "unknown".
/// @return endpoint, zero callback, residual, interval, stagnation, iteration limit, no bracket, or unknown.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::bisection_stop_name(dm::algorithms::BisectionStop::Residual)); // residual
/// @endcode
/// @see BisectionStop
std::string_view bisection_stop_name(BisectionStop stop);

/// @brief Bisect a sign-changing bracket using the historical combined tolerance.
/// Shares the recorded loop with bisection_trace(), using x_tolerance=tol and
/// residual_tolerance=tol. A small residual can stop with a wide bracket; it is
/// not generally a coordinate-error guarantee. Finite endpoints are ordered
/// internally. Same-sign nonzero endpoints return the default unconverged result.
/// A zero endpoint returns immediately. Continuity and reliable callback signs
/// are caller assumptions. Finite discontinuities can pass the interval test.
/// O(max_iter) callback calls and constant auxiliary storage, with no recording.
/// No shared state; callback side effects and exceptions remain the caller's.
/// @param f Nonempty callable returning finite values at evaluated coordinates.
/// @param a Finite bracket endpoint, in either order relative to b.
/// @param b Finite other bracket endpoint.
/// @param tol Positive finite tolerance applied to both coordinate and residual tests; default 1e-10.
/// @param max_iter Positive midpoint-evaluation budget; default 200.
/// @return Method stopping flag, last estimate, midpoint count and absolute callback value.
/// @throws std::invalid_argument For empty f, nonfinite endpoints/tolerance, nonpositive tol or max_iter.
/// @throws std::domain_error If f returns a nonfinite value; callback exceptions also propagate.
/// @par Example
/// @code
/// const auto r=dm::algorithms::bisection([](double x){return x*x-2;},1,2);
/// std::print("{} {:.6f}\n",r.converged,r.root); // true 1.414214
/// @endcode
/// @par References
/// - NIST DLMF, section 3.8(iii), Bisection Method: https://dlmf.nist.gov/3.8#iii.
/// @see bisection_trace(), BisectionOptions
RootResult bisection(const std::function<double(double)>& f,double a,double b,
                     double tol=1e-10,int max_iter=200);

/// @brief Record canonical bisection with independent position and residual tolerances.
/// Default options disable residual-based stopping. Stopping on a callback zero
/// is retained. Continuity and trustworthy function evaluations are required to
/// infer a root from the bracket; the routine cannot check those assumptions.
/// Opposite signs are compared without multiplying callback values. Uses an
/// overflow-safe midpoint and reports exhausted precision instead of repeating
/// an endpoint indefinitely. Same-sign endpoints are an ordinary failed result.
/// O(max_iterations) callback calls and owned trace storage; no input mutation or
/// shared state. Callback side effects and exceptions remain the caller's.
/// @param f Nonempty finite-valued callback on evaluated coordinates.
/// @param a Finite endpoint; reversed endpoints are accepted.
/// @param b Finite endpoint.
/// @param options Independent tolerances and positive iteration budget, default BisectionOptions.
/// @return Owned result, stop reason, endpoint bracket, callback count and midpoint snapshots.
/// @throws std::invalid_argument For invalid callback, endpoints or options.
/// @throws std::domain_error For nonfinite callback values; callback exceptions propagate.
/// @par Example
/// @code
/// const auto t=dm::algorithms::bisection_trace([](double x){return 1e-20*(x-.75);},0,2);
/// std::print("{}\n",t.result.root); // 0.75
/// @endcode
/// @par References
/// - NIST DLMF, section 3.8(iii), Bisection Method: https://dlmf.nist.gov/3.8#iii.
/// @see bisection(), BisectionStep
BisectionTrace bisection_trace(const std::function<double(double)>& f,double a,double b,
                               BisectionOptions options={});
} // namespace datamunge::algorithms
