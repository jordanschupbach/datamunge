#pragma once
/// @file itp.hpp
/// @brief Checked interpolate-truncate-project root search with actual candidate histories.
#include <cstddef>
#include <datamunge/algorithms/root_result.hpp>
#include <functional>
#include <limits>
#include <string_view>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual ITP stopping reason, not an independently certified root claim.
/// Continuity and reliable callback signs remain caller assumptions.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::itp_stop_name(dm::algorithms::ItpStop::Interval)); // interval
/// @endcode
/// @see itp_stop_name(), ItpRun
enum class ItpStop {
  NoBracket,      ///< Same-sign nonzero endpoint values; no candidate evaluated.
  Endpoint,       ///< A callback returned zero at an initial endpoint.
  ExactValue,     ///< A candidate callback returned zero; not a mathematical exactness certificate.
  Residual,       ///< Positive strict residual tolerance passed, possibly in a wide bracket.
  Interval,       ///< Actual returned midpoint's maximum endpoint distance passed x_tolerance.
  IterationLimit, ///< Caller projection-trial budget exhausted without other acceptance.
  ProjectionLimit, ///< Computed exact-arithmetic projection bound exhausted without rounded interval acceptance.
  Stagnation ///< No interior representable midpoint remained before tolerance acceptance.
};
/// @brief Independent tolerances, projection allowance and optional retained-record ceiling.
/// Fixed kappa2=2 and kappa1=0.2/initial_width. extra_iterations=n0 allows
/// n_half+n0 projection trials in exact arithmetic; only n0=0 has bisection's
/// worst-case trial bound. This is not a floating interval certificate. A zero
/// residual tolerance disables that test; callback zeros still stop. max_records
/// is an online scalar-record count, admitted before each candidate callback;
/// zero disables it and untraced runs ignore it. Excludes endpoints, capacities,
/// callback-owned storage and byte/peak RAM. No partial trace returns on error.
/// @par Example
/// @code
/// const dm::algorithms::ItpOptions o;
/// std::print("{} {}\n",o.extra_iterations,o.max_records); // 1 1000000
/// @endcode
/// @see itp_run(), itp_trace()
struct ItpOptions {
  double x_tolerance{1e-10}; ///< Positive finite coordinate maximum-distance tolerance.
  double residual_tolerance{
      0}; ///< Nonnegative finite strict function-value tolerance; zero disables.
  int max_iterations{
      100}; ///< Nonnegative projection-trial cap, excluding endpoint/final-midpoint calls.
  int         extra_iterations{1}; ///< Nonnegative n0 added to the checked bisection count.
  std::size_t max_records{
      1000000}; ///< Actual candidate records, including final midpoint; zero disables.
};
/// @brief One actual pre-update bracket and evaluated candidate in the producing loop.
/// projected=true means a projection trial; false is a terminal midpoint call,
/// whose interpolation/truncation/projection fields are zero and unused. Wide
/// scalars avoid double span/product overflow where long double has extra range;
/// precision/range remain platform dependent. radius is max(x-lower,upper-x),
/// not always half the width. All snapshots are owned scalars.
/// @par Example
/// @code
/// const auto t=dm::algorithms::itp_trace([](double x){return x*x-2;},1,2);
/// std::print("{}\n",t.steps.front().projected); // true
/// @endcode
/// @see ItpTrace
struct ItpStep {
  int iteration{
      0}; ///< Number of actual projection trials including this one, or completed before final midpoint.
  bool        projected{false}; ///< True for I/T/P trial; false for final midpoint evaluation.
  double      lower{0};         ///< Actual ordered lower endpoint before this evaluation.
  double      upper{0};         ///< Actual upper endpoint before this evaluation.
  double      f_lower{0};       ///< Actual finite lower endpoint callback value.
  double      f_upper{0};       ///< Actual finite upper endpoint callback value.
  long double midpoint{0};     ///< Working-precision rounded bracket centre for a projection trial.
  long double interpolated{0}; ///< Actual wide regula-falsi point, only for projected trials.
  long double delta{0};     ///< Actual kappa1*width^2 truncation amount, only for projected trials.
  long double truncated{0}; ///< Actual wide truncated estimate, only for projected trials.
  long double projection_radius{0}; ///< Actual nonnegative feasible radius, capped at half-width.
  double      x{0};      ///< Actual evaluated double coordinate after rounding/interior fallback.
  double      f_x{0};    ///< Actual finite callback value at x.
  long double radius{0}; ///< Actual working-precision maximum distance from x to endpoints.
  bool        rounding_adjusted{
      false}; ///< Trial required an interior midpoint fallback or lies outside the wide projection band after rounding.
};
/// @brief Final ITP summary, bracket, actual callbacks and theoretical trial allowance.
/// Terminal endpoints contain the root only under continuity/reliable-sign
/// assumptions. Endpoint/candidate zeros collapse them. Residual acceptance can
/// leave a wide bracket. No retained callable/input references or shared state.
/// @par Example
/// @code
/// const auto r=dm::algorithms::itp_run([](double x){return x-1;},0,2);
/// std::print("{} {}\n",r.result.root,r.evaluations); // 1 3
/// @endcode
/// @see itp_run(), ItpStop
struct ItpRun {
  RootResult  result; ///< Legacy layout; iterations counts only actual projection trials.
  ItpStop     stop{ItpStop::NoBracket}; ///< Actual stopping reason, not a root certificate.
  std::size_t evaluations{
      0}; ///< All actual f calls, including evaluated endpoints and terminal midpoint.
  int bisection_bound{
      0}; ///< Least nonnegative halvings making the initial wide half-width <=x_tolerance.
  int         projection_bound{0}; ///< Checked bisection_bound+extra_iterations.
  double      lower{0};            ///< Terminal actual lower endpoint.
  double      upper{0};            ///< Terminal actual upper endpoint.
  long double radius{
      std::numeric_limits<long double>::
          infinity()}; ///< Maximum endpoint distance for evaluated estimate, infinity for no bracket.
};
/// @brief Owned same-execution scalar history and final ITP diagnostics.
/// No callback replay or repeated shorter runs. Endpoints are not step records;
/// every actual projected/final-midpoint candidate call has one record.
/// @par Example
/// @code
/// const auto t=dm::algorithms::itp_trace([](double x){return x-1;},0,2);
/// std::print("{}\n",t.steps.size()); // 1
/// @endcode
/// @see itp_trace()
struct ItpTrace {
  ItpRun               run;   ///< Actual final result, stop, bracket and counts.
  std::vector<ItpStep> steps; ///< Owned chronological projected/final-midpoint candidate records.
};
/// @brief Return a stable program-lifetime literal for an ITP stopping reason.
/// No allocation/mutation/shared state; unknown enum values return "unknown".
/// @param stop Actual stop enumerator.
/// @return Stable readable reason name.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::itp_stop_name(dm::algorithms::ItpStop::ProjectionLimit)); // projection limit
/// @endcode
/// @see ItpStop
std::string_view itp_stop_name(ItpStop stop);
/// @brief Solve a finite sign-changing bracket with the historical combined tolerance.
/// Uses the same checked kernel as itp_run(), x_tolerance=tol and
/// residual_tolerance=tol, fixed kappa2=2, scaled kappa1=0.2/initial_width,
/// n0=1. Unlike the old blanket claim, its exact-arithmetic worst case allows
/// one extra projection trial over bisection. Rounded near-boundary arithmetic
/// may stop at a projection/precision limit instead of claiming convergence.
/// Inputs are ordered internally; same-sign endpoints return failed defaults,
/// callback zero endpoints return immediately. Continuity and accurate callback
/// signs are unchecked assumptions. Callback zeros/residuals need not certify
/// a mathematical root or coordinate accuracy. Original signature/layout retained;
/// safer wide arithmetic and consistent guards can change last digits/counts.
/// No input mutation, retained callback or shared state; callback side effects
/// and exceptions remain the caller's. O(K+N_half+1) scalar work
/// plus at most K+3 callback calls, constant auxiliary storage, no history.
/// @param f Nonempty callable with finite values at evaluated coordinates.
/// @param a Finite first endpoint, either order.
/// @param b Finite other endpoint.
/// @param tol Positive finite combined coordinate/function tolerance, default1e-10.
/// @param max_iter Nonnegative projection-trial budget, default100; zero still permits endpoints/final midpoint.
/// @return Owned legacy result; converged means a numerical stopping criterion passed.
/// @throws std::invalid_argument On invalid controls, endpoints or empty callback, before callbacks.
/// @throws std::domain_error On a nonfinite callback return; callback exceptions propagate.
/// @throws std::overflow_error On unrepresentable wide arithmetic.
/// @throws std::length_error On count/exponent overflow.
/// @par Example
/// @code
/// const auto r=dm::algorithms::itp([](double x){return x*x-2;},1,2);
/// std::print("{} {:.6f}\n",r.converged,r.root); // true 1.414214
/// @endcode
/// @par References
/// - Oliveira and Takahashi (2021), ACM TOMS47(1), article5: https://doi.org/10.1145/3423597.
/// - Maintained implementation documentation: https://paulnorthrop.github.io/itp/reference/itp.html.
/// @see itp_run(), itp_trace()
RootResult itp(
    const std::function<double(double)>& f,
    double                               a,
    double                               b,
    double                               tol      = 1e-10,
    int                                  max_iter = 100);
/// @brief Run checked ITP with separate coordinate/residual criteria and no history.
/// Same finite/sign/rounding/ownership/numerical policies as itp(). Options expose
/// n0 and independent stopping; no derivative or monotonicity is required.
/// Fixed kappa2=2 and kappa1=0.2/initial_width. Reported residual uses an actual
/// callback; neither continuity nor reliable signs can be checked. A final
/// midpoint callback is additional to the projection-trial budget. max_records
/// is ignored. No mutation, retained references or shared state; independent
/// calls except caller-owned callback state. O(K+N_half+1) scalar
/// work plus callback costs, constant storage. Bounds apply in exact arithmetic,
/// not as an independently rounded interval certificate.
/// @param f Borrowed nonempty finite-valued callback; exceptions/side effects propagate.
/// @param a Finite bracket endpoint, either order.
/// @param b Finite other endpoint.
/// @param options Independent positive coordinate/nonnegative residual tolerances, nonnegative trial/n0 counts and history policy.
/// @return Owned actual result, reason, bracket, selected counts and exact-arithmetic allowance.
/// @throws std::invalid_argument On invalid controls, endpoints or empty callback before callbacks.
/// @throws std::domain_error On nonfinite callback return; callback exceptions propagate.
/// @throws std::overflow_error On unrepresentable wide arithmetic.
/// @throws std::length_error On count/exponent overflow.
/// @par Example
/// @code
/// const auto r=dm::algorithms::itp_run([](double x){return 1e-20*(x-.75);},0,2,{1e-8,1e-10});
/// std::print("{}\n",dm::algorithms::itp_stop_name(r.stop)); // residual
/// @endcode
/// @see itp(), ItpOptions
ItpRun itp_run(const std::function<double(double)>& f, double a, double b, ItpOptions options = {});
/// @brief Capture actual I/T/P stages and all terminal-midpoint candidate calls.
/// Same numerical and callback policies as itp_run(). Each actual candidate
/// record is admitted before its callback; max_records=0 disables this optional
/// scalar-record cap. No partial result returns on errors; callback side effects
/// already executed persist. Untraced runs ignore the cap. Additional O(K)
/// history storage, excluding endpoints, capacities, callback storage and byte
/// peak RAM. All snapshots are owned; no solve replay or retained callable.
/// @param f Borrowed nonempty finite-valued callback.
/// @param a Finite endpoint, either order.
/// @param b Finite other endpoint.
/// @param options Independent criteria, projection allowance, trial and retained-record limits.
/// @return Owned same-execution chronology and final run diagnostics.
/// @throws std::invalid_argument On invalid controls, endpoints or empty callback before callbacks.
/// @throws std::domain_error On nonfinite callback return; callback exceptions propagate.
/// @throws std::overflow_error On unrepresentable wide arithmetic.
/// @throws std::length_error On enabled record cap or count/exponent overflow.
/// @throws std::bad_alloc On history allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::itp_trace([](double x){return x-1;},0,2,{1e-10,0,100,0});
/// std::print("{} {}\n",t.run.result.root,t.run.projection_bound); // 1 34
/// @endcode
/// @see itp_run(), ItpStep
ItpTrace
itp_trace(const std::function<double(double)>& f, double a, double b, ItpOptions options = {});
} // namespace datamunge::algorithms
