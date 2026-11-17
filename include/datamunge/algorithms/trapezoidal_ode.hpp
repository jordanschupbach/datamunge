#pragma once
/// @file trapezoidal_ode.hpp
/// @brief Actual checked implicit trapezoidal steps and explicit Newton residual convergence.
#include <cstddef>
#include <functional>
#include <string>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Own TrapezoidOptions values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TrapezoidOptions r{};
/// (void)r;
/// @endcode
struct TrapezoidOptions {
  int max_newton_iterations{
      60}; ///< Nonnegative actual correction-assignment cap per implicit step.
  double absolute_tolerance{1e-14}; ///< Finite nonnegative implicit residual tolerance component.
  double relative_tolerance{
      1e-12}; ///< Finite nonnegative multiplier of max(abs(previous y),abs(candidate y)).
  std::size_t max_steps{1000000};        ///< Complete requested grid steps; zero disables.
  std::size_t max_evaluations{20000000}; ///< Actual RHS callback invocations; zero disables.
  std::size_t max_records{1000000};      ///< Actual optional Newton attempt records; zero disables.
};
/// @brief Own TrapezoidNewton values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TrapezoidNewton r{};
/// (void)r;
/// @endcode
struct TrapezoidNewton {
  std::size_t step{}; ///< Actual zero-based attempted time step.
  int iteration{}; ///< Actual one-based correction attempt; zero derivative can prevent its assignment.
  double guess{};          ///< Actual candidate before this Newton attempt.
  double residual{};       ///< Actual implicit equation residual at guess.
  double perturbation{};   ///< Actual finite representable derivative-sample argument.
  double perturbed_rhs{};  ///< Actual RHS sampled at that argument and next time.
  double derivative{};     ///< Actual numerical derivative of the implicit equation.
  bool   corrected{};      ///< Actual correction assignment occurred; false for zero derivative.
  double next{};           ///< Actual candidate after correction, or same guess when no assignment.
  double residual_after{}; ///< Actual recomputed implicit residual at next.
};
/// @brief Own TrapezoidStep values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TrapezoidStep r{};
/// (void)r;
/// @endcode
struct TrapezoidStep {
  double time_before{}; ///< Actual accepted previous time.
  double time_after{};  ///< Actual next equal-grid time; final requested time pinned exactly to t1.
  double previous{};    ///< Actual accepted previous y.
  double rhs_before{};  ///< Actual f(time_before,previous).
  double predictor{};   ///< Actual forward-Euler predictor.
  double candidate{};   ///< Actual final implicit candidate, accepted only when converged.
  double residual{};    ///< Actual final implicit residual at candidate.
  int    iterations{};  ///< Actual completed Newton correction assignments.
  bool   converged{};   ///< Actual final residual meets configured absolute/relative threshold.
};
/// @brief Own TrapezoidRun values for one independent native execution.
/// Fields describe actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record alone performs no algorithm work and retains no caller storage or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::TrapezoidRun r{};
/// (void)r;
/// @endcode
struct TrapezoidRun {
  std::vector<std::pair<double, double>>
       trajectory;  ///< Initial state plus actually accepted converged states only.
  bool converged{}; ///< All requested implicit steps accepted.
  std::string
      stop; ///< Actual complete, Newton iteration limit, zero Newton derivative or precision exhausted.
  std::size_t completed_steps{}; ///< Actual accepted time steps.
  std::size_t evaluations{};     ///< Actual RHS callback invocations.
  std::size_t newton_updates{}; ///< Actual completed correction assignments across attempted steps.
  std::vector<TrapezoidStep>
      steps; ///< Actual per-attempt predictor/final residual/status, including failed attempt.
  std::vector<TrapezoidNewton>
      newton; ///< Optional actual derivative/correction histories; no callback replay.
};
/// @brief Integrate actual implicit trapezoidal steps with honest Newton outcomes.
/// Requires a nonempty finite-valued synchronous scalar RHS, finite y0/t0/t1 and positive requested steps. Backward time is supported. Grid endpoints use stable interpolation and the last time is exactly t1. Each step solves Y-y-h/2*(f(t,y)+f(t_next,Y))=0 using a forward-Euler predictor and finite-difference Newton derivative. Accept a step only if the actual recomputed equation residual is within atol+rtol*max(abs(y),abs(Y)); a small correction alone is insufficient. Zero Newton cap still checks the predictor residual. Zero span copies identical states without RHS calls. Run output distinguishes iteration exhaustion, zero derivative and representational stagnation, retaining only converged accepted trajectory states. Callback exceptions/side effects propagate without rollback and actual values are not replayed. Trapezoidal A-stability is a property of exactly solved linear steps; this finite nonlinear Newton implementation does not certify universal nonlinear convergence and is not L-stable. O(steps*Newton cap) callback work, owned linear step diagnostics plus optional histories, no internal shared state.
/// @param f Borrowed synchronous finite scalar RHS callback; assumed mathematical/deterministic for ODE interpretation.
/// @param y0 Finite initial y.
/// @param t0 Finite initial time.
/// @param t1 Finite final time, either direction allowed.
/// @param steps Complete strictly positive equal-grid step count.
/// @param options Nonnegative finite residual tolerances, Newton cap and complete/actual policies.
/// @return Owned accepted trajectory, actual implicit residual/status/work and optional Newton attempts.
/// @throws std::invalid_argument Malformed callback, finite inputs, nonpositive steps or invalid controls.
/// @throws std::domain_error Actual RHS callback result is nonfinite.
/// @throws std::overflow_error Actual predictor/equation/derivative/correction arithmetic not representable.
/// @throws std::length_error Complete grid or actual callback/history/counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::trapezoidal_ode_run([](double,double){return 0.;},2,0,1,1);
/// std::print("{} {}\n",r.converged,r.trajectory.back().second); // true 2
/// @endcode
/// @par References
/// - Implicit trapezoidal rule; A-stability applies to solved linear test equations.
/// @see trapezoidal_ode()
TrapezoidRun trapezoidal_ode_run(
    const std::function<double(double, double)>& f,
    double                                       y0,
    double                                       t0,
    double                                       t1,
    int                                          steps,
    const TrapezoidOptions&                      options = {});
/// @brief Integrate actual implicit trapezoidal steps with honest Newton outcomes.
/// Requires a nonempty finite-valued synchronous scalar RHS, finite y0/t0/t1 and positive requested steps. Backward time is supported. Grid endpoints use stable interpolation and the last time is exactly t1. Each step solves Y-y-h/2*(f(t,y)+f(t_next,Y))=0 using a forward-Euler predictor and finite-difference Newton derivative. Accept a step only if the actual recomputed equation residual is within atol+rtol*max(abs(y),abs(Y)); a small correction alone is insufficient. Zero Newton cap still checks the predictor residual. Zero span copies identical states without RHS calls. Run output distinguishes iteration exhaustion, zero derivative and representational stagnation, retaining only converged accepted trajectory states. Callback exceptions/side effects propagate without rollback and actual values are not replayed. Trapezoidal A-stability is a property of exactly solved linear steps; this finite nonlinear Newton implementation does not certify universal nonlinear convergence and is not L-stable. O(steps*Newton cap) callback work, owned linear step diagnostics plus optional histories, no internal shared state.
/// @param f Borrowed synchronous finite scalar RHS callback; assumed mathematical/deterministic for ODE interpretation.
/// @param y0 Finite initial y.
/// @param t0 Finite initial time.
/// @param t1 Finite final time, either direction allowed.
/// @param steps Complete strictly positive equal-grid step count.
/// @param options Nonnegative finite residual tolerances, Newton cap and complete/actual policies.
/// @return Owned accepted trajectory, actual implicit residual/status/work and optional Newton attempts.
/// @throws std::invalid_argument Malformed callback, finite inputs, nonpositive steps or invalid controls.
/// @throws std::domain_error Actual RHS callback result is nonfinite.
/// @throws std::overflow_error Actual predictor/equation/derivative/correction arithmetic not representable.
/// @throws std::length_error Complete grid or actual callback/history/counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::trapezoidal_ode_trace([](double,double){return 0.;},2,0,1,1);
/// std::print("{} {}\n",r.converged,r.trajectory.back().second); // true 2
/// @endcode
/// @par References
/// - Implicit trapezoidal rule; A-stability applies to solved linear test equations.
/// @see trapezoidal_ode()
TrapezoidRun trapezoidal_ode_trace(
    const std::function<double(double, double)>& f,
    double                                       y0,
    double                                       t0,
    double                                       t1,
    int                                          steps,
    const TrapezoidOptions&                      options = {});
/// @brief Preserve a complete implicit trapezoidal trajectory or fail on unresolved Newton work.
/// Requires a nonempty finite-valued synchronous scalar RHS, finite y0/t0/t1 and positive requested steps. Backward time is supported. Grid endpoints use stable interpolation and the last time is exactly t1. Each step solves Y-y-h/2*(f(t,y)+f(t_next,Y))=0 using a forward-Euler predictor and finite-difference Newton derivative. Accept a step only if the actual recomputed equation residual is within atol+rtol*max(abs(y),abs(Y)); a small correction alone is insufficient. Zero Newton cap still checks the predictor residual. Zero span copies identical states without RHS calls. Run output distinguishes iteration exhaustion, zero derivative and representational stagnation, retaining only converged accepted trajectory states. Callback exceptions/side effects propagate without rollback and actual values are not replayed. Trapezoidal A-stability is a property of exactly solved linear steps; this finite nonlinear Newton implementation does not certify universal nonlinear convergence and is not L-stable. O(steps*Newton cap) callback work, owned linear step diagnostics plus optional histories, no internal shared state.
/// @param f Borrowed synchronous finite scalar RHS callback; assumed mathematical/deterministic for ODE interpretation.
/// @param y0 Finite initial y.
/// @param t0 Finite initial time.
/// @param t1 Finite final time, either direction allowed.
/// @param steps Complete strictly positive equal-grid step count.
/// @return Owned steps+1 accepted(time,y) pairs when every implicit step converges.
/// @throws std::invalid_argument Malformed complete inputs.
/// @throws std::domain_error Nonfinite actual callback value.
/// @throws std::overflow_error Actual arithmetic not representable.
/// @throws std::length_error Default grid/work capacity exceeded.
/// @throws std::runtime_error An implicit step remains unresolved; use run for partial trajectory and status.
/// @par Example
/// @code
/// auto r=dm::algorithms::trapezoidal_ode([](double,double){return 0.;},2,0,1,1);
/// std::print("{}\n",r.back().second); // 2
/// @endcode
/// @see trapezoidal_ode_run()
std::vector<std::pair<double, double>> trapezoidal_ode(
    const std::function<double(double, double)>& f,
    double                                       y0,
    double                                       t0,
    double                                       t1,
    int                                          steps);
} // namespace datamunge::algorithms
