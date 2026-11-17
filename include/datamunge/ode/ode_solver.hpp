#pragma once
/// @file ode_solver.hpp
/// @brief Checked forward ODE integration with actual callback/stage and accepted/rejected attempt histories.
#include <cstddef>
#include <datamunge/ode/rhs.hpp>
#include <string>
#include <vector>
/// @brief Canonical checked forward initial-value ODE integration and native studies.
namespace datamunge::ode {
/// @brief Original native explicit/adaptive/PECE method selectors.
/// @par Example
/// @code
/// dm::ode::ODEOptions o;std::print("{}\n",o.method==dm::ode::StepMethod::RK4); // true
/// @endcode
/// @see ODESolver
enum class StepMethod {
  Euler,          ///< One-stage forward Euler.
  Midpoint,       ///< Two-stage explicit midpoint.
  RK4,            ///< Four-stage classical Runge-Kutta.
  RK45,           ///< Seven-evaluation Dormand-Prince embedded4/5, no FSAL reuse.
  AdamsBashforth, ///< Explicit constant-step Adams order2through4, RK4 startup/tail.
  AdamsMoulton    ///< One PECE correction after Adams prediction, not an iterated implicit solve.
};
/// @brief Original-order numerical controls for all native forward methods.
/// @par Example
/// @code
/// dm::ode::ODEOptions o;std::print("{}\n",o.step_size); // 0.01
/// @endcode
/// @see ODESolver
struct ODEOptions {
  StepMethod  method{StepMethod::RK4}; ///< Original selected known method.
  std::size_t multistep_order{4};      ///< Adams order clamped to2..4, ignored by one-step methods.
  double      step_size{.01};          ///< Finite positive fixed step or initial RK45 trial step.
  double      abs_tol{
      1e-6}; ///< Finite positive RK45 local absolute tolerance; validated even if inactive.
  double rel_tol{
      1e-6}; ///< Finite nonnegative RK45 local relative tolerance; validated even if inactive.
  double max_step{
      0}; ///< Finite nonnegative RK45 maximum step;0 unbounded, inactive for fixed methods.
};
/// @brief Optional actual work, retained solution and trace policies, all0values disable their quota.
/// @par Example
/// @code
/// dm::ode::ODEExecutionLimits l;std::print("{}\n",l.max_attempts); // 1000000
/// @endcode
/// @see ODESolver
struct ODEExecutionLimits {
  std::size_t max_attempts{
      1000000}; ///< Actual fixed/startup/adaptive/Adams trial count, including rejected trials.
  std::size_t max_evaluations{
      7000000}; ///< Actual RHS callback count, admitted before callback effects.
  std::size_t max_solution_entries{
      1000000}; ///< Actual retained time/state scalar count,(1+D)*samples, including initial sample.
  std::size_t max_trace_entries{
      1000000}; ///< Actual owned input/solution/callback/attempt logical scalar/record quota;0 disables.
};
/// @brief Owned actual forward trajectory including initial and exact requested final time.
/// Public vectors are mutable by C++ callers; accessors retain the original
/// cross-language-friendly shape without promising every binding installation.
/// @par Example
/// @code
/// dm::ode::ODESolution s;std::print("{}\n",s.size()); // 0
/// @endcode
/// @see ODESolver
struct ODESolution {
  std::vector<double>
      t; ///< Owned strictly increasing accepted times, including both endpoints on success.
  std::vector<std::vector<double>>
              y;              ///< Owned matching finite fixed-dimensional accepted state vectors.
  std::size_t steps_taken{0}; ///< Actual accepted complete updates, excluding rejected RK45 trials.
  std::size_t function_evaluations{
      0}; ///< Actual successful RHS callbacks, including startup/history and rejected-trial stages.

  /// @brief Read actual retained time/state sample count.
  /// @return t.size(), without mutation.
  /// @par Example
  /// @code
  /// dm::ode::ODESolution s;std::print("{}\n",s.size()); // 0
  /// @endcode
  /// @see time_at(), state_at()
  [[nodiscard]] std::size_t size() const noexcept { return t.size(); }
  /// @brief Copy one actual retained sample time, checking its index.
  /// @param index Original retained sample position.
  /// @return Actual time value, without mutation.
  /// @throws std::out_of_range If index is outside t.
  /// @par Example
  /// @code
  /// dm::ode::ODESolution s;s.t={0};std::print("{}\n",s.time_at(0)); // 0
  /// @endcode
  /// @see state_at()
  [[nodiscard]] double time_at(std::size_t index) const { return t.at(index); }
  /// @brief Borrow one actual retained full state, checking its index.
  /// @param index Original retained sample position.
  /// @return Read-only state reference, valid until this solution is mutated/destroyed.
  /// @throws std::out_of_range If index is outside y.
  /// @par Example
  /// @code
  /// dm::ode::ODESolution s;s.y={{1}};std::print("{}\n",s.state_at(0)); // [1]
  /// @endcode
  /// @see time_at()
  [[nodiscard]] const std::vector<double>& state_at(std::size_t index) const { return y.at(index); }
};
/// @brief Actual selected step kernel, distinguishing startup/shortened Adams tails.
/// @par Example
/// @code
/// std::print("{}\n",dm::ode::ODEAttemptKind::rk4_tail==dm::ode::ODEAttemptKind::rk4_tail); // true
/// @endcode
/// @see ODEAttempt
enum class ODEAttemptKind {
  fixed_step,      ///< Actual configured Euler/midpoint/RK4 update.
  rk45_trial,      ///< Actual adaptive embedded Dormand-Prince trial.
  rk4_startup,     ///< Actual Adams self-starting classical RK4 update.
  adams_bashforth, ///< Actual full constant-step explicit Adams update.
  adams_pece,      ///< Actual full constant-step predictor plus one corrector.
  rk4_tail         ///< Actual shortened final Adams step, safely handled by RK4.
};
/// @brief Actual finite RHS input state/time and returned derivative, copied from the executing callback.
/// @par Example
/// @code
/// dm::ode::ODEEvaluation e;std::print("{}\n",e.time); // 0
/// @endcode
/// @see ODESolver
struct ODEEvaluation {
  double              time{0};    ///< Actual finite callback evaluation time.
  std::vector<double> state;      ///< Owned actual finite input state.
  std::vector<double> derivative; ///< Owned actual returned finite matching-dimensional derivative.
};
/// @brief Actual step proposal, acceptance and callback range, not a replay.
/// @par Example
/// @code
/// dm::ode::ODEAttempt a;std::print("{}\n",a.accepted); // false
/// @endcode
/// @see ODESolver
struct ODEAttempt {
  ODEAttemptKind      kind{ODEAttemptKind::fixed_step}; ///< Actual kernel used for this trial.
  double              time_before{0}; ///< Actual accepted time before the attempt.
  double              step{0};        ///< Actual finite positive selected step.
  std::vector<double> before;         ///< Owned actual unchanged old state.
  std::vector<double>
         proposed; ///< Owned actual finite proposed new state, including rejected RK45 proposals.
  double time_after{0}; ///< Actual committed time when accepted, old time when rejected.
  double error_norm{
      0}; ///< Actual RK45 RMS scaled embedded error;0 placeholder for other kernels,may+inf when rejecting.
  bool error_widened{false}; ///< Actual widened error normalization fallback.
  bool accepted{
      false}; ///< Actual commit decision; RK45 requires error_norm<=1 without forced tiny-step acceptance.
  std::size_t evaluation_begin{0}; ///< Inclusive actual callback index before the attempt.
  std::size_t evaluation_end{
      0}; ///< Exclusive actual callback index after the attempt,including Adams derivative caching.
};
/// @brief Owned original controls/state and actual callback/attempt/accepted-solution histories.
/// @par Example
/// @code
/// dm::ode::ODETrace t;std::print("{}\n",t.evaluations.size()); // 0
/// @endcode
/// @see ODESolver
struct ODETrace {
  ODEOptions          options; ///< Original numerical controls.
  ODEExecutionLimits  limits;  ///< Original selected policies, excluded from logical scalar quota.
  std::vector<double> initial_state; ///< Owned original finite initial state.
  double              start{0};      ///< Original finite start time.
  double              end{0};        ///< Original finite strictly later end time.
  ODESolution         solution;      ///< Actual complete accepted trajectory.
  std::vector<ODEEvaluation>
      evaluations; ///< Every actual successful finite callback, including rejected trials/history initialization.
  std::vector<ODEAttempt>
              attempts; ///< Every actual fixed/startup/Adams/adaptive proposal and decision.
  std::size_t captured_entries{
      0}; ///< Actual root8+D + each solution sample(1+D)+evaluation(2+2D)+attempt(1+2D).
};
/// @brief Owned-options forward solver with canonical one-step/adaptive/PECE paths.
/// Different solver objects own their options; concurrency also depends on the
/// caller's RHS thread safety. Methods use local work buffers and no global state.
/// @par Example
/// @code
/// dm::ode::ODESolver s;auto r=s.solve_builtin("exponential_decay",{1.},{1.},0.,.01);std::print("{}\n",r.t.back()); // 0.01
/// @endcode
/// @par References
/// - NIST DLMF3.7(v), Runge-Kutta Method, https://dlmf.nist.gov/3.7.v.
/// - J. R. Dormand and P. J. Prince(1980), A family of embedded Runge-Kutta formulae, doi:10.1016/0771-050X(80)90013-3. Native controller does not reuse FSAL and rejects nonconforming tiny trials.
/// @see RHS, ODETrace
class ODESolver {
 public:
  /// @brief Own and validate the complete original numerical options.
  /// @param options Known method and finite admissible controls, defaults RK4/h.01.
  /// @throws std::invalid_argument If any complete method/numerical control fails.
  /// @par Example
  /// @code
  /// dm::ode::ODESolver s;std::print("{}\n",s.options().step_size); // 0.01
  /// @endcode
  /// @par References
  /// - NIST DLMF3.7(v), Runge-Kutta Method, https://dlmf.nist.gov/3.7.v.
  /// - J. R. Dormand and P. J. Prince(1980), A family of embedded Runge-Kutta formulae, doi:10.1016/0771-050X(80)90013-3. Native controller does not reuse FSAL and rejects nonconforming tiny trials.
  /// @see options()
  explicit ODESolver(ODEOptions options = {});
  /// @brief Integrate one complete checked forward initial-value problem.
  /// Complete finite nonempty y0 and finite forward span t_end>t_start with
  /// representable difference are checked before callbacks. Known method, finite
  /// positive step/abs_tol, finite rel_tol>=0 and max_step>=0 are validated, including
  /// inactive numeric controls. Each callback receives a finite fixed-dimensional
  /// stage and must return finite matching dimension; exceptions propagate.
  /// Steps must advance representable time; tiny/nonadvancing steps throw rather
  /// than loop. No absolute1e-14 endpoint skip; successful output reaches t_end.
  /// Adams uses original RK4 startup and clamped order2..4; a shortened final step
  /// uses RK4 instead of applying constant-grid coefficients to unequal spacing.
  /// RK45 does not reuse FSAL and accepts only scaled error<=1, not a tiny-step bypass.
  /// Local error control is not a certified global error, stability or energy bound.
  /// State/control/input references are not retained. Callback effects before a
  /// later error/quota failure persist; no partial solution/trace is returned.
  /// Default work policies:1000000 attempts,7000000 RHS calls,1000000 retained
  /// solution scalar entries; limits overloads allow0 to disable each policy.
  /// @param rhs Live derivative callback, finite matching-dimensional outputs required.
  /// @param y0 Original finite nonempty fixed-dimensional state.
  /// @param t_start Original finite initial time.
  /// @param t_end Original finite strictly later terminal time.
  /// @return Owned actual ODESolution, including both requested endpoint times on success.
  /// @throws std::invalid_argument If complete controls/state/builtin or callback dimension fails.
  /// @throws std::length_error If complete/actual count geometry or selected policies fail.
  /// @throws std::overflow_error If a stage/derivative/proposal or advancing time is not representable.
  /// @throws std::bad_alloc If owned storage allocation fails.
  /// @par Example
  /// @code
  /// struct F:dm::ode::RHS{std::vector<double>evaluate(double,const std::vector<double>&y)override{return {-y[0]};}};F f;dm::ode::ODESolver s({dm::ode::StepMethod::Euler,4,.25});std::print("{}\n",s.solve(f,{1.},0.,.25).y.back()[0]); // 0.75
  /// @endcode
  /// @par References
  /// - NIST DLMF3.7(v), Runge-Kutta Method, https://dlmf.nist.gov/3.7.v.
  /// - J. R. Dormand and P. J. Prince(1980), A family of embedded Runge-Kutta formulae, doi:10.1016/0771-050X(80)90013-3. Native controller does not reuse FSAL and rejects nonconforming tiny trials.
  /// @see ODEAttempt, ODEEvaluation
  [[nodiscard]] ODESolution
  solve(RHS& rhs, const std::vector<double>& y0, double t_start, double t_end) const;
  /// @brief Integrate one complete checked forward initial-value problem.
  /// Complete finite nonempty y0 and finite forward span t_end>t_start with
  /// representable difference are checked before callbacks. Known method, finite
  /// positive step/abs_tol, finite rel_tol>=0 and max_step>=0 are validated, including
  /// inactive numeric controls. Each callback receives a finite fixed-dimensional
  /// stage and must return finite matching dimension; exceptions propagate.
  /// Steps must advance representable time; tiny/nonadvancing steps throw rather
  /// than loop. No absolute1e-14 endpoint skip; successful output reaches t_end.
  /// Adams uses original RK4 startup and clamped order2..4; a shortened final step
  /// uses RK4 instead of applying constant-grid coefficients to unequal spacing.
  /// RK45 does not reuse FSAL and accepts only scaled error<=1, not a tiny-step bypass.
  /// Local error control is not a certified global error, stability or energy bound.
  /// State/control/input references are not retained. Callback effects before a
  /// later error/quota failure persist; no partial solution/trace is returned.
  /// Default work policies:1000000 attempts,7000000 RHS calls,1000000 retained
  /// solution scalar entries; limits overloads allow0 to disable each policy.
  /// @param rhs Live derivative callback, finite matching-dimensional outputs required.
  /// @param y0 Original finite nonempty fixed-dimensional state.
  /// @param t_start Original finite initial time.
  /// @param t_end Original finite strictly later terminal time.
  /// @param limits Actual work/output/history policies, defaults documented above;0 disables each.
  /// @return Owned actual ODESolution, including both requested endpoint times on success.
  /// @throws std::invalid_argument If complete controls/state/builtin or callback dimension fails.
  /// @throws std::length_error If complete/actual count geometry or selected policies fail.
  /// @throws std::overflow_error If a stage/derivative/proposal or advancing time is not representable.
  /// @throws std::bad_alloc If owned storage allocation fails.
  /// @par Example
  /// @code
  /// struct F:dm::ode::RHS{std::vector<double>evaluate(double,const std::vector<double>&y)override{return {-y[0]};}};F f;dm::ode::ODESolver s;std::print("{}\n",s.solve_with_limits(f,{1.},0.,.01,{}).t.back()); // 0.01
  /// @endcode
  /// @par References
  /// - NIST DLMF3.7(v), Runge-Kutta Method, https://dlmf.nist.gov/3.7.v.
  /// - J. R. Dormand and P. J. Prince(1980), A family of embedded Runge-Kutta formulae, doi:10.1016/0771-050X(80)90013-3. Native controller does not reuse FSAL and rejects nonconforming tiny trials.
  /// @see ODEAttempt, ODEEvaluation
  [[nodiscard]] ODESolution solve_with_limits(
      RHS&                       rhs,
      const std::vector<double>& y0,
      double                     t_start,
      double                     t_end,
      ODEExecutionLimits         limits = {}) const;
  /// @brief Capture actual canonical forward ODE callback and step histories.
  /// Complete finite nonempty y0 and finite forward span t_end>t_start with
  /// representable difference are checked before callbacks. Known method, finite
  /// positive step/abs_tol, finite rel_tol>=0 and max_step>=0 are validated, including
  /// inactive numeric controls. Each callback receives a finite fixed-dimensional
  /// stage and must return finite matching dimension; exceptions propagate.
  /// Steps must advance representable time; tiny/nonadvancing steps throw rather
  /// than loop. No absolute1e-14 endpoint skip; successful output reaches t_end.
  /// Adams uses original RK4 startup and clamped order2..4; a shortened final step
  /// uses RK4 instead of applying constant-grid coefficients to unequal spacing.
  /// RK45 does not reuse FSAL and accepts only scaled error<=1, not a tiny-step bypass.
  /// Local error control is not a certified global error, stability or energy bound.
  /// State/control/input references are not retained. Callback effects before a
  /// later error/quota failure persist; no partial solution/trace is returned.
  /// Default work policies:1000000 attempts,7000000 RHS calls,1000000 retained
  /// solution scalar entries; limits overloads allow0 to disable each policy.
  /// Trace units root8+D(options6,times2,initialD), each accepted solution
  /// sample1+D, each callback2+2D, each actual attempt1+2D. Default1000000,
  ///0 disables; admission precedes each retained copy/append. This is not RAM.
  /// @param rhs Live derivative callback, finite matching-dimensional outputs required.
  /// @param y0 Original finite nonempty fixed-dimensional state.
  /// @param t_start Original finite initial time.
  /// @param t_end Original finite strictly later terminal time.
  /// @param limits Actual work/output/history policies, defaults documented above;0 disables each.
  /// @return Owned actual ODETrace, including both requested endpoint times on success.
  /// @throws std::invalid_argument If complete controls/state/builtin or callback dimension fails.
  /// @throws std::length_error If complete/actual count geometry or selected policies fail.
  /// @throws std::overflow_error If a stage/derivative/proposal or advancing time is not representable.
  /// @throws std::bad_alloc If owned storage allocation fails.
  /// @par Example
  /// @code
  /// struct F:dm::ode::RHS{std::vector<double>evaluate(double,const std::vector<double>&y)override{return {-y[0]};}};F f;dm::ode::ODESolver s;std::print("{}\n",s.trace(f,{1.},0.,.01).evaluations.size()); // 4
  /// @endcode
  /// @par References
  /// - NIST DLMF3.7(v), Runge-Kutta Method, https://dlmf.nist.gov/3.7.v.
  /// - J. R. Dormand and P. J. Prince(1980), A family of embedded Runge-Kutta formulae, doi:10.1016/0771-050X(80)90013-3. Native controller does not reuse FSAL and rejects nonconforming tiny trials.
  /// @see ODEAttempt, ODEEvaluation
  [[nodiscard]] ODETrace trace(
      RHS&                       rhs,
      const std::vector<double>& y0,
      double                     t_start,
      double                     t_end,
      ODEExecutionLimits         limits = {}) const;
  /// @brief Integrate one complete checked forward initial-value problem.
  /// Complete finite nonempty y0 and finite forward span t_end>t_start with
  /// representable difference are checked before callbacks. Known method, finite
  /// positive step/abs_tol, finite rel_tol>=0 and max_step>=0 are validated, including
  /// inactive numeric controls. Each callback receives a finite fixed-dimensional
  /// stage and must return finite matching dimension; exceptions propagate.
  /// Steps must advance representable time; tiny/nonadvancing steps throw rather
  /// than loop. No absolute1e-14 endpoint skip; successful output reaches t_end.
  /// Adams uses original RK4 startup and clamped order2..4; a shortened final step
  /// uses RK4 instead of applying constant-grid coefficients to unequal spacing.
  /// RK45 does not reuse FSAL and accepts only scaled error<=1, not a tiny-step bypass.
  /// Local error control is not a certified global error, stability or energy bound.
  /// State/control/input references are not retained. Callback effects before a
  /// later error/quota failure persist; no partial solution/trace is returned.
  /// Builtins:exponential_decay(k=1,dim1),logistic_growth(r=1,K=1,dim1),
  /// harmonic_oscillator(omega=1,dim2),van_der_pol(mu=1,dim2),
  /// lorenz(sigma=10,rho=28,beta=8/3,dim3).Missing parameters use defaults;
  /// finite extras ignored. All supplied parameters finite, logistic K>0, finite
  /// omega squared required. Exact state dimension is checked before integration.
  /// Default work policies:1000000 attempts,7000000 RHS calls,1000000 retained
  /// solution scalar entries; limits overloads allow0 to disable each policy.
  /// @param system Existing builtin identifier, exact spellings above.
  /// @param params Complete finite positional parameters, missing entries defaulted.
  /// @param y0 Original finite nonempty fixed-dimensional state.
  /// @param t_start Original finite initial time.
  /// @param t_end Original finite strictly later terminal time.
  /// @return Owned actual ODESolution, including both requested endpoint times on success.
  /// @throws std::invalid_argument If complete controls/state/builtin or callback dimension fails.
  /// @throws std::length_error If complete/actual count geometry or selected policies fail.
  /// @throws std::overflow_error If a stage/derivative/proposal or advancing time is not representable.
  /// @throws std::bad_alloc If owned storage allocation fails.
  /// @par Example
  /// @code
  /// dm::ode::ODESolver s;std::print("{}\n",s.solve_builtin("exponential_decay",{1.},{1.},0.,.01).function_evaluations); // 4
  /// @endcode
  /// @par References
  /// - NIST DLMF3.7(v), Runge-Kutta Method, https://dlmf.nist.gov/3.7.v.
  /// - J. R. Dormand and P. J. Prince(1980), A family of embedded Runge-Kutta formulae, doi:10.1016/0771-050X(80)90013-3. Native controller does not reuse FSAL and rejects nonconforming tiny trials.
  /// @see ODEAttempt, ODEEvaluation
  [[nodiscard]] ODESolution solve_builtin(
      const std::string&         system,
      const std::vector<double>& params,
      const std::vector<double>& y0,
      double                     t_start,
      double                     t_end) const;
  /// @brief Integrate one complete checked forward initial-value problem.
  /// Complete finite nonempty y0 and finite forward span t_end>t_start with
  /// representable difference are checked before callbacks. Known method, finite
  /// positive step/abs_tol, finite rel_tol>=0 and max_step>=0 are validated, including
  /// inactive numeric controls. Each callback receives a finite fixed-dimensional
  /// stage and must return finite matching dimension; exceptions propagate.
  /// Steps must advance representable time; tiny/nonadvancing steps throw rather
  /// than loop. No absolute1e-14 endpoint skip; successful output reaches t_end.
  /// Adams uses original RK4 startup and clamped order2..4; a shortened final step
  /// uses RK4 instead of applying constant-grid coefficients to unequal spacing.
  /// RK45 does not reuse FSAL and accepts only scaled error<=1, not a tiny-step bypass.
  /// Local error control is not a certified global error, stability or energy bound.
  /// State/control/input references are not retained. Callback effects before a
  /// later error/quota failure persist; no partial solution/trace is returned.
  /// Builtins:exponential_decay(k=1,dim1),logistic_growth(r=1,K=1,dim1),
  /// harmonic_oscillator(omega=1,dim2),van_der_pol(mu=1,dim2),
  /// lorenz(sigma=10,rho=28,beta=8/3,dim3).Missing parameters use defaults;
  /// finite extras ignored. All supplied parameters finite, logistic K>0, finite
  /// omega squared required. Exact state dimension is checked before integration.
  /// Default work policies:1000000 attempts,7000000 RHS calls,1000000 retained
  /// solution scalar entries; limits overloads allow0 to disable each policy.
  /// @param system Existing builtin identifier, exact spellings above.
  /// @param params Complete finite positional parameters, missing entries defaulted.
  /// @param y0 Original finite nonempty fixed-dimensional state.
  /// @param t_start Original finite initial time.
  /// @param t_end Original finite strictly later terminal time.
  /// @param limits Actual work/output/history policies, defaults documented above;0 disables each.
  /// @return Owned actual ODESolution, including both requested endpoint times on success.
  /// @throws std::invalid_argument If complete controls/state/builtin or callback dimension fails.
  /// @throws std::length_error If complete/actual count geometry or selected policies fail.
  /// @throws std::overflow_error If a stage/derivative/proposal or advancing time is not representable.
  /// @throws std::bad_alloc If owned storage allocation fails.
  /// @par Example
  /// @code
  /// dm::ode::ODESolver s;std::print("{}\n",s.solve_builtin_with_limits("exponential_decay",{1.},{1.},0.,.01,{}).size()); // 2
  /// @endcode
  /// @par References
  /// - NIST DLMF3.7(v), Runge-Kutta Method, https://dlmf.nist.gov/3.7.v.
  /// - J. R. Dormand and P. J. Prince(1980), A family of embedded Runge-Kutta formulae, doi:10.1016/0771-050X(80)90013-3. Native controller does not reuse FSAL and rejects nonconforming tiny trials.
  /// @see ODEAttempt, ODEEvaluation
  [[nodiscard]] ODESolution solve_builtin_with_limits(
      const std::string&         system,
      const std::vector<double>& params,
      const std::vector<double>& y0,
      double                     t_start,
      double                     t_end,
      ODEExecutionLimits         limits = {}) const;
  /// @brief Capture actual canonical forward ODE callback and step histories.
  /// Complete finite nonempty y0 and finite forward span t_end>t_start with
  /// representable difference are checked before callbacks. Known method, finite
  /// positive step/abs_tol, finite rel_tol>=0 and max_step>=0 are validated, including
  /// inactive numeric controls. Each callback receives a finite fixed-dimensional
  /// stage and must return finite matching dimension; exceptions propagate.
  /// Steps must advance representable time; tiny/nonadvancing steps throw rather
  /// than loop. No absolute1e-14 endpoint skip; successful output reaches t_end.
  /// Adams uses original RK4 startup and clamped order2..4; a shortened final step
  /// uses RK4 instead of applying constant-grid coefficients to unequal spacing.
  /// RK45 does not reuse FSAL and accepts only scaled error<=1, not a tiny-step bypass.
  /// Local error control is not a certified global error, stability or energy bound.
  /// State/control/input references are not retained. Callback effects before a
  /// later error/quota failure persist; no partial solution/trace is returned.
  /// Builtins:exponential_decay(k=1,dim1),logistic_growth(r=1,K=1,dim1),
  /// harmonic_oscillator(omega=1,dim2),van_der_pol(mu=1,dim2),
  /// lorenz(sigma=10,rho=28,beta=8/3,dim3).Missing parameters use defaults;
  /// finite extras ignored. All supplied parameters finite, logistic K>0, finite
  /// omega squared required. Exact state dimension is checked before integration.
  /// Default work policies:1000000 attempts,7000000 RHS calls,1000000 retained
  /// solution scalar entries; limits overloads allow0 to disable each policy.
  /// Trace units root8+D(options6,times2,initialD), each accepted solution
  /// sample1+D, each callback2+2D, each actual attempt1+2D. Default1000000,
  ///0 disables; admission precedes each retained copy/append. This is not RAM.
  /// @param system Existing builtin identifier, exact spellings above.
  /// @param params Complete finite positional parameters, missing entries defaulted.
  /// @param y0 Original finite nonempty fixed-dimensional state.
  /// @param t_start Original finite initial time.
  /// @param t_end Original finite strictly later terminal time.
  /// @param limits Actual work/output/history policies, defaults documented above;0 disables each.
  /// @return Owned actual ODETrace, including both requested endpoint times on success.
  /// @throws std::invalid_argument If complete controls/state/builtin or callback dimension fails.
  /// @throws std::length_error If complete/actual count geometry or selected policies fail.
  /// @throws std::overflow_error If a stage/derivative/proposal or advancing time is not representable.
  /// @throws std::bad_alloc If owned storage allocation fails.
  /// @par Example
  /// @code
  /// dm::ode::ODESolver s;std::print("{}\n",s.trace_builtin("exponential_decay",{1.},{1.},0.,.01).attempts.size()); // 1
  /// @endcode
  /// @par References
  /// - NIST DLMF3.7(v), Runge-Kutta Method, https://dlmf.nist.gov/3.7.v.
  /// - J. R. Dormand and P. J. Prince(1980), A family of embedded Runge-Kutta formulae, doi:10.1016/0771-050X(80)90013-3. Native controller does not reuse FSAL and rejects nonconforming tiny trials.
  /// @see ODEAttempt, ODEEvaluation
  [[nodiscard]] ODETrace trace_builtin(
      const std::string&         system,
      const std::vector<double>& params,
      const std::vector<double>& y0,
      double                     t_start,
      double                     t_end,
      ODEExecutionLimits         limits = {}) const;
  /// @brief Borrow the owned validated numerical options without mutation.
  /// @return Read-only reference valid until this solver is destroyed.
  /// @par Example
  /// @code
  /// dm::ode::ODESolver s;std::print("{}\n",s.options().multistep_order); // 4
  /// @endcode
  /// @see ODEOptions
  [[nodiscard]] const ODEOptions& options() const noexcept { return options_; }

 private:
  ODEOptions options_; ///< Owned complete validated original numerical controls.
};
} // namespace datamunge::ode
