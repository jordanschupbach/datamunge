#pragma once
/// @file nth_root.hpp
/// @brief Checked nonnegative integer-degree roots with actual Newton states and explicit large-degree initialization.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual update/state-change/capture policies, with zero tolerance disabling early acceptance.
/// @par Example
/// @code
/// dm::algorithms::NthRootOptions o;std::print("{}\n",o.max_iterations); // 100
/// @endcode
/// @see nth_root_trace()
struct NthRootOptions {
  int    max_iterations{100};       ///< Nonnegative actual update cap below INT_MAX.
  double relative_tolerance{1e-15}; ///< Finite nonnegative relative consecutive-state threshold.
  std::size_t max_trace_entries{1000000}; ///< Actual4+updates logical capture cap, zero disables.
};
/// @brief Actual finite root approximation and selected seed/update work.
/// @par Example
/// @code
/// auto r=dm::algorithms::nth_root_run(27,3);std::print("{:.6f}\n",r.value); // 3.000000
/// @endcode
/// @see nth_root_trace()
struct NthRootRun {
  double      value{0};    ///< Actual final finite state, exact a for degree1.
  double      initial{0};  ///< Actual initial state, or shortcut output.
  double      mantissa{0}; ///< Actual frexp(a) mantissa, zero placeholder for shortcuts.
  int         exponent{0}; ///< Actual frexp(a) exponent, zero placeholder for shortcuts.
  std::size_t updates{0};  ///< Actual Newton assignments performed.
  std::size_t power_multiplications{
      0}; ///< Actual attempted/fallback power-loop multiplies for degree<=64, excludes transcendental work.
  std::size_t widened_updates{
      0}; ///< Actual small-degree updates requiring widened arithmetic fallback.
  std::size_t log_ratio_updates{0}; ///< Actual degree>64 updates using a logarithmic ratio.
  bool        log_seed{
      false}; ///< Actual degree>64 log/exp initial approximation, not a multiplication-only seed.
  bool converged{
      false}; ///< Consecutive-state test passed, or a mathematically exact identity/zero shortcut.
  bool identity_shortcut{false}; ///< Degree1 returned the original finite input exactly.
  bool zero_shortcut{false};     ///< Degree>1 and a==0 returned0 without iterations.
};
/// @brief Actual Newton power/ratio and assigned state, with arithmetic route explicit.
/// @par Example
/// @code
/// auto t=dm::algorithms::nth_root_trace(50,3);std::print("{}\n",t.steps.front().before); // 4
/// @endcode
/// @see nth_root_trace()
struct NthRootStep {
  double before{0}; ///< Actual positive finite state before the update.
  double power{
      0}; ///< Actual small-degree finite double power if power_available, otherwise0 placeholder.
  bool power_available{false}; ///< Whether the small-degree attempted power was positive finite.
  long double ratio{0};        ///< Actual a/x^(n-1) value used, widened or log-derived as flagged.
  double      after{0};        ///< Actual assigned finite next state.
  double      relative_change{0}; ///< Actual abs(after-before)/abs(after), not a root-error bound.
  std::size_t power_multiplications{
      0};                ///< Actual attempted/fallback power multiplies during this update.
  bool widened{false};   ///< Actual small-degree widened fallback route.
  bool log_ratio{false}; ///< Actual large-degree log/exp ratio route.
};
/// @brief Owned original controls and every actual same-producer root update.
/// @par Example
/// @code
/// auto t=dm::algorithms::nth_root_trace(0,3);std::print("{}\n",t.captured_entries); // 4
/// @endcode
/// @see nth_root_trace()
struct NthRootTrace {
  double                   radicand{0}; ///< Original complete finite nonnegative a.
  int                      degree{0};   ///< Original positive integer root degree.
  NthRootOptions           options;     ///< Original actual budget/tolerance/capture policies.
  NthRootRun               run;         ///< Actual same-producer output, routes and selected work.
  std::vector<NthRootStep> steps;       ///< Actual chronological assignments, no replay.
  std::size_t              captured_entries{
      0}; ///< Logical input controls4 plus actual update records; capture policy metadata excluded.
};
/// @brief Approximate the nonnegative nth root with checked seed and actual Newton updates.
/// Requires finite a>=0 and integer n>=1; negative radicands are not accepted
/// even for odd n. Degree1 returns a exactly; degree>1 zero returns positive0.
/// Degree<=64 starts at2^ceil(frexp_exponent/n), preserving the original50/3
/// seed4 while avoiding below-root binary starts. Original finite-double power
/// and ((n-1)*x+a/power)/n order is retained when safe; subnormal/nonpositive/nonfinite
/// powers or nonfinite arithmetic fall back to long-double power/update work.
/// Degree>64 uses an explicit log/exp root approximation as its seed and a
/// log-derived ratio for each refinement, avoiding an O(n) power loop. This
/// branch is not presented as a multiplication-only initialization/iteration.
/// Finite states and a small relative consecutive-state change do NOT certify
/// correct rounding or a root-error/residual bound. Default100 actual updates,
/// threshold1e-15; exhausted budgets return finite state with converged=false.
/// Small-degree work O(I*n), n<=64, large-degree O(I) log/exp evaluations;
/// constant driver state plus optional actual owned records. No input mutation.
/// @param a Complete finite nonnegative radicand.
/// @param n Positive degree, through INT_MAX; degree1 is identity.
/// @return Actual finite double approximation/result with stated shortcut conventions.
/// @throws std::invalid_argument If complete radicand/degree/options fail before arithmetic.
/// @throws std::overflow_error If finite positive seed/update/ratio cannot be represented.
/// @throws std::length_error If actual history quota/count geometry is exceeded (trace only).
/// @par Example
/// @code
/// auto r=dm::algorithms::nth_root(1024,10);std::print("{:.6f}\n",r); // 2.000000
/// @endcode
/// @see NthRootRun, NthRootTrace
double nth_root(double a, int n);
/// @brief Approximate the nonnegative nth root with checked seed and actual Newton updates.
/// Requires finite a>=0 and integer n>=1; negative radicands are not accepted
/// even for odd n. Degree1 returns a exactly; degree>1 zero returns positive0.
/// Degree<=64 starts at2^ceil(frexp_exponent/n), preserving the original50/3
/// seed4 while avoiding below-root binary starts. Original finite-double power
/// and ((n-1)*x+a/power)/n order is retained when safe; subnormal/nonpositive/nonfinite
/// powers or nonfinite arithmetic fall back to long-double power/update work.
/// Degree>64 uses an explicit log/exp root approximation as its seed and a
/// log-derived ratio for each refinement, avoiding an O(n) power loop. This
/// branch is not presented as a multiplication-only initialization/iteration.
/// Finite states and a small relative consecutive-state change do NOT certify
/// correct rounding or a root-error/residual bound. Default100 actual updates,
/// threshold1e-15; exhausted budgets return finite state with converged=false.
/// Small-degree work O(I*n), n<=64, large-degree O(I) log/exp evaluations;
/// constant driver state plus optional actual owned records. No input mutation.
/// @param a Complete finite nonnegative radicand.
/// @param n Positive degree, through INT_MAX; degree1 is identity.
/// @param options Actual update/tolerance/history policies; history cap ignored without tracing.
/// @return Actual finite NthRootRun approximation/result with stated shortcut conventions.
/// @throws std::invalid_argument If complete radicand/degree/options fail before arithmetic.
/// @throws std::overflow_error If finite positive seed/update/ratio cannot be represented.
/// @throws std::length_error If actual history quota/count geometry is exceeded (trace only).
/// @par Example
/// @code
/// auto r=dm::algorithms::nth_root_run(1024,10);std::print("{:.6f}\n",r.value); // 2.000000
/// @endcode
/// @see NthRootRun, NthRootTrace
NthRootRun nth_root_run(double a, int n, NthRootOptions options = {});
/// @brief Capture actual checked nonnegative integer-degree Newton root states.
/// Requires finite a>=0 and integer n>=1; negative radicands are not accepted
/// even for odd n. Degree1 returns a exactly; degree>1 zero returns positive0.
/// Degree<=64 starts at2^ceil(frexp_exponent/n), preserving the original50/3
/// seed4 while avoiding below-root binary starts. Original finite-double power
/// and ((n-1)*x+a/power)/n order is retained when safe; subnormal/nonpositive/nonfinite
/// powers or nonfinite arithmetic fall back to long-double power/update work.
/// Degree>64 uses an explicit log/exp root approximation as its seed and a
/// log-derived ratio for each refinement, avoiding an O(n) power loop. This
/// branch is not presented as a multiplication-only initialization/iteration.
/// Finite states and a small relative consecutive-state change do NOT certify
/// correct rounding or a root-error/residual bound. Default100 actual updates,
/// threshold1e-15; exhausted budgets return finite state with converged=false.
/// Small-degree work O(I*n), n<=64, large-degree O(I) log/exp evaluations;
/// constant driver state plus optional actual owned records. No input mutation.
/// @param a Complete finite nonnegative radicand.
/// @param n Positive degree, through INT_MAX; degree1 is identity.
/// @param options Actual update/tolerance/history policies; history cap ignored without tracing.
/// @return Owned original inputs, actual state history and run metadata.
/// @throws std::invalid_argument If complete radicand/degree/options fail before arithmetic.
/// @throws std::overflow_error If finite positive seed/update/ratio cannot be represented.
/// @throws std::length_error If actual history quota/count geometry is exceeded (trace only).
/// @par Example
/// @code
/// auto r=dm::algorithms::nth_root_trace(1024,10);std::print("{:.6f}\n",r.run.value); // 2.000000
/// @endcode
/// @see NthRootRun, NthRootTrace
NthRootTrace nth_root_trace(double a, int n, NthRootOptions options = {});
/// @brief Original50/cube-root fixture with actual producer states and native cbrt comparison.
/// @par Example
/// @code
/// auto s=dm::algorithms::nth_root_demo_study();std::print("{}\n",s.trace.run.initial); // 4
/// @endcode
/// @see nth_root_trace()
struct NthRootStudy {
  NthRootTrace trace;        ///< Actual run for a50,n3, default policies unless supplied.
  double       reference{0}; ///< Native cbrt(50) comparison value, not an independent exact oracle.
  std::vector<double> indices;          ///< Actual state index0 through number of assigned updates.
  std::vector<double> values;           ///< Actual initial and assigned states.
  std::vector<double> reference_errors; ///< Actual absolute differences from native cbrt(50).
};
/// @brief Preserve the original a50,n3 state/error experiment through the actual canonical producer.
/// With tolerance0 and budget8, performs exactly the old eight educational
/// assignments (including stationary ones); this differs from default stopping.
/// @param options Actual update/tolerance/history policies, default ordinary values.
/// @return Owned original experiment states and cbrt comparison errors.
/// @throws std::invalid_argument If complete options fail.
/// @throws std::length_error If trace policy is exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::nth_root_demo_study({8,0,1000}).trace.run.updates); // 8
/// @endcode
/// @see nth_root_trace()
NthRootStudy nth_root_demo_study(NthRootOptions options = {});
} // namespace datamunge::algorithms
