#pragma once
/// @file halley.hpp
/// @brief Checked shared Halley root iteration with actual callback and update records.
#include <cstddef>
#include <datamunge/algorithms/root_result.hpp>
#include <functional>
#include <vector>
namespace datamunge::algorithms {
/// @brief Distinguish residual acceptance, a zero update denominator and exhausted updates.
/// These are numerical outcomes, not existence or uniqueness certificates.
/// @par Example
/// @code
/// std::print("{}\n", int(dm::algorithms::HalleyStop::Residual)); // 0
/// @endcode
/// @see HalleyRun
enum class HalleyStop {
  Residual,        ///< Strict absolute callback residual is below the requested tolerance.
  ZeroDenominator, ///< Actual 2*d*d-f*d2 is exactly zero; no update applied.
  IterationLimit   ///< All allowed updates used and final residual test failed.
};
/// @brief Actual selected callback calls and applied updates, not FLOPs or elapsed time.
/// Callback bodies can have arbitrary costs; no timing comparison follows from these counts.
/// @par Example
/// @code
/// const dm::algorithms::HalleyWork w;
/// std::print("{}\n", w.updates); // 0
/// @endcode
/// @see HalleyRun
struct HalleyWork {
  std::size_t f_calls{0};   ///< Actual f evaluations, including final cap evaluation.
  std::size_t df_calls{0};  ///< Actual first derivative evaluations.
  std::size_t d2f_calls{0}; ///< Actual second derivative evaluations.
  std::size_t updates{0};   ///< Actual completed x-minus-correction assignments.
};
/// @brief Owned final legacy summary, precise stopping reason and producing callback counts.
/// No callback or input references survive. Legacy iterations count loop inspections,
/// including the successful residual inspection, rather than updates; with cap zero
/// the final callback is evaluated but iterations remains zero.
/// @par Example
/// @code
/// const dm::algorithms::HalleyRun r;
/// std::print("{}\n", r.work.f_calls); // 0
/// @endcode
/// @see halley_run(), HalleyTrace
struct HalleyRun {
  RootResult result; ///< Actual final estimate, strict residual flag and legacy count.
  HalleyStop stop{HalleyStop::IterationLimit}; ///< Actual terminating branch.
  HalleyWork work;                             ///< Selected actual callback/update counts.
};
/// @brief Owned actual callback inspection and optional Halley update from one execution.
/// Derivative and denominator fields are meaningful only when derivatives_evaluated;
/// next is x unless updated. The final budget inspection evaluates only f, even if
/// its residual is not accepted. No extra callback evaluations reconstruct records.
/// @par Example
/// @code
/// const dm::algorithms::HalleyStep s;
/// std::print("{}\n", s.updated); // false
/// @endcode
/// @see HalleyTrace
struct HalleyStep {
  int    inspection{0}; ///< One-based f callback inspection, independent of legacy iteration count.
  double x{0};          ///< Actual input to this f callback.
  double fx{0};         ///< Actual signed callback value.
  bool   derivatives_evaluated{false}; ///< Whether df and d2f were called at x.
  double derivative{0};                ///< Actual df value, meaningful only when evaluated.
  double second_derivative{0};         ///< Actual d2f value, meaningful only when evaluated.
  double denominator{0}; ///< Actual sequential 2*d*d-f*d2, when derivatives evaluated.
  double next{0};        ///< Actual assigned next iterate, or x when no update.
  bool   updated{false}; ///< Whether a finite correction was applied.
};
/// @brief Owned chronological actual inspections and producing run; no replay or truncation.
/// One record per f call, including final budget check, is admitted before that
/// callback. Default cap one million records, zero disables it. Record counts are
/// not bytes, capacities or peak RAM. Copying is deep; no retained callable state.
/// @par Example
/// @code
/// const dm::algorithms::HalleyTrace t;
/// std::print("{}\n", t.steps.size()); // 0
/// @endcode
/// @see halley_trace()
struct HalleyTrace {
  HalleyRun               run;   ///< Actual producing result, stop and work.
  std::vector<HalleyStep> steps; ///< Owned actual callback and update records.
};
/// @brief Solve f(x)=0 using the retained Halley update and strict residual stopping.
/// Uses x-2*f*df/(2*df*df-f*d2f) in the original arithmetic order. Cubic local
/// convergence assumes a sufficiently smooth simple root, consistent derivatives
/// and a sufficiently close start; multiple roots and distant starts need not be
/// cubic or converge. No bracket, step-size test or stagnation acceptance. tol=0
/// disables acceptance even at callback zero. A zero denominator returns failure.
/// All controls/callables are checked before callbacks; nonfinite callback/update
/// arithmetic throws. Callback exceptions propagate; their side effects persist.
/// Calls are independent except for caller-owned callback state. O(K) scalar work
/// plus callback costs; constant storage. max_iter=0 still evaluates f(x0).
/// @param f Nonempty scalar function returning finite values at visited points.
/// @param df Nonempty first derivative returning finite values when called.
/// @param d2f Nonempty second derivative returning finite values when called.
/// @param x0 Finite initial iterate.
/// @param tol Finite nonnegative strict absolute residual tolerance in function units,
/// default1e-10.
/// @param max_iter Nonnegative update budget, default100, below INT_MAX.
/// @return Actual legacy root summary; iterations counts loop inspections, not updates.
/// @throws std::invalid_argument On invalid controls/empty callbacks before any callback.
/// @throws std::domain_error On a nonfinite callback result.
/// @throws std::overflow_error On a nonfinite denominator/correction/next iterate.
/// @par Example
/// @code
/// const auto r = dm::algorithms::halley(
///     [](double x) { return x * x - 2; },
///     [](double x) { return 2 * x; },
///     [](double) { return 2.; },
///     2.);
/// std::print("{} {}\n", r.converged, r.iterations); // true 4
/// @endcode
/// @par References
/// - E. Halley (1694), A new, exact, and easy method of finding the roots of equations generally,
///   Philosophical Transactions 18, 136–148, https://doi.org/10.1098/rstl.1694.0029 .
/// @see halley_run(), halley_trace()
RootResult halley(
    const std::function<double(double)>& f,
    const std::function<double(double)>& df,
    const std::function<double(double)>& d2f,
    double                               x0,
    double                               tol      = 1e-10,
    int                                  max_iter = 100);
/// @brief Execute the same Halley kernel with explicit stop and actual selected work.
/// Same domains, arithmetic, callback order, errors and stopping rules as halley().
/// No extra callback evaluations or trace storage. No input references retained.
/// @param f Finite-valued nonempty function.
/// @param df Finite-valued nonempty first derivative.
/// @param d2f Finite-valued nonempty second derivative.
/// @param x0 Finite initial iterate.
/// @param tol Finite nonnegative strict residual tolerance, default1e-10; zero disables acceptance.
/// @param max_iter Nonnegative update budget below INT_MAX, default100.
/// @return Owned producing summary, stop and actual counts.
/// @throws std::invalid_argument On invalid controls/empty callbacks.
/// @throws std::domain_error On nonfinite callback values.
/// @throws std::overflow_error On nonfinite update arithmetic.
/// @par Example
/// @code
/// const auto r = dm::algorithms::halley_run(
///     [](double x) { return x - 1; },
///     [](double) { return 1.; },
///     [](double) { return 0.; },
///     2.);
/// std::print("{} {}\n", r.work.updates, r.work.f_calls); // 1 2
/// @endcode
/// @see halley(), HalleyRun
HalleyRun halley_run(
    const std::function<double(double)>& f,
    const std::function<double(double)>& df,
    const std::function<double(double)>& d2f,
    double                               x0,
    double                               tol      = 1e-10,
    int                                  max_iter = 100);
/// @brief Capture actual Halley inspections from the producing execution without replay.
/// Same contracts as halley_run(). Online whole-record admission precedes each
/// callback; no partial trace returns on cap/error, but earlier callback side
/// effects persist. O(K) owned records in addition to scalar working storage.
/// @param f Finite-valued nonempty function.
/// @param df Finite-valued nonempty first derivative.
/// @param d2f Finite-valued nonempty second derivative.
/// @param x0 Finite initial iterate.
/// @param tol Finite nonnegative strict residual tolerance, default1e-10.
/// @param max_iter Nonnegative update budget below INT_MAX, default100.
/// @param max_entries Online record cap, default1000000; zero disables policy cap.
/// @return Owned complete producing run and chronological actual inspections.
/// @throws std::invalid_argument On invalid controls/empty callbacks, before callbacks/caps.
/// @throws std::domain_error On nonfinite callback values.
/// @throws std::overflow_error On nonfinite update arithmetic.
/// @throws std::length_error On online capture/container limit before next callback.
/// @throws std::bad_alloc On record allocation failure.
/// @par Example
/// @code
/// const auto t = dm::algorithms::halley_trace(
///     [](double x) { return x - 1; },
///     [](double) { return 1.; },
///     [](double) { return 0.; },
///     2.);
/// std::print("{}\n", t.steps.size()); // 2
/// @endcode
/// @see HalleyTrace, halley_run()
HalleyTrace halley_trace(
    const std::function<double(double)>& f,
    const std::function<double(double)>& df,
    const std::function<double(double)>& d2f,
    double                               x0,
    double                               tol         = 1e-10,
    int                                  max_iter    = 100,
    std::size_t                          max_entries = 1000000);
} // namespace datamunge::algorithms
