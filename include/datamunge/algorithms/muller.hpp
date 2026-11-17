#pragma once
/// @file muller.hpp
/// @brief Checked real-valued Muller updates with actual callback and parabola records.
#include <array>
#include <cstddef>
#include <datamunge/algorithms/root_result.hpp>
#include <functional>
#include <string_view>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual numerical stop, distinct from a root/existence certificate.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::muller_stop_name(dm::algorithms::MullerStop::Residual)); // residual
/// @endcode
/// @see MullerRun
enum class MullerStop {
  Residual,        ///< Strict absolute callback residual is below tolerance.
  CoincidentNodes, ///< A required interpolation spacing is exactly zero.
  ZeroDenominator, ///< Both quadratic-root denominator choices reduce to zero.
  IterationLimit   ///< All requested updates completed and final residual test failed.
};
/// @brief Return a stable display name for a numerical stop.
/// @param stop Native stop enum; unknown cast values are allowed.
/// @return residual, coincident-nodes, zero-denominator, iteration-limit or unknown.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::muller_stop_name(dm::algorithms::MullerStop::IterationLimit)); // iteration-limit
/// @endcode
/// @see MullerStop
std::string_view muller_stop_name(MullerStop stop) noexcept;
/// @brief Actual selected callbacks, residual inspections, updates and discriminant clamps.
/// Initial three callbacks count even when x2 already passes or update cap is zero.
/// Inspections include the final cap check, which reuses the last callback value.
/// @par Example
/// @code
/// auto r=dm::algorithms::muller_run([](double x){return x-1;},0,.5,1);
/// std::print("{} {}\n",r.evaluations,r.updates); // 3 0
/// @endcode
/// @see muller_run()
struct MullerRun {
  RootResult  result; ///< Actual legacy summary; iterations counts loop inspections, not updates.
  MullerStop  stop{MullerStop::IterationLimit}; ///< Actual numerical termination reason.
  std::size_t evaluations{0}; ///< Actual function callbacks, exactly3+updates on success.
  std::size_t inspections{0}; ///< Actual strict residual checks, including final cap check.
  std::size_t updates{0};     ///< Actual finite next-point assignments and corresponding callbacks.
  std::size_t clamped_discriminants{0}; ///< Actual negative raw discriminants replaced by zero.
};
/// @brief One actual finite callback observation, in invocation order.
/// @par Example
/// @code
/// auto t=dm::algorithms::muller_trace([](double x){return x-1;},0,.5,1);
/// std::print("{}\n",t.evaluations.size()); // 3
/// @endcode
/// @see MullerTrace
struct MullerEvaluation {
  double x{0};           ///< Actual finite callback coordinate.
  double value{0};       ///< Actual finite returned function value.
  bool   initial{false}; ///< True for the first three seed callbacks.
};
/// @brief Actual residual inspection and, when needed, its quadratic proposal.
/// quadratic=false means complete coefficients/discriminant are unavailable;
/// spacing fields can retain an attempted coincident-node check, while unused
/// coefficient fields are zero placeholders and next=x2. updated=false means no
/// callback at next was performed. A final cap
/// check records at_limit=true without fabricating a new quadratic iteration.
/// @par Example
/// @code
/// auto t=dm::algorithms::muller_trace([](double x){return x*x-2;},0,1,2);
/// std::print("{} {}\n",t.steps.front().a,t.steps.front().b); // 1 4
/// @endcode
/// @see MullerTrace
struct MullerStep {
  int  legacy_iteration{0};   ///< One-based loop inspection, or max_iter at final cap check.
  bool at_limit{false};       ///< True for the explicit final cap inspection.
  std::array<double, 3> x{};  ///< Actual three current coordinates.
  std::array<double, 3> f{};  ///< Actual already evaluated callback values at those coordinates.
  bool   quadratic{false};    ///< Whether finite quadratic coefficients/discriminant were produced.
  double h0{0};               ///< Actual x1-x0 when interpolation is attempted.
  double h1{0};               ///< Actual x2-x1 when interpolation is attempted.
  double a{0};                ///< Actual quadratic coefficient for h=x-x2.
  double b{0};                ///< Actual linear coefficient for h=x-x2.
  double c{0};                ///< Actual constant coefficient, f(x2).
  double discriminant_raw{0}; ///< Actual b*b-4*a*c before the real-variant clamp.
  double discriminant_used{0}; ///< Actual max(raw,0).
  bool   clamped{false};       ///< Actual negative discriminant clamp.
  double denominator{0}; ///< Actual larger-magnitude b±sqrt(discriminant) choice; minus on ties.
  double next{0};        ///< Actual proposed finite coordinate, or current x2 when no update.
  bool   updated{false}; ///< Whether next was assigned and evaluated.
};
/// @brief Owned actual seed/controls, callback history and quadratic inspections.
/// Capture units are five seed/control scalars, each actual callback record and
/// each actual inspection record. Optional quota is online; earlier callback
/// side effects remain if later admission fails. No partial trace is returned.
/// @par Example
/// @code
/// auto t=dm::algorithms::muller_trace([](double x){return x-1;},0,.5,1);
/// std::print("{}\n",t.captured_entries); // 9
/// @endcode
/// @see muller_trace()
struct MullerTrace {
  std::array<double, 3> initial{};    ///< Owned original finite seed coordinates.
  double                tolerance{0}; ///< Original finite nonnegative strict residual threshold.
  int                   max_iterations{0};           ///< Original nonnegative update budget.
  MullerRun             run;                         ///< Actual result, stop and work.
  std::vector<MullerEvaluation> evaluations;         ///< Actual callback order.
  std::vector<MullerStep>       steps;               ///< Actual residual checks and proposals.
  std::size_t                   captured_entries{0}; ///< Actual admitted logical history units.
};
/// @brief Solve with the retained real Muller quadratic update and residual stopping.
/// Fits the three-point parabola in local h=x-x2 coordinates and chooses the
/// larger-magnitude quadratic denominator to reduce cancellation. Negative
/// discriminants are clamped to zero: this is a real heuristic, not complex
/// Muller iteration. No bracket, step-size acceptance or global convergence
/// guarantee. Smooth simple-root local order~1.839 requires additional regularity,
/// close distinct iterates and the appropriate unmodified quadratic branch.
/// Nonfinite produced arithmetic/callbacks throw. Repeated interpolation nodes
/// stop numerically if a quadratic is needed, after the residual check.
/// Legacy arithmetic order and inspection count are retained on finite valid
/// runs. tol=0 disables acceptance even at callback zero. O(K) scalar work plus
/// callbacks, constant driver storage; zero cap still performs all three seed calls.
/// @param f Nonempty callable returning finite values; its exceptions propagate.
/// @param x0 First finite seed, evaluated first.
/// @param x1 Second finite seed, evaluated second.
/// @param x2 Third finite seed, first candidate tested for acceptance.
/// @param tol Finite nonnegative strict function-residual tolerance, default1e-10.
/// @param max_iter Nonnegative update cap below INT_MAX, default100.
/// @return Actual legacy root summary, not a mathematical certificate.
/// @throws std::invalid_argument If callable/seeds/controls are invalid, before callbacks.
/// @throws std::domain_error If a callback returns nonfinite data.
/// @throws std::overflow_error If produced spacing, coefficients or update are nonfinite.
/// @par Example
/// @code
/// auto r=dm::algorithms::muller([](double x){return x-1;},0,.5,1);
/// std::print("{} {}\n",r.converged,r.root); // true 1
/// @endcode
/// @par References
/// - D. E. Muller (1956), A method for solving algebraic equations using an automatic computer, Mathematical Tables and Other Aids to Computation10,208–215, doi:10.1090/S0025-5718-1956-0083822-0 (real discriminant-clamping variant here).
/// @see muller_run(), muller_trace()
RootResult muller(
    const std::function<double(double)>& f,
    double                               x0,
    double                               x1,
    double                               x2,
    double                               tol      = 1e-10,
    int                                  max_iter = 100);
/// @brief Return actual stop and work from the same ordinary Muller producer.
/// No replay or callback changes; same contracts/complexity as muller().
/// @param f Nonempty finite-valued callback.
/// @param x0 First finite seed.
/// @param x1 Second finite seed.
/// @param x2 Third finite seed.
/// @param tol Finite nonnegative residual tolerance, default1e-10.
/// @param max_iter Nonnegative update cap below INT_MAX, default100.
/// @return Actual result, stop, callback/update/inspection/clamp counts.
/// @throws std::invalid_argument If callable/input controls are invalid.
/// @throws std::domain_error If callback values are nonfinite.
/// @throws std::overflow_error If produced arithmetic is nonfinite.
/// @par Example
/// @code
/// auto r=dm::algorithms::muller_run([](double x){return 1.;},0,1,2);
/// std::print("{}\n",dm::algorithms::muller_stop_name(r.stop)); // zero-denominator
/// @endcode
/// @see MullerRun
MullerRun muller_run(
    const std::function<double(double)>& f,
    double                               x0,
    double                               x1,
    double                               x2,
    double                               tol      = 1e-10,
    int                                  max_iter = 100);
/// @brief Capture actual callbacks and inspections from the ordinary producer.
/// Admits five base scalars then each actual record before its callback/append;
/// no replay. Default1000000 entry cap, zero disables. Storage is proportional
/// to admitted records. Earlier callback effects persist if a later quota fails.
/// @param f Nonempty finite callback; exceptions propagate.
/// @param x0 First finite seed.
/// @param x1 Second finite seed.
/// @param x2 Third finite seed.
/// @param tol Finite nonnegative residual tolerance, default1e-10.
/// @param max_iter Nonnegative update budget below INT_MAX, default100.
/// @param max_entries Actual logical capture quota, default1000000; zero disables.
/// @return Owned actual seed/controls, callbacks, quadratic records and run.
/// @throws std::invalid_argument If callable/seeds/controls are invalid before callbacks.
/// @throws std::domain_error If a callback result is nonfinite.
/// @throws std::overflow_error If arithmetic is nonfinite.
/// @throws std::length_error If actual quota or record counts exceed limits.
/// @throws std::bad_alloc If history allocation fails.
/// @par Example
/// @code
/// auto t=dm::algorithms::muller_trace([](double x){return x-1;},0,.5,1);
/// std::print("{}\n",t.run.updates); // 0
/// @endcode
/// @see MullerTrace
MullerTrace muller_trace(
    const std::function<double(double)>& f,
    double                               x0,
    double                               x1,
    double                               x2,
    double                               tol         = 1e-10,
    int                                  max_iter    = 100,
    std::size_t                          max_entries = 1000000);
} // namespace datamunge::algorithms
