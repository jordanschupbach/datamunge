#pragma once
/// @file root_finding.hpp
/// @brief Scalar root methods and checked same-pass plain/Illinois bracketing evidence.
#include <datamunge/algorithms/bisection.hpp>

#include <functional>
#include <cstddef>
#include <limits>
#include <string_view>
#include <vector>

namespace datamunge::algorithms {

/// @brief *Newton's method* (Newton-Raphson): from a guess @p x0, repeatedly step to the zero of the
///        tangent line, @c x <- x - f(x)/f'(x). It converges *quadratically* near a simple root (the
///        number of correct digits roughly doubles each step) but needs the derivative @p df and can
///        diverge from a poor start or where @c f'(x)=0.
///
/// @param f the function.
/// @param df its derivative.
/// @param x0 the initial guess.
/// @param tol convergence tolerance on the residual.
/// @param max_iter iteration cap.
/// @return the @ref RootResult; @c converged is false if a zero derivative is hit or the cap is
///         reached without meeting @p tol.
RootResult newton_raphson(const std::function<double(double)>& f, const std::function<double(double)>& df,
                          double x0, double tol = 1e-10, int max_iter = 100);

/// @brief The *secant method*: like Newton's, but approximates the derivative with a finite
///        difference over the two most recent points, needing no explicit @c f'. It converges
///        *superlinearly* (order ≈ 1.618, the golden ratio) using one function evaluation per step.
///
/// @param f the function.
/// @param x0,x1 two initial points (need not bracket the root).
/// @param tol convergence tolerance on the residual.
/// @param max_iter iteration cap.
/// @return the @ref RootResult.
RootResult secant(const std::function<double(double)>& f, double x0, double x1, double tol = 1e-10,
                  int max_iter = 100);

/// @brief *Ridders' method*: a bracketing root-finder that, per step, evaluates @c f at the interval
///        midpoint and then applies an exponential correction that fits @c f·e^{Q} to a straight
///        line, landing far closer to the root than bisection. It keeps the guaranteed convergence
///        of a bracketing method while achieving *quadratic* convergence, using two function
///        evaluations per iteration. Requires a sign-changing bracket @c [a,b].
///
/// @param f the continuous function.
/// @param a,b endpoints bracketing the root.
/// @param tol convergence tolerance on the residual / bracket width.
/// @param max_iter iteration cap.
/// @return the @ref RootResult; @c converged is false if @c [a,b] does not bracket a sign change.
RootResult ridders(const std::function<double(double)>& f, double a, double b, double tol = 1e-10,
                   int max_iter = 100);

/// @brief Select plain secant bracketing or the legacy retained-twice Illinois variant.
/// Illinois starts with no previous replacement; damping begins only when the
/// same endpoint is replaced twice consecutively. This differs from the paper's
/// most-recent-point initialization, while retaining its stale-weight mechanism.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::false_position_policy_name(dm::algorithms::FalsePositionPolicy::Illinois)); // illinois
/// @endcode
/// @see FalsePositionOptions, false_position_policy_name()
enum class FalsePositionPolicy {
  Plain,   ///< Ordinary regula falsi: interpolation weights remain actual endpoint values.
  Illinois ///< Halve the opposite retained weight after consecutive replacements of the same
           ///< endpoint.
};

/// @brief Distinguish numerical stopping conditions from a mathematical root certificate.
/// Continuity, existence and trustworthy callback signs are caller assumptions;
/// a callback zero or tolerance test alone does not establish those assumptions.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::false_position_stop_name(dm::algorithms::FalsePositionStop::Stagnation)); // stagnation
/// @endcode
/// @see FalsePositionOutcome, false_position_stop_name()
enum class FalsePositionStop {
  NoBracket,  ///< Nonzero actual endpoint values have the same sign; no trial evaluated.
  Endpoint,   ///< An initial callback returned zero; both endpoints were evaluated first.
  ExactValue, ///< A trial callback returned zero, not necessarily exact mathematical zero.
  Residual,   ///< Positive residual tolerance passed at a trial.
  Interval,   ///< Positive coordinate tolerance passed using the rounded trial-to-endpoint radius.
  Stagnation, ///< Trial rounded to an endpoint without passing a stopping test.
  WeightUnderflow, ///< Actual Illinois halving rounded a nonzero artificial weight to zero.
  IterationLimit ///< The trial-evaluation budget was exhausted before another stopping test passed.
};

/// @brief Independent policy, function-unit and coordinate-unit controls for secant bracketing.
/// Zero residual or coordinate tolerance disables that test; callback zeros always
/// stop. Endpoint zeros are checked before sign rejection; endpoint residual-only
/// acceptance is not performed. All controls are validated before callbacks/capture.
/// @par Example
/// @code
/// dm::algorithms::FalsePositionOptions o;o.policy=dm::algorithms::FalsePositionPolicy::Plain;
/// std::print("{} {}\n",dm::algorithms::false_position_policy_name(o.policy),o.max_iterations); // plain 200
/// @endcode
/// @see false_position_work(), false_position_trace()
struct FalsePositionOptions {
  FalsePositionPolicy policy = FalsePositionPolicy::Illinois; ///< Plain or retained-twice Illinois;
                                                              ///< other enum values are invalid.
  double residual_tolerance =
      1e-10; ///< Finite nonnegative tolerance in callback-value units; zero disables.
  double x_tolerance =
      0; ///< Finite nonnegative rounded-radius tolerance in coordinate units; zero disables.
  int max_iterations = 200; ///< Positive maximum number of trial callback evaluations.
};

/// @brief Owned result, terminal reason, actual work and true-versus-weighted bracket state.
/// For a trial stop, a/b and their values/weights describe the bracket used to
/// compute the returned estimate, before its terminal update. Endpoint zero
/// collapses that state to the zero endpoint; same-sign rejection keeps original
/// inputs, default RootResult and infinite radius. Input order is retained, so a
/// need not be less than b. Radius is the rounded maximum estimate-to-endpoint
/// distance, possibly infinity for a full-width interval; not an outward-rounded
/// certified error bound. Small residual does not imply small coordinate error.
/// No callback or input reference is retained; independent calls share no state.
/// @par Example
/// @code
/// const auto o=dm::algorithms::false_position_work([](double x){return x-1;},0,2);
/// std::print("{} {} {}\n",o.result.root,o.evaluations,dm::algorithms::false_position_stop_name(o.stop)); // 1 3 zero callback
/// @endcode
/// @see RootResult, FalsePositionTrace
struct FalsePositionOutcome {
  RootResult result; ///< Legacy result layout: flag, last estimate, trial count and absolute actual
                     ///< callback value.
  FalsePositionStop stop = FalsePositionStop::NoBracket; ///< Actual terminal reason, distinct from
                                                         ///< proof of a mathematical root.
  std::size_t evaluations =
      0; ///< Actual callback calls, including both initial endpoint evaluations.
  std::size_t guarded_interpolations = 0; ///< Actual trials using scaled positive-weight
                                          ///< interpolation instead of the safe legacy expression.
  std::size_t weight_halvings =
      0; ///< Actual artificial-weight halvings, including a terminal underflow attempt.
  double a =
      0; ///< First endpoint in the bracket used for the last trial, retaining input orientation.
  double b        = 0; ///< Second endpoint in that bracket, retaining input orientation.
  double f_a      = 0; ///< Actual callback value at a; never an artificially halved weight.
  double f_b      = 0; ///< Actual callback value at b; never an artificially halved weight.
  double weight_a = 0; ///< Artificial interpolation weight for a in that bracket.
  double weight_b = 0; ///< Artificial interpolation weight for b in that bracket.
  double radius =
      std::numeric_limits<double>::infinity(); ///< Rounded maximum last-estimate distance to a/b,
                                               ///< or infinity when no estimate exists.
};

/// @brief One owned actual callback evaluation from endpoint initialization or a trial.
/// Records the returned finite value without retaining or calling the function again.
/// @par Example
/// @code
/// const auto t=dm::algorithms::false_position_trace([](double x){return x-1;},0,2);
/// std::print("{} {} {}\n",t.evaluations.front().role,t.evaluations.front().coordinate,t.evaluations.front().value); // -1 0 -1
/// @endcode
/// @see FalsePositionTrace
struct FalsePositionEvaluation {
  int    role       = 0; ///< -1 for initial a, +1 for initial b, 0 for an actual trial.
  double coordinate = 0; ///< Actual coordinate passed to the callback.
  double value      = 0; ///< Finite callback value returned at that coordinate.
};

/// @brief Actual pre-update bracket, true values, artificial weights and one evaluated trial.
/// Pre-update terminal tests leave replaced=0; weight underflow can follow an
/// actual replacement/halving. Raw
/// signs certify the bracket under caller assumptions; weights only choose the
/// secant. Snapshots own all values and are never reconstructed by another solve.
/// @par Example
/// @code
/// const auto t=dm::algorithms::false_position_trace([](double x){return x-1;},0,2);
/// std::print("{} {} {}\n",t.steps.front().estimate,t.steps.front().f_estimate,t.steps.front().replaced); // 1 0 0
/// @endcode
/// @see FalsePositionTrace
struct FalsePositionStep {
  double a          = 0; ///< First actual endpoint before this trial's update.
  double b          = 0; ///< Second actual endpoint before this trial's update.
  double f_a        = 0; ///< True callback value at a, before any artificial weighting.
  double f_b        = 0; ///< True callback value at b.
  double weight_a   = 0; ///< Weight used for a in this trial's secant.
  double weight_b   = 0; ///< Weight used for b in this trial's secant.
  double estimate   = 0; ///< Actual evaluated interpolation point, possibly rounded to an endpoint.
  double f_estimate = 0; ///< Actual finite callback value at estimate.
  double radius     = 0; ///< Rounded maximum distance from estimate to either pre-update endpoint.
  int    replaced =
      0; ///< -1 for replacing a, +1 for replacing b, 0 for a terminal trial with no update.
  bool damped =
      false; ///< Whether this trial actually halved the opposite retained weight after evaluation.
  bool guarded = false; ///< Whether this trial actually used guarded scaled-weight interpolation.
};

/// @brief Owned outcome and callback/step records captured during the same actual solve.
/// Logical optional capture C=evaluations.size()+steps.size() counts WHOLE records:
/// both endpoints cost2 and each evaluated trial adds2. Metadata, ordinary result,
/// capacities and allocator/runtime bytes are excluded; this is not peak RAM.
/// Quota is admitted before both endpoint calls and before each trial's arithmetic/
/// callback; zero disables quota only. No partial trace escapes an exception.
/// @par Example
/// @code
/// const auto t=dm::algorithms::false_position_trace([](double x){return x-1;},0,2,{},4);
/// std::print("{} {} {}\n",t.evaluations.size(),t.steps.size(),t.captured_entries); // 3 1 4
/// @endcode
/// @see false_position_trace(), FalsePositionOutcome
struct FalsePositionTrace {
  FalsePositionOutcome outcome; ///< Owned actual result, terminal state and callback/work counts.
  std::vector<FalsePositionEvaluation>
      evaluations; ///< Actual endpoint-a, endpoint-b, then trial calls in order.
  std::vector<FalsePositionStep> steps; ///< One actual pre-update record per trial callback.
  std::size_t captured_entries = 0;     ///< Whole evaluation-plus-step records actually retained.
};

/// @brief Return a stable name for a supported false-position policy.
/// Program-lifetime literal, no allocation or shared mutation; unknown values map to unknown.
/// @param policy Policy enumerator, including manually cast values.
/// @return plain, illinois or unknown.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::false_position_policy_name(dm::algorithms::FalsePositionPolicy::Plain)); // plain
/// @endcode
/// @see FalsePositionPolicy
std::string_view false_position_policy_name(FalsePositionPolicy policy) noexcept;

/// @brief Return a stable name for a false-position stopping reason.
/// Program-lifetime literal, no allocation/shared mutation; unknown values map to unknown.
/// @param stop Reason enumerator, including manually cast values.
/// @return no bracket, endpoint, zero callback, residual, interval, stagnation, weight underflow,
/// iteration limit or unknown.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::false_position_stop_name(dm::algorithms::FalsePositionStop::WeightUnderflow)); // weight underflow
/// @endcode
/// @see FalsePositionStop
std::string_view false_position_stop_name(FalsePositionStop stop) noexcept;

/// @brief Solve through the historical compiled retained-twice Illinois interface.
/// Shares false_position_work() with policy Illinois, residual tolerance tol and
/// coordinate tolerance disabled. Keeps input orientation and f(a),f(b) evaluation
/// order, including evaluating both before endpoint-zero acceptance. Safe legacy
/// secant arithmetic is retained when nonzero products remain normal; scaled-weight
/// interpolation avoids finite overflow or spurious product-underflow trials otherwise. Exact callback zeros and trial residuals
/// strictly below tol pass. Same-sign endpoints return the default unconverged
/// record. Floating stagnation/weight underflow/budget exhaustion return unconverged.
/// Raw endpoint values stay separate from artificial weights. Continuity and reliable
/// signs are caller assumptions; a small residual alone is no coordinate certificate.
/// O(max_iter) calls and constant auxiliary state, no optional history/quota/shared
/// mutable state. Callback exceptions and side effects belong to the caller.
/// @param f Nonempty callable returning finite values at every evaluated coordinate.
/// @param a Finite first endpoint, in either order relative to b.
/// @param b Finite second endpoint; actual nonzero values must have opposite signs to iterate.
/// @param tol Positive finite trial-residual tolerance in callback units; default1e-10.
/// @param max_iter Positive trial-evaluation budget; default200.
/// @return Legacy result with actual last trial count and absolute callback residual.
/// @throws std::invalid_argument For empty f, nonfinite endpoints/tol, nonpositive tol or budget.
/// @throws std::domain_error If a callback returns nonfinite; callback exceptions also propagate.
/// @par Example
/// @code
/// const auto r=dm::algorithms::false_position([](double x){return std::exp(x)-2;},0,1);
/// std::print("{} {:.6f}\n",r.converged,r.root); // true 0.693147
/// @endcode
/// @par References
/// - Dowell and Jarratt (1971), A modified regula falsi method for computing the root
///   of an equation, BIT11,168–174. https://personal.math.ubc.ca/~loew/mech2/Dowell+Jarratt.pdf .
///   Native retained-twice initialization delays the first damping relative to their
///   most-recent-point rule; finite arithmetic, explicit stops and residual test are native
///   choices.
/// @see false_position_work(), false_position_trace(), RootResult
RootResult false_position(
    const std::function<double(double)>& f,
    double                               a,
    double                               b,
    double                               tol      = 1e-10,
    int                                  max_iter = 200);

/// @brief Execute checked plain or retained-twice Illinois bracketing with actual work counts.
/// Uses the same kernel as false_position(); no optional history/quota. Options
/// specify independent residual and rounded-radius tolerances; zero disables a
/// test. A callback zero stops first, then residual, radius, stagnation and budget.
/// Pre-update stopping tests skip the terminal update; weight underflow instead
/// reports the actual halving that failed. Input orientation and both initial calls
/// are retained. Endpoint-zero/no-bracket outcomes follow FalsePositionOutcome.
/// Finite interpolation guards avoid intermediate overflow; if double precision
/// cannot move the point or retain a nonzero weight, the unconverged stop is explicit.
/// O(budget) callbacks/constant storage. No callback/input alias or shared state.
/// @param f Nonempty finite-valued callback; continuity/reliable signs are caller assumptions.
/// @param a Finite first endpoint, in either order relative to b.
/// @param b Finite second endpoint.
/// @param options Supported policy, finite nonnegative tolerances and positive budget; defaults as
/// declared.
/// @return Owned actual result, stopping reason, callback work and terminal pre-update state.
/// @throws std::invalid_argument For empty callback, nonfinite endpoints/controls, negative
/// tolerance, invalid policy or nonpositive budget.
/// @throws std::domain_error If a callback returns nonfinite; callback exceptions also propagate.
/// @par Example
/// @code
/// dm::algorithms::FalsePositionOptions o;o.policy=dm::algorithms::FalsePositionPolicy::Plain;
/// const auto r=dm::algorithms::false_position_work([](double x){return x-1;},0,2,o);
/// std::print("{} {}\n",r.result.root,r.evaluations); // 1 3
/// @endcode
/// @par References
/// - Dowell and Jarratt (1971), BIT11,168–174; retained-twice variant as false_position().
///   https://personal.math.ubc.ca/~loew/mech2/Dowell+Jarratt.pdf .
/// @see FalsePositionOptions, FalsePositionOutcome, false_position_trace()
FalsePositionOutcome false_position_work(
    const std::function<double(double)>& f,
    double                               a,
    double                               b,
    const FalsePositionOptions&          options = {});

/// @brief Capture actual finite callback evaluations and steps during one checked bracketing solve.
/// Shares false_position_work()'s kernel, controls, actual calls and terminal rules;
/// no replay or callback retained. C=2+2K whole records for K actual trials, as in
/// FalsePositionTrace. Known initial2 records admitted before either callback;
/// each trial's2 records admitted before interpolation/callback. Default quota1m,
/// zero disables only quota. O(K) owned optional history, no partial result on
/// exceptions; callback side effects may already have occurred before a later failure.
/// @param f Nonempty finite-valued callback, with caller-owned continuity/sign assumptions.
/// @param a Finite first endpoint, retaining input orientation.
/// @param b Finite second endpoint.
/// @param options Supported policy, finite nonnegative tolerances and positive budget; defaults as
/// declared.
/// @param max_entries Whole evaluation-plus-step record quota, default1000000; zero disables quota
/// only.
/// @return Owned same-pass outcome, all actual callback/step snapshots and exact capture count.
/// @throws std::invalid_argument For malformed callback, endpoints or options before
/// capture/callbacks.
/// @throws std::domain_error For nonfinite returned callback values; callback exceptions propagate.
/// @throws std::length_error If quota, checked record-count addition or vector capacity is
/// exceeded.
/// @throws std::bad_alloc If optional history storage cannot be allocated.
/// @par Example
/// @code
/// const auto t=dm::algorithms::false_position_trace([](double x){return x-1;},0,2,{},4);
/// std::print("{} {}\n",t.outcome.evaluations,t.captured_entries); // 3 4
/// @endcode
/// @par References
/// - https://personal.math.ubc.ca/~loew/mech2/Dowell+Jarratt.pdf ; native retained-twice
/// initialization as false_position().
/// @see FalsePositionTrace, FalsePositionStep, false_position_work()
FalsePositionTrace false_position_trace(
    const std::function<double(double)>& f,
    double                               a,
    double                               b,
    const FalsePositionOptions&          options     = {},
    std::size_t                          max_entries = 1000000);

/// @brief *Halley's method*: a third-order (cubically convergent) root-finder that uses the first
///        *and second* derivatives. It fits a hyperbola (rather than Newton's tangent line) to the
///        function at each iterate, giving @c x_{n+1}=x_n-2 f f'/(2 f'^2 - f f''). Near a simple root
///        it roughly *triples* the number of correct digits each step -- faster than Newton -- at the
///        cost of needing @c f''.
///
/// @param f the function.
/// @param df its first derivative.
/// @param d2f its second derivative.
/// @param x0 the initial guess.
/// @param tol convergence tolerance on the residual.
/// @param max_iter iteration cap.
/// @return the @ref RootResult.
RootResult halley(const std::function<double(double)>& f, const std::function<double(double)>& df,
                  const std::function<double(double)>& d2f, double x0, double tol = 1e-10,
                  int max_iter = 100);

/// @brief *Muller's method*: generalizes the secant method by fitting a *parabola* through the three
///        most recent points and taking the root of that parabola nearest the last iterate (choosing
///        the quadratic-formula sign that maximizes the denominator for stability). It converges at
///        order ~1.84, needs no derivative, and -- because a parabola can cross zero where a line
///        cannot -- can find roots that trap the secant method (this real-valued implementation
///        clamps a negative discriminant to zero, so it targets real roots).
///
/// @param f the function.
/// @param x0,x1,x2 three initial points.
/// @param tol convergence tolerance on the residual.
/// @param max_iter iteration cap.
/// @return the @ref RootResult.
RootResult muller(const std::function<double(double)>& f, double x0, double x1, double x2,
                  double tol = 1e-10, int max_iter = 100);

/// @brief The *ITP method* (Interpolate-Truncate-Project; Oliveira & Takahashi, 2020): a modern
///        bracketing root-finder that is *minmax optimal* in worst-case iterations yet *superlinear*
///        on average -- getting the best of bisection and secant methods simultaneously. Each step
///        computes the regula-falsi point, *truncates* it toward the interval midpoint by a
///        shrinking margin, then *projects* it into a minmax interval around the midpoint, guaranteeing
///        it never does worse than bisection while usually doing much better.
///
/// @param f the continuous function.
/// @param a,b endpoints bracketing the root.
/// @param tol convergence tolerance on the half-interval width.
/// @param max_iter iteration cap.
/// @return the @ref RootResult; @c converged is false if @c [a,b] does not bracket a sign change.
RootResult itp(const std::function<double(double)>& f, double a, double b, double tol = 1e-10,
               int max_iter = 100);

} // namespace datamunge::algorithms
