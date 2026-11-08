#pragma once
/// @file ellipsoid.hpp
/// @brief Factored central ellipsoid cuts, actual convex-oracle trajectories and explicit stop reasons.
#include <datamunge/plot/diagram.hpp>
#include <cstddef>
#include <functional>
#include <vector>
namespace datamunge::algorithms {

/// @brief Best evaluated point, its objective and attempted oracle iterations.
/// Retains the original three-field layout. iterations counts subgradient calls,
/// including a terminal zero/tolerance/stagnation check; it is not the number of
/// successful cuts. A best point alone does not certify convergence. Copies own x.
/// @par Example
/// @code
/// const auto f=[](const std::vector<double>& x){return x[0]*x[0];};
/// const auto g=[](const std::vector<double>& x){return std::vector<double>{2*x[0]};};
/// const auto r=dm::algorithms::ellipsoid_minimize(f,g,{0},1);
/// std::print("{} {}\n",r.value,r.iterations); // 0 1
/// @endcode
/// @see EllipsoidOutcome, ellipsoid_minimize()
struct EllipsoidResult {
    std::vector<double> x; ///< Best finite evaluated point, owned.
    double value{0.0}; ///< Objective at x; default object is uncomputed.
    int iterations{0}; ///< Attempted subgradient iterations, at most max_iter.
};

/// @brief Owned ellipsoid E={center+lower*u : ||u||<=1} with a triangular shape factor.
/// The lower matrix is n by n, finite, exactly zero above the diagonal, with
/// positive diagonal. Thus P=lower*lower^T is positive definite mathematically;
/// forming P explicitly may lose this property through floating cancellation.
/// Public values are mutable; entry points validate shape before access.
/// @par Example
/// @code
/// const auto e=dm::algorithms::ellipsoid_ball({0,0},2);
/// std::print("{} {}\n",e.center.size(),e.lower[1][1]); // 2 2
/// @endcode
/// @see ellipsoid_ball(), ellipsoid_central_cut(), ellipsoid_outline()
struct EllipsoidState {
    std::vector<double> center; ///< Finite n-vector, with n>=1.
    std::vector<std::vector<double>> lower; ///< Owned n by n lower factor, including stored upper zeros.
};

/// @brief One actual factored central cut, without an objective or convergence claim.
/// support_bound=||lower^T*g|| is computed at the input state. Log volumes are
/// measured from the input/output positive diagonals. center_changed is false if
/// rounding left every coordinate unchanged; an optimizer treats that as stagnation.
/// @par Example
/// @code
/// const auto c=dm::algorithms::ellipsoid_central_cut(dm::algorithms::ellipsoid_ball({0,0},1),{1,0});
/// std::print("{:.6f} {}\n",c.state.center[0],c.center_changed); // -0.333333 true
/// @endcode
/// @see ellipsoid_central_cut()
struct EllipsoidCut {
    EllipsoidState state; ///< Candidate enclosing ellipsoid, computed by the cut formula.
    double support_bound=0; ///< Positive support norm at the input state, in g's units.
    double log_volume_before=0; ///< Measured natural log volume before the cut.
    double log_volume_after=0; ///< Measured natural log volume after the cut.
    bool center_changed=false; ///< Whether at least one represented center coordinate changed.
};

/// @brief Why the finite optimizer stopped; only two reasons are conditional optimality checks.
/// Valid convex subgradients and an initial enclosing ellipsoid are caller
/// assumptions, not verified by the implementation. Iteration limit and rounded
/// center stagnation do not certify an optimum or the requested tolerance.
/// @par Example
/// @code
/// const auto reason=dm::algorithms::EllipsoidStop::center_stagnation;
/// std::print("{}\n",reason==dm::algorithms::EllipsoidStop::iteration_limit); // false
/// @endcode
/// @see EllipsoidOutcome
enum class EllipsoidStop {
    iteration_limit, ///< Reached max_iter, including zero.
    zero_subgradient, ///< Oracle returned exactly zero; certifies a minimum for a valid convex oracle.
    support_tolerance, ///< Conditional current objective-gap bound <= requested absolute tolerance.
    center_stagnation ///< Positive proposed shift rounded to no coordinate change; no convergence claim.
};

/// @brief Checked logical counts from the actual callback/cut execution.
/// Counts exclude validation, geometry preparation, allocations, copying, logs,
/// candidate arithmetic and CPU time. support_products counts lower-factor terms
/// in L^T*g for nonzero oracle gradients, including terminal tolerance attempts.
/// factor_entries_updated counts committed lower-triangle entries, not all writes.
/// @par Example
/// @code
/// const auto f=[](const std::vector<double>& x){return x[0]*x[0];};
/// const auto g=[](const std::vector<double>&){return std::vector<double>{0};};
/// const auto r=dm::algorithms::ellipsoid_minimize_work(f,g,{0},1);
/// std::print("{} {} {}\n",r.work.objective_calls,r.work.subgradient_calls,r.work.updates); // 1 1 0
/// @endcode
/// @see EllipsoidOutcome
struct EllipsoidWork {
    std::size_t objective_calls=0; ///< One initial call plus one per committed update.
    std::size_t subgradient_calls=0; ///< Attempted iterations, equal to result.iterations.
    std::size_t updates=0; ///< Committed central cuts with newly evaluated centers.
    std::size_t support_products=0; ///< Terms used in actual L^T*g projections of nonzero gradients.
    std::size_t factor_entries_updated=0; ///< updates*n*(n+1)/2 committed lower entries.
};

/// @brief Detailed finite outcome with best point, last state and an explicit stop reason.
/// last_support_bound belongs to the most recent subgradient query, which can
/// precede the final state if the last allowed iteration performed an update.
/// has_support_bound is false when max_iter=0. Ordinary/work calls retain no
/// input or iteration history, have no optional capture quota, and own their output.
/// @par Example
/// @code
/// const auto f=[](const std::vector<double>& x){return x[0]*x[0];};
/// const auto g=[](const std::vector<double>& x){return std::vector<double>{2*x[0]};};
/// const auto r=dm::algorithms::ellipsoid_minimize_work(f,g,{0},1);
/// std::print("{} {}\n",r.result.value,r.stop==dm::algorithms::EllipsoidStop::zero_subgradient); // 0 true
/// @endcode
/// @see ellipsoid_minimize_work(), ellipsoid_minimize_trace()
struct EllipsoidOutcome {
    EllipsoidResult result; ///< Best evaluated point/objective and attempted iteration count.
    EllipsoidState state; ///< Last committed ellipsoid, not necessarily centered at best x.
    EllipsoidStop stop=EllipsoidStop::iteration_limit; ///< Actual terminal reason.
    bool has_support_bound=false; ///< Whether a subgradient query occurred.
    double last_support_bound=0; ///< Most recent support norm, zero for an exact zero gradient.
    EllipsoidWork work; ///< Counters from this run.
};

/// @brief Actual objective evaluation at an owned ellipsoid state.
/// Initial sample is followed by exactly one sample for each committed cut.
/// objective is the center value; best_value is the smallest evaluated value so
/// far. The ellipsoid algorithm need not decrease the current center objective.
/// @par Example
/// @code
/// const auto f=[](const std::vector<double>& x){return x[0]*x[0];};
/// const auto g=[](const std::vector<double>& x){return std::vector<double>{2*x[0]};};
/// const auto t=dm::algorithms::ellipsoid_minimize_trace(f,g,{0},1);
/// std::print("{} {}\n",t.samples[0].objective,t.samples[0].best_value); // 0 0
/// @endcode
/// @see EllipsoidTrace
struct EllipsoidSample {
    EllipsoidState state; ///< Owned state where objective was called.
    double objective=0; ///< Actual finite callback result.
    double best_value=0; ///< Best objective observed through this sample.
    double log_volume=0; ///< Measured natural log volume of this factored state.
};

/// @brief Actual subgradient query at one recorded sample, with its accepted-cut decision.
/// No callbacks are retained and no explanation is replayed. A terminal iteration
/// has updated=false. state_index identifies the pre-cut objective sample.
/// @par Example
/// @code
/// const auto f=[](const std::vector<double>& x){return x[0]*x[0];};
/// const auto g=[](const std::vector<double>& x){return std::vector<double>{2*x[0]};};
/// const auto t=dm::algorithms::ellipsoid_minimize_trace(f,g,{0},1);
/// std::print("{} {}\n",t.iterations[0].state_index,t.iterations[0].updated); // 0 false
/// @endcode
/// @see EllipsoidTrace
struct EllipsoidIteration {
    std::size_t state_index=0; ///< Sample queried before this attempted cut.
    std::vector<double> gradient; ///< Exact owned finite oracle vector.
    double support_bound=0; ///< Actual norm ||L^T*g||, or zero for g=0.
    bool updated=false; ///< Whether this attempt committed a cut and a new objective sample.
};

/// @brief Owned actual objective samples and subgradient attempts from the same optimizer run.
/// For dimension n, each sample costs S=1+n+n*n entries (record, center and all
/// stored factor entries) and each iteration I=1+n (record and gradient). Thus
/// C=samples*S+iterations*I. The returned outcome, callbacks, runtime buffers,
/// row/container metadata, capacities and allocator bytes are excluded; quota is
/// retained history, not peak RAM. No input/callback views survive.
/// @par Example
/// @code
/// const auto f=[](const std::vector<double>& x){return x[0]*x[0];};
/// const auto g=[](const std::vector<double>& x){return std::vector<double>{2*x[0]};};
/// const auto t=dm::algorithms::ellipsoid_minimize_trace(f,g,{0},1);
/// std::print("{} {} {}\n",t.samples.size(),t.iterations.size(),t.captured_entries); // 1 1 5
/// @endcode
/// @see ellipsoid_minimize_trace()
struct EllipsoidTrace {
    EllipsoidOutcome outcome; ///< Detailed ordinary outcome, outside the optional history count.
    std::vector<EllipsoidSample> samples; ///< Actual f evaluations, initially one.
    std::vector<EllipsoidIteration> iterations; ///< Actual subgradient calls and cut decisions.
    std::size_t captured_entries=0; ///< Exact retained logical entries as above.
};

/// @brief Construct an owned ball using a positive triangular factor, without squaring radius.
///
/// Validates nonempty finite center, positive finite radius and complete matrix
/// geometry before copying/allocation. The initial lower factor is radius*I; no
/// P matrix is formed. Work/storage O(n*n). Input is unchanged.
/// @param center Finite nonempty coordinates.
/// @param radius Positive finite ball radius.
/// @return Owned center and n by n triangular factor.
/// @throws std::invalid_argument On invalid center/radius.
/// @throws std::length_error On count arithmetic or container limits.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto e=dm::algorithms::ellipsoid_ball({0,0},2);
/// std::print("{} {}\n",e.lower[0][0],e.lower[0][1]); // 2 0
/// @endcode
/// @see EllipsoidState, ellipsoid_central_cut()
EllipsoidState ellipsoid_ball(const std::vector<double>& center,double radius);

/// @brief Measure log volume from the actual positive factor diagonal.
///
/// Returns log(unit-ball volume)+sum(log(L_ii)), avoiding determinant overflow
/// or underflow. Validates the complete finite triangular state first: O(n*n)
/// validation and O(n) log work, O(1) extra storage. Input is unchanged.
/// @param state Nonempty finite center and valid positive lower factor.
/// @return Finite natural log volume.
/// @throws std::invalid_argument On invalid state shape/entries.
/// @throws std::length_error On state geometry limits.
/// @throws std::overflow_error If the computed log is outside finite double range.
/// @par Example
/// @code
/// std::print("{:.6f}\n",dm::algorithms::ellipsoid_log_volume(dm::algorithms::ellipsoid_ball({0,0},1))); // 1.144730
/// @endcode
/// @par References
/// - S. Boyd (2018), Ellipsoid Method, Stanford EE364b notes, https://web.stanford.edu/class/ee364b/lectures/ellipsoid_method_notes.pdf . The implementation uses a triangular rank-one factor update of the central-cut formula.
/// @see ellipsoid_log_volume_ratio()
double ellipsoid_log_volume(const EllipsoidState& state);

/// @brief Return the exact-arithmetic central-cut log-volume ratio as a stable numeric value.
///
/// For n>1, rho=(n*n/(n*n-1))^(n/2)*sqrt((n-1)/(n+1)). A convergent
/// log series avoids large-n subtraction cancellation. For n=1, ratio is one
/// half. This describes volume, not maximum-axis or iterate-distance shrinkage.
/// No allocation or shared state; floating rounding still applies.
/// @param dimension Positive dimension, including one.
/// @return Approximate log(rho), strictly negative.
/// @throws std::invalid_argument If dimension is zero.
/// @par Example
/// @code
/// std::print("{:.6f}\n",std::exp(dm::algorithms::ellipsoid_log_volume_ratio(2))); // 0.769800
/// @endcode
/// @par References
/// - S. Boyd (2018), Ellipsoid Method, Stanford EE364b notes, https://web.stanford.edu/class/ee364b/lectures/ellipsoid_method_notes.pdf . The implementation uses a triangular rank-one factor update of the central-cut formula.
/// @see ellipsoid_log_volume(), ellipsoid_central_cut()
double ellipsoid_log_volume_ratio(std::size_t dimension);

/// @brief Enclose a central half ellipsoid using a positive triangular shape update.
///
/// For nonzero g, retains g^T(x-center)<=0 in exact arithmetic. Scaled
/// projections and a rank-one triangular update avoid forming P=L*L^T.
/// The one-dimensional case halves its interval. O(n*n) word work/storage.
/// Finite diagonal positivity is checked after update; arithmetic failures expose
/// no partial result. center_changed=false signals all-coordinate rounding
/// stagnation. This is numerical geometry, not a formal enclosure certificate.
/// Inputs are unchanged and independent calls share no mutable state.
/// @param state Valid owned-shape description, borrowed unchanged.
/// @param gradient Finite n-vector, not all zero.
/// @return Owned candidate state, input support norm and measured log volumes.
/// @throws std::invalid_argument On invalid state/gradient or zero gradient.
/// @throws std::overflow_error On nonfinite/out-of-range computed geometry or support.
/// @throws std::runtime_error On positive-factor/support underflow or lost representability.
/// @throws std::length_error On count arithmetic or container limits.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto c=dm::algorithms::ellipsoid_central_cut(dm::algorithms::ellipsoid_ball({0},4),{-1});
/// std::print("{} {}\n",c.state.center[0],c.state.lower[0][0]); // 2 2
/// @endcode
/// @par References
/// - S. Boyd (2018), Ellipsoid Method, Stanford EE364b notes, https://web.stanford.edu/class/ee364b/lectures/ellipsoid_method_notes.pdf . The implementation uses a triangular rank-one factor update of the central-cut formula.
/// @see EllipsoidCut, ellipsoid_minimize_work()
EllipsoidCut ellipsoid_central_cut(const EllipsoidState& state,const std::vector<double>& gradient);

/// @brief Sample a two-dimensional factored ellipsoid boundary in native diagram coordinates.
///
/// Maps equally spaced unit-circle points through the actual factor; last point
/// is an exact copy of the first, so paths close. State/segment geometry and the
/// optional sample cap are checked before output allocation. O(segments) output
/// work/storage after constant-dimensional validation. Inputs are unchanged.
/// @param state Valid two-dimensional ellipsoid.
/// @param segments Boundary segments, at least eight, default 128.
/// @param max_segments Optional segment limit, default 1000000; zero disables it.
/// @return Owned segments+1 finite native points, counterclockwise for positive diagonal.
/// @throws std::invalid_argument On invalid state/dimension or fewer than eight segments.
/// @throws std::length_error On output geometry or optional segment limit.
/// @throws std::overflow_error On nonfinite sampled coordinates.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto p=dm::algorithms::ellipsoid_outline(dm::algorithms::ellipsoid_ball({0,0},1),8);
/// std::print("{} {}\n",p.size(),p.front().x==p.back().x); // 9 true
/// @endcode
/// @see graphics::Diagram, ellipsoid_central_cut()
std::vector<graphics::DiagramPoint> ellipsoid_outline(const EllipsoidState& state,std::size_t segments=128,
                                                       std::size_t max_segments=1000000);

/// @brief Minimize with factored central cuts and an explicit finite stopping outcome.
///
/// Validates parameters/geometry before callbacks. Evaluates f once initially
/// and after each committed cut, subgradient once per attempted iteration.
/// Zero subgradient stops; otherwise ||L^T*g||<=tol is a conditional absolute
/// objective-gap test, requiring a valid convex oracle and minimizer enclosure.
/// Those assumptions are not certified. Center stagnation and iteration limit
/// do not certify convergence. The method is not a descent method; best point
/// is retained. n=1 uses the same support criterion, correcting the old width
/// criterion and wasted zero-gradient iterations. Finite positive lower factors
/// replace the old unchecked possibly indefinite P. Ordinary calls keep no
/// optional history/quota: O(n*n) working storage and O(n*n) per attempt plus
/// callback cost. Input is unchanged; caller callbacks determine concurrency.
/// @param f Finite convex objective callback, defined at every queried center.
/// @param subgradient Finite n-vector oracle; must be a valid convex subgradient.
/// @param center Nonempty finite initial center.
/// @param radius Positive finite initial ball radius, assumed to enclose a minimizer.
/// @param max_iter Nonnegative subgradient-call cap, default 1000.
/// @param tol Finite nonnegative absolute objective-gap tolerance, default 1e-10.
/// @return Owned best result, final state, actual reason, latest query bound and work.
/// @throws std::invalid_argument On invalid parameters, callback shape or nonfinite callback values.
/// @throws std::length_error On count, matrix or container geometry.
/// @throws std::overflow_error On nonfinite computed geometry or support.
/// @throws std::runtime_error On factor/support underflow or lost numerical representability.
/// @throws std::bad_alloc If allocation fails; callback exceptions propagate unchanged.
/// @par Example
/// @code
/// const auto f=[](const std::vector<double>& x){return x[0]*x[0];};
/// const auto g=[](const std::vector<double>& x){return std::vector<double>{2*x[0]};};
/// const auto r=dm::algorithms::ellipsoid_minimize_work(f,g,{0},1);
/// std::print("{} {}\n",r.result.iterations,r.work.updates); // 1 0
/// @endcode
/// @par References
/// - S. Boyd (2018), Ellipsoid Method, Stanford EE364b notes, https://web.stanford.edu/class/ee364b/lectures/ellipsoid_method_notes.pdf . The implementation uses a triangular rank-one factor update of the central-cut formula.
/// @see EllipsoidOutcome, ellipsoid_minimize_trace()
EllipsoidOutcome ellipsoid_minimize_work(const std::function<double(const std::vector<double>&)>& f,
    const std::function<std::vector<double>(const std::vector<double>&)>& subgradient,
    const std::vector<double>& center,double radius,int max_iter=1000,double tol=1e-10);

/// @brief Capture actual objective states and subgradient decisions without replay.
///
/// Same kernel/callback order as ellipsoid_minimize_work(). Reserves initial
/// S=1+n+n*n history before shape allocation, owned copies or f. Each attempt
/// reserves I=1+n before its subgradient callback. A candidate cut can be formed
/// before its next sample quota check; S is reserved before the next f callback
/// and history copy. Default one million entries; zero disables only quota.
/// No partial trace escapes domain, callback, quota or numeric failure. See
/// EllipsoidTrace for excluded output/runtime memory; this is not a peak-RAM cap.
/// @param f Finite convex objective callback, defined at every queried center.
/// @param subgradient Finite n-vector oracle; must be a valid convex subgradient.
/// @param center Nonempty finite initial center.
/// @param radius Positive finite initial ball radius, assumed to enclose a minimizer.
/// @param max_iter Nonnegative subgradient-call cap, default 1000.
/// @param tol Finite nonnegative absolute objective-gap tolerance, default 1e-10.
/// @param max_entries Retained-history entry limit, default 1000000; zero disables it.
/// @return Owned actual samples/attempts and the same detailed outcome.
/// @throws std::invalid_argument On invalid parameters, callback shape or nonfinite callback values.
/// @throws std::length_error On count, matrix or container geometry.
/// @throws std::overflow_error On nonfinite computed geometry or support.
/// @throws std::runtime_error On factor/support underflow or lost numerical representability.
/// @throws std::bad_alloc If allocation fails; callback exceptions propagate unchanged.
/// @throws std::length_error Also on optional history quota exhaustion.
/// @par Example
/// @code
/// const auto f=[](const std::vector<double>& x){return x[0]*x[0];};
/// const auto g=[](const std::vector<double>& x){return std::vector<double>{2*x[0]};};
/// const auto t=dm::algorithms::ellipsoid_minimize_trace(f,g,{0},1,10,1e-10,5);
/// std::print("{} {}\n",t.captured_entries,t.outcome.result.value); // 5 0
/// @endcode
/// @par References
/// - S. Boyd (2018), Ellipsoid Method, Stanford EE364b notes, https://web.stanford.edu/class/ee364b/lectures/ellipsoid_method_notes.pdf . The implementation uses a triangular rank-one factor update of the central-cut formula.
/// @see EllipsoidTrace, ellipsoid_minimize_work()
EllipsoidTrace ellipsoid_minimize_trace(const std::function<double(const std::vector<double>&)>& f,
    const std::function<std::vector<double>(const std::vector<double>&)>& subgradient,
    const std::vector<double>& center,double radius,int max_iter=1000,double tol=1e-10,
    std::size_t max_entries=1000000);

/// @brief Return the best evaluated point through the original six-argument interface.
///
/// Forwards to the factored detailed kernel, retaining the original result
/// layout and best-point meaning. No convergence status is present in this legacy
/// result; inspect ellipsoid_minimize_work() for the actual stop reason.
/// Invalid callbacks/geometry now fail before unsafe access; zero gradients stop
/// and n=1 uses objective-gap tolerance. Floating trajectories/counts can differ
/// from the former unstable direct-matrix update. No optional history/quota.
/// @param f Finite convex objective callback, defined at every queried center.
/// @param subgradient Finite n-vector oracle; must be a valid convex subgradient.
/// @param center Nonempty finite initial center.
/// @param radius Positive finite initial ball radius, assumed to enclose a minimizer.
/// @param max_iter Nonnegative subgradient-call cap, default 1000.
/// @param tol Finite nonnegative absolute objective-gap tolerance, default 1e-10.
/// @return Best evaluated point/objective and number of attempted subgradient calls.
/// @throws std::invalid_argument On invalid parameters, callback shape or nonfinite callback values.
/// @throws std::length_error On count, matrix or container geometry.
/// @throws std::overflow_error On nonfinite computed geometry or support.
/// @throws std::runtime_error On factor/support underflow or lost numerical representability.
/// @throws std::bad_alloc If allocation fails; callback exceptions propagate unchanged.
/// @par Example
/// @code
/// const auto f=[](const std::vector<double>& x){return (x[0]-2)*(x[0]-2);};
/// const auto g=[](const std::vector<double>& x){return std::vector<double>{2*(x[0]-2)};};
/// const auto r=dm::algorithms::ellipsoid_minimize(f,g,{2},4);
/// std::print("{} {}\n",r.x[0],r.value); // 2 0
/// @endcode
/// @par References
/// - S. Boyd (2018), Ellipsoid Method, Stanford EE364b notes, https://web.stanford.edu/class/ee364b/lectures/ellipsoid_method_notes.pdf . The implementation uses a triangular rank-one factor update of the central-cut formula.
/// @see EllipsoidResult, ellipsoid_minimize_work()
EllipsoidResult ellipsoid_minimize(const std::function<double(const std::vector<double>&)>& f,
    const std::function<std::vector<double>(const std::vector<double>&)>& subgradient,
    std::vector<double> center,double radius,int max_iter=1000,double tol=1e-10);

} // namespace datamunge::algorithms
