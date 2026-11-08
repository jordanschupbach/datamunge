#pragma once

/// @file
/// @brief Checked fixed-ratio maximization, actual bracket states and online histories.

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace datamunge::algorithms {

/// @brief Midpoint estimate, evaluated objective and iteration count from golden-section maximization.
///
/// Owns three scalars; the existing layout is retained. argmax is the final
/// midpoint, not the best sampled point. A small bracket is NOT an unconditional
/// accuracy certificate: strict unimodality and faithful comparisons are assumed.
/// No callable or captured state is retained. Defaults are zero.
///
/// @par Example
/// @code
/// const auto r = dm::algorithms::golden_section_search([](double x){return x;}, 0., 1., 1., 0);
/// std::print("{} {} {}\n", r.argmax, r.value, r.iterations); // 0.5 0.5 0
/// @endcode
///
/// @par References
/// - D. P. Bertsekas, Nonlinear Programming, Third Edition, Appendix C.3, pp.812–814:
///   https://web.mit.edu/dimitrib/www/NLP_Appendixes_Refs.pdf
/// @see golden_section_search(), GoldenRun
struct GoldenResult {
    double argmax{0}; ///< Final 0.5*(a+b), in input-coordinate units.
    double value{0}; ///< Actual fresh callback result at argmax, in objective units.
    int iterations{0}; ///< Completed bracket updates, excluding initial and final calls.
};

/// @brief Actual loop termination condition, without an optimization-success claim.
///
/// Width takes precedence when both conditions hold. These reasons describe
/// floating arithmetic, not whether the bracket contains a true maximizer.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::algorithms::GoldenStop::width == dm::algorithms::GoldenStop::width); // true
/// @endcode
/// @see GoldenRun, golden_stop_name()
enum class GoldenStop {
    width, ///< Actual final b-a <= tol, including an initially short or equal bracket.
    iteration_limit ///< Actual final b-a > tol and the requested update limit reached.
};

/// @brief Names a native golden-section termination condition.
/// @param stop Condition to name; an unrecognized enum value is allowed.
/// @return Static string "width", "iteration_limit", or "unknown"; never null.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::algorithms::golden_stop_name(dm::algorithms::GoldenStop::width)); // width
/// @endcode
/// @see GoldenStop
inline const char* golden_stop_name(GoldenStop stop) noexcept {
    switch (stop) {
    case GoldenStop::width: return "width";
    case GoldenStop::iteration_limit: return "iteration_limit";
    }
    return "unknown";
}

/// @brief Actual endpoints, two cached probes and their evaluated values.
///
/// Owns six doubles, default zero. Endpoints/probes have input units; fc/fd have
/// objective units. Cached points may coincide after rounding. No reconstructed
/// ideal positions or callable references are stored.
///
/// @par Example
/// @code
/// const dm::algorithms::GoldenState s{};
/// std::print("{} {}\n", s.a, s.fd); // 0 0
/// @endcode
/// @see GoldenRun, GoldenEvent
struct GoldenState {
    double a{0}; ///< Actual left endpoint.
    double b{0}; ///< Actual right endpoint.
    double c{0}; ///< Actual left probe, initially b-rho*(b-a).
    double d{0}; ///< Actual right probe, initially a+rho*(b-a).
    double fc{0}; ///< Actual cached f(c).
    double fd{0}; ///< Actual cached f(d).
};

/// @brief Owned ordinary result plus actual final bracket, callback count and stop reason.
///
/// evaluations counts actual callback invocations: two initial, one per completed
/// update and one FINAL MIDPOINT CALL, hence iterations+3 even for a zero-width
/// interval or zero update limit. Copying/validation/allocation/FLOPs/time and work
/// inside the callback are excluded. No history or callable is retained; auxiliary
/// solver storage is constant, excluding state owned by the passed callable.
/// Defaults describe an unevaluated empty record, not a completed search.
///
/// @par Example
/// @code
/// const auto r = dm::algorithms::golden_section_run([](double x){return x;}, 0., 1., 1., 0);
/// std::print("{} {}\n", r.evaluations, dm::algorithms::golden_stop_name(r.stop)); // 3 width
/// @endcode
/// @see golden_section_run(), GoldenTrace
struct GoldenRun {
    GoldenResult result{}; ///< Preserved midpoint/value/completed-update layout.
    GoldenState final_state{}; ///< Actual final cached bracket and probes.
    std::size_t evaluations{0}; ///< Successful-run callback invocations only.
    GoldenStop stop{GoldenStop::width}; ///< Actual loop condition, not a truth certificate.
};

/// @brief Kind of complete fixed-size event emitted by a native golden-section search.
///
/// An initial record and a final stop record count even with no bracket updates.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::algorithms::GoldenEventKind::step == dm::algorithms::GoldenEventKind::step); // true
/// @endcode
/// @see GoldenEvent, GoldenTrace
enum class GoldenEventKind {
    initial, ///< Two initial callback results are available.
    step, ///< One actual branch, cached-probe reuse and new callback have completed.
    stop ///< Actual final midpoint callback has completed.
};

/// @brief Owned before/after states and the actual new callback for one search event.
///
/// Initial and stop records have identical before/after states. Initial has
/// iteration=0, evaluations=2 and unused evaluated_x/evaluated_value=0. A step
/// records the strict before.fc < before.fd decision, the actual new probe/value
/// and cumulative calls=iteration+2. Stop records the fresh midpoint call and
/// cumulative calls=iteration+3. kept_right is unused false for initial/stop.
/// All fields are primitive fixed-size values; no callback or numerical replay.
///
/// @par Example
/// @code
/// const dm::algorithms::GoldenEvent e{};
/// std::print("{} {}\n", e.iteration, e.evaluations); // 0 0
/// @endcode
/// @see golden_section_trace(), GoldenState
struct GoldenEvent {
    GoldenEventKind kind{GoldenEventKind::initial}; ///< Actual event kind.
    int iteration{0}; ///< Completed updates at this event.
    bool kept_right{false}; ///< For a step, true means old [c,b] was retained.
    GoldenState before{}; ///< Actual state before this step; current state for metadata.
    GoldenState after{}; ///< Actual state after this step; current state for metadata.
    double evaluated_x{0}; ///< Actual fresh probe/midpoint, unused zero for initial.
    double evaluated_value{0}; ///< Actual fresh objective value, unused zero for initial.
    std::size_t evaluations{0}; ///< Actual cumulative callback invocations.
};

/// @brief Owned run and complete online initial/update/stop records.
///
/// H=iterations+2 fixed-size complete records; no full-vector snapshots, replay,
/// truncation or partial return. Copying owns independent record storage. No
/// callable/captures are stored. Callback side effects already performed are not
/// undone on exception; a copied callable can still share borrowed capture state.
/// Root/input/callable/ordinary run and final state/constant working state/allocator
/// overhead are excluded from the quota. Records are not numeric entries, bytes
/// or peak RAM. Recording takes O(H) space; run/output-only storage is constant.
///
/// @par Example
/// @code
/// const auto t = dm::algorithms::golden_section_trace([](double x){return x;}, 0., 1., 1., 0, 2);
/// std::print("{} {}\n", t.events.size(), t.run.evaluations); // 2 3
/// @endcode
/// @see golden_section_trace(), GoldenEvent
struct GoldenTrace {
    GoldenRun run{}; ///< Actual result and selected callback work.
    std::vector<GoldenEvent> events; ///< Complete owned records in actual execution order.
};

/// @brief Shared implementation details for the native templated search.
namespace golden_detail {

/// @brief Rejects a nonfinite evaluated solver coordinate without repairing it.
/// @param x Evaluated coordinate or width.
/// @return The unchanged finite value.
/// @throws std::overflow_error If x is nonfinite.
inline double coordinate(double x) {
    if (!std::isfinite(x)) throw std::overflow_error("golden_section_search: nonfinite coordinate arithmetic");
    return x;
}

/// @brief Admits one complete next record before its associated callback work.
/// @param events Null for ordinary execution, otherwise the owned output records.
/// @param max_records Optional complete-record quota; zero disables the quota.
/// @throws std::length_error If the next whole record exceeds the quota or capacity.
inline void admit(const std::vector<GoldenEvent>* events, std::size_t max_records) {
    if (events && ((max_records && events->size() >= max_records) || events->size() == events->max_size()))
        throw std::length_error("golden_section_trace: complete record limit");
}

/// @brief Calls the actual supplied objective once and validates its converted result.
/// @tparam F Callable type accepting one double and returning a double-convertible value.
/// @param f Borrowed local callable; its own exceptions propagate unchanged.
/// @param x Actual finite evaluation point.
/// @return The unchanged finite converted objective value.
/// @throws std::domain_error If the evaluated objective is nonfinite.
template <typename F>
double evaluate(F& f, double x) {
    const double value = f(x);
    if (!std::isfinite(value)) throw std::domain_error("golden_section_search: nonfinite objective");
    return value;
}

/// @brief Executes the original ordered probes, strict branch, reuse and final call.
/// @tparam F Value-passed public callable's actual local type.
/// @param f Borrowed local callable; no further callable copies are made.
/// @param a Finite left endpoint, at most b.
/// @param b Finite right endpoint, at least a.
/// @param tol Finite nonnegative absolute width tolerance.
/// @param max_iter Nonnegative maximum completed bracket updates.
/// @param events Optional owned output vector; null disables recording.
/// @param max_records Complete-record quota; zero disables the optional quota.
/// @return Actual final result, states, calls and loop stop condition.
/// @throws std::invalid_argument For invalid endpoints, tolerance or update limit.
/// @throws std::overflow_error For nonfinite evaluated solver arithmetic.
/// @throws std::domain_error For a nonfinite evaluated objective value.
/// @throws std::length_error For an inadmissible complete next record.
/// @throws std::bad_alloc If record allocation fails; callback exceptions propagate.
template <typename F>
GoldenRun execute(F& f, double a, double b, double tol, int max_iter,
                  std::vector<GoldenEvent>* events, std::size_t max_records) {
    if (!std::isfinite(a) || !std::isfinite(b) || a > b || !std::isfinite(tol) || tol < 0 || max_iter < 0)
        throw std::invalid_argument("golden_section_search: ordered finite endpoints, nonnegative finite tolerance and update limit required");
    coordinate(b-a);
    const double invphi = (std::sqrt(5.0)-1.0)/2.0;
    double c = coordinate(b-invphi*(b-a));
    double d = coordinate(a+invphi*(b-a));
    admit(events,max_records);
    double fc = evaluate(f,c), fd = evaluate(f,d);
    int it = 0;
    if (events) events->push_back({GoldenEventKind::initial,0,false,{a,b,c,d,fc,fd},{a,b,c,d,fc,fd},0,0,2});
    while (b-a > tol && it < max_iter) {
        admit(events,max_records);
        const GoldenState before{a,b,c,d,fc,fd};
        const bool kept_right = fc < fd;
        double new_x, new_value;
        if (fc < fd) {
            a = c;
            c = d; fc = fd;
            d = coordinate(a+invphi*(b-a)); fd = evaluate(f,d);
            new_x=d; new_value=fd;
        } else {
            b = d;
            d = c; fd = fc;
            c = coordinate(b-invphi*(b-a)); fc = evaluate(f,c);
            new_x=c; new_value=fc;
        }
        ++it;
        if (events) events->push_back({GoldenEventKind::step,it,kept_right,before,{a,b,c,d,fc,fd},new_x,new_value,static_cast<std::size_t>(it)+2});
    }
    admit(events,max_records);
    const double x = coordinate(0.5*(a+b));
    const double value = evaluate(f,x);
    const GoldenState final_state{a,b,c,d,fc,fd};
    const auto calls=static_cast<std::size_t>(it)+3;
    if (events) events->push_back({GoldenEventKind::stop,it,false,final_state,final_state,x,value,calls});
    return {{x,value,it},final_state,calls,b-a<=tol ? GoldenStop::width : GoldenStop::iteration_limit};
}
} // namespace golden_detail

/// @brief Maximizes an assumed strictly unimodal function using the preserved fixed-ratio loop.
///
/// Uses rho=(sqrt(5)-1)/2, two initial calls, one fresh call per bracket update,
/// and a fresh final midpoint call. Strict fc<fd retains [c,b]; ties take [a,d].
/// No endpoints are evaluated, no best-sampled-point substitution or stagnation
/// early exit occurs. Ordinary auxiliary storage is O(1), excluding callable state.
/// The callable is passed by value exactly as in the original API; captured
/// references can still mutate external state. Its exceptions propagate unchanged.
/// Unimodality, deterministic values and faithful comparisons are NOT validated.
/// A small final width is NOT an unconditional accuracy certificate; roundoff,
/// flat tails, noise or multiple peaks can discard a true maximizer. Zero tol
/// can reach the iteration limit after probe stagnation. No history allocation.
///
/// @tparam F Copy/move-constructible callable accepting double and convertible result.
/// @param f Objective callable passed by value; evaluations must return finite values.
/// @param a Finite left endpoint, at most b; equal endpoints are allowed.
/// @param b Finite right endpoint, at least a; b-a must evaluate finite.
/// @param tol Finite nonnegative absolute bracket width, default 1e-9; zero allowed.
/// @param max_iter Nonnegative maximum bracket updates, default 500; zero still makes three calls.
/// @return Final midpoint, its fresh objective value and completed update count.
/// @throws std::invalid_argument For invalid endpoints, tolerance or update limit.
/// @throws std::overflow_error For nonfinite evaluated width/probes/midpoint arithmetic.
/// @throws std::domain_error For a nonfinite objective value; callable exceptions propagate.
///
/// @par Example
/// @code
/// const auto r = dm::algorithms::golden_section_search([](double x){return x;}, 0., 1., 1., 0);
/// std::print("{} {} {}\n", r.argmax, r.value, r.iterations); // 0.5 0.5 0
/// @endcode
///
/// @par References
/// - D. P. Bertsekas, Nonlinear Programming, Third Edition, Appendix C.3, pp.812–814:
///   https://web.mit.edu/dimitrib/www/NLP_Appendixes_Refs.pdf
///   The native maximizer keeps its original one-sided tie rule and omits the
///   text's endpoint checks; this is not an exact port of that minimizer.
/// @see golden_section_run(), golden_section_trace(), GoldenResult
template <typename F>
GoldenResult golden_section_search(F f, double a, double b, double tol=1e-9, int max_iter=500) {
    return golden_detail::execute(f,a,b,tol,max_iter,nullptr,0).result;
}

/// @brief Runs the same native maximizer with actual final states and callback counts.
///
/// Same validation, value-passed callable, strict branch, numerical limitations
/// and exception propagation as golden_section_search(). No retained history.
/// Evaluations count actual calls (iterations+3), not elapsed time or work inside f.
/// Stop names the actual width/iteration condition and does not assert success.
///
/// @tparam F Callable accepting double and returning a double-convertible finite value.
/// @param f Objective callable passed by value; captured references remain borrowed.
/// @param a Finite left endpoint, at most b.
/// @param b Finite right endpoint, at least a, with finite b-a.
/// @param tol Finite nonnegative absolute width tolerance, default 1e-9.
/// @param max_iter Nonnegative completed update limit, default 500; zero allowed.
/// @return Owned result, final cached state, actual callback count and stop condition.
/// @throws std::invalid_argument For invalid endpoints, tolerance or update limit.
/// @throws std::overflow_error For nonfinite evaluated solver arithmetic.
/// @throws std::domain_error For nonfinite objective values; callable exceptions propagate.
///
/// @par Example
/// @code
/// const auto r = dm::algorithms::golden_section_run([](double x){return x;}, 0., 1., 1., 0);
/// std::print("{} {}\n", r.evaluations, dm::algorithms::golden_stop_name(r.stop)); // 3 width
/// @endcode
///
/// @par References
/// - Fixed-ratio method and native variant: golden_section_search().
/// @see golden_section_search(), GoldenRun, golden_section_trace()
template <typename F>
GoldenRun golden_section_run(F f, double a, double b, double tol=1e-9, int max_iter=500) {
    return golden_detail::execute(f,a,b,tol,max_iter,nullptr,0);
}

/// @brief Records complete actual bracket events online, with a whole-record quota.
///
/// Same ordered kernel and numerical limitations as golden_section_search().
/// H=iterations+2, including initial and stop metadata. Before each record's
/// callback work, the next complete append is admitted; no replay, truncation or
/// partial trace is returned. No-step searches need two records: reject one,
/// admit two. One update needs three, including its final stop. Zero disables
/// the optional quota and still records everything. Earlier callback side effects
/// cannot be rolled back on quota rejection or another exception. The trace owns
/// its scalar records, not callable captures. Root/input/callable/ordinary result/
/// final state/working storage/allocator overhead are excluded; complete records
/// are not entries, bytes or peak RAM. Parameters are validated before admission;
/// numerical objective checks occur at their actual execution points. O(H) space.
///
/// @tparam F Callable accepting double and returning a double-convertible finite value.
/// @param f Objective callable passed by value; borrowed capture state can be mutated.
/// @param a Finite left endpoint, at most b.
/// @param b Finite right endpoint, at least a, with finite b-a.
/// @param tol Finite nonnegative absolute width tolerance, default 1e-9.
/// @param max_iter Nonnegative completed update limit, default 500.
/// @param max_records Complete initial/update/stop records, default 1,000,000; zero optional-off.
/// @return Owned ordinary run and complete actual execution events.
/// @throws std::invalid_argument For invalid endpoints, tolerance or update limit.
/// @throws std::overflow_error For nonfinite evaluated solver arithmetic.
/// @throws std::domain_error For nonfinite objective values.
/// @throws std::length_error Before an inadmissible whole next record's callback work.
/// @throws std::bad_alloc If history allocation fails; callable exceptions propagate.
///
/// @par Example
/// @code
/// const auto t = dm::algorithms::golden_section_trace([](double x){return x;}, 0., 1., 1., 0, 2);
/// std::print("{} {}\n", t.events.size(), t.run.evaluations); // 2 3
/// @endcode
///
/// @par References
/// - Fixed-ratio method and native variant: golden_section_search().
/// @see GoldenTrace, GoldenEvent, golden_section_run()
template <typename F>
GoldenTrace golden_section_trace(F f, double a, double b, double tol=1e-9, int max_iter=500,
                                 std::size_t max_records=1000000) {
    GoldenTrace trace;
    trace.run=golden_detail::execute(f,a,b,tol,max_iter,&trace.events,max_records);
    return trace;
}

} // namespace datamunge::algorithms
