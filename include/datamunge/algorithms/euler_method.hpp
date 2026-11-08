#pragma once
/// @file euler_method.hpp
/// @brief Checked fixed-step forward Euler, actual derivatives and explicit logical history geometry.
#include <datamunge/algorithms/ode_types.hpp>
#include <cstddef>
namespace datamunge::algorithms {
/// @brief Logical work from the actual fixed-step Euler execution.
/// Counts exclude validation, time-grid construction, allocation, history copying
/// and CPU time. rhs_calls counts actual callback invocations; scalar_updates counts
/// committed component updates, not floating instructions. Complete results have
/// steps=rhs_calls=N and scalar_updates=N*d. Exceptions expose no partial result.
/// @par Example
/// @code
/// const auto r=dm::algorithms::euler_method_work([](double,const std::vector<double>& y){return y;},{1,2},0,1,4);
/// std::print("{} {} {}\n",r.work.rhs_calls,r.work.steps,r.work.scalar_updates); // 4 4 8
/// @endcode
/// @see EulerOutcome
struct EulerWork {
    std::size_t rhs_calls=0; ///< Actual derivative callback calls.
    std::size_t steps=0; ///< Committed steps, equal to the requested positive count on success.
    std::size_t scalar_updates=0; ///< Committed individual state-coordinate updates.
};

/// @brief Owned intrinsic trajectory, nominal step size and actual work.
/// Every Euler call returns N+1 samples, including both endpoints. Ordinary/work
/// calls keep the intrinsic trajectory but no optional derivative history or
/// capture quota. step_size is the finite rounded nominal h; sample times follow
/// euler_method()'s documented grid rounding. A successful run is not an error
/// estimate or a guarantee of stability/accuracy for the supplied ODE.
/// @par Example
/// @code
/// const auto r=dm::algorithms::euler_method_work([](double,const std::vector<double>& y){return std::vector<double>{-y[0]};},{1},0,1,2);
/// std::print("{} {}\n",r.step_size,r.solution.y.back()[0]); // 0.5 0.25
/// @endcode
/// @see euler_method_work(), EulerTrace
struct EulerOutcome {
    OdeSolution solution; ///< Owned time/state samples from this execution.
    double step_size=0; ///< Rounded nominal (t1-t0)/N, negative for reverse integration.
    EulerWork work; ///< Actual callback/step/component counts.
};

/// @brief Owned actual derivative vectors alongside the same execution's intrinsic trajectory.
/// derivatives[i] is the exact callback vector at solution.t[i], solution.y[i].
/// The final sample has no derivative query. No callbacks are retained or replayed.
/// Logical capture C=(2N+1)(d+1) counts N+1 sample slots with d state entries and
/// N derivative slots with d derivative entries. All C is checked before history
/// copies or callbacks. Default one million; zero disables only that quota.
/// Other outcome metadata, runtime state/callback storage, vector headers,
/// capacities and allocator bytes are excluded: C is not peak RAM. Copies own all
/// samples/derivatives. Failures return no partial trace; callback side effects
/// before a later callback/arithmetic failure remain the caller's responsibility.
/// @par Example
/// @code
/// const auto t=dm::algorithms::euler_method_trace([](double,const std::vector<double>& y){return std::vector<double>{-y[0]};},{1},0,1,2);
/// std::print("{} {} {}\n",t.outcome.solution.y.back()[0],t.derivatives[0][0],t.captured_entries); // 0.25 -1 10
/// @endcode
/// @see euler_method_trace(), euler_trace_size()
struct EulerTrace {
    EulerOutcome outcome; ///< Intrinsic trajectory and actual counters from this run.
    std::vector<std::vector<double>> derivatives; ///< Owned actual derivative vectors, one per step.
    std::size_t captured_entries=0; ///< Checked logical sample/derivative capture count.
};

/// @brief Check Euler state/history geometry and return the complete logical trace size.
/// Computes (2N+1)(d+1) with checked counts. Also checks intrinsic state cells and
/// required per-vector/container limits before allocation. This is a geometric
/// count check, not a promise of available memory. It allocates no trajectory.
/// @param dimension Positive state dimension d.
/// @param steps Positive number N of fixed steps.
/// @return Logical captured sample/derivative slots and scalar entries.
/// @throws std::invalid_argument For zero dimension or steps<1.
/// @throws std::length_error For count arithmetic or container geometry limits.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::euler_trace_size(1,2)); // 10
/// @endcode
/// @see EulerTrace
std::size_t euler_trace_size(std::size_t dimension,int steps);

/// @brief Integrate a first-order vector ODE using checked fixed-step forward Euler.
/// Uses one actual derivative per step: y_next=y+h*f(t,y). Under suitable C2
/// solution and Lipschitz assumptions, one-step defect is O(h^2) and global error
/// over a fixed finite interval is O(h); neither stability nor an error estimate
/// is provided by the API. Runtime/callback storage aside, time is O(N*d+N*cost(f))
/// and the intrinsic returned history uses O((N+1)*d) entries.
/// Nominal h is converted from an extended-precision endpoint difference/N.
/// Interior times use extended t0+i*h; the last time is exactly t1. Rounded grid
/// differences need not equal h exactly; state updates always use nominal h.
/// For a nonzero interval every stored time must advance strictly in its direction,
/// checked before any callback. Reverse integration is supported. A zero interval
/// preserves N callbacks and N+1 repeated-time states with h=0.
/// Inputs and every returned derivative are checked before indexing. Each double
/// product h*f_i and subsequent double sum must be finite; callback exceptions
/// propagate unchanged. No callback or input storage is retained. Concurrent calls
/// are independent if the caller's callback/captured state is safe for concurrency.
/// @param f Nonempty borrowed callback; returns d finite derivatives in state/time units.
/// @param y0 Nonempty finite initial state, owned by this legacy by-value signature.
/// @param t0 Finite initial time.
/// @param t1 Finite final time; may be below or equal to t0.
/// @param steps Positive number N of nominal equal steps.
/// @return Owned N+1 time/state samples, including initial and final samples.
/// @throws std::invalid_argument For invalid inputs, derivative shape or nonfinite derivative.
/// @throws std::length_error For checked count/container geometry limits.
/// @throws std::overflow_error For nonfinite step/grid/product/state arithmetic.
/// @throws std::underflow_error If a nonzero interval's nominal h rounds to zero.
/// @throws std::runtime_error If a nonzero interval's rounded grid cannot advance strictly.
/// @throws std::bad_alloc If required storage cannot be allocated.
/// @par Example
/// @code
/// const auto s=dm::algorithms::euler_method([](double,const std::vector<double>& y){return std::vector<double>{-y[0]};},{1},0,1,2);
/// std::print("{} {}\n",s.t.back(),s.y.back()[0]); // 1 0.25
/// @endcode
/// @par References
/// - E. Hairer and C. Lubich. Numerical solution of ordinary differential equations, section 1.1. https://www.unige.ch/~hairer/preprints/pcam-ode.pdf
/// - C. Moler. Numerical Computing with MATLAB, chapter 7 (2013 PDF version). https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/moler/odes.pdf
/// @see euler_method_work(), euler_method_trace()
OdeSolution euler_method(const OdeSystem& f,std::vector<double> y0,double t0,double t1,int steps);

/// @brief Return the intrinsic Euler trajectory, nominal h and actual logical work.
/// Uses the same checked kernel and grid rounding as euler_method(), including
/// reverse/zero intervals and full derivative validation. Keeps no optional
/// derivative history or capture quota. Intrinsic output uses O((N+1)*d) memory;
/// runtime is O(N*d+N*cost(f)). Caller callback exceptions propagate unchanged.
/// @param f Nonempty callback returning d finite derivatives; borrowed only during this call.
/// @param y0 Borrowed nonempty finite initial state; output owns its copies.
/// @param t0 Finite initial time.
/// @param t1 Finite final time, including reverse or zero intervals.
/// @param steps Positive nominal step count N.
/// @return Owned intrinsic trajectory, actual step size and work counters.
/// @throws std::invalid_argument For invalid inputs or callback shape/finiteness.
/// @throws std::length_error For count/container geometry limits.
/// @throws std::overflow_error For nonfinite step/grid/product/state arithmetic.
/// @throws std::underflow_error If a nonzero interval's h rounds to zero.
/// @throws std::runtime_error If a nonzero interval's stored grid cannot advance.
/// @throws std::bad_alloc If required storage cannot be allocated.
/// @par Example
/// @code
/// const auto r=dm::algorithms::euler_method_work([](double,const std::vector<double>&){return std::vector<double>{2};},{1},0,1,4);
/// std::print("{} {}\n",r.solution.y.back()[0],r.work.rhs_calls); // 3 4
/// @endcode
/// @par References
/// - E. Hairer and C. Lubich. Numerical solution of ordinary differential equations. https://www.unige.ch/~hairer/preprints/pcam-ode.pdf
/// @see euler_method(), EulerOutcome
EulerOutcome euler_method_work(const OdeSystem& f,const std::vector<double>& y0,double t0,double t1,int steps);

/// @brief Capture the exact derivatives from the actual Euler trajectory with a checked quota.
/// Same numerical kernel, validation, intrinsic storage and grid rounding as
/// euler_method(). Optional capture C=(2N+1)(d+1) follows EulerTrace; the full known
/// quota is checked before copies or callbacks. A quota failure therefore performs
/// no callback. Other later failures can follow earlier callback side effects and
/// expose no partial output. Capture adds O(N*d) derivative storage and copies.
/// @param f Nonempty borrowed callback, returning d finite derivatives per actual step.
/// @param y0 Borrowed nonempty finite initial state; trace owns all copies.
/// @param t0 Finite initial time.
/// @param t1 Finite final time, including reverse or zero intervals.
/// @param steps Positive nominal step count N.
/// @param max_entries Logical capture limit (default 1000000); zero disables only quota.
/// @return Owned actual trajectory, derivative history, work and logical capture count.
/// @throws std::invalid_argument For invalid inputs or derivative shape/finiteness.
/// @throws std::length_error For count/container limits or C>max_entries when enabled.
/// @throws std::overflow_error For nonfinite step/grid/product/state arithmetic.
/// @throws std::underflow_error If a nonzero interval's h rounds to zero.
/// @throws std::runtime_error If a nonzero interval's grid cannot advance strictly.
/// @throws std::bad_alloc If required storage cannot be allocated.
/// @par Example
/// @code
/// const auto t=dm::algorithms::euler_method_trace([](double,const std::vector<double>& y){return std::vector<double>{-y[0]};},{1},0,1,2,10);
/// std::print("{} {}\n",t.derivatives.size(),t.captured_entries); // 2 10
/// @endcode
/// @par References
/// - E. Hairer and C. Lubich. Numerical solution of ordinary differential equations. https://www.unige.ch/~hairer/preprints/pcam-ode.pdf
/// @see EulerTrace, euler_trace_size(), euler_method_work()
EulerTrace euler_method_trace(const OdeSystem& f,const std::vector<double>& y0,double t0,double t1,int steps,std::size_t max_entries=1000000);
} // namespace datamunge::algorithms
