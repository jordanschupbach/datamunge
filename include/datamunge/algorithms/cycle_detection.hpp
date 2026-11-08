#pragma once
/// @file cycle_detection.hpp
/// @brief Constant-state Brent/Floyd cycle finding, actual diagnostic records and decimal-map helpers.
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace datamunge::algorithms {
/// @brief Deterministic uint64 transition; repeated equal inputs must have equal outputs.
/// Callback exceptions propagate; a detector never owns or rewinds caller state.
/// @par Example
/// @code
/// const dm::algorithms::CycleFunction next=[](std::uint64_t x){return (x+1)%3;};
/// std::print("{}\n",next(1)); // 2
/// @endcode
/// @see brent_cycle_detection()
using CycleFunction=std::function<std::uint64_t(std::uint64_t)>;

/// @brief Owned least positive period lambda and least cycle-entry index mu.
/// The historical layout/types are unchanged. Successful detection has lambda>=1;
/// a default zero record is a placeholder, not a detected cycle.
/// @par Example
/// @code
/// const auto cycle=dm::algorithms::brent_cycle_detection(0,[](std::uint64_t x){return (x+1)%3;});
/// std::print("{} {}\n",cycle.lambda,cycle.mu); // 3 0
/// @endcode
/// @see brent_cycle_detection(), CycleWork
struct CycleResult {
    std::size_t lambda{0}; ///< Smallest positive period of the eventual cycle.
    std::size_t mu{0}; ///< Least index of the first state on that cycle.
};

/// @brief Optional completed-transition-call limit for diagnostic cycle finding.
/// Zero means no caller limit. A positive limit stops before the next call that
/// would exceed it; completed callback calls are not undone. It is not a timeout.
/// @par Example
/// @code
/// const dm::algorithms::CycleOptions options{100};
/// std::print("{}\n",options.max_evaluations); // 100
/// @endcode
/// @see brent_cycle_work(), floyd_cycle_work(), brent_cycle_trace()
struct CycleOptions {
    std::size_t max_evaluations{0}; ///< Maximum successful transition calls, or zero for no caller limit.
};

/// @brief Actual detector phase; counters separate discovery from recovering entry/period.
/// @par Example
/// @code
/// const auto phase=dm::algorithms::CyclePhase::align;
/// std::print("{}\n",phase==dm::algorithms::CyclePhase::align); // true
/// @endcode
/// @see CycleWork, BrentCycleStep
enum class CyclePhase {
    detect, ///< Brent moving blocks or Floyd's meeting search.
    align, ///< Brent advances one pointer lambda steps before locating the entry.
    locate, ///< Both pointers advance together to determine mu.
    measure ///< Floyd walks one complete cycle to determine lambda.
};

/// @brief Owned cycle, its entry state and exact completed-call/equality counts.
/// Evaluations are the sum of the four phase counters, including initial calls.
/// Comparisons count only state-equality tests, excluding loop/limit checks.
/// Entry is obtained by the canonical recovery phase without replaying the orbit.
/// @par Example
/// @code
/// const auto work=dm::algorithms::brent_cycle_work(42,[](std::uint64_t x){return x;});
/// std::print("{} {} {}\n",work.entry,work.evaluations,work.result.lambda); // 42 2 1
/// @endcode
/// @see brent_cycle_work(), floyd_cycle_work()
struct CycleWork {
    CycleResult result; ///< Same least period/entry index as the corresponding ordinary detector.
    std::uint64_t entry{0}; ///< Actual state f^mu(x0) at the start of the cycle.
    std::size_t evaluations{0}; ///< All completed transition calls in this invocation.
    std::size_t detection_evaluations{0}; ///< Calls used to discover a meeting/period.
    std::size_t alignment_evaluations{0}; ///< Brent's lambda-step gap; zero for Floyd.
    std::size_t entry_evaluations{0}; ///< Two calls per lockstep entry-search move.
    std::size_t period_evaluations{0}; ///< Floyd's lambda-step period measurement; zero for Brent.
    std::size_t comparisons{0}; ///< State equality tests, including the final equal tests.
};

/// @brief One actual Brent state after movement, phase entry or checkpoint reset.
/// A checkpoint copies runner to reference without calling f; their equality at
/// that row is not a detected collision, since movement precedes the next test.
/// Indices identify positions in the original orbit, independent of equal state labels.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::brent_cycle_trace(0,[](std::uint64_t x){return (x+1)%3;});
/// std::print("{} {}\n",trace.steps[0].reference_index,trace.steps[0].runner_index); // 0 1
/// @endcode
/// @see BrentCycleTrace, CyclePhase
struct BrentCycleStep {
    CyclePhase phase{CyclePhase::detect}; ///< Detection, gap alignment or entry location; Brent never uses measure.
    std::uint64_t reference{0}; ///< Stationary reference in detection, rear pointer in later phases.
    std::uint64_t runner{0}; ///< Moving state in detection, forward pointer in later phases.
    std::size_t reference_index{0}; ///< Orbit index of reference.
    std::size_t runner_index{0}; ///< Orbit index of runner.
    std::size_t power{0}; ///< Current detection block length; zero in later phases.
    std::size_t offset{0}; ///< Detection block progress, alignment gap progress or entry candidate mu.
    std::size_t evaluations{0}; ///< Completed calls at this state, unchanged by a checkpoint/reset.
    bool checkpoint{false}; ///< This is a no-call reference-copy and block-doubling event.
};

/// @brief Owned actual Brent states and summary from the same kernel as ordinary detection.
/// Records add O(mu+lambda) storage; they do not cache the function's state space
/// or trigger extra transition calls. Callback storage is borrowed only during the call.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::brent_cycle_trace(0,[](std::uint64_t x){return (x+1)%3;});
/// std::print("{} {}\n",trace.work.result.lambda,trace.work.entry); // 3 0
/// @endcode
/// @see brent_cycle_trace(), BrentCycleStep
struct BrentCycleTrace {
    CycleWork work; ///< Period, entry and actual phase/equality counts.
    std::vector<BrentCycleStep> steps; ///< Chronological post-movement, phase-entry and checkpoint states.
};

/// @brief One actual Floyd pair after a complete movement or a no-call phase reset.
/// Detection pairs are x_t and x_(2t), with t starting at one. Locate resets
/// tortoise to x0; measure holds tortoise at the recovered entry. Original-orbit
/// indices distinguish equal labels reached at different times. Each row precedes
/// that pair's equality test; phase resets are not callback evaluations.
/// @par Example
/// @code
/// const auto t=dm::algorithms::floyd_cycle_trace(0,[](std::uint64_t x){return (x+1)%3;});
/// std::print("{} {}\n",t.steps.front().tortoise_index,t.steps.front().hare_index); // 1 2
/// @endcode
/// @see floyd_cycle_trace(), CyclePhase
struct FloydCycleStep {
    CyclePhase phase{CyclePhase::detect}; ///< Detection, entry location or period measurement; never align.
    std::uint64_t tortoise{}; ///< Actual slow/reference state.
    std::uint64_t hare{}; ///< Actual fast/moving state.
    std::size_t tortoise_index{}; ///< Current slow state's original-orbit index.
    std::size_t hare_index{}; ///< Current moving state's original-orbit index.
    std::size_t progress{}; ///< Detection t, locate candidate mu or measure candidate lambda.
    std::size_t evaluations{}; ///< Completed callback calls before this pair's comparison.
    bool phase_entry{}; ///< First pair of this phase, including the initial detection pair.
};
/// @brief Owned actual Floyd pairs and work from the ordinary detector kernel.
/// Recording adds O(mu+lambda) storage and no callback calls or orbit replay.
/// No retained callback/input views. Failure returns no partial trace.
/// @par Example
/// @code
/// const auto t=dm::algorithms::floyd_cycle_trace(0,[](std::uint64_t x){return (x+1)%3;});
/// std::print("{} {}\n",t.work.result.lambda,t.work.evaluations); // 3 12
/// @endcode
/// @see FloydCycleStep, floyd_cycle_trace()
struct FloydCycleTrace {
    CycleWork work; ///< Actual entry, least period/index and phase counts.
    std::vector<FloydCycleStep> steps; ///< Chronological pairs before their equality tests.
};

/// @brief Find an eventual period and entry by Brent's power-of-two block method.
/// The deterministic callback is borrowed and may close over a larger state model.
/// Reference indices are 0,1,3,7,..., with block lengths 1,2,4,8,... . After period
/// detection, align a second pointer lambda ahead and move both to recover mu.
/// O(mu+lambda) transition/equality work and O(1) auxiliary storage, excluding the
/// callback's own cost/storage. No trace allocation; only caller-supplied callback effects can mutate external state.
/// This legacy overload has no caller limit; arithmetic counters are checked.
/// @param x0 Initial uint64 state.
/// @param f Nonempty deterministic uint64-to-uint64 callback; exceptions propagate.
/// @return Least positive lambda and least nonnegative mu, with unchanged result layout.
/// @throws std::bad_function_call If f is empty.
/// @throws std::overflow_error If period, indices, block length or diagnostics exceed size_t.
/// @par Example
/// @code
/// const auto result=dm::algorithms::brent_cycle_detection(0,[](std::uint64_t x){return (x+1)%3;});
/// std::print("{} {}\n",result.mu,result.lambda); // 0 3
/// @endcode
/// @par References
/// - R. P. Brent (1980), An improved Monte Carlo factorization algorithm, BIT 20,
///   176-184, doi:10.1007/BF01933190; deterministic doubling blocks and entry recovery here.
/// @see brent_cycle_work(), brent_cycle_trace(), floyd_cycle_detection()
CycleResult brent_cycle_detection(std::uint64_t x0,const CycleFunction& f);

/// @brief Find an eventual cycle by Floyd's meeting, entry-location and period-measurement phases.
/// Preserves the legacy transition-call order, including the repeated initial f(x0).
/// Deterministic callbacks and ownership/mutation rules match Brent's overload.
/// O(mu+lambda) transition/equality work and O(1) auxiliary storage, excluding f.
/// No caller limit; counts are checked. Call counts alone are not elapsed-time claims.
/// @param x0 Initial state.
/// @param f Nonempty deterministic transition; callback exceptions propagate.
/// @return Least period lambda and cycle-entry index mu.
/// @throws std::bad_function_call If f is empty.
/// @throws std::overflow_error If period, entry index or diagnostics exceed size_t.
/// @par Example
/// @code
/// const auto result=dm::algorithms::floyd_cycle_detection(0,[](std::uint64_t x){return (x+1)%3;});
/// std::print("{} {}\n",result.lambda,result.mu); // 3 0
/// @endcode
/// @par References
/// - R. P. Brent (1980), doi:10.1007/BF01933190, discusses Floyd's detector as attributed by Knuth.
/// @see floyd_cycle_work(), floyd_cycle_trace(), brent_cycle_detection()
CycleResult floyd_cycle_detection(std::uint64_t x0,const CycleFunction& f);

/// @brief Return Brent's entry state and actual counts from the ordinary detection kernel.
/// Uses O(1) auxiliary storage, with no trace records or orbit replay. Costs and
/// deterministic-callback requirements match brent_cycle_detection(). Limits count
/// completed calls across every phase; exceptions do not undo callback effects.
/// @param x0 Initial state.
/// @param f Borrowed nonempty deterministic transition.
/// @param options Optional global evaluation limit; default zero is unlimited.
/// @return Owned cycle, entry and actual diagnostic counts.
/// @throws std::runtime_error If the positive evaluation budget is exhausted.
/// @throws std::bad_function_call If f is empty; callback exceptions propagate.
/// @throws std::overflow_error If arithmetic counters exceed size_t.
/// @par Example
/// @code
/// const auto work=dm::algorithms::brent_cycle_work(0,[](std::uint64_t x){return (x+1)%3;},{100});
/// std::print("{} {}\n",work.result.lambda,work.evaluations); // 3 9
/// @endcode
/// @par References
/// - R. P. Brent (1980), doi:10.1007/BF01933190.
/// @see brent_cycle_detection(), brent_cycle_trace(), CycleOptions
CycleWork brent_cycle_work(std::uint64_t x0,const CycleFunction& f,CycleOptions options={});

/// @brief Return Floyd's actual phase counts and entry without extra transition calls.
/// The shared ordinary kernel uses O(1) storage and O(mu+lambda) work; callback
/// lifetime, determinism, limits and mutation rules match brent_cycle_work().
/// @param x0 Initial state.
/// @param f Borrowed nonempty deterministic transition.
/// @param options Optional completed-call limit, default unlimited.
/// @return Owned cycle, entry and diagnostic counters.
/// @throws std::runtime_error If the positive evaluation limit is exhausted.
/// @throws std::bad_function_call If f is empty; callback exceptions propagate.
/// @throws std::overflow_error If counters exceed size_t.
/// @par Example
/// @code
/// const auto work=dm::algorithms::floyd_cycle_work(0,[](std::uint64_t x){return (x+1)%3;});
/// std::print("{}\n",work.evaluations); // 12
/// @endcode
/// @par References
/// - R. P. Brent (1980), doi:10.1007/BF01933190, comparison with Floyd's cycle finding.
/// @see floyd_cycle_detection(), CycleWork
CycleWork floyd_cycle_work(std::uint64_t x0,const CycleFunction& f,CycleOptions options={});

/// @brief Record actual Brent checkpoints and all three phases without changing callback calls.
/// Work/result match brent_cycle_work(); state recording adds O(mu+lambda) owned
/// storage. Each row is a movement, checkpoint or phase entry, not necessarily a
/// function call. In particular, checkpoint equality is followed by an advance.
/// @param x0 Initial state.
/// @param f Borrowed nonempty deterministic transition.
/// @param options Optional completed-call budget, default unlimited.
/// @return Owned work summary and chronological states, with no borrowed references.
/// @throws std::runtime_error If the positive evaluation limit is exhausted.
/// @throws std::bad_function_call If f is empty; callback exceptions propagate.
/// @throws std::overflow_error If counters exceed size_t; allocation errors propagate.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::brent_cycle_trace(0,[](std::uint64_t x){return (x+1)%3;});
/// std::print("{}\n",trace.steps.back().offset); // 0
/// @endcode
/// @par References
/// - R. P. Brent (1980), doi:10.1007/BF01933190; the ordinary kernel's actual doubling schedule is recorded.
/// @see brent_cycle_work(), BrentCycleTrace
BrentCycleTrace brent_cycle_trace(std::uint64_t x0,const CycleFunction& f,CycleOptions options={});

/// @brief Record actual Floyd pairs in all three phases without extra callback calls.
/// Shares the ordinary/work kernel and preserves the repeated initial f(x0).
/// O(mu+lambda) transition work and owned record storage, excluding callback cost.
/// Global evaluation limits use CycleOptions; max_records is a separate online
/// row cap, default one million, zero unlimited. Checks each next row's admission
/// before its movement/reset, including phase-entry rows. Complete trace size
/// cannot be known without traversing; this is not an upfront whole-orbit cap.
/// Allocation failures or callback/evaluation limits can occur after earlier
/// callback effects; those effects are not undone. No partial trace is returned.
/// Empty callbacks reject before any movement. Independent calls share no state.
/// @param x0 Initial uint64 state.
/// @param f Borrowed nonempty deterministic transition; exceptions propagate.
/// @param options Global successful-call budget; default zero is unlimited.
/// @param max_records Online row cap, default one million; zero disables only this cap.
/// @return Owned actual pairs and the ordinary detector's result/counts.
/// @throws std::bad_function_call Empty callback.
/// @throws std::runtime_error Positive evaluation budget exhausted before next call.
/// @throws std::length_error Next row would exceed cap or vector capacity.
/// @throws std::overflow_error Checked indices or counters cannot be represented.
/// @throws std::bad_alloc Record storage allocation fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::floyd_cycle_trace(0,[](std::uint64_t x){return (x+1)%3;},{12},100);
/// std::print("{} {}\n",t.work.result.mu,t.steps.back().progress); // 0 3
/// @endcode
/// @par References
/// - R. P. Brent (1980), An improved Monte Carlo factorization algorithm,
///   doi:10.1007/BF01933190, discusses Floyd's detector as attributed by Knuth.
/// @see floyd_cycle_detection(), floyd_cycle_work(), FloydCycleTrace
FloydCycleTrace floyd_cycle_trace(std::uint64_t x0,const CycleFunction& f,
    CycleOptions options={},std::size_t max_records=1000000);

/// @brief Compute the mathematical quadratic residue (state squared plus constant) modulo m.
/// Uses widened unsigned arithmetic, without uint64 square/add wrap. State and
/// constant need not already be residues; modulus one returns zero. O(1) machine
/// arithmetic, no allocation, no mutation/shared state. This is a transition,
/// not a factorization algorithm or a random-map guarantee.
/// @param state Any uint64 input state.
/// @param modulus Positive modulus; zero is invalid.
/// @param constant Any uint64 additive constant, default one.
/// @return Canonical residue in [0,modulus).
/// @throws std::invalid_argument Modulus is zero.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::quadratic_cycle_step(2,1009)); // 5
/// @endcode
/// @par References
/// - J. M. Pollard (1975), A Monte Carlo method for factorization,
///   BIT15,331-334, doi:10.1007/BF01933667; polynomial iteration context.
///   The report uses plus one rather than Pollard's illustrative minus one.
/// @see floyd_cycle_work(), brent_cycle_work()
std::uint64_t quadratic_cycle_step(std::uint64_t state,std::uint64_t modulus,
    std::uint64_t constant=1);

/// @brief Materialise count successive states x0,f(x0),..., making exactly count-1 calls for count>0.
/// Zero count returns empty, one returns only x0; these cases do not invoke f.
/// Costs O(count) work/storage; inputs/shared state are unchanged except for callback
/// effects. Returned values own their storage. This is an orbit renderer, not a detector.
/// @param x0 Initial state.
/// @param f Borrowed deterministic transition; need not be callable when count<=1.
/// @param count Number of states to retain, including x0.
/// @return Owned orbit prefix, without an extra call beyond its final stored state.
/// @throws std::bad_function_call If f is empty and count>1; callback exceptions propagate.
/// @throws std::length_error If count exceeds vector capacity; allocation errors propagate.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::iterate_states(0,[](std::uint64_t x){return (x+1)%3;},5)); // [0, 1, 2, 0, 1]
/// @endcode
/// @see brent_cycle_trace()
std::vector<std::uint64_t> iterate_states(std::uint64_t x0,const CycleFunction& f,std::size_t count);

/// @brief Build a labelled rho map with the requested tail and cycle lengths.
/// States 0..mu+lambda-1 advance consecutively; the last returns to mu. Starting
/// from zero therefore has precisely the supplied lengths. O(mu+lambda) work/storage.
/// Output owns its entries; no randomness or shared state is used.
/// @param mu Nonnegative tail length.
/// @param lambda Positive period.
/// @return Owned successor table indexed by state label.
/// @throws std::invalid_argument If lambda is zero.
/// @throws std::overflow_error If mu+lambda overflows size_t.
/// @throws std::length_error If the table exceeds vector capacity; allocation errors propagate.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::rho_successors(3,4)); // [1, 2, 3, 4, 5, 6, 3]
/// @endcode
/// @see brent_cycle_work()
std::vector<std::uint64_t> rho_successors(std::size_t mu,std::size_t lambda);

/// @brief Sum squares of decimal digits, with zero mapping to zero.
/// O(number of decimal digits) work and O(1) storage; every uint64 input maps to
/// at most 20*81=1620, so accumulation cannot overflow. No shared/input mutation.
/// @param value Any uint64 integer, including zero.
/// @return Sum of squared base-ten digits.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::sum_squared_decimal_digits(19)); // 82
/// @endcode
/// @par References
/// - OEIS A007770, https://oeis.org/A007770, decimal happy-number iteration.
/// @see is_happy_number()
std::uint64_t sum_squared_decimal_digits(std::uint64_t value) noexcept;

/// @brief Whether a positive integer reaches one under repeated decimal digit-square summation.
/// Uses the canonical Brent kernel's recovered entry; zero is not happy. No orbit
/// is stored and no caller/shared data are changed. The first transition is <=1620,
/// making the subsequent finite search independent of the magnitude of the input.
/// @param value Any uint64 integer; zero returns false.
/// @return True exactly when iteration reaches the fixed point one.
/// @par Example
/// @code
/// std::print("{} {}\n",dm::algorithms::is_happy_number(19),dm::algorithms::is_happy_number(4)); // true false
/// @endcode
/// @par References
/// - OEIS A007770, https://oeis.org/A007770, positive integers reaching one by this map.
/// @see sum_squared_decimal_digits(), brent_cycle_work()
bool is_happy_number(std::uint64_t value);
} // namespace datamunge::algorithms
