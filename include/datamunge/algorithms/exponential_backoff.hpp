#pragma once
/// @file exponential_backoff.hpp
/// @brief Checked finite capped windows, explicit/seeded jitter and owned cumulative idle-wait schedules.
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {

/// @brief One actual checked retry window, applied unit variate and finite delay.
/// Attempt k is the zero-based wait before retry k+1. Parameters base/cap use one
/// caller-chosen common time unit, must be finite and nonnegative, and can be zero.
/// The window is min(base*2^k,cap) mathematically, including immediate clamping when
/// base>=cap. Exponent/mantissa comparison precedes ldexp, so no intermediate
/// infinity or retry-index loop is needed. Scalar work uses a fixed number of math
/// operations even at INT_MAX; this is not a CPU-time or real-clock guarantee.
/// u is finite in [0,1], including the legacy endpoint one. delay=u*window is
/// finite in [0,window]; underflow/rounding can erase tiny waits or round them to a
/// window endpoint. capped/scaled describe the actual scalar branch, not instruction
/// counts. No sleeping, networking, random draws or optional history is involved.
///
///
/// @par Example
/// @code
/// const auto s=dm::algorithms::exponential_backoff_step(3,1,5,.5);
/// std::print("{:.1f} {:.1f} {} {}\n",s.window,s.delay,s.capped,s.scaled); // 5.0 2.5 true false
/// @endcode
///
/// @par References
/// - Marc Brooker (2015), Exponential Backoff And Jitter, AWS Architecture Blog. https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/. This API computes numeric windows/waits only, not an RPC retry policy or TCP timer.
///
/// @see exponential_backoff_step(), BackoffSchedule
struct BackoffStep {
    int attempt=0; ///< Zero-based retry-wait index.
    double window=0; ///< Actual checked capped maximum wait, in caller time units.
    double u=1; ///< Actual caller/seed-stream unit variate; one denotes a full window.
    double delay=0; ///< Actual finite product u*window, before cumulative summation.
    bool capped=false; ///< True when the scalar kernel returned the cap branch.
    bool scaled=false; ///< True only when the scalar kernel actually used ldexp for positive k.
};

/// @brief Owned actual steps and cumulative idle waits, excluding request durations.
/// steps and retry_times have equal length N; retry_times[i] is the ordinary
/// floating sum of delays through i, starting at zero. They are idle-wait offsets,
/// not wall-clock deadlines or a simulation of request success/latency. Zero waits
/// can give equal offsets. Addition rejects delay>DBL_MAX-current before summing;
/// thus no cumulative infinity escapes, though ordinary rounding still applies.
/// total_delay is the final offset, or zero when empty. Intrinsic output owns O(N)
/// records/values; no optional replay/history quota is imposed. Caller data is copied
/// as scalars and independent results share no mutable RNG state.
///
///
/// @par Example
/// @code
/// const auto s=dm::algorithms::exponential_backoff_schedule(3,1,5,{.25,.5,.75});
/// std::print("{} {:.2f} {}\n",s.steps.size(),s.total_delay,s.jittered); // 3 4.25 true
/// @endcode
///
/// @par References
/// - Marc Brooker (2015), Exponential Backoff And Jitter, AWS Architecture Blog. https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/. This API computes numeric windows/waits only, not an RPC retry policy or TCP timer.
///
/// @see exponential_backoff_schedule(), exponential_backoff_seeded()
struct BackoffSchedule {
    std::vector<BackoffStep> steps; ///< Owned actual checked windows/uniforms/delays and branch flags.
    std::vector<double> retry_times; ///< Owned cumulative idle waits, one per emitted step.
    double total_delay=0; ///< Final cumulative idle wait, zero for no steps.
    bool jittered=false; ///< True for explicit nonempty uniforms or a seeded schedule, even seeded empty input.
};

/// @brief Compute one finite capped backoff window and apply an explicit unit variate.
/// Attempt k is the zero-based wait before retry k+1. Parameters base/cap use one
/// caller-chosen common time unit, must be finite and nonnegative, and can be zero.
/// The window is min(base*2^k,cap) mathematically, including immediate clamping when
/// base>=cap. Exponent/mantissa comparison precedes ldexp, so no intermediate
/// infinity or retry-index loop is needed. Scalar work uses a fixed number of math
/// operations even at INT_MAX; this is not a CPU-time or real-clock guarantee.
/// u is a caller-supplied finite number in inclusive [0,1]; one is retained for
/// legacy endpoint callers, not a uniform-generator output. The function does not
/// sample a distribution. Calls are pure, own scalars and share no mutable state.
///
/// @param k Nonnegative zero-based wait index.
/// @param base Finite nonnegative initial window, in caller time units.
/// @param cap Finite nonnegative maximum window, in the same units.
/// @param u Finite applied variate in [0,1], default one for a full-window delay.
/// @return Owned actual window, variate, product and scalar branch flags.
/// @throws std::invalid_argument On negative attempt/count, nonfinite/negative base/cap, or invalid supplied uniforms.
///
/// @par Example
/// @code
/// const auto s=dm::algorithms::exponential_backoff_step(2,1,30,.75);
/// std::print("{} {:.0f} {:.0f} {}\n",s.attempt,s.window,s.delay,s.scaled); // 2 4 3 true
/// @endcode
///
/// @par References
/// - Marc Brooker (2015), Exponential Backoff And Jitter, AWS Architecture Blog. https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/. This API computes numeric windows/waits only, not an RPC retry policy or TCP timer.
///
/// @see BackoffStep, exponential_backoff_delay(), exponential_backoff_jittered()
BackoffStep exponential_backoff_step(int k,double base,double cap,double u=1);

/// @brief Return N pure capped windows without requiring a cumulative timeline.
/// Attempt k is the zero-based wait before retry k+1. Parameters base/cap use one
/// caller-chosen common time unit, must be finite and nonnegative, and can be zero.
/// The window is min(base*2^k,cap) mathematically, including immediate clamping when
/// base>=cap. Exponent/mantissa comparison precedes ldexp, so no intermediate
/// infinity or retry-index loop is needed. Scalar work uses a fixed number of math
/// operations even at INT_MAX; this is not a CPU-time or real-clock guarantee.
/// Validates count/base/cap even for empty output. Uses the shared scalar kernel
/// once per output, O(N) work/owned storage. Does not sum windows, apply jitter,
/// retain step records or fail merely because a hypothetical sum would overflow.
/// Borrowed/global state is not mutated; independent calls are safe.
///
/// @param n Nonnegative number of windows, indices zero through n-1.
/// @param base Finite nonnegative initial window.
/// @param cap Finite nonnegative maximum window.
/// @return Owned N finite windows; empty for n=0.
/// @throws std::invalid_argument On negative n or nonfinite/negative base/cap.
/// @throws std::length_error On output-container geometry limits.
/// @throws std::bad_alloc If owned output storage cannot be allocated.
///
/// @par Example
/// @code
/// const auto windows=dm::algorithms::exponential_backoff_windows(4,1,5);
/// std::print("{:.0f} {:.0f}\n",windows[2],windows[3]); // 4 5
/// @endcode
///
/// @par References
/// - Marc Brooker (2015), Exponential Backoff And Jitter, AWS Architecture Blog. https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/. This API computes numeric windows/waits only, not an RPC retry policy or TCP timer.
///
/// @see exponential_backoff_sequence(), exponential_backoff_schedule()
std::vector<double> exponential_backoff_windows(int n,double base,double cap);

/// @brief Emit actual windows/delays and their checked cumulative idle waits.
/// Attempt k is the zero-based wait before retry k+1. Parameters base/cap use one
/// caller-chosen common time unit, must be finite and nonnegative, and can be zero.
/// The window is min(base*2^k,cap) mathematically, including immediate clamping when
/// base>=cap. Exponent/mantissa comparison precedes ldexp, so no intermediate
/// infinity or retry-index loop is needed. Scalar work uses a fixed number of math
/// operations even at INT_MAX; this is not a CPU-time or real-clock guarantee.
/// Empty uniforms selects deterministic full windows (u=1). Otherwise exactly N
/// finite values in [0,1] are required and all are checked before output allocation.
/// The shared emitter creates each record and prefix offset once, without replay.
/// Known vector geometry is checked before reserve. O(N) work/owned storage; no
/// sleeping, callbacks, request durations, clock operations or random draws.
/// Cumulative overflow throws without exposing a partial result; ordinary windows
/// remain available independently when no representable timeline exists.
///
/// @param n Nonnegative number of retry waits.
/// @param base Finite nonnegative initial window.
/// @param cap Finite nonnegative maximum window.
/// @param uniforms Empty by default for full windows, or exactly N explicit finite [0,1] variates; borrowed unchanged.
/// @return Owned actual records, cumulative idle waits and final total.
/// @throws std::invalid_argument On negative attempt/count, nonfinite/negative base/cap, or invalid supplied uniforms.
/// @throws std::length_error On output-container geometry limits.
/// @throws std::bad_alloc If owned output storage cannot be allocated.
/// @throws std::overflow_error If a delay exceeds the remaining finite cumulative capacity.
///
/// @par Example
/// @code
/// const auto s=dm::algorithms::exponential_backoff_schedule(4,1,5,{.25,.5,.75,1});
/// std::print("{:.2f} {:.2f}\n",s.retry_times[1],s.total_delay); // 1.25 9.25
/// @endcode
///
/// @par References
/// - Marc Brooker (2015), Exponential Backoff And Jitter, AWS Architecture Blog. https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/. This API computes numeric windows/waits only, not an RPC retry policy or TCP timer.
///
/// @see BackoffSchedule, exponential_backoff_seeded()
BackoffSchedule exponential_backoff_schedule(int n,double base,double cap,const std::vector<double>& uniforms={});

/// @brief Emit a reproducible jitter schedule from the original local xorshift64 stream.
/// Attempt k is the zero-based wait before retry k+1. Parameters base/cap use one
/// caller-chosen common time unit, must be finite and nonnegative, and can be zero.
/// The window is min(base*2^k,cap) mathematically, including immediate clamping when
/// base>=cap. Exponent/mantissa comparison precedes ldexp, so no intermediate
/// infinity or retry-index loop is needed. Scalar work uses a fixed number of math
/// operations even at INT_MAX; this is not a CPU-time or real-clock guarantee.
/// Starting from a nonzero uint64 seed, each actual step applies x^=x<<13,
/// x^=x>>7, x^=x<<17 modulo2^64, then u=(x>>11)*2^-53. This preserves the
/// original report stream. It is deterministic, noncryptographic and not a proof
/// of independence across clients. Reusing a seed reproduces the same waits;
/// use a suitable seeding source, such as random::SplitMix64, for distinct fixtures.
/// Unit outputs lie in [0,1); floating delay products can round to an endpoint.
/// The shared emitter records the actual draw/window/wait and cumulative offset.
/// All scalar/seed/geometry validation precedes allocation or draws. O(N) time and
/// owned storage, no global RNG, borrowed state, optional history or partial result.
///
/// @param n Nonnegative number of emitted retry waits.
/// @param base Finite nonnegative initial window.
/// @param cap Finite nonnegative maximum window.
/// @param seed Nonzero local uint64 xorshift state, default88172645463325252 from the original report.
/// @return Owned actual jittered records and cumulative idle-wait offsets; jittered=true even when empty.
/// @throws std::invalid_argument On invalid count/base/cap or zero seed.
/// @throws std::length_error On output-container geometry limits.
/// @throws std::bad_alloc If owned output storage cannot be allocated.
/// @throws std::overflow_error If a delay exceeds remaining finite cumulative capacity.
///
/// @par Example
/// @code
/// const auto s=dm::algorithms::exponential_backoff_seeded(8,1,30);
/// std::print("{:.6f} {:.6f}\n",s.steps[0].delay,s.steps[1].delay); // 0.474259 0.329695
/// @endcode
///
/// @par References
/// - Marc Brooker (2015), Exponential Backoff And Jitter, AWS Architecture Blog. https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/. This API computes numeric windows/waits only, not an RPC retry policy or TCP timer.
///
/// @see exponential_backoff_schedule(), random::SplitMix64
BackoffSchedule exponential_backoff_seeded(int n,double base,double cap,std::uint64_t seed=88172645463325252ULL);

/// @brief Compute the original delay API through the checked constant-operation kernel.
/// Attempt k is the zero-based wait before retry k+1. Parameters base/cap use one
/// caller-chosen common time unit, must be finite and nonnegative, and can be zero.
/// The window is min(base*2^k,cap) mathematically, including immediate clamping when
/// base>=cap. Exponent/mantissa comparison precedes ldexp, so no intermediate
/// infinity or retry-index loop is needed. Scalar work uses a fixed number of math
/// operations even at INT_MAX; this is not a CPU-time or real-clock guarantee.
/// Retains the original inline name/signature. Returns a window only, with no
/// randomness or cumulative idle sum; separate calls share no mutable state.
///
/// @param k Nonnegative zero-based retry-wait index.
/// @param base Finite nonnegative initial window.
/// @param cap Finite nonnegative maximum window.
/// @return Finite min(base*2^k,cap), in caller time units.
/// @throws std::invalid_argument On negative k or nonfinite/negative base/cap.
///
/// @par Example
/// @code
/// std::print("{:.0f} {:.0f}\n",dm::algorithms::exponential_backoff_delay(3,1,30),dm::algorithms::exponential_backoff_delay(7,1,30)); // 8 30
/// @endcode
///
/// @par References
/// - Marc Brooker (2015), Exponential Backoff And Jitter, AWS Architecture Blog. https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/. This API computes numeric windows/waits only, not an RPC retry policy or TCP timer.
///
/// @see exponential_backoff_step(), exponential_backoff_sequence()
inline double exponential_backoff_delay(int k,double base,double cap){return exponential_backoff_step(k,base,cap).window;}

/// @brief Return the original sequence API with O(N) shared-kernel work.
/// Validates nonnegative count and finite nonnegative base/cap even for an empty
/// sequence. Full windows, including zero, are supported. Does not compute prefix
/// sums or require their representability. Output owns its N values; no global
/// state/history or caller mutation. Retains original inline name/signature.
///
/// @param n Nonnegative number of windows.
/// @param base Finite nonnegative initial window.
/// @param cap Finite nonnegative maximum window.
/// @return Owned N windows for zero-based indices; empty when n=0.
/// @throws std::invalid_argument On negative n or nonfinite/negative base/cap.
/// @throws std::length_error On output-container geometry limits.
/// @throws std::bad_alloc If owned output storage cannot be allocated.
///
/// @par Example
/// @code
/// const auto s=dm::algorithms::exponential_backoff_sequence(8,1,30);
/// std::print("{} {:.0f}\n",s.size(),s.back()); // 8 30
/// @endcode
///
/// @par References
/// - Marc Brooker (2015), Exponential Backoff And Jitter, AWS Architecture Blog. https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/. This API computes numeric windows/waits only, not an RPC retry policy or TCP timer.
///
/// @see exponential_backoff_windows(), exponential_backoff_schedule()
inline std::vector<double> exponential_backoff_sequence(int n,double base,double cap){return exponential_backoff_windows(n,base,cap);}

/// @brief Apply explicit full-window jitter through the original inline API.
/// Returns u*min(base*2^k,cap) with checked finite nonnegative base/cap and k>=0.
/// u is finite in inclusive [0,1], preserving the originally tested u=1 endpoint.
/// The caller provides randomness; a single supplied u does not establish a uniform
/// law. Product rounding/underflow follows double arithmetic; output stays within
/// [0,window]. Pure scalar output, no sleeping, cumulative sum or global RNG.
///
/// @param k Nonnegative zero-based retry-wait index.
/// @param base Finite nonnegative initial window.
/// @param cap Finite nonnegative maximum window.
/// @param u Finite caller variate in inclusive [0,1], including both endpoints.
/// @return Finite checked jittered delay, in caller time units.
/// @throws std::invalid_argument On negative attempt/count, nonfinite/negative base/cap, or invalid supplied uniforms.
///
/// @par Example
/// @code
/// std::print("{:.1f} {:.1f}\n",dm::algorithms::exponential_backoff_jittered(3,1,30,.5),dm::algorithms::exponential_backoff_jittered(3,1,30,1)); // 4.0 8.0
/// @endcode
///
/// @par References
/// - Marc Brooker (2015), Exponential Backoff And Jitter, AWS Architecture Blog. https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/. This API computes numeric windows/waits only, not an RPC retry policy or TCP timer.
///
/// @see exponential_backoff_step(), exponential_backoff_seeded()
inline double exponential_backoff_jittered(int k,double base,double cap,double u){return exponential_backoff_step(k,base,cap,u).delay;}
} // namespace datamunge::algorithms
