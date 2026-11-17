#pragma once
/// @file grover.hpp
/// @brief Checked floating state-vector Grover iterations, actual phase snapshots and selected
/// work.
#include <datamunge/algorithms/quantum_sim.hpp>

namespace datamunge::algorithms {
/// @brief Legacy deterministic summary, without sampling or state collapse.
/// measured retains its historical name but is an argmax; success_prob is the
/// actual squared target amplitude, unclamped and not renormalized. Roundoff can
/// make the state norm differ from one. No quantum device or hidden-target search
/// is performed: the simulator is given the marked index.
/// @par Example
/// @code
/// const auto r=dm::algorithms::grover(6,42);
/// std::print("{} {}\n",r.measured,r.iterations); // 42 6
/// @endcode
/// @see grover(), GroverRun
struct GroverResult {
  std::size_t measured;     ///< First maximum-probability basis index, not a sampled outcome.
  double      success_prob; ///< Actual std::norm of the marked amplitude.
  int         iterations;   ///< Actual complete oracle/diffusion iterations applied.
};
/// @brief Complete state and full-phase-capture caps, each zero disabling its policy limit.
/// State units are complex amplitudes, not bytes. Capture counts the initial N
/// amplitudes and 2N+1 per iteration (oracle state, diffused state and oracle mean).
/// Metadata, ordinary final state, capacities and allocation overhead are excluded.
/// @par Example
/// @code
/// const dm::algorithms::GroverOptions o;
/// std::print("{}\n",o.max_amplitudes); // 1048576
/// @endcode
/// @see grover_trace()
struct GroverOptions {
  std::size_t max_amplitudes{1048576}; ///< Complete state cap, default2^20; zero disables.
  std::size_t max_trace_entries{
      1000000}; ///< Complete complex-value capture cap; zero disables; ignored without capture.
};
/// @brief Selected actual state-operator visits, excluding snapshots and probability extraction.
/// One oracle call is one known-target phase flip in this simulator. It is not a
/// gate synthesis, physical device timing or a classical black-box query count.
/// Hadamard pairs count the actual two-amplitude gate pairs, not scalar FLOPs.
/// @par Example
/// @code
/// const auto r=dm::algorithms::grover_run(2,3,1);
/// std::print("{} {}\n",r.work.oracle_calls,r.work.mean_terms); // 1 4
/// @endcode
/// @see GroverRun
struct GroverWork {
  std::size_t initialized_slots{0}; ///< Actual ground-state amplitude slots initialized.
  std::size_t hadamard_gates{0};    ///< Actual h_gate calls during uniform preparation.
  std::size_t hadamard_pairs{0};    ///< Actual two-amplitude pairs across preparation gates.
  std::size_t oracle_calls{0};      ///< Actual target phase flips, one per complete iteration.
  std::size_t mean_terms{0};        ///< Actual complex amplitudes summed after phase flips.
  std::size_t diffusion_updates{0}; ///< Actual amplitude inversion updates.
  std::size_t argmax_terms{0}; ///< Actual probability candidates visited in final most_probable().
};
/// @brief Owned final native state, deterministic summary and selected producing work.
/// State preparation uses the existing Hadamard gate arithmetic; iterations use
/// actual complex-vector reflection loops. No analytic probability substitution,
/// normalization, probability clamping, sampling or input references.
/// @par Example
/// @code
/// const auto r=dm::algorithms::grover_run(2,3,1);
/// std::print("{} {}\n",r.state.size(),r.result.measured); // 4 3
/// @endcode
/// @see grover_run()
struct GroverRun {
  GroverResult result{}; ///< Actual final summary, including first argmax.
  QuantumState state;    ///< Owned actual final amplitudes.
  GroverWork   work;     ///< Selected actual state-operator work.
};
/// @brief Actual oracle and diffusion states captured during one iteration.
/// before is the preceding step's after_diffusion, or trace.initial on step one.
/// oracle_mean is the actual sequential complex mean used by diffusion. Each
/// vector owns storage; no second execution reconstructs these phase states.
/// @par Example
/// @code
/// const auto t=dm::algorithms::grover_trace(2,3,1);
/// std::print("{}\n",t.steps[0].after_oracle.size()); // 4
/// @endcode
/// @see GroverTrace
struct GroverStep {
  int          iteration{0};    ///< One-based producing iteration.
  QuantumState after_oracle;    ///< Actual state immediately after the target sign flip.
  Complex      oracle_mean{};   ///< Actual mean after the oracle, before diffusion.
  QuantumState after_diffusion; ///< Actual complete state after inversion about mean.
  double marked_probability{0}; ///< Actual squared target amplitude after diffusion, unclamped.
};
/// @brief Owned initial state, actual full-phase snapshots and final producing result.
/// Known complete requirement N+K(2N+1) is preflighted before state allocation or
/// gates. No truncation, replay, partial result or retained references. Capture
/// adds O(KN) owned complex values to ordinary O(N) working/output storage; units
/// exclude metadata, final ordinary output, capacity and peak RAM. Copying is deep.
/// @par Example
/// @code
/// const auto t=dm::algorithms::grover_trace(2,3,1);
/// std::print("{}\n",t.captured_entries); // 13
/// @endcode
/// @see GroverOptions, grover_trace()
struct GroverTrace {
  GroverRun               run;     ///< Actual producing final state/result/work.
  QuantumState            initial; ///< Actual post-Hadamard preparation, before any oracle.
  std::vector<GroverStep> steps;   ///< Actual chronological complete phase records.
  std::size_t captured_entries{0}; ///< Complete retained complex-value units, not bytes.
};
/// @brief Internal single-step compatibility interface.
namespace detail {
/// @brief Apply one checked phase flip then actual sequential inversion about the mean.
/// Accepts any nonempty finite complex vector, not necessarily normalized or a
/// power-of-two state. Checks input before mutation; arithmetic failure can leave
/// the oracle or a partial diffusion applied. Same original arithmetic order.
/// @param s Borrowed mutable amplitudes, finite and nonempty.
/// @param target Valid marked index into s.
/// @throws std::invalid_argument On empty/nonfinite state or invalid target, before mutation.
/// @throws std::overflow_error If mean or updated amplitudes become nonfinite.
/// @see grover_run()
void grover_step(QuantumState& s, std::size_t target);
} // namespace detail
/// @brief Run the preserved legacy round(pi/4*sqrt(2^n)) iteration schedule.
/// This schedule is an approximation, not an optimal integer stopping rule: for
/// n=2 it uses two iterations while one succeeds with near-unit probability.
/// measured is a first argmax, not measurement. Preserves ordinary legacy output
/// and arithmetic; validates n/target before shifts/allocation. n=0 is supported.
/// No optional state/history policy caps; container limits still apply. Floating
/// state-vector work O(nN+KN), storage O(N); these are classical simulation costs.
/// @param n Qubit count, nonnegative and below size_t bit width, with N container-representable.
/// @param target Marked basis index, less than N=2^n.
/// @return Actual legacy summary, without random sampling.
/// @throws std::invalid_argument On invalid n or target.
/// @throws std::length_error On state/count/container limits.
/// @throws std::overflow_error On nonfinite arithmetic.
/// @throws std::bad_alloc On state allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::grover(8,42).iterations); // 13
/// @endcode
/// @par References
/// - L. K. Grover (1996), A fast quantum mechanical algorithm for database search,
///   https://arxiv.org/abs/quant-ph/9605043 . This is a floating classical simulation.
/// @see grover_run(), grover_first_peak_iterations()
GroverResult grover(int n, std::size_t target);
/// @brief Execute a requested nonnegative count through the shared native state-vector kernel.
/// Uniform preparation uses qstate_zero and h_gate, then actual oracle/diffusion
/// loops. No analytic substitutions or sampling. Complete input/capture geometry
/// validation precedes state allocation. Trace cap is ignored without capture.
/// @param n Nonnegative qubit count with representable state size.
/// @param target Marked index below 2^n.
/// @param iterations Nonnegative complete iteration count, zero permitted.
/// @param options Complete amplitude cap, default2^20; zero disables.
/// @return Owned actual final state, deterministic summary and selected work.
/// @throws std::invalid_argument On invalid n/target/iterations, before policy caps.
/// @throws std::length_error On state cap, counts or container limits.
/// @throws std::overflow_error On nonfinite arithmetic.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto r=dm::algorithms::grover_run(2,3,0);
/// std::print("{} {}\n",r.result.iterations,r.work.oracle_calls); // 0 0
/// @endcode
/// @see GroverRun, grover_trace()
GroverRun grover_run(int n, std::size_t target, int iterations, GroverOptions options = {});
/// @brief Capture the shared execution's initial, oracle and diffusion states without replay.
/// Complete known complex-value requirement follows GroverTrace. Preflight checks
/// policy/container/count limits before allocating states or invoking gates.
/// Operator work equals the untraced run; probability extraction/copies are excluded.
/// @param n Valid nonnegative qubit count.
/// @param target Marked index below 2^n.
/// @param iterations Nonnegative complete iteration count.
/// @param options State and complete capture caps, default2^20/1000000; zero disables either.
/// @return Owned actual full-phase states, producing result and selected work.
/// @throws std::invalid_argument On invalid n/target/iterations before caps.
/// @throws std::length_error On state/capture/count/container limits.
/// @throws std::overflow_error On nonfinite arithmetic.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::grover_trace(2,3,1,{4,13});
/// std::print("{}\n",t.steps.size()); // 1
/// @endcode
/// @see GroverTrace, grover_run()
GroverTrace grover_trace(int n, std::size_t target, int iterations, GroverOptions options = {});
/// @brief Return actual target probabilities at steps zero through max_iters, without full
/// snapshots. Shared kernel arithmetic preserves legacy outputs. Negative max_iters now throws
/// rather than silently returning only step zero. No sampling, clamping or normalization. O(nN+KN)
/// work and O(N+K) working/output storage; no optional policy caps, but checked containers/counts.
/// n=0 is valid.
/// @param n Nonnegative representable qubit count.
/// @param target Marked basis index below 2^n.
/// @param max_iters Nonnegative complete iteration count.
/// @return Owned actual squared target amplitudes, length max_iters+1.
/// @throws std::invalid_argument On invalid n/target/max_iters before allocation.
/// @throws std::length_error On counts/container limits.
/// @throws std::overflow_error On nonfinite arithmetic.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::grover_success_trace(2,3,2).size()); // 3
/// @endcode
/// @see grover_trace()
std::vector<double> grover_success_trace(int n, std::size_t target, int max_iters);
/// @brief Estimate the first-lobe integer stopping point from the single-marked-state rotation
/// model. Uses floor(pi/(4*asin(1/sqrt(N)))); returns zero for n=0 and n=1, where ideal success is
/// constant across integer iterations. O(1) scalar theory calculation, not a native run, exhaustive
/// window maximization or global-time optimum claim.
/// @param n Nonnegative qubit count with container-representable N.
/// @return Estimated first-lobe maximizing nonnegative integer iteration count.
/// @throws std::invalid_argument On invalid n.
/// @throws std::length_error On state/count representability limits.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::grover_first_peak_iterations(8)); // 12
/// @endcode
/// @par References
/// - Grover (1996), https://arxiv.org/abs/quant-ph/9605043 .
/// @see grover(), grover_run()
int grover_first_peak_iterations(int n);
} // namespace datamunge::algorithms
