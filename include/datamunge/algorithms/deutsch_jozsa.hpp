#pragma once
/// @file deutsch_jozsa.hpp
/// @brief Deutsch–Jozsa decisions, actual circuit states and query accounting.
/// One logical oracle application requires 2^n classical callback calls in this
/// binary64 simulator. Integer outputs retain the original low-bit f(x)&1 convention.
#include <datamunge/algorithms/quantum_sim.hpp>
#include <cstddef>
#include <functional>
#include <vector>
namespace datamunge::algorithms {
/// @brief Original two-field numerical Deutsch–Jozsa decision.
/// constant is the compatibility threshold p0>0.5, not a promise certificate.
/// Probabilities are unclipped and may slightly exceed one through rounding.
/// @par Example
/// @code
/// const auto r=dm::algorithms::deutsch_jozsa(2,[](std::size_t){return 1;});
/// std::print("{} {:.6f}\n",r.constant,r.prob_all_zero); // true 1.000000
/// @endcode
/// @see deutsch_jozsa(), DeutschJozsaTrace
struct DeutschJozsaResult {
  bool constant; ///< True exactly when computed p0 exceeds 0.5.
  double prob_all_zero; ///< Squared final all-zero amplitude, without clipping.
};
/// @brief Promise category determined from every observed effective bit.
/// Constant includes n=0; balanced requires exactly half ones in an even domain.
/// Neither explicitly identifies a table outside the promise.
/// @par Example
/// @code
/// const auto t=dm::algorithms::deutsch_jozsa_trace(2,[](std::size_t x){return x==3;});
/// std::print("{}\n",t.promise==dm::algorithms::DeutschJozsaPromise::neither); // true
/// @endcode
/// @see deutsch_jozsa_trace()
enum class DeutschJozsaPromise {
  constant, ///< All effective outputs equal.
  balanced, ///< Exactly half the effective outputs are one.
  neither ///< Neither promised category; the decision guarantee does not apply.
};
/// @brief Circuit oracle, callback, gate and amplitude-pair counts.
/// Excludes allocation, copying, probability extraction and callback internals.
/// A butterfly updates one pair per single-qubit Hadamard.
/// @par Example
/// @code
/// const auto t=dm::algorithms::deutsch_jozsa_trace(2,[](std::size_t x){return x&1;});
/// std::print("{} {} {} {}\n",t.work.oracle_applications,t.work.callback_calls,t.work.hadamard_gates,t.work.butterflies); // 1 4 4 8
/// @endcode
/// @see DeutschJozsaTrace
struct DeutschJozsaWork {
  std::size_t oracle_applications{0}; ///< Whole-register logical phase applications.
  std::size_t callback_calls{0}; ///< Actual ascending C++ oracle evaluations.
  std::size_t hadamard_gates{0}; ///< Actual single-qubit Hadamards: 2n on success.
  std::size_t butterflies{0}; ///< Actual amplitude-pair updates: n*2^n on success.
  std::size_t phase_flips{0}; ///< Actual amplitude sign flips; number of effective ones.
};
/// @brief Owned executing-circuit snapshots and observed promise evidence.
/// No replay or repeated oracle evaluation. Basis bit q is qubit q, with q=0
/// least significant. mean_phase is count-derived (zeros-ones)/2^n in binary64,
/// independently of gate accumulation. No shared mutable state; all arrays own
/// their data. Stateful callbacks describe an observed table, whereas a physical
/// Boolean oracle must describe a fixed function.
/// @par Example
/// @code
/// const auto t=dm::algorithms::deutsch_jozsa_trace(2,[](std::size_t x){return x&1;});
/// std::print("{} {} {:.6f}\n",t.truth_table,t.ones,t.mean_phase); // [0, 1, 0, 1] 2 0.000000
/// @endcode
/// @see deutsch_jozsa_trace(), walsh_spectrum()
struct DeutschJozsaTrace {
  DeutschJozsaResult result{false,0}; ///< Compatibility numerical result.
  int n{0}; ///< Input qubit count; zero means one basis input.
  std::vector<int> truth_table; ///< Actual f(x)&1, in ascending x order.
  QuantumState before_oracle; ///< Actual state after initial n Hadamards.
  QuantumState after_oracle; ///< Actual phase-marked state before final Hadamards.
  QuantumState final_state; ///< Actual state after final n Hadamards.
  std::size_t ones{0}; ///< Count of effective ones in truth_table.
  double mean_phase{0}; ///< Count-derived signed all-zero amplitude before squaring.
  DeutschJozsaPromise promise{DeutschJozsaPromise::neither}; ///< Exact table-count category.
  DeutschJozsaWork work; ///< Logical and classical simulator work kept distinct.
};
/// @brief Simulate one phase-oracle query, returning the original result type.
/// Applies n initial Hadamards, f(x)&1 once per ascending x, then n final Hadamards.
/// No promise verification or random measurement is performed. n=0 calls f(0)
/// once and returns constant. O(2^n) storage; O((n+1)*2^n) work plus callback cost.
/// Exceptions do not undo callback side effects. Separate calls share only f.
/// @param n Nonnegative qubits below size_t's value-bit count; state must fit memory/capacity.
/// @param f Nonempty callable; effective value f(x)&1, including non-Boolean integer returns.
/// @return Unclipped squared zero amplitude and its original >0.5 decision.
/// @throws std::invalid_argument For invalid n or empty f, before calls.
/// @throws std::length_error For vector capacity overflow, before allocation/calls.
/// @throws std::bad_alloc For allocation failure. Callback exceptions propagate.
/// @par Example
/// @code
/// const auto r=dm::algorithms::deutsch_jozsa(4,[](std::size_t x){return (x>>3)&1;});
/// std::print("{} {:.6f}\n",r.constant,r.prob_all_zero); // false 0.000000
/// @endcode
/// @par References
/// - D. Deutsch and R. Jozsa (1992), Rapid solution of problems by quantum computation, doi:10.1098/rspa.1992.0167.
/// - R. Cleve et al. (1998), Quantum algorithms revisited, doi:10.1098/rspa.1998.0164; https://arxiv.org/abs/quant-ph/9708016 (one-query phase-kickback form).
/// @see deutsch_jozsa_trace(), deutsch_jozsa_classical(), fourier_sampling_state()
DeutschJozsaResult deutsch_jozsa(int n,const std::function<int(std::size_t)>& f);
/// @brief Record actual gate-boundary states, promise counts and simulator work.
/// Same kernel and gate order as deutsch_jozsa(); no replay. Stores three complex
/// arrays and one integer table, each of length 2^n: O(2^n) total storage.
/// Complete state-count cap is checked before allocation/calls. Zero disables
/// only this cap; dimension, capacity and counter-overflow guards remain.
/// Exceptions return no partial trace; earlier callback side effects remain.
/// @param n Qubits with deutsch_jozsa()'s dimension constraints.
/// @param f Nonempty callable, evaluated exactly 2^n times on success; low bit used.
/// @param max_basis_states Complete length cap, default 1048576; zero disables.
/// @return Owned actual snapshots, exact observed promise category and counts.
/// @throws std::invalid_argument For invalid n or empty f, before cap checks.
/// @throws std::length_error For cap, capacity or n*2^n counter overflow, before calls.
/// @throws std::bad_alloc For allocation failure. Callback exceptions propagate.
/// @par Example
/// @code
/// const auto t=dm::algorithms::deutsch_jozsa_trace(2,[](std::size_t x){return x==3;},4);
/// std::print("{} {:.6f} {:.6f}\n",t.ones,t.mean_phase,t.result.prob_all_zero); // 1 0.500000 0.250000
/// @endcode
/// @par References
/// - R. Cleve et al. (1998), Quantum algorithms revisited, doi:10.1098/rspa.1998.0164.
/// @see deutsch_jozsa(), DeutschJozsaTrace, walsh_spectrum()
DeutschJozsaTrace deutsch_jozsa_trace(int n,const std::function<int(std::size_t)>& f,std::size_t max_basis_states=1048576);
/// @brief Sequential classical promised decision and actual callback count.
/// Agreement of 2^(n-1)+1 values implies constant only under the promise.
/// n=0 needs one observation. The full table is not certified.
/// @par Example
/// @code
/// const auto r=dm::algorithms::deutsch_jozsa_classical(4,[](std::size_t x){return (x>>3)&1;});
/// std::print("{} {}\n",r.constant,r.queries); // false 9
/// @endcode
/// @see deutsch_jozsa_classical()
struct DeutschJozsaClassicalResult {
  bool constant{false}; ///< Promised decision; not an arbitrary-table certificate.
  std::size_t queries{0}; ///< Actual evaluations before stopping.
};
/// @brief Decide the promised problem by querying distinct ascending inputs.
/// Stops at the first differing low bit or 2^(n-1)+1 agreeing outputs. n=0
/// queries its single input once. O(1) storage, work proportional to queries
/// plus callback cost. No promise verification or random sampling occurs.
/// @param n Nonnegative qubits below size_t's value-bit count.
/// @param f Nonempty callable; effective f(x)&1, never repeated for one input.
/// @return Promised decision and actual calls, at most 2^(n-1)+1 for n>0.
/// @throws std::invalid_argument For invalid n or empty f, before calls.
/// Callback exceptions propagate without undoing earlier callback effects.
/// @par Example
/// @code
/// const auto r=dm::algorithms::deutsch_jozsa_classical(4,[](std::size_t x){return x&1;});
/// std::print("{} {}\n",r.constant,r.queries); // false 2
/// @endcode
/// @par References
/// - D. Deutsch and R. Jozsa (1992), Rapid solution of problems by quantum computation, doi:10.1098/rspa.1992.0167 (deterministic classical query separation).
/// @see deutsch_jozsa(), DeutschJozsaClassicalResult
DeutschJozsaClassicalResult deutsch_jozsa_classical(int n,const std::function<int(std::size_t)>& f);
} // namespace datamunge::algorithms
