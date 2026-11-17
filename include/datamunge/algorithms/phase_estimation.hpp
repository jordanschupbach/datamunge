#pragma once
/// @file phase_estimation.hpp
/// @brief Deterministic counting-register phase estimation, retaining actual simulated states.
#include <cstddef>
#include <datamunge/algorithms/quantum_sim.hpp>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned modal outcome of a simulated counting register; no measurement is sampled.
/// Fields retain their historical names and order. Numerical ties follow the
/// actual first strict maximum found in computed binary64 probabilities.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::phase_estimation(3,.375).measured); // 3
/// @endcode
struct PhaseEstimationResult {
  double      estimate;    ///< Modal integer divided by 2^t, in [0,1).
  std::size_t measured;    ///< Deterministically selected modal basis index, not a random draw.
  double      probability; ///< Actual squared magnitude of that computed modal amplitude.
};
/// @brief Optional basis-size and retained-amplitude policies, independently disabled by zero.
/// @par Example
/// @code
/// const dm::algorithms::PhaseEstimationOptions o;std::print("{}\n",o.max_basis_states); // 1048576
/// @endcode
struct PhaseEstimationOptions {
  std::size_t max_basis_states{1048576}; ///< Allowed 2^t amplitude count before allocation.
  std::size_t max_trace_entries{
      1048576}; ///< Three retained complex states plus final probabilities; counts entries, not bytes.
};
/// @brief Owned actual simulator states captured at producing stage boundaries.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::phase_estimation_trace(3,.375).after_kickback.size()); // 8
/// @endcode
struct PhaseEstimationTrace {
  PhaseEstimationResult result{};          ///< Actual final deterministic argmax result.
  int                   qubits{0};         ///< Counting-register bit count, possibly zero.
  double                phase{0};          ///< Original finite phase in [0,1).
  QuantumState          after_hadamards;   ///< Actual state after applying all t Hadamards.
  QuantumState          after_kickback;    ///< Actual state after applying all t phase gates.
  QuantumState          after_inverse_qft; ///< Actual state after the native inverse QFT.
  std::vector<double>   distribution;      ///< Actual final squared magnitudes from that state.
  std::size_t hadamards{0}; ///< Actual preparation Hadamard calls, excluding inverse-QFT internals.
  std::size_t phase_gates{0}; ///< Actual phase-kickback gate calls.
};
/// @brief Simulate t counting qubits and return the deterministic modal phase estimate.
/// Prepares uniform amplitudes, applies phase 2*pi*phi*2^q to each bit q, then
/// calls the native inverse QFT. This models an already supplied eigenphase;
/// it constructs neither an unknown unitary nor its eigenstate. State-vector
/// memory is exponential; simulated gate work is O(t^2*2^t), not a classical
/// exponential speedup. No random measurement is performed. Borrows no state,
/// retains no references and uses no shared mutable state.
/// @param t Nonnegative counting qubits below size_t bit width; zero is a one-state register.
/// @param phi Finite eigenphase in [0,1); no implicit wrapping.
/// @return Modal index, index/2^t and actual modal probability.
/// @throws std::invalid_argument If t or phase is invalid.
/// @throws std::length_error If default basis policy or representable storage is exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::phase_estimation(3,.375).measured); // 3
/// @endcode
/// @par References
/// - A. Kitaev (1995), Quantum measurements and the Abelian Stabilizer Problem,
///   https://arxiv.org/abs/quant-ph/9511026 . This is a classical counting-register simulation.
PhaseEstimationResult phase_estimation(int t, double phi);
/// @brief Capture actual intermediate states and probabilities from the same producing simulation.
/// O(4*2^t) retained entries; no independent reconstruction populates the history.
/// @param t Valid nonnegative counting-register size.
/// @param phi Finite eigenphase in [0,1).
/// @param options Basis-state and retained-entry policies, admitted before allocation.
/// @return Owned stage states, actual preparation gate counts and modal result.
/// @throws std::invalid_argument If t or phase is invalid.
/// @throws std::length_error If basis/history policies or representable storage are exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::phase_estimation_trace(3,.375).distribution.size()); // 8
/// @endcode
PhaseEstimationTrace phase_estimation_trace(int t, double phi, PhaseEstimationOptions options = {});
} // namespace datamunge::algorithms
