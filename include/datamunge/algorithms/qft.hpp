#pragma once
/// @file qft.hpp
/// @brief Checked native Fourier gate networks with actual state histories and classical simulation limits.
#include <datamunge/algorithms/quantum_sim.hpp>
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual Fourier-network gate kind.
/// The same native simulator gate calls produce ordinary and captured transforms.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::qft_trace(dm::algorithms::qstate_zero(1),1).gates[0].kind==dm::algorithms::FourierGateKind::Hadamard); // true
/// @endcode
enum class FourierGateKind {
  Hadamard,        ///< Actual single-qubit Hadamard call.
  ControlledPhase, ///< Actual controlled complex phase call.
  Swap             ///< Actual bit-order swap call.
};
/// @brief Classical state-vector and retained-entry policies.
/// Zero independently disables either cap; coordinate counts are not peak-byte budgets.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::QFTOptions{}.max_basis_states); // 1048576
/// @endcode
struct QFTOptions {
  std::size_t max_basis_states{1048576}; ///< Maximum actual2^n amplitudes admitted before copying.
  std::size_t max_trace_entries{
      1000000}; ///< Initial/final amplitudes plus gate records and each captured post-gate amplitude.
};
/// @brief One actual gate call and its immediately resulting complete state.
/// For Hadamard the control equals target and angle is zero; for Swap control/target are the swapped bits.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::qft_trace(dm::algorithms::qstate_zero(1),1).gates[0].after.size()); // 2
/// @endcode
struct FourierGate {
  FourierGateKind kind;    ///< Actual native gate kind.
  int             control; ///< Actual control bit, first swap bit, or target for a Hadamard.
  int             target;  ///< Actual transformed bit or second swap bit.
  double          angle;   ///< Actual controlled phase angle in radians; zero for other gates.
  QuantumState    after;   ///< Actual native simulator amplitudes immediately after this gate.
};
/// @brief Owned actual transform and selected native gate counts.
/// The vector simulation takes O(n*n*2^n) time and exponential state storage, despite the abstract quantum circuit having O(n*n) gates.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::qft_run(dm::algorithms::qstate_zero(2),2).phase_gates); // 1
/// @endcode
struct QFTRun {
  QuantumState state;          ///< Actual complete transformed amplitudes.
  std::size_t  hadamards{0};   ///< Actual native h_gate calls.
  std::size_t  phase_gates{0}; ///< Actual native controlled_phase calls.
  std::size_t  swaps{0};       ///< Actual native swap_qubits calls.
};
/// @brief Owned input, actual post-gate histories and ordinary transform.
/// All post-gate states are copied at actual producing call boundaries; no direct DFT populates history.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::qft_trace(dm::algorithms::qstate_zero(2),2).gates.size()); // 4
/// @endcode
struct QFTTrace {
  QFTRun                   run;     ///< Actual result and native gate counts.
  QuantumState             initial; ///< Owned actual input amplitudes, not normalized or altered.
  std::vector<FourierGate> gates;   ///< Actual gate execution order and post-gate states.
};
/// @brief Apply the native Fourier gate network in place.
/// Uses the existing Hadamard, controlled-phase and bit-swap simulator calls. Forward sign is positive2*pi*i*j*k/N; inverse sign is negative and both use unitary1/sqrt(N) scaling. Complete finite/shape validation precedes gates; n=0 accepts one amplitude. Input need not have unit norm and is not renormalized. Classical work is O(n*n*2^n), ordinary additional storage O(2^n); this does not provide a classical exponential speedup. State is local and no references are retained. Uses default basis policy; successful return replaces state, errors leave it unchanged.
/// @param s Mutable complete finite state vector of length2^n.
/// @param n Counting qubits, nonnegative and below size_t bit width.
/// @throws std::invalid_argument If n is invalid, state length differs from2^n, or an input amplitude is nonfinite.
/// @throws std::length_error If basis/history policy, size arithmetic or representable storage is exceeded.
/// @throws std::overflow_error If a native gate produces a nonfinite amplitude.
/// @par Example
/// @code
/// auto s=dm::algorithms::qstate_zero(1);dm::algorithms::qft(s,1);std::print("{}\n",s.size()); // 2
/// @endcode
/// @par References
/// - M. A. Nielsen and I. L. Chuang, Quantum Computation and Quantum Information, tenth anniversary edition, Cambridge University Press,2010, section5.1 (quantum Fourier transform).
void qft(QuantumState& s, int n);
/// @brief Apply the inverse native Fourier gate network in place.
/// Uses the existing Hadamard, controlled-phase and bit-swap simulator calls. Forward sign is positive2*pi*i*j*k/N; inverse sign is negative and both use unitary1/sqrt(N) scaling. Complete finite/shape validation precedes gates; n=0 accepts one amplitude. Input need not have unit norm and is not renormalized. Classical work is O(n*n*2^n), ordinary additional storage O(2^n); this does not provide a classical exponential speedup. State is local and no references are retained. Uses default basis policy; successful return replaces state, errors leave it unchanged.
/// @param s Mutable complete finite state vector of length2^n.
/// @param n Counting qubits, nonnegative and below size_t bit width.
/// @throws std::invalid_argument If n is invalid, state length differs from2^n, or an input amplitude is nonfinite.
/// @throws std::length_error If basis/history policy, size arithmetic or representable storage is exceeded.
/// @throws std::overflow_error If a native gate produces a nonfinite amplitude.
/// @par Example
/// @code
/// auto s=dm::algorithms::qstate_zero(1);dm::algorithms::inverse_qft(s,1);std::print("{}\n",s.size()); // 2
/// @endcode
/// @par References
/// - M. A. Nielsen and I. L. Chuang, Quantum Computation and Quantum Information, tenth anniversary edition, Cambridge University Press,2010, section5.1 (quantum Fourier transform).
void inverse_qft(QuantumState& s, int n);
/// @brief Execute the same Fourier network with actual selected gate counts.
/// Uses the existing Hadamard, controlled-phase and bit-swap simulator calls. Forward sign is positive2*pi*i*j*k/N; inverse sign is negative and both use unitary1/sqrt(N) scaling. Complete finite/shape validation precedes gates; n=0 accepts one amplitude. Input need not have unit norm and is not renormalized. Classical work is O(n*n*2^n), ordinary additional storage O(2^n); this does not provide a classical exponential speedup. State is local and no references are retained. No gate histories are retained.
/// @param s Borrowed complete finite input, unchanged.
/// @param n Nonnegative qubits matching the complete state length.
/// @param inverse Choose inverse network, defaultfalse.
/// @param options Basis and retained-entry policies.
/// @return Owned actual transform/counts.
/// @throws std::invalid_argument If n is invalid, state length differs from2^n, or an input amplitude is nonfinite.
/// @throws std::length_error If basis/history policy, size arithmetic or representable storage is exceeded.
/// @throws std::overflow_error If a native gate produces a nonfinite amplitude.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::qft_run(dm::algorithms::qstate_zero(2),2).hadamards); // 2
/// @endcode
/// @par References
/// - M. A. Nielsen and I. L. Chuang, Quantum Computation and Quantum Information, tenth anniversary edition, Cambridge University Press,2010, section5.1 (quantum Fourier transform).
QFTRun qft_run(const QuantumState& s, int n, bool inverse = false, QFTOptions options = {});
/// @brief Execute the same Fourier network with actual selected gate counts.
/// Uses the existing Hadamard, controlled-phase and bit-swap simulator calls. Forward sign is positive2*pi*i*j*k/N; inverse sign is negative and both use unitary1/sqrt(N) scaling. Complete finite/shape validation precedes gates; n=0 accepts one amplitude. Input need not have unit norm and is not renormalized. Classical work is O(n*n*2^n), ordinary additional storage O(2^n); this does not provide a classical exponential speedup. State is local and no references are retained. Captured history includes complete states at every gate and can require O(n*n*2^n) entries.
/// @param s Borrowed complete finite input, unchanged.
/// @param n Nonnegative qubits matching the complete state length.
/// @param inverse Choose inverse network, defaultfalse.
/// @param options Basis and retained-entry policies.
/// @return Owned actual transform/counts and post-gate histories.
/// @throws std::invalid_argument If n is invalid, state length differs from2^n, or an input amplitude is nonfinite.
/// @throws std::length_error If basis/history policy, size arithmetic or representable storage is exceeded.
/// @throws std::overflow_error If a native gate produces a nonfinite amplitude.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::qft_trace(dm::algorithms::qstate_zero(2),2).run.hadamards); // 2
/// @endcode
/// @par References
/// - M. A. Nielsen and I. L. Chuang, Quantum Computation and Quantum Information, tenth anniversary edition, Cambridge University Press,2010, section5.1 (quantum Fourier transform).
QFTTrace qft_trace(const QuantumState& s, int n, bool inverse = false, QFTOptions options = {});
/// @brief Construct a normalized finite comb state at offset+k*period.
/// Uses all represented positions below2^n; period need not divide2^n. A nondividing period gives a truncated comb, so exact equally spaced frequency peaks are not promised. O(2^n) storage/initialization; the original N16,period4,offset1 support is1,5,9,13.
/// @param n Nonnegative qubits below size_t bit width.
/// @param period Positive basis-index step.
/// @param offset First support index below2^n.
/// @param options Basis-state policy; history cap is unused.
/// @return Owned normalized equal-amplitude comb state.
/// @throws std::invalid_argument If n,period or offset is invalid.
/// @throws std::length_error If the basis policy or vector storage is exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::qft_comb(4,4,1)[1].real()); // 0.5
/// @endcode
/// @see qft_trace()
QuantumState qft_comb(int n, std::size_t period, std::size_t offset = 0, QFTOptions options = {});
} // namespace datamunge::algorithms
