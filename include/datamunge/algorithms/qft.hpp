#pragma once

/// \file qft.hpp
/// \brief Quantum Fourier Transform on a state-vector register.
///
/// The QFT is the quantum analogue of the discrete Fourier transform: it maps a basis state
/// \f$|j\rangle\f$ to \f$\frac{1}{\sqrt N}\sum_k e^{2\pi i jk/N}|k\rangle\f$ with
/// \f$N=2^n\f$, and by linearity applies the DFT to the amplitude vector of any state. On a
/// quantum computer it costs only \f$O(n^2)\f$ gates -- Hadamards interleaved with controlled
/// phase rotations \f$R_m\f$ of angle \f$2\pi/2^m\f$ -- versus \f$O(N\log N)\f$ for the classical
/// FFT, an exponential gate-count saving that powers phase estimation and Shor's algorithm.
/// This module implements the QFT and its inverse on the shared state-vector simulator.

#include <datamunge/algorithms/quantum_sim.hpp>

namespace datamunge::algorithms {

/// \brief Apply the Quantum Fourier Transform to an \f$n\f$-qubit register (in place).
///
/// Uses the standard Hadamard + controlled-phase network followed by the bit-reversal swaps,
/// so the output amplitude at basis state \c k equals the (unitary, \f$+2\pi i\f$-convention)
/// DFT of the input amplitudes.
inline void qft(QuantumState& s, int n) {
    for (int j = n - 1; j >= 0; --j) {
        h_gate(s, j);
        for (int k = j - 1; k >= 0; --k)
            controlled_phase(s, k, j, M_PI / static_cast<double>(std::size_t{1} << (j - k)));
    }
    for (int i = 0; i < n / 2; ++i) swap_qubits(s, i, n - 1 - i);
}

/// \brief Apply the inverse QFT to an \f$n\f$-qubit register (in place).
inline void inverse_qft(QuantumState& s, int n) {
    for (int i = 0; i < n / 2; ++i) swap_qubits(s, i, n - 1 - i);
    for (int j = 0; j < n; ++j) {
        for (int k = 0; k < j; ++k)
            controlled_phase(s, k, j, -M_PI / static_cast<double>(std::size_t{1} << (j - k)));
        h_gate(s, j);
    }
}

}  // namespace datamunge::algorithms
